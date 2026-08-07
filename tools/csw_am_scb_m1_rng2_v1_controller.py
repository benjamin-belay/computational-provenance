#!/usr/bin/env python3
"""One-shot controller for the sealed RNG2 validation role."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-v1/CONFIG.json"
HEX = frozenset("0123456789abcdef")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(path: Path, commit: str) -> bool:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT
    ) == path.read_bytes()


def open_allocation(config: dict[str, Any], secret: dict[str, Any]) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    envelope = read_json(ROOT / config["bindings"]["sealed_allocation_path"])
    aad = config["controller"]["sealed_aad"].encode()
    if envelope["aad_sha256"] != sha(aad):
        raise ValueError("AAD")
    return json.loads(AESGCM(bytes.fromhex(secret["encryption_key"])).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    ))


def levels(leaves: list[str]) -> list[list[str]]:
    result = [list(leaves)]
    while len(result[-1]) > 1:
        values = list(result[-1])
        if len(values) % 2:
            values.append(values[-1])
        result.append([
            sha(b"NODE\x00" + bytes.fromhex(values[index]) + bytes.fromhex(values[index + 1]))
            for index in range(0, len(values), 2)
        ])
    return result


def proof(tree: list[list[str]], index: int) -> list[dict[str, str]]:
    result = []
    for level in tree[:-1]:
        values = list(level)
        if len(values) % 2:
            values.append(values[-1])
        sibling = index ^ 1
        result.append({
            "side": "LEFT" if sibling < index else "RIGHT",
            "sha256": values[sibling],
        })
        index //= 2
    return result


def release(secret_path: Path, key_path: Path, source_commit: str, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    output, key_path = output.resolve(), key_path.resolve()
    if output.exists() or key_path.exists():
        raise FileExistsError("FINAL_ONE_SHOT_BOUNDARY")
    if len(source_commit) != 40 or any(value not in HEX for value in source_commit):
        raise ValueError("SOURCE_COMMIT")
    checks = {
        name: file_sha(ROOT / config["bindings"][name + "_path"])
        == config["bindings"][name + "_sha256"]
        for name in (
            "ccal1_main", "ccal1_terminal", "frozen_v1", "rng2_r0_main",
            "role_bindings", "sealed_allocation", "agg_s0_config", "detector_protocol",
        )
    }
    checks["source_population"] = (
        file_sha(ROOT / config["source_population"]["path"])
        == config["source_population"]["sha256"]
    )
    checks["checkpoint"] = (
        file_sha(ROOT / config["surface_channel"]["checkpoint_path"])
        == config["surface_channel"]["checkpoint_sha256"]
    )
    checks["source_commit"] = all(
        committed(ROOT / relative, source_commit) for relative in config["source_paths"]
    )
    audit = read_json(ROOT / config["bindings"]["ccal1_main_path"])
    frozen = read_json(ROOT / config["bindings"]["frozen_v1_path"])
    checks["calibration_ready"] = (
        audit["main_verdict"] == config["bindings"]["ccal1_terminal"]
        and audit["authorizations"]["separate_RNG2_V1_request"] is True
        and frozen["T"] == config["thresholds"]["T"]
        and frozen["M"] == config["thresholds"]["M"]
        and frozen["validation_release_authorized"] is False
    )
    if not all(checks.values()):
        raise ValueError("SOURCE_BINDING")

    allocation = open_allocation(config, read_json(secret_path.resolve()))
    records = allocation["roles"][config["role"]["sealed_role"]]
    tree = levels([record["leaf_sha256"] for record in records])
    if len(records) != 512 or tree[-1][0] != config["role"]["membership_root"]:
        raise ValueError("ROLE_ROOT")
    source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(ROOT / config["source_population"]["path"]):
        source[row["closure_id"]].append(row)
    closures, rows = [], []
    for index, record in enumerate(records):
        selected = sorted(source[record["closure_id"]], key=lambda row: row["orientation"])
        if len(selected) != 2 or sha(canonical(selected)) != record["closure_content_sha256"]:
            raise ValueError("CLOSURE_CONTENT")
        closures.append({
            "closure_id": record["closure_id"],
            "closure_content_sha256": record["closure_content_sha256"],
            "leaf_sha256": record["leaf_sha256"],
            "merkle_proof": proof(tree, index),
        })
        rows.extend(selected)
    key = os.urandom(32)
    key_commitment = hmac.new(
        config["controller"]["key_commitment_domain"].encode(), key, hashlib.sha256
    ).hexdigest()
    write_json(key_path, {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.private_key.v1",
        "sampler_key_hex": key.hex(),
        "key_commitment": key_commitment,
    })
    os.chmod(key_path, 0o600)
    output.mkdir(parents=True)
    write_json(output / "AUTHORIZATION.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.authorization.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "checks": checks,
        "role": config["role"]["public_role"],
        "membership_root": tree[-1][0],
        "denominators": config["denominators"],
        "thresholds": config["thresholds"],
        "frozen_v1_sha256": config["bindings"]["frozen_v1_sha256"],
        "release_count": 1,
        "reserve_release": False,
        "receipt_integration_release": False,
        "protected_release": False,
    })
    write_json(output / "KEY_COMMITMENT.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.key_commitment.v1",
        "key_commitment": key_commitment,
        "commitment_domain": config["controller"]["key_commitment_domain"],
        "committed_before_release": True,
    })
    write_rows(output / "VALIDATION_CLOSURES.jsonl", closures)
    write_rows(output / "VALIDATION_ROWS.jsonl", rows)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.release.v1",
        "identity": config["identity"],
        "role": config["role"]["public_role"],
        "release_count": 1,
        "closures": len(closures),
        "oriented_rows": len(rows),
        "membership_root": tree[-1][0],
        "closure_manifest_sha256": file_sha(output / "VALIDATION_CLOSURES.jsonl"),
        "row_manifest_sha256": file_sha(output / "VALIDATION_ROWS.jsonl"),
        "key_commitment_sha256": file_sha(output / "KEY_COMMITMENT.json"),
        "reserve_release": 0,
        "receipt_integration_release": 0,
        "protected_release": 0,
    }
    write_json(output / "RELEASE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-secret", type=Path, required=True)
    parser.add_argument("--sampler-key", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(release(
        args.allocation_secret, args.sampler_key, args.source_commit, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
