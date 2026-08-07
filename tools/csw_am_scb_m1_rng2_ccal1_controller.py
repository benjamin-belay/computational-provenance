#!/usr/bin/env python3
"""One-shot release controller for RNG2-CCAL1."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-ccal1/CONFIG.json"
HEX = frozenset("0123456789abcdef")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


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


def decrypt(config: dict[str, Any], secret: dict[str, Any]) -> dict[str, Any]:
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
            sha(b"NODE\x00" + bytes.fromhex(values[index])
                + bytes.fromhex(values[index + 1]))
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


def validate(config: dict[str, Any], source_commit: str) -> dict[str, bool]:
    checks = {}
    for name in (
        "R0_main_audit", "R0_terminal", "sealed_allocation",
        "role_commitments", "D0_main_audit", "D0_terminal",
        "D0_executable_manifest", "corrected_wrapper", "calibration_protocol",
        "denominators", "error_control", "chronology",
    ):
        checks[name] = (
            file_sha(ROOT / config["bindings"][f"{name}_path"])
            == config["bindings"][f"{name}_sha256"]
        )
    checks["source_population"] = (
        file_sha(ROOT / config["source_population"]["path"])
        == config["source_population"]["sha256"]
    )
    for name in ("source", "checkpoint"):
        checks["surface_" + name] = (
            file_sha(ROOT / config["surface_channel"][f"{name}_path"])
            == config["surface_channel"][f"{name}_sha256"]
        )
    r0 = read_json(ROOT / config["bindings"]["R0_main_audit_path"])
    d0 = read_json(ROOT / config["bindings"]["D0_main_audit_path"])
    checks["lineage_ready"] = (
        r0["main_verdict"] == config["bindings"]["R0_terminal"]
        and d0["main_verdict"] == config["bindings"]["D0_terminal"]
        and r0["authorizations"]["clean_calibration_release"] is False
        and d0["authorizations"]["clean_calibration_release"] is False
    )
    if (
        len(source_commit) != 40
        or any(character not in HEX for character in source_commit)
    ):
        raise ValueError("SOURCE_COMMIT")
    checks["source_commit"] = all(
        committed(ROOT / relative, source_commit)
        for relative in config["source_paths"]
    )
    return checks


def release(
    allocation_secret_path: Path, sampler_key_path: Path,
    source_commit: str, output_argument: Path,
) -> dict[str, Any]:
    output = output_argument.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    sampler_key_path = sampler_key_path.expanduser().resolve()
    if output.exists() or sampler_key_path.exists():
        raise FileExistsError("FINAL_ONE_SHOT_BOUNDARY")
    config = read_json(CONFIG)
    checks = validate(config, source_commit)
    if not all(checks.values()):
        raise ValueError("SOURCE_BINDING")
    allocation = decrypt(config, read_json(allocation_secret_path.expanduser().resolve()))
    role = config["role"]["name"]
    records = allocation["roles"][role]
    tree = levels([record["leaf_sha256"] for record in records])
    if (
        len(records) != 512
        or tree[-1][0] != config["role"]["merkle_root"]
    ):
        raise ValueError("ROLE_ROOT")
    source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(ROOT / config["source_population"]["path"]):
        source[row["closure_id"]].append(row)
    closures, rows = [], []
    for index, record in enumerate(records):
        closure_rows = sorted(
            source[record["closure_id"]], key=lambda row: row["orientation"]
        )
        if (
            len(closure_rows) != 2
            or sha(canonical(closure_rows)) != record["closure_content_sha256"]
        ):
            raise ValueError("CLOSURE_CONTENT")
        closures.append({
            "closure_id": record["closure_id"],
            "closure_content_sha256": record["closure_content_sha256"],
            "leaf_sha256": record["leaf_sha256"],
            "merkle_proof": proof(tree, index),
        })
        rows.extend(closure_rows)
    forbidden = {
        record["closure_id"]
        for record in allocation["roles"]["RNG2_REMAINING_PUBLIC_RESERVE"]
    }
    key = os.urandom(32)
    key_commitment = hmac.new(
        config["controller"]["key_commitment_domain"].encode(),
        key, hashlib.sha256,
    ).hexdigest()
    private = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.private_key.v1",
        "sampler_key_hex": key.hex(),
        "key_commitment": key_commitment,
    }
    sampler_key_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(sampler_key_path, private)
    os.chmod(sampler_key_path, 0o600)
    output.mkdir(parents=True)
    write_json(output / "AUTHORIZATION.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.authorization.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "checks": checks,
        "role": role,
        "closures": 512,
        "oriented_rows": 1024,
        "membership_root": tree[-1][0],
        "denominators": config["denominators"],
        "release_count": 1,
        "validation_release": False,
        "remaining_reserve_release": False,
        "protected_release": False,
    })
    write_json(output / "KEY_COMMITMENT.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.key_commitment.v1",
        "key_commitment": key_commitment,
        "commitment_domain": config["controller"]["key_commitment_domain"],
        "committed_before_release": True,
        "key_revealed": False,
    })
    write_rows(output / "CALIBRATION_CLOSURES.jsonl", closures)
    write_rows(output / "CALIBRATION_ROWS.jsonl", rows)
    public_bytes = b"".join(path.read_bytes() for path in output.iterdir())
    if any(closure_id.encode() in public_bytes for closure_id in forbidden):
        raise ValueError("FUTURE_IDENTITY_LEAK")
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.release.v1",
        "identity": config["identity"],
        "role": role,
        "release_count": 1,
        "closures": len(closures),
        "oriented_rows": len(rows),
        "membership_root": tree[-1][0],
        "closure_manifest_sha256": file_sha(output / "CALIBRATION_CLOSURES.jsonl"),
        "row_manifest_sha256": file_sha(output / "CALIBRATION_ROWS.jsonl"),
        "key_commitment_sha256": file_sha(output / "KEY_COMMITMENT.json"),
        "validation_release": 0,
        "remaining_reserve_release": 0,
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
