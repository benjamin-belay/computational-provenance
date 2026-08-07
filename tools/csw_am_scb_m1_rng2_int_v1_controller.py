#!/usr/bin/env python3
"""One-shot controller for the sealed receipt-bound integration validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-int-v1/CONFIG.json"
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


def key_commitment(key: bytes, domain: str) -> str:
    return sha(domain.encode() + b"\x00" + key)


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


def merkle_root(leaves: list[str]) -> str:
    values = list(leaves)
    while len(values) > 1:
        if len(values) % 2:
            values.append(values[-1])
        values = [
            sha(b"NODE\x00" + bytes.fromhex(values[i]) + bytes.fromhex(values[i + 1]))
            for i in range(0, len(values), 2)
        ]
    return values[0]


def release(secret_arg: Path, source_commit: str, output_arg: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    output = output_arg.resolve()
    receipt_path = Path(config["controller"]["receipt_key_path"])
    sampler_path = Path(config["controller"]["sampler_key_path"])
    if output.exists() or receipt_path.exists() or sampler_path.exists():
        raise FileExistsError("FINAL_ONE_SHOT_BOUNDARY")
    if len(source_commit) != 40 or any(c not in HEX for c in source_commit):
        raise ValueError("SOURCE_COMMIT")
    checks = {}
    for name in (
        "S0_main", "D0_main", "D0_terminal", "D0_manifest", "D0_production",
        "D0_reference", "sealed_allocation", "role_commitments",
    ):
        checks[name] = (
            file_sha(ROOT / config["bindings"][name + "_path"])
            == config["bindings"][name + "_sha256"]
        )
    checks["source_population"] = (
        file_sha(ROOT / config["source_population"]["path"])
        == config["source_population"]["sha256"]
    )
    checks["source_commit"] = all(
        committed(ROOT / relative, source_commit) for relative in config["source_paths"]
    )
    s0 = read_json(ROOT / config["bindings"]["S0_main_path"])
    d0 = read_json(ROOT / config["bindings"]["D0_main_path"])
    checks["lineage"] = (
        s0["main_verdict"] == config["bindings"]["S0_terminal"]
        and d0["main_verdict"] == config["bindings"]["D0_terminal"]
        and d0["authorizations"]["separate_INT_V1_request"] is True
        and s0["authorizations"]["protected_release"] is False
    )
    if not all(checks.values()):
        raise ValueError("SOURCE_BINDING")

    allocation = decrypt(config, read_json(secret_arg.resolve()))
    records = allocation["roles"][config["role"]["sealed_role"]]
    if (
        len(records) != config["role"]["closures"]
        or merkle_root([row["leaf_sha256"] for row in records])
        != config["role"]["membership_root"]
    ):
        raise ValueError("ROLE_ROOT")
    source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(ROOT / config["source_population"]["path"]):
        source[row["closure_id"]].append(row)
    closures, all_rows, execution_rows = [], [], []
    for record in records:
        rows = sorted(source[record["closure_id"]], key=lambda row: row["orientation"])
        if len(rows) != 2 or sha(canonical(rows)) != record["closure_content_sha256"]:
            raise ValueError("CLOSURE_CONTENT")
        selected = [
            row for row in rows
            if row["orientation"] == config["role"]["registered_orientation"]
        ]
        if len(selected) != 1:
            raise ValueError("REGISTERED_ORIENTATION")
        closures.append({
            "closure_id": record["closure_id"],
            "closure_content_sha256": record["closure_content_sha256"],
            "leaf_sha256": record["leaf_sha256"],
        })
        all_rows.extend(rows)
        execution_rows.extend(selected)

    receipt_key, sampler_key = os.urandom(32), os.urandom(32)
    for path, key in ((receipt_path, receipt_key), (sampler_path, sampler_key)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    output.mkdir(parents=True)
    write_json(output / "AUTHORIZATION.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.authorization.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "checks": checks,
        "role": config["role"]["public_role"],
        "membership_root": config["role"]["membership_root"],
        "denominators": config["denominators"],
        "thresholds": {"T": config["carrier"]["T"], "M": config["carrier"]["M"]},
        "protected_release": False,
    })
    write_json(output / "KEY_COMMITMENTS.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.key_commitments.v1",
        "receipt_key_commitment": key_commitment(
            receipt_key, config["controller"]["receipt_commitment_domain"]
        ),
        "sampler_key_commitment": key_commitment(
            sampler_key, config["controller"]["sampler_commitment_domain"]
        ),
        "committed_before_release": True,
        "keys_revealed": False,
    })
    write_rows(output / "VALIDATION_CLOSURES.jsonl", closures)
    write_rows(output / "RELEASED_ORIENTED_ROWS.jsonl", all_rows)
    write_rows(output / "REGISTERED_EXECUTION_ROWS.jsonl", execution_rows)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.release.v1",
        "identity": config["identity"],
        "role": config["role"]["public_role"],
        "release_count": 1,
        "closures": len(closures),
        "oriented_rows_released": len(all_rows),
        "registered_execution_rows": len(execution_rows),
        "membership_root": config["role"]["membership_root"],
        "closure_manifest_sha256": file_sha(output / "VALIDATION_CLOSURES.jsonl"),
        "released_rows_sha256": file_sha(output / "RELEASED_ORIENTED_ROWS.jsonl"),
        "execution_rows_sha256": file_sha(output / "REGISTERED_EXECUTION_ROWS.jsonl"),
        "key_commitments_sha256": file_sha(output / "KEY_COMMITMENTS.json"),
        "protected_release": 0,
    }
    write_json(output / "RELEASE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-secret", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(release(
        args.allocation_secret, args.source_commit, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
