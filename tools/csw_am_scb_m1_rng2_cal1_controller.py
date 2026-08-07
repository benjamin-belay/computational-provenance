#!/usr/bin/env python3
"""One-shot controller release for RNG2-CAL1."""
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
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rng2-cal1/CONFIG.json"
HEX = frozenset("0123456789abcdef")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("JSON_OBJECT:" + str(path))
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
    relative = str(path.relative_to(ROOT))
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    ) == path.read_bytes()


def merkle_levels(leaves: list[str]) -> list[list[str]]:
    if not leaves:
        raise ValueError("EMPTY_MERKLE")
    levels = [leaves]
    while len(levels[-1]) > 1:
        values = list(levels[-1])
        if len(values) % 2:
            values.append(values[-1])
        levels.append([
            sha(b"NODE\x00" + bytes.fromhex(values[index])
                + bytes.fromhex(values[index + 1]))
            for index in range(0, len(values), 2)
        ])
    return levels


def merkle_proof(levels: list[list[str]], index: int) -> list[dict[str, str]]:
    proof = []
    for level in levels[:-1]:
        values = list(level)
        if len(values) % 2:
            values.append(values[-1])
        sibling = index ^ 1
        proof.append({
            "side": "LEFT" if sibling < index else "RIGHT",
            "sha256": values[sibling],
        })
        index //= 2
    return proof


def role_leaf(old_config: dict[str, Any], salt: bytes, record: dict[str, Any]) -> str:
    preimage = {
        key: record[key]
        for key in (
            "protocol_identity", "task_identity", "role", "closure_id",
            "closure_content_sha256", "schema_version",
        )
    }
    return hmac.new(
        salt,
        old_config["secret_material"]["role_leaf_domain"].encode()
        + b"\x00" + canonical(preimage),
        hashlib.sha256,
    ).hexdigest()


def open_allocation(config: dict[str, Any], secret: dict[str, Any]) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    envelope = read_json(ROOT / config["controller"]["old_full_allocation_path"])
    aad = config["controller"]["old_aad"].encode()
    if envelope["algorithm"] != "AES-256-GCM" or envelope["aad_sha256"] != sha(aad):
        raise ValueError("SEALED_ALLOCATION_ENVELOPE")
    plain = AESGCM(bytes.fromhex(secret["encryption_key"])).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    )
    value = json.loads(plain)
    if type(value) is not dict:
        raise ValueError("SEALED_ALLOCATION_OBJECT")
    return value


def validate_bindings(config: dict[str, Any], source_commit: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name in (
        "CAL_S0_main_audit", "CAL_S0_terminal", "calibration_protocol",
        "denominators", "chronology", "role_bindings", "exposure_status",
        "D0_config", "D0_sampler", "D0_reference", "D0_executable_manifest",
        "D0_main_audit", "D0_terminal",
    ):
        checks[name] = (
            file_sha(ROOT / config["bindings"][f"{name}_path"])
            == config["bindings"][f"{name}_sha256"]
        )
    for name in ("old_config", "old_full_allocation", "old_role_commitments"):
        checks[name] = (
            file_sha(ROOT / config["controller"][f"{name}_path"])
            == config["controller"][f"{name}_sha256"]
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
    s0 = read_json(ROOT / config["bindings"]["CAL_S0_main_audit_path"])
    d0 = read_json(ROOT / config["bindings"]["D0_main_audit_path"])
    checks["lineage_ready"] = (
        s0["main_verdict"] == config["bindings"]["CAL_S0_terminal"]
        and d0["main_verdict"] == "MAIN_CSW_AM_SCB_M1_RNG2_D0_IMPLEMENTATION_READY"
    )
    role_bindings = read_json(ROOT / config["bindings"]["role_bindings_path"])
    calibration = [
        row for row in role_bindings["bindings"]
        if row["new_role"] == "RNG2_CALIBRATION"
    ]
    checks["role_binding"] = (
        len(calibration) == 1
        and calibration[0]["closures"] == 512
        and calibration[0]["exposure"] == "SEALED_UNSEEN"
        and calibration[0]["prior_execution"] == "NONE"
        and calibration[0]["inherited_membership_root"]
        == "7518f095e22dafe7c7bd5fb7baaf05818e77c0c823aab4b851fb01ef79af311a"
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
    allocation_secret_path: Path,
    calibration_key_path: Path,
    source_commit: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or calibration_key_path.exists():
        raise FileExistsError("ONE_SHOT_BOUNDARY")
    config = read_json(CONFIG_PATH)
    checks = validate_bindings(config, source_commit)
    if not all(checks.values()):
        raise ValueError("SOURCE_OR_LINEAGE_BINDING")
    allocation_secret = read_json(allocation_secret_path)
    sealed = open_allocation(config, allocation_secret)
    old_config = read_json(ROOT / config["controller"]["old_config_path"])
    commitments = read_json(ROOT / config["controller"]["old_role_commitments_path"])
    records = sealed["roles"]["AGG_CALIBRATION"]
    salt = bytes.fromhex(sealed["role_salts"]["AGG_CALIBRATION"])
    leaves = [role_leaf(old_config, salt, row) for row in records]
    levels = merkle_levels(leaves)
    expected_root = commitments["roles"]["AGG_CALIBRATION"]["merkle_root"]
    if (
        len(records) != 512
        or levels[-1][0] != expected_root
        or expected_root
        != "7518f095e22dafe7c7bd5fb7baaf05818e77c0c823aab4b851fb01ef79af311a"
        or any(leaf != row["leaf_sha256"] for leaf, row in zip(leaves, records))
    ):
        raise ValueError("CALIBRATION_ROLE_ROOT")

    source_by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(ROOT / config["source_population"]["path"]):
        source_by_closure[row["closure_id"]].append(row)
    release_closures, release_rows = [], []
    for index, record in enumerate(records):
        rows = sorted(
            source_by_closure[record["closure_id"]],
            key=lambda row: row["orientation"],
        )
        if (
            len(rows) != 2
            or sha(canonical(rows)) != record["closure_content_sha256"]
        ):
            raise ValueError("CALIBRATION_CLOSURE_CONTENT")
        release_closures.append({
            "closure_id": record["closure_id"],
            "closure_content_sha256": record["closure_content_sha256"],
            "leaf_sha256": record["leaf_sha256"],
            "merkle_proof": merkle_proof(levels, index),
        })
        release_rows.extend(rows)
    calibration_ids = {row["closure_id"] for row in records}
    forbidden_ids = {
        row["closure_id"]
        for role in ("AGG_VALIDATION", "PUBLIC_RESERVE")
        for row in sealed["roles"][role]
    }
    if not calibration_ids.isdisjoint(forbidden_ids):
        raise ValueError("ROLE_OVERLAP")

    key = os.urandom(32)
    key_commitment = hmac.new(
        config["controller"]["key_commitment_domain"].encode(),
        key,
        hashlib.sha256,
    ).hexdigest()
    private = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.private_key.v1",
        "identity": config["identity"],
        "sampler_key_hex": key.hex(),
        "key_commitment": key_commitment,
        "revealed": False,
    }
    calibration_key_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(calibration_key_path, private)
    os.chmod(calibration_key_path, 0o600)
    output.mkdir(parents=True)
    authorization = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.authorization_record.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "binding_checks": checks,
        "authorized_role": "RNG2_CALIBRATION",
        "authorized_closures": 512,
        "authorized_root": expected_root,
        "release_number": 1,
        "validation_authorized": False,
        "reserve_authorized": False,
        "protected_authorized": False,
    }
    key_record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.key_commitment.v1",
        "identity": config["identity"],
        "algorithm": "HMAC-SHA256",
        "commitment_domain": config["controller"]["key_commitment_domain"],
        "key_commitment": key_commitment,
        "key_revealed": False,
        "committed_before_release": True,
    }
    write_json(output / "AUTHORIZATION.json", authorization)
    write_json(output / "KEY_COMMITMENT.json", key_record)
    write_rows(output / "CALIBRATION_CLOSURES.jsonl", release_closures)
    write_rows(output / "CALIBRATION_ROWS.jsonl", release_rows)
    release_record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.release.v1",
        "identity": config["identity"],
        "role": "RNG2_CALIBRATION",
        "release_number": 1,
        "closures": len(release_closures),
        "oriented_rows": len(release_rows),
        "membership_root": expected_root,
        "closure_manifest_sha256": file_sha(output / "CALIBRATION_CLOSURES.jsonl"),
        "row_manifest_sha256": file_sha(output / "CALIBRATION_ROWS.jsonl"),
        "key_commitment_sha256": file_sha(output / "KEY_COMMITMENT.json"),
        "future_role_identities_exported": 0,
        "validation_release": 0,
        "reserve_release": 0,
        "protected_release": 0,
    }
    public_bytes = b"".join(path.read_bytes() for path in output.iterdir())
    if any(closure_id.encode() in public_bytes for closure_id in forbidden_ids):
        raise ValueError("FUTURE_IDENTITY_LEAK")
    write_json(output / "RELEASE.json", release_record)
    return release_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation-secret", type=Path, required=True)
    parser.add_argument("--calibration-key", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(release(
        args.allocation_secret, args.calibration_key, args.source_commit, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
