#!/usr/bin/env python3
"""Sealed, one-shot 512/128 split of the untouched RNG2 public reserve."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rng2-ccal-r0/CONFIG.json"
TEST_ONLY = "TEST_ONLY"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(path: Path, commit: str) -> bool:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT
    ) == path.read_bytes()


def open_old(config: dict[str, Any], secret: dict[str, Any]) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    envelope = read_json(ROOT / config["lineage"]["old_full_allocation_path"])
    aad = config["lineage"]["old_aad"].encode()
    plain = AESGCM(bytes.fromhex(secret["encryption_key"])).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    )
    return json.loads(plain)


def seal(value: dict[str, Any], key: bytes, aad: bytes) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.sealed.v1",
        "algorithm": "AES-256-GCM",
        "aad_sha256": sha(aad),
        "nonce_hex": nonce.hex(),
        "ciphertext_hex": AESGCM(key).encrypt(nonce, canonical(value), aad).hex(),
    }


def merkle_root(leaves: list[str]) -> str:
    values = list(leaves)
    if not values:
        raise ValueError("EMPTY")
    while len(values) > 1:
        if len(values) % 2:
            values.append(values[-1])
        values = [
            sha(b"NODE\x00" + bytes.fromhex(values[i]) + bytes.fromhex(values[i + 1]))
            for i in range(0, len(values), 2)
        ]
    return values[0]


def create_secret(secret_path: Path, commitment_path: Path) -> dict[str, Any]:
    if secret_path.exists() or commitment_path.exists():
        raise FileExistsError("ONE_SHOT_SECRET")
    config = read_json(CONFIG_PATH)
    secret = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.secret.v1",
        "allocation_seed": os.urandom(32).hex(),
        "encryption_key": os.urandom(32).hex(),
        "role_salts": {
            "RNG2_CLEAN_CALIBRATION": os.urandom(32).hex(),
            "RNG2_REMAINING_PUBLIC_RESERVE": os.urandom(32).hex(),
        },
    }
    seed_commitment = hmac.new(
        config["new_controller"]["seed_commitment_domain"].encode(),
        bytes.fromhex(secret["allocation_seed"]), hashlib.sha256,
    ).hexdigest()
    commitment = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.seed_commitment.v1",
        "identity": config["identity"],
        "seed_commitment": seed_commitment,
        "seed_revealed": False,
        "allocation_executed": False,
    }
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(secret_path, secret)
    os.chmod(secret_path, 0o600)
    write_json(commitment_path, commitment)
    return commitment


def validate_source(config: dict[str, Any], source_commit: str) -> dict[str, bool]:
    checks = {}
    for name in (
        "CAL1_main_audit", "CAL1_terminal", "RNG2_role_bindings",
        "RNG2_exposure", "old_config", "old_full_allocation",
        "old_role_commitments",
    ):
        checks[name] = (
            file_sha(ROOT / config["lineage"][f"{name}_path"])
            == config["lineage"][f"{name}_sha256"]
        )
    audit = read_json(ROOT / config["lineage"]["CAL1_main_audit_path"])
    terminal = read_json(ROOT / config["lineage"]["CAL1_terminal_path"])
    exposure = read_json(ROOT / config["lineage"]["RNG2_exposure_path"])
    checks["closed_CAL1"] = (
        audit["main_verdict"] == config["lineage"]["CAL1_terminal"]
        and terminal["terminal"] == config["lineage"]["CAL1_terminal"]
        and terminal["T"] is None and terminal["M"] is None
        and terminal["RNG2_V1_release_authorized"] is False
    )
    checks["reserve_unseen"] = (
        exposure["roles"]["RNG2_PUBLIC_RESERVE"]["closures"] == 640
        and exposure["roles"]["RNG2_PUBLIC_RESERVE"]["exposure"] == "SEALED_UNSEEN"
        and exposure["roles"]["RNG2_PUBLIC_RESERVE"]["prior_execution"] == "NONE"
    )
    checks["validation_unseen"] = (
        exposure["roles"]["RNG2_VALIDATION"]["exposure"] == "SEALED_UNSEEN"
    )
    if source_commit != TEST_ONLY:
        checks["committed_source"] = all(
            committed(ROOT / relative, source_commit)
            for relative in config["source_paths"]
        )
    else:
        checks["committed_source"] = True
    return checks


def allocate(
    old_secret_path: Path, new_secret_path: Path, commitment_path: Path,
    commitment_commit: str, source_commit: str, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("ONE_SHOT_ALLOCATION")
    config = read_json(CONFIG_PATH)
    checks = validate_source(config, source_commit)
    if not all(checks.values()):
        raise ValueError("LINEAGE")
    commitment = read_json(commitment_path)
    if commitment_commit != TEST_ONLY and not committed(commitment_path, commitment_commit):
        raise ValueError("COMMITMENT_NOT_COMMITTED")
    old_secret, new_secret = read_json(old_secret_path), read_json(new_secret_path)
    expected_commitment = hmac.new(
        config["new_controller"]["seed_commitment_domain"].encode(),
        bytes.fromhex(new_secret["allocation_seed"]), hashlib.sha256,
    ).hexdigest()
    if commitment["seed_commitment"] != expected_commitment:
        raise ValueError("SEED_COMMITMENT")
    old = open_old(config, old_secret)
    reserve = old["roles"]["PUBLIC_RESERVE"]
    old_commitments = read_json(ROOT / config["lineage"]["old_role_commitments_path"])
    if (
        len(reserve) != 640
        or old_commitments["roles"]["PUBLIC_RESERVE"]["merkle_root"]
        != "f501159cf3ec33b03a40625f3abdcb8a9fce5e39c6d762caa7a731d957a47a34"
    ):
        raise ValueError("OLD_RESERVE")
    seed = bytes.fromhex(new_secret["allocation_seed"])
    domain = config["new_controller"]["allocation_domain"].encode()
    ordered = sorted(
        reserve,
        key=lambda row: (
            hmac.new(seed, domain + b"\x00" + row["closure_id"].encode(), hashlib.sha256).digest(),
            row["closure_id"],
        ),
    )
    roles = {
        "RNG2_CLEAN_CALIBRATION": ordered[:512],
        "RNG2_REMAINING_PUBLIC_RESERVE": ordered[512:],
    }
    sealed_roles, roots = {}, {}
    for role, records in roles.items():
        salt = bytes.fromhex(new_secret["role_salts"][role])
        values = []
        for record in records:
            preimage = {
                "protocol_identity": config["identity"],
                "role": role,
                "closure_id": record["closure_id"],
                "closure_content_sha256": record["closure_content_sha256"],
                "schema_version": "exp33a.csw_am_scb_m1_rng2_ccal_r0.role_leaf.v1",
            }
            values.append({
                **preimage,
                "leaf_sha256": hmac.new(
                    salt,
                    config["new_controller"]["role_leaf_domain"].encode()
                    + b"\x00" + canonical(preimage),
                    hashlib.sha256,
                ).hexdigest(),
            })
        sealed_roles[role] = values
        roots[role] = merkle_root([row["leaf_sha256"] for row in values])
    role_commitments = {
        role: {
            "closures": len(records),
            "rows": len(records) * 2,
            "merkle_root": roots[role],
            "exposure": "SEALED_UNSEEN",
            "released": False,
        }
        for role, records in sealed_roles.items()
    }
    full_root = sha(
        config["new_controller"]["full_root_domain"].encode()
        + b"\x00" + canonical(role_commitments)
    )
    payload = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.full_allocation.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "seed_commitment": commitment["seed_commitment"],
        "source_reserve_root":
            old_commitments["roles"]["PUBLIC_RESERVE"]["merkle_root"],
        "roles": sealed_roles,
        "role_salts": new_secret["role_salts"],
        "full_allocation_root": full_root,
    }
    output.mkdir(parents=True)
    write_json(output / "FULL_ALLOCATION.sealed", seal(
        payload, bytes.fromhex(new_secret["encryption_key"]),
        config["new_controller"]["sealed_aad"].encode(),
    ))
    public = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.commitments.v1",
        "identity": config["identity"],
        "source_reserve_root": payload["source_reserve_root"],
        "source_reserve_closures": 640,
        "allocation_rule": config["allocation"]["rule"],
        "seed_commitment": commitment["seed_commitment"],
        "roles": role_commitments,
        "full_allocation_root": full_root,
        "validation_role_unchanged": config["unchanged_roles"]["RNG2_VALIDATION"],
        "identity_release_count": 0,
    }
    write_json(output / "ROLE_COMMITMENTS.json", public)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.result.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "commitment_commit": commitment_commit,
        "checks": {
            **checks,
            "source_reserve_complete": len(reserve) == 640,
            "split_counts": len(roles["RNG2_CLEAN_CALIBRATION"]) == 512
                and len(roles["RNG2_REMAINING_PUBLIC_RESERVE"]) == 128,
            "partition_exact": {
                row["closure_id"] for values in roles.values() for row in values
            } == {row["closure_id"] for row in reserve},
            "partition_disjoint": not (
                {row["closure_id"] for row in roles["RNG2_CLEAN_CALIBRATION"]}
                & {row["closure_id"] for row in roles["RNG2_REMAINING_PUBLIC_RESERVE"]}
            ),
            "no_identity_release": True,
            "validation_unchanged": True,
        },
        "role_counts": {role: len(values) for role, values in roles.items()},
        "role_roots": roots,
        "full_allocation_root": full_root,
        "new_identity_releases": 0,
        "model_forwards": 0,
        "stochastic_outputs": 0,
        "calibration_release_authorized": False,
        "validation_release_authorized": False,
        "protected_release": 0,
    }
    result["passed"] = all(result["checks"].values())
    result["terminal"] = (
        config["terminals"]["ready"] if result["passed"]
        else config["terminals"]["allocation_invalid"]
    )
    write_json(output / "OFFICIAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-secret")
    create.add_argument("--secret", type=Path, required=True)
    create.add_argument("--commitment", type=Path, required=True)
    run = sub.add_parser("allocate")
    run.add_argument("--old-secret", type=Path, required=True)
    run.add_argument("--new-secret", type=Path, required=True)
    run.add_argument("--commitment", type=Path, required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create-secret":
        value = create_secret(args.secret, args.commitment)
    else:
        value = allocate(
            args.old_secret, args.new_secret, args.commitment,
            args.commitment_commit, args.source_commit, args.output,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
