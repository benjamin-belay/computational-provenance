#!/usr/bin/env python3
"""Independent controller reconstruction of the root-preserving RNG2 migration."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]


def canon(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


def root(leaves: list[str]) -> str:
    values = list(leaves)
    while len(values) > 1:
        if len(values) % 2:
            values.append(values[-1])
        values = [
            sha(b"NODE\x00" + bytes.fromhex(values[i]) + bytes.fromhex(values[i + 1]))
            for i in range(0, len(values), 2)
        ]
    return values[0]


def assess(evidence: Path, secret_path: Path) -> dict[str, Any]:
    config = obj(ROOT / "protocol/csw-am-scb-m1-rng2-r0/CONFIG.json")
    old_config = obj(ROOT / config["lineage"]["agg_r0_config_path"])
    secrets = obj(secret_path)
    envelope = obj(ROOT / config["lineage"]["old_full_allocation_path"])
    plain = AESGCM(bytes.fromhex(secrets["encryption_key"])).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        config["controller"]["old_aad"].encode(),
    )
    sealed = json.loads(plain)
    old_commitments = obj(ROOT / config["lineage"]["old_role_commitments_path"])
    bindings = obj(evidence / "ROLE_MIGRATION_BINDINGS.json")
    attestation = obj(evidence / "CONTROLLER_ATTESTATION.json")
    exposure = obj(evidence / "EXPOSURE_STATUS.json")
    authorization = obj(evidence / "DEVELOPMENT_AUTHORIZATION.json")
    official = obj(evidence / "OFFICIAL_RESULT.json")
    recomputed_roots = {}
    all_ids = set()
    duplicate = False
    for role, records in sealed["roles"].items():
        salt = bytes.fromhex(sealed["role_salts"][role])
        leaves = []
        for record in records:
            closure_id = record["closure_id"]
            duplicate = duplicate or closure_id in all_ids
            all_ids.add(closure_id)
            preimage = {
                key: record[key]
                for key in (
                    "protocol_identity", "task_identity", "role", "closure_id",
                    "closure_content_sha256", "schema_version",
                )
            }
            leaves.append(hmac.new(
                salt,
                old_config["secret_material"]["role_leaf_domain"].encode()
                + b"\x00" + canon(preimage),
                hashlib.sha256,
            ).hexdigest())
        recomputed_roots[role] = root(leaves)
    binding_by_old = {row["old_role"]: row for row in bindings["bindings"]}
    roots_inherited = all(
        binding_by_old[role]["inherited_membership_root"]
        == recomputed_roots[role]
        == old_commitments["roles"][role]["merkle_root"]
        for role in recomputed_roots
    )
    binding_root = sha(
        config["controller"]["full_binding_domain"].encode()
        + b"\x00" + canon(bindings["bindings"])
    )
    attestation_preimage = {
        key: value for key, value in attestation.items()
        if key != "attestation_tag"
    }
    attestation_key = hmac.new(
        bytes.fromhex(secrets["encryption_key"]),
        config["controller"]["attestation_key_domain"].encode(),
        hashlib.sha256,
    ).digest()
    tag = hmac.new(
        attestation_key,
        config["controller"]["attestation_domain"].encode()
        + b"\x00" + canon(attestation_preimage),
        hashlib.sha256,
    ).hexdigest()
    public = b"\n".join(
        path.read_bytes() for path in evidence.iterdir() if path.is_file()
    )
    future_ids = {
        record["closure_id"]
        for role in ("AGG_CALIBRATION", "AGG_VALIDATION", "PUBLIC_RESERVE")
        for record in sealed["roles"][role]
    }
    gates = {
        "complete_partition": len(all_ids) == 1920 and not duplicate,
        "role_counts": Counter(sealed["assignments"].values())
        == {
            "AGG_DEVELOPMENT": 256,
            "AGG_CALIBRATION": 512,
            "AGG_VALIDATION": 512,
            "PUBLIC_RESERVE": 640,
        },
        "roots_inherited": roots_inherited,
        "binding_root": binding_root
        == bindings["full_migration_binding_root"]
        == attestation["full_migration_binding_root"],
        "attestation": tag == attestation["attestation_tag"],
        "future_ids_absent": not any(identifier.encode() in public for identifier in future_ids),
        "exposure": (
            exposure["roles"]["RNG2_DEVELOPMENT"]["exposure"] == "PUBLICLY_RELEASED"
            and all(
                exposure["roles"][role]["exposure"] == "SEALED_UNSEEN"
                for role in (
                    "RNG2_CALIBRATION", "RNG2_VALIDATION", "RNG2_PUBLIC_RESERVE"
                )
            )
        ),
        "no_new_release": all(
            not value["new_identity_release"] for value in exposure["roles"].values()
        ),
        "development_only": (
            authorization["authorized_use"]
            == "RNG2_D0_PUBLIC_IMPLEMENTATION_AND_DEBUGGING_ONLY"
            and not authorization["calibration_access"]
            and not authorization["validation_access"]
            and not authorization["reserve_access"]
            and not authorization["protected_access"]
        ),
        "official": (
            official["passed"]
            and official["RNG2_D0_authorized"]
            and not official["RNG2_CAL1_authorized"]
            and not official["protected_release_authorized"]
        ),
    }
    valid = all(gates.values())
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_r0.reference.v1",
        "gates": gates,
        "valid": valid,
        "role_counts": attestation["role_counts"],
        "full_migration_binding_root": binding_root,
        "new_identity_releases": 0,
        "protected_access": 0,
        "terminal": (
            "MAIN_CSW_AM_SCB_M1_RNG2_R0_PUBLIC_ROLE_MIGRATION_READY"
            if valid else "MAIN_CSW_AM_SCB_M1_RNG2_R0_IMPLEMENTATION_INVALID"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--secret-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.evidence, args.secret_path)
    data = canon(result) + b"\n"
    if args.output:
        if args.output.exists():
            raise ValueError("OUTPUT_BOUNDARY")
        args.output.write_bytes(data)
    else:
        print(data.decode(), end="")


if __name__ == "__main__":
    main()
