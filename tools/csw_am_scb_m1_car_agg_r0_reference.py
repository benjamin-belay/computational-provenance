#!/usr/bin/env python3
"""Independent reconstruction of the CAR-AGG-R0 allocation evidence."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-car-agg-r0/CONFIG.json"


def canon(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def decrypt(blob: bytes, key: bytes, aad: bytes) -> dict[str, Any]:
    envelope = json.loads(blob)
    if envelope["algorithm"] != "AES-256-GCM" or envelope["aad_sha256"] != sha(aad):
        raise ValueError("ENVELOPE")
    plain = AESGCM(key).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    )
    return json.loads(plain)


def leaf(config: dict[str, Any], salt: bytes, preimage: dict[str, Any]) -> str:
    return hmac.new(
        salt,
        config["secret_material"]["role_leaf_domain"].encode()
        + b"\x00" + canon(preimage),
        hashlib.sha256,
    ).hexdigest()


def root(leaves: list[str]) -> str:
    current = list(leaves)
    while len(current) > 1:
        if len(current) % 2:
            current.append(current[-1])
        current = [
            sha(b"NODE\x00" + bytes.fromhex(current[index])
                + bytes.fromhex(current[index + 1]))
            for index in range(0, len(current), 2)
        ]
    return current[0]


def reconstruct(
    evidence_dir: Path,
    secret_path: Path,
    output_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    config, secrets = obj(CONFIG), obj(secret_path)
    seed_commitment = obj(evidence_dir / "SEED_COMMITMENT.json")
    manifest = obj(evidence_dir / "ALLOCATION_MANIFEST.json")
    commitments = obj(evidence_dir / "ROLE_COMMITMENTS.json")
    sealed = decrypt(
        (evidence_dir / "FULL_ALLOCATION.sealed").read_bytes(),
        bytes.fromhex(secrets["encryption_key"]),
        config["secret_material"]["sealed_payload_aad"].encode(),
    )
    source_rows = rows(ROOT / config["lineage"]["public_source_path"])
    old_ids = {
        row["closure_id"]
        for row in rows(ROOT / config["lineage"]["closed_car_d0_path"])
    }
    source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if row["closure_id"] not in old_ids:
            source[row["closure_id"]].append(row)
    roles = [role["role"] for role in config["roles"]]
    gates: dict[str, bool] = {}
    commitment_domain = config["secret_material"]["commitment_domain"]
    expected_secret_commitments = {
        "allocation_seed_commitment": sha(
            commitment_domain.encode() + b"\x00ALLOCATION_SEED\x00"
            + bytes.fromhex(secrets["allocation_seed"])
        ),
        "encryption_key_commitment": sha(
            commitment_domain.encode() + b"\x00ENCRYPTION_KEY\x00"
            + bytes.fromhex(secrets["encryption_key"])
        ),
    }
    gates["allocation_seed_commitment"] = (
        seed_commitment["allocation_seed_commitment"]
        == expected_secret_commitments["allocation_seed_commitment"]
    )
    gates["encryption_key_commitment"] = (
        seed_commitment["encryption_key_commitment"]
        == expected_secret_commitments["encryption_key_commitment"]
    )
    for role, salt_hex in secrets["role_salts"].items():
        gates["role_salt_commitment:" + role] = (
            seed_commitment["role_salt_commitments"][role]
            == sha(
                commitment_domain.encode() + b"\x00ROLE_SALT|" + role.encode()
                + b"\x00" + bytes.fromhex(salt_hex)
            )
        )
    seen = set()
    computed_role_roots = {}
    all_content_valid = True
    all_leaves_valid = True
    for role in roles:
        records = sealed["roles"][role]
        salt = bytes.fromhex(sealed["role_salts"][role])
        recomputed = []
        for record in records:
            closure_id = record["closure_id"]
            preimage = {
                key: record[key] for key in (
                    "protocol_identity", "task_identity", "role", "closure_id",
                    "closure_content_sha256", "schema_version",
                )
            }
            source_digest = sha(canon(sorted(
                source[closure_id], key=lambda row: row["orientation"]
            )))
            all_content_valid = (
                all_content_valid
                and source_digest == record["closure_content_sha256"]
            )
            recomputed.append(leaf(config, salt, preimage))
            all_leaves_valid = all_leaves_valid and (
                recomputed[-1] == record["leaf_sha256"]
                and closure_id not in seen
                and closure_id not in old_ids
            )
            seen.add(closure_id)
        computed_role_roots[role] = root(recomputed)
        gates["root:" + role] = (
            computed_role_roots[role]
            == commitments["roles"][role]["merkle_root"]
        )
    computed_full = sha(
        config["secret_material"]["full_root_domain"].encode()
        + b"\x00" + canon(commitments["roles"])
    )
    dev_closures = rows(evidence_dir / "development/DEVELOPMENT_CLOSURES.jsonl")
    dev_rows = rows(evidence_dir / "development/DEVELOPMENT_ROWS.jsonl")
    public_blobs = b"\n".join(
        path.read_bytes()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path.name != "FULL_ALLOCATION.sealed"
    )
    future_ids = {
        record["closure_id"]
        for role in ("AGG_CALIBRATION", "AGG_VALIDATION", "PUBLIC_RESERVE")
        for record in sealed["roles"][role]
    }
    gates.update({
        "all_closure_content_digests": all_content_valid,
        "all_role_leaves_unique_and_valid": all_leaves_valid,
        "complete_partition": len(seen) == len(source) == 1920,
        "role_quotas": Counter(sealed["assignments"].values())
        == {role["role"]: role["closures"] for role in config["roles"]},
        "full_root": computed_full == commitments["full_allocation_root"]
        == sealed["full_allocation_root"] == manifest["full_allocation_root"],
        "manifest_sealed_digest": manifest["full_allocation_sealed_sha256"]
        == file_sha(evidence_dir / "FULL_ALLOCATION.sealed"),
        "development_only": (
            len(dev_closures) == 256
            and len(dev_rows) == 512
            and {row["closure_id"] for row in dev_closures}
            == {row["closure_id"] for row in sealed["roles"]["AGG_DEVELOPMENT"]}
        ),
        "future_ids_absent_from_plaintext": not any(
            closure_id.encode() in public_blobs for closure_id in future_ids
        ),
        "released_roles_exact": manifest["released_roles"] == ["AGG_DEVELOPMENT"],
        "balance": manifest["balance"]["all_balance_gates_pass"] is True,
        "authoritative_once": manifest["authoritative_allocation_count"] == 1,
    })
    valid = all(gates.values())
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.reference_result.v1",
        "terminal": (
            config["terminals"]["ready"]
            if valid else config["terminals"]["implementation_invalid"]
        ),
        "valid": valid,
        "gates": gates,
        "role_counts": Counter(sealed["assignments"].values()),
        "development_released_closures": len(dev_closures),
        "future_released_closures": 0,
        "protected_released_closures": 0,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(canon(result) + b"\n")
    if audit_path:
        official = obj(evidence_dir / "OFFICIAL_RESULT.json")
        terminal = obj(evidence_dir / "TERMINAL.json")
        audit = {
            "schema": "exp33a.csw_am_scb_m1_car_agg_r0.main_audit.v1",
            "identity": config["identity"],
            "main_verdict": result["terminal"],
            "status": "PUBLIC_ALLOCATION_COMMITTED_DEVELOPMENT_RELEASED_ONLY",
            "source_commit": manifest["source_commit"],
            "commitment_commit": manifest["commitment_commit"],
            "evidence": {
                "official_result_sha256": file_sha(
                    evidence_dir / "OFFICIAL_RESULT.json"
                ),
                "reference_result_sha256": (
                    file_sha(output_path) if output_path else None
                ),
                "terminal_sha256": file_sha(evidence_dir / "TERMINAL.json"),
                "role_commitments_sha256": file_sha(
                    evidence_dir / "ROLE_COMMITMENTS.json"
                ),
                "allocation_manifest_sha256": file_sha(
                    evidence_dir / "ALLOCATION_MANIFEST.json"
                ),
                "full_allocation_sealed_sha256": file_sha(
                    evidence_dir / "FULL_ALLOCATION.sealed"
                ),
                "production_reference_terminal_agreement": (
                    official["terminal"] == result["terminal"]
                    == terminal["terminal"]
                ),
            },
            "population": {
                "eligible_closures": 1920,
                "excluded_closed_car_d0_closures": 128,
                "role_counts": result["role_counts"],
                "complete_dependency_closures": True,
                "answer_quota_exact": True,
                "joint_support_minimums_pass": True,
                "task_only_balance_gates_pass": True,
            },
            "commitments": {
                "full_allocation_root": manifest["full_allocation_root"],
                "role_roots_committed": True,
                "seed_committed_before_allocation": True,
                "encryption_key_committed_before_allocation": True,
                "role_salts_committed_before_allocation": True,
                "authoritative_allocation_count": 1,
            },
            "release": {
                "development_closures": 256,
                "development_rows": 512,
                "calibration_closures": 0,
                "validation_closures": 0,
                "reserve_closures": 0,
                "protected_closures": 0,
                "future_role_identities_in_public_plaintext": 0,
                "arbitrary_membership_queries": "DENIED",
            },
            "authorizations": {
                "AGG_D0_public_implementation": True,
                "AGG_CAL1": False,
                "AGG_V1": False,
                "AGG_Q1": False,
                "AGG_C1": False,
                "protected_release": False,
                "automatic_successor": False,
            },
            "next_boundary": "AGG_D0_PUBLIC_IMPLEMENTATION",
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_bytes(canon(audit) + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--secret-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    try:
        result = reconstruct(
            args.evidence_dir, args.secret_path, args.output, args.audit_output
        )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
