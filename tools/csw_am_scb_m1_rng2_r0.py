#!/usr/bin/env python3
"""Membership-preserving migration from the sealed AGG roles to RNG2 roles."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol/csw-am-scb-m1-rng2-r0"
CONFIG_PATH = PROTOCOL / "CONFIG.json"
SPEC_PATH = PROTOCOL / "MIGRATION_SPEC.json"
TEST_COMMIT = "TEST_ONLY"


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("DUPLICATE_KEY:" + key)
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, object_pairs_hook=strict_pairs)
        for line in path.read_bytes().splitlines()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def open_payload(blob: bytes, key: bytes, aad: bytes) -> dict[str, Any]:
    envelope = json.loads(blob, object_pairs_hook=strict_pairs)
    if (
        envelope["algorithm"] != "AES-256-GCM"
        or envelope["aad_sha256"] != sha(aad)
    ):
        raise ValueError("SEALED_ENVELOPE")
    plain = AESGCM(key).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    )
    value = json.loads(plain, object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("SEALED_OBJECT")
    return value


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        raise ValueError("EMPTY_MERKLE")
    current = list(leaves)
    while len(current) > 1:
        if len(current) % 2:
            current.append(current[-1])
        current = [
            sha(
                b"NODE\x00"
                + bytes.fromhex(current[index])
                + bytes.fromhex(current[index + 1])
            )
            for index in range(0, len(current), 2)
        ]
    return current[0]


def old_leaf(
    old_config: dict[str, Any], salt: bytes, record: dict[str, Any]
) -> str:
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


def validate_source(source_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config, spec = read_json(CONFIG_PATH), read_json(SPEC_PATH)
    for name in (
        "rng2_s0_audit", "rng2_s0_terminal", "agg_r0_config", "agg_r0_audit",
        "old_role_commitments", "old_allocation_manifest", "old_full_allocation",
        "development_release", "development_closures", "development_rows",
        "scb_protected_boundary_audit",
    ):
        path = ROOT / config["lineage"][f"{name}_path"]
        if file_sha(path) != config["lineage"][f"{name}_sha256"]:
            raise ValueError("LINEAGE_HASH:" + name)
    s0_audit = read_json(ROOT / config["lineage"]["rng2_s0_audit_path"])
    s0_terminal = read_json(ROOT / config["lineage"]["rng2_s0_terminal_path"])
    if (
        s0_audit["main_verdict"] != config["lineage"]["rng2_s0_terminal"]
        or s0_terminal["terminal"] != config["lineage"]["rng2_s0_terminal"]
        or not s0_audit["authorizations"]["RNG2_R0"]
        or s0_audit["authorizations"]["RNG2_D0"]
    ):
        raise ValueError("S0_BOUNDARY")
    if source_commit != TEST_COMMIT:
        if (
            len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise ValueError("SOURCE_COMMIT")
        for relative in config["source_paths"]:
            committed = subprocess.check_output(
                ["git", "show", source_commit + ":" + relative], cwd=ROOT
            )
            if committed != (ROOT / relative).read_bytes():
                raise ValueError("SOURCE_BYTES:" + relative)
    return config, spec


def public_population(
    old_config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(ROOT / old_config["lineage"]["public_source_path"]):
        source[row["closure_id"]].append(row)
    excluded = {
        row["closure_id"]
        for row in read_rows(ROOT / old_config["lineage"]["closed_car_d0_path"])
    }
    eligible = {
        closure_id: rows
        for closure_id, rows in source.items()
        if closure_id not in excluded
    }
    if len(source) != 2048 or len(excluded) != 128 or len(eligible) != 1920:
        raise ValueError("PUBLIC_POPULATION")
    if any(
        len(rows) != 2
        or {row["orientation"] for row in rows}
        != {"LOWER_C_ORIENTATION", "UPPER_C_ORIENTATION"}
        for rows in eligible.values()
    ):
        raise ValueError("CLOSURE_INTEGRITY")
    return eligible, excluded


def migration_binding(
    config: dict[str, Any],
    migration: dict[str, Any],
    old_commitment: dict[str, Any],
) -> dict[str, Any]:
    preimage = {
        "schema": "exp33a.csw_am_scb_m1_rng2_r0.role_binding.v1",
        "protocol_identity": config["identity"],
        "old_protocol_identity": "CSW-AM-SCB-M1-CAR-AGG-R0",
        "old_role": migration["old_role"],
        "new_role": migration["new_role"],
        "membership_commitment_kind": "INHERITED_OLD_MERKLE_ROOT",
        "inherited_membership_root": old_commitment["merkle_root"],
        "closures": migration["closures"],
        "rows": migration["rows"],
        "exposure": migration["exposure"],
        "prior_execution": migration["prior_execution"],
        "new_release": False,
    }
    return {
        **preimage,
        "binding_sha256": sha(
            config["controller"]["migration_binding_domain"].encode()
            + b"\x00" + canonical(preimage)
        ),
    }


def controller_validate(
    secret_path: Path,
    source_commit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    config, spec = validate_source(source_commit)
    old_config = read_json(ROOT / config["lineage"]["agg_r0_config_path"])
    commitments = read_json(
        ROOT / config["lineage"]["old_role_commitments_path"]
    )
    manifest = read_json(
        ROOT / config["lineage"]["old_allocation_manifest_path"]
    )
    secrets = read_json(secret_path)
    sealed = open_payload(
        (ROOT / config["lineage"]["old_full_allocation_path"]).read_bytes(),
        bytes.fromhex(secrets["encryption_key"]),
        config["controller"]["old_aad"].encode(),
    )
    eligible, excluded = public_population(old_config)
    protected_audit = read_json(
        ROOT / config["lineage"]["scb_protected_boundary_audit_path"]
    )
    migrations = {item["old_role"]: item for item in spec["migrations"]}
    bindings = []
    all_ids: set[str] = set()
    future_ids: set[str] = set()
    roots_recomputed: dict[str, str] = {}
    content_valid = True
    leaves_valid = True
    for old_role, migration in migrations.items():
        records = sealed["roles"][old_role]
        salt = bytes.fromhex(sealed["role_salts"][old_role])
        leaves = []
        for record in records:
            closure_id = record["closure_id"]
            if closure_id in all_ids:
                leaves_valid = False
            all_ids.add(closure_id)
            if old_role != "AGG_DEVELOPMENT":
                future_ids.add(closure_id)
            expected_content = sha(canonical(sorted(
                eligible[closure_id], key=lambda row: row["orientation"]
            )))
            content_valid = (
                content_valid
                and expected_content == record["closure_content_sha256"]
            )
            leaf = old_leaf(old_config, salt, record)
            leaves_valid = leaves_valid and leaf == record["leaf_sha256"]
            leaves.append(leaf)
        roots_recomputed[old_role] = merkle_root(leaves)
        if (
            len(records) != migration["closures"]
            or roots_recomputed[old_role]
            != commitments["roles"][old_role]["merkle_root"]
        ):
            raise ValueError("OLD_ROLE_ROOT:" + old_role)
        bindings.append(
            migration_binding(config, migration, commitments["roles"][old_role])
        )
    development_public = {
        row["closure_id"]
        for row in read_rows(ROOT / config["lineage"]["development_closures_path"])
    }
    assignments = sealed["assignments"]
    full_binding_root = sha(
        config["controller"]["full_binding_domain"].encode()
        + b"\x00" + canonical(bindings)
    )
    gates = {
        "old_sealed_payload_authenticated": sealed["identity"]
        == "CSW-AM-SCB-M1-CAR-AGG-R0",
        "old_full_root": sealed["full_allocation_root"]
        == commitments["full_allocation_root"]
        == manifest["full_allocation_root"],
        "old_roots_recomputed": all(
            roots_recomputed[role] == commitments["roles"][role]["merkle_root"]
            for role in migrations
        ),
        "closure_content": content_valid,
        "leaves_unique_and_valid": leaves_valid,
        "complete_partition": all_ids == set(eligible) and len(all_ids) == 1920,
        "assignments_match": (
            set(assignments) == all_ids
            and Counter(assignments.values())
            == {role: migration["closures"] for role, migration in migrations.items()}
        ),
        "historical_exclusion": all_ids.isdisjoint(excluded) and len(excluded) == 128,
        "development_membership": development_public
        == {record["closure_id"] for record in sealed["roles"]["AGG_DEVELOPMENT"]},
        "protected_disjointness_bound": (
            protected_audit["public_training_bindings"][
                "overlap_with_protected_scb_closures"
            ] == 0
            and protected_audit["public_training_bindings"][
                "checkpoint_selection_closures"
            ] == 2048
        ),
        "no_redraw": manifest["authoritative_allocation_count"] == 1,
        "balance_inherited": manifest["balance"]["all_balance_gates_pass"] is True,
        "binding_count": len(bindings) == 4,
    }
    attestation = {
        "schema": "exp33a.csw_am_scb_m1_rng2_r0.controller_attestation.v1",
        "source_commit": source_commit,
        "old_full_allocation_sha256": file_sha(
            ROOT / config["lineage"]["old_full_allocation_path"]
        ),
        "old_full_allocation_root": commitments["full_allocation_root"],
        "full_migration_binding_root": full_binding_root,
        "role_counts": {
            binding["new_role"]: binding["closures"] for binding in bindings
        },
        "old_roots_recomputed": roots_recomputed,
        "gates": gates,
        "future_role_identity_count_examined_inside_controller": len(future_ids),
        "future_role_identity_count_exported": 0,
        "protected_identity_count_accessed": 0,
        "new_allocation_seed_created": False,
        "new_allocation_executed": False,
        "new_identity_release_count": 0,
    }
    attestation_key = hmac.new(
        bytes.fromhex(secrets["encryption_key"]),
        config["controller"]["attestation_key_domain"].encode(),
        hashlib.sha256,
    ).digest()
    tag = hmac.new(
        attestation_key,
        config["controller"]["attestation_domain"].encode()
        + b"\x00" + canonical(attestation),
        hashlib.sha256,
    ).hexdigest()
    return {**attestation, "attestation_tag": tag}, bindings, future_ids


def public_records(
    config: dict[str, Any],
    spec: dict[str, Any],
    attestation: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    binding_root = attestation["full_migration_binding_root"]
    return {
        "ROLE_MIGRATION_BINDINGS.json": {
            "schema": "exp33a.csw_am_scb_m1_rng2_r0.bindings.v1",
            "identity": config["identity"],
            "method": spec["method"],
            "old_membership_roots_inherited_without_reallocation": True,
            "bindings": bindings,
            "full_migration_binding_root": binding_root,
        },
        "RETIREMENT_DECLARATION.json": {
            "schema": "exp33a.csw_am_scb_m1_rng2_r0.retirement.v1",
            "closed_AGG_terminal_preserved":
                "MAIN_CSW_AM_SCB_M1_CAR_AGG_D0_CLOSURE_CONFIRMED",
            "retirements": spec["retirements"],
            "old_membership_commitments_preserved": True,
            "old_scientific_role_authorities_active": False,
            "old_lineage_reopened": False,
        },
        "EXPOSURE_STATUS.json": {
            "schema": "exp33a.csw_am_scb_m1_rng2_r0.exposure.v1",
            "roles": {
                binding["new_role"]: {
                    "closures": binding["closures"],
                    "exposure": binding["exposure"],
                    "prior_execution": binding["prior_execution"],
                    "new_identity_release": False,
                }
                for binding in bindings
            },
            "development_identities_previously_public": 256,
            "development_fixture_closures_previously_executed": 1,
            "future_public_identities_released": 0,
            "protected_identities_accessed": 0,
        },
        "DEVELOPMENT_AUTHORIZATION.json": {
            "schema": "exp33a.csw_am_scb_m1_rng2_r0.development_authorization.v1",
            "role": "RNG2_DEVELOPMENT",
            "closures": 256,
            "identity_source": "EXISTING_PUBLIC_AGG_DEVELOPMENT_MANIFEST",
            "development_closures_sha256":
                config["lineage"]["development_closures_sha256"],
            "development_rows_sha256": config["lineage"]["development_rows_sha256"],
            "new_identity_release": False,
            "authorized_use": "RNG2_D0_PUBLIC_IMPLEMENTATION_AND_DEBUGGING_ONLY",
            "calibration_access": False,
            "validation_access": False,
            "reserve_access": False,
            "protected_access": False,
        },
        "CONTROLLER_ATTESTATION.json": attestation,
    }


def materialize(secret_path: Path, source_commit: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("OUTPUT_BOUNDARY")
    config, spec = validate_source(source_commit)
    attestation, bindings, future_ids = controller_validate(secret_path, source_commit)
    records = public_records(config, spec, attestation, bindings)
    public_bytes = b"\n".join(canonical(value) for value in records.values())
    if any(closure_id.encode() in public_bytes for closure_id in future_ids):
        raise ValueError("FUTURE_IDENTITY_LEAK")
    if not all(attestation["gates"].values()):
        raise ValueError("CONTROLLER_GATE")
    output.mkdir(parents=True)
    for name, value in records.items():
        write_json(output / name, value)
    official = {
        "schema": "exp33a.csw_am_scb_m1_rng2_r0.official.v1",
        "source_commit": source_commit,
        "gates": {
            **attestation["gates"],
            "no_future_identity_leak": True,
            "zero_new_release": True,
            "zero_scientific_execution": True,
            "protected_boundary": True,
        },
        "full_migration_binding_root": attestation["full_migration_binding_root"],
        "role_counts": attestation["role_counts"],
        "new_allocation_executed": False,
        "new_identity_releases": 0,
        "model_forwards": 0,
        "stochastic_outputs": 0,
        "receipts": 0,
        "detector_scores": 0,
        "protected_access": 0,
        "RNG2_D0_authorized": True,
        "RNG2_CAL1_authorized": False,
        "protected_release_authorized": False,
        "passed": True,
        "terminal": config["terminals"]["ready"],
    }
    write_json(output / "OFFICIAL_RESULT.json", official)
    return official


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-path", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.secret_path, args.source_commit, args.output)
    print((canonical(result) + b"\n").decode(), end="")


if __name__ == "__main__":
    main()
