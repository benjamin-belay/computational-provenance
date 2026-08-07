#!/usr/bin/env python3
"""One-shot public closure allocator with staged role release."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol/csw-am-scb-m1-car-agg-r0"
CONFIG_PATH = PROTOCOL_DIR / "CONFIG.json"
SPEC_PATH = PROTOCOL_DIR / "ALLOCATION_SPEC.json"
TEST_COMMIT = "TEST_ONLY"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("DUPLICATE_KEY:" + key)
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line, object_pairs_hook=strict_pairs)
        if type(value) is not dict:
            raise ValueError("ROW_REQUIRED:" + path.name)
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_tree(commit: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", commit + "^{tree}"], cwd=ROOT, text=True
    ).strip()


def validate_source(source_commit: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    config, spec = read_json(CONFIG_PATH), read_json(SPEC_PATH)
    audit_path = ROOT / config["lineage"]["agg_s0_main_audit_path"]
    if (
        config["identity"] != "CSW-AM-SCB-M1-CAR-AGG-R0"
        or file_sha(audit_path) != config["lineage"]["agg_s0_main_audit_sha256"]
        or read_json(audit_path)["main_verdict"] != config["lineage"]["agg_s0_terminal"]
        or sum(role["closures"] for role in config["roles"])
        != config["population"]["eligible_closures"]
    ):
        raise ValueError("PROTOCOL_BINDING")
    if source_commit == TEST_COMMIT:
        return config, spec, "0" * 40
    if len(source_commit) != 40 or any(
        char not in "0123456789abcdef" for char in source_commit
    ):
        raise ValueError("SOURCE_COMMIT")
    for relative in config["source_paths"]:
        committed = subprocess.check_output(
            ["git", "show", source_commit + ":" + relative], cwd=ROOT
        )
        if committed != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    return config, spec, git_tree(source_commit)


def secret_commitment(domain: str, label: str, value: bytes) -> str:
    return sha(domain.encode() + b"\x00" + label.encode() + b"\x00" + value)


def ensure_external_new(path: Path) -> None:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("SECRET_INSIDE_REPOSITORY")
    if path.exists():
        raise FileExistsError("SECRET_PATH_EXISTS")


def create_secrets(
    secret_path: Path, commitment_path: Path, source_commit: str
) -> dict[str, Any]:
    config, _, source_tree = validate_source(source_commit)
    ensure_external_new(secret_path)
    if commitment_path.exists():
        raise FileExistsError("COMMITMENT_EXISTS")
    roles = [role["role"] for role in config["roles"]]
    secrets = {
        "allocation_seed": os.urandom(
            config["secret_material"]["allocation_seed_bytes"]
        ).hex(),
        "encryption_key": os.urandom(
            config["secret_material"]["encryption_key_bytes"]
        ).hex(),
        "role_salts": {
            role: os.urandom(config["secret_material"]["role_salt_bytes"]).hex()
            for role in roles
        },
    }
    domain = config["secret_material"]["commitment_domain"]
    commitment = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.seed_commitment.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "source_tree": source_tree,
        "config_sha256": file_sha(CONFIG_PATH),
        "allocation_spec_sha256": file_sha(SPEC_PATH),
        "commitment_domain": domain,
        "allocation_seed_commitment": secret_commitment(
            domain, "ALLOCATION_SEED", bytes.fromhex(secrets["allocation_seed"])
        ),
        "encryption_key_commitment": secret_commitment(
            domain, "ENCRYPTION_KEY", bytes.fromhex(secrets["encryption_key"])
        ),
        "role_salt_commitments": {
            role: secret_commitment(
                domain, "ROLE_SALT|" + role,
                bytes.fromhex(secrets["role_salts"][role]),
            )
            for role in roles
        },
        "secret_material_revealed": False,
        "authoritative_allocation_count": 0,
    }
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    with secret_path.open("xb") as handle:
        handle.write(canonical(secrets) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    secret_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    write_json(commitment_path, commitment)
    return commitment


def validate_commitment(
    commitment_path: Path,
    commitment_commit: str,
    source_commit: str,
    source_tree: str,
    secrets: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    domain = config["secret_material"]["commitment_domain"]
    roles = [role["role"] for role in config["roles"]]
    expected = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.seed_commitment.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "source_tree": source_tree,
        "config_sha256": file_sha(CONFIG_PATH),
        "allocation_spec_sha256": file_sha(SPEC_PATH),
        "commitment_domain": domain,
        "allocation_seed_commitment": secret_commitment(
            domain, "ALLOCATION_SEED", bytes.fromhex(secrets["allocation_seed"])
        ),
        "encryption_key_commitment": secret_commitment(
            domain, "ENCRYPTION_KEY", bytes.fromhex(secrets["encryption_key"])
        ),
        "role_salt_commitments": {
            role: secret_commitment(
                domain, "ROLE_SALT|" + role,
                bytes.fromhex(secrets["role_salts"][role]),
            )
            for role in roles
        },
        "secret_material_revealed": False,
        "authoritative_allocation_count": 0,
    }
    if read_json(commitment_path) != expected:
        raise ValueError("COMMITMENT_CONTENT")
    if commitment_commit != TEST_COMMIT:
        relative = str(commitment_path.resolve().relative_to(ROOT))
        committed = subprocess.check_output(
            ["git", "show", commitment_commit + ":" + relative], cwd=ROOT
        )
        parent = subprocess.check_output(
            ["git", "rev-parse", commitment_commit + "^"], cwd=ROOT, text=True
        ).strip()
        if committed != commitment_path.read_bytes():
            raise ValueError("COMMITMENT_NOT_IMMUTABLE")
        if parent != source_commit:
            raise ValueError("COMMITMENT_CHRONOLOGY")
    return expected


def keyed_rank(seed: bytes, domain: str, *parts: Any) -> str:
    return hmac.new(
        seed, domain.encode() + b"\x00" + canonical(parts), hashlib.sha256
    ).hexdigest()


def load_population(
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    source = ROOT / config["lineage"]["public_source_path"]
    old_path = ROOT / config["lineage"]["closed_car_d0_path"]
    if (
        file_sha(source) != config["lineage"]["public_source_sha256"]
        or file_sha(old_path) != config["lineage"]["closed_car_d0_sha256"]
    ):
        raise ValueError("POPULATION_HASH")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(source):
        grouped[row["closure_id"]].append(row)
    old = {row["closure_id"] for row in read_rows(old_path)}
    if len(grouped) != config["population"]["source_closures"]:
        raise ValueError("SOURCE_DENOMINATOR")
    eligible = {key: value for key, value in grouped.items() if key not in old}
    if (
        len(old) != config["population"]["excluded_closed_car_d0_closures"]
        or len(eligible) != config["population"]["eligible_closures"]
    ):
        raise ValueError("ELIGIBLE_DENOMINATOR")
    for closure_id, rows in eligible.items():
        if (
            len(rows) != 2
            or {row["orientation"] for row in rows}
            != {"LOWER_C_ORIENTATION", "UPPER_C_ORIENTATION"}
            or any(row["closure_id"] != closure_id for row in rows)
        ):
            raise ValueError("CLOSURE_INTEGRITY:" + closure_id)
    return eligible, old


def lower(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["orientation"] == "LOWER_C_ORIENTATION")


def primary_cell(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    row = lower(rows)
    return (
        row["natural"]["z2"] % 8,
        row["natural"]["answer"],
        row["template_id"],
    )


def closure_digest(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["orientation"])
    return sha(canonical(ordered))


def allocate_roles(
    eligible: dict[str, list[dict[str, Any]]],
    seed: bytes,
    config: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, str]:
    domain = config["secret_material"]["allocation_domain"]
    nonreserve = ("AGG_DEVELOPMENT", "AGG_CALIBRATION", "AGG_VALIDATION")
    groups: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for closure_id, rows in eligible.items():
        groups[primary_cell(rows)].append(closure_id)
    if len(groups) != spec["reachable_primary_cells"]:
        raise ValueError("PRIMARY_SUPPORT")
    assignments: dict[str, str] = {}
    for cell in sorted(groups):
        ordered = sorted(
            groups[cell],
            key=lambda closure_id: keyed_rank(
                seed, domain, "CELL", cell, closure_id
            ),
        )
        if len(ordered) < len(nonreserve):
            raise ValueError("CELL_CAPACITY")
        role_order = sorted(
            nonreserve,
            key=lambda role: keyed_rank(seed, domain, "CELL_ROLE", cell, role),
        )
        for closure_id, role in zip(ordered[:3], role_order):
            assignments[closure_id] = role
    answer_quota = spec["answer_quota_per_role"]
    by_answer: dict[int, list[str]] = defaultdict(list)
    for closure_id, rows in eligible.items():
        if closure_id not in assignments:
            by_answer[lower(rows)["natural"]["answer"]].append(closure_id)
    for answer in range(8):
        used = Counter(
            role for closure_id, role in assignments.items()
            if lower(eligible[closure_id])["natural"]["answer"] == answer
        )
        closures = sorted(
            by_answer[answer],
            key=lambda closure_id: keyed_rank(
                seed, domain, "ANSWER_CLOSURE", answer, closure_id
            ),
        )
        # Equal role strings need unique slot identities before their keyed shuffle.
        indexed_slots = [
            (role, index) for role in answer_quota
            for index in range(answer_quota[role] - used[role])
        ]
        indexed_slots.sort(
            key=lambda item: keyed_rank(
                seed, domain, "ANSWER_SLOT", answer, item[0], item[1]
            )
        )
        if len(closures) != len(indexed_slots):
            raise ValueError("ANSWER_DENOMINATOR")
        for closure_id, (role, _) in zip(closures, indexed_slots):
            assignments[closure_id] = role
    if len(assignments) != len(eligible):
        raise ValueError("ALLOCATION_INCOMPLETE")
    expected = {role["role"]: role["closures"] for role in config["roles"]}
    if Counter(assignments.values()) != expected:
        raise ValueError("ROLE_QUOTA")
    return assignments


def balance_report(
    eligible: dict[str, list[dict[str, Any]]],
    assignments: dict[str, str],
    config: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    roles = [role["role"] for role in config["roles"]]
    eligible_count = len(eligible)
    eligible_features = {
        "high_bit": Counter(),
        "template": Counter(),
        "z2_mod8": Counter(),
    }
    role_features = {
        role: {name: Counter() for name in eligible_features} for role in roles
    }
    answer_counts = {role: Counter() for role in roles}
    cell_counts = {role: Counter() for role in roles}
    for closure_id, rows in eligible.items():
        row = lower(rows)
        values = {
            "high_bit": row["natural"]["z2"] // 8,
            "template": row["template_id"],
            "z2_mod8": row["natural"]["z2"] % 8,
        }
        role = assignments[closure_id]
        for name, value in values.items():
            eligible_features[name][value] += 1
            role_features[role][name][value] += 1
        answer_counts[role][row["natural"]["answer"]] += 1
        cell_counts[role][primary_cell(rows)] += 1
    maximums = {
        "high_bit": spec["balance_gates"][
            "maximum_natural_z2_high_bit_fraction_deviation_from_eligible"
        ],
        "template": spec["balance_gates"][
            "maximum_template_fraction_deviation_from_eligible"
        ],
        "z2_mod8": spec["balance_gates"][
            "maximum_z2_mod8_fraction_deviation_from_eligible"
        ],
    }
    deviations: dict[str, dict[str, float]] = {}
    gates: dict[str, bool] = {}
    expected_answer = spec["answer_quota_per_role"]
    for role in roles:
        role_total = Counter(assignments.values())[role]
        deviations[role] = {}
        for name, counter in role_features[role].items():
            max_deviation = max(
                abs(counter[value] / role_total - count / eligible_count)
                for value, count in eligible_features[name].items()
            )
            deviations[role][name] = max_deviation
            gates[f"{role}:{name}"] = max_deviation <= maximums[name]
        gates[f"{role}:answer"] = all(
            answer_counts[role][answer] == expected_answer[role]
            for answer in range(8)
        )
        minimum = spec["minimum_per_primary_cell"][role]
        gates[f"{role}:primary_cell"] = (
            len(cell_counts[role]) == spec["reachable_primary_cells"]
            and min(cell_counts[role].values()) >= minimum
        ) if minimum else True
    return {
        "eligible_feature_counts": {
            name: {str(key): value for key, value in sorted(counter.items())}
            for name, counter in eligible_features.items()
        },
        "role_answer_counts": {
            role: {str(key): value for key, value in sorted(counter.items())}
            for role, counter in answer_counts.items()
        },
        "role_primary_cell_support": {
            role: {
                "cells": len(counter),
                "minimum": min(counter.values()) if counter else 0,
                "maximum": max(counter.values()) if counter else 0,
            }
            for role, counter in cell_counts.items()
        },
        "maximum_fraction_deviations": deviations,
        "gates": gates,
        "all_balance_gates_pass": all(gates.values()),
    }


def merkle_levels(leaves: list[str]) -> list[list[str]]:
    if not leaves:
        raise ValueError("EMPTY_MERKLE")
    levels = [leaves]
    while len(levels[-1]) > 1:
        current = levels[-1]
        if len(current) % 2:
            current = current + [current[-1]]
        levels.append([
            sha(b"NODE\x00" + bytes.fromhex(current[index])
                + bytes.fromhex(current[index + 1]))
            for index in range(0, len(current), 2)
        ])
    return levels


def merkle_proof(levels: list[list[str]], index: int) -> list[dict[str, str]]:
    proof = []
    for level in levels[:-1]:
        padded = level if len(level) % 2 == 0 else level + [level[-1]]
        sibling = index ^ 1
        proof.append({
            "side": "LEFT" if sibling < index else "RIGHT",
            "sha256": padded[sibling],
        })
        index //= 2
    return proof


def verify_proof(leaf: str, proof: list[dict[str, str]], root: str) -> bool:
    value = leaf
    for item in proof:
        sibling = item["sha256"]
        if item["side"] == "LEFT":
            value = sha(b"NODE\x00" + bytes.fromhex(sibling) + bytes.fromhex(value))
        elif item["side"] == "RIGHT":
            value = sha(b"NODE\x00" + bytes.fromhex(value) + bytes.fromhex(sibling))
        else:
            return False
    return hmac.compare_digest(value, root)


def leaf_preimage(
    config: dict[str, Any], role: str, closure_id: str, content_sha: str
) -> dict[str, Any]:
    return {
        "protocol_identity": config["identity"],
        "task_identity": "GATED_ALIAS16",
        "role": role,
        "closure_id": closure_id,
        "closure_content_sha256": content_sha,
        "schema_version": "exp33a.csw_am_scb_m1_car_agg_r0.role_leaf.v1",
    }


def role_leaf(
    config: dict[str, Any], salt: bytes, preimage: dict[str, Any]
) -> str:
    return hmac.new(
        salt,
        config["secret_material"]["role_leaf_domain"].encode()
        + b"\x00" + canonical(preimage),
        hashlib.sha256,
    ).hexdigest()


def seal_payload(payload: dict[str, Any], key: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, canonical(payload), aad)
    envelope = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.sealed.v1",
        "algorithm": "AES-256-GCM",
        "nonce_hex": nonce.hex(),
        "aad_sha256": sha(aad),
        "ciphertext_hex": ciphertext.hex(),
    }
    return canonical(envelope) + b"\n"


def open_payload(blob: bytes, key: bytes, aad: bytes) -> dict[str, Any]:
    envelope = json.loads(blob, object_pairs_hook=strict_pairs)
    if envelope["aad_sha256"] != sha(aad):
        raise ValueError("SEALED_AAD")
    plaintext = AESGCM(key).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    )
    value = json.loads(plaintext, object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("SEALED_OBJECT")
    return value


def allocate(
    secret_path: Path,
    commitment_path: Path,
    commitment_commit: str,
    source_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    config, spec, source_tree = validate_source(source_commit)
    secrets = read_json(secret_path)
    validate_commitment(
        commitment_path, commitment_commit, source_commit, source_tree,
        secrets, config,
    )
    if output_dir.exists():
        allowed = (
            commitment_path.parent.resolve() == output_dir.resolve()
            and {path.name for path in output_dir.iterdir()}
            == {commitment_path.name}
        )
        if not allowed:
            raise FileExistsError("OUTPUT_EXISTS")
    if commitment_commit != TEST_COMMIT:
        lock_path = secret_path.with_name(secret_path.name + ".allocation-lock")
        with lock_path.open("xb") as handle:
            handle.write(canonical({
                "schema": "exp33a.csw_am_scb_m1_car_agg_r0.allocation_lock.v1",
                "source_commit": source_commit,
                "commitment_commit": commitment_commit,
                "output_dir": str(output_dir.resolve()),
            }) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    seed = bytes.fromhex(secrets["allocation_seed"])
    eligible, old = load_population(config)
    assignments = allocate_roles(eligible, seed, config, spec)
    balance = balance_report(eligible, assignments, config, spec)
    if not balance["all_balance_gates_pass"]:
        raise ValueError("BALANCE_FAILED")
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = [role["role"] for role in config["roles"]]
    role_records: dict[str, list[dict[str, Any]]] = {}
    role_levels: dict[str, list[list[str]]] = {}
    role_salts = {
        role: bytes.fromhex(secrets["role_salts"][role]) for role in roles
    }
    for role in roles:
        records = []
        closure_ids = sorted(
            (cid for cid, assigned in assignments.items() if assigned == role),
            key=lambda cid: keyed_rank(seed, config["secret_material"][
                "allocation_domain"], "ROLE_ORDER", role, cid),
        )
        for closure_id in closure_ids:
            content_sha = closure_digest(eligible[closure_id])
            preimage = leaf_preimage(config, role, closure_id, content_sha)
            records.append({
                **preimage,
                "leaf_sha256": role_leaf(config, role_salts[role], preimage),
            })
        role_records[role] = records
        role_levels[role] = merkle_levels([row["leaf_sha256"] for row in records])
    role_commitments = {
        role: {
            "closures": len(role_records[role]),
            "rows": len(role_records[role]) * 2,
            "merkle_root": role_levels[role][-1][0],
            "role_salt_commitment": read_json(commitment_path)[
                "role_salt_commitments"
            ][role],
            "released_in_r0": role == "AGG_DEVELOPMENT",
        }
        for role in roles
    }
    full_root = sha(
        config["secret_material"]["full_root_domain"].encode()
        + b"\x00" + canonical(role_commitments)
    )
    sealed_payload = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.full_allocation.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "commitment_commit": commitment_commit,
        "full_allocation_root": full_root,
        "role_salts": secrets["role_salts"],
        "roles": role_records,
        "assignments": assignments,
    }
    aad = config["secret_material"]["sealed_payload_aad"].encode()
    sealed_path = output_dir / "FULL_ALLOCATION.sealed"
    sealed_path.write_bytes(seal_payload(
        sealed_payload, bytes.fromhex(secrets["encryption_key"]), aad
    ))
    development = role_records["AGG_DEVELOPMENT"]
    development_rows = []
    development_closures = []
    for index, record in enumerate(development):
        closure_id = record["closure_id"]
        proof = merkle_proof(role_levels["AGG_DEVELOPMENT"], index)
        if not verify_proof(
            record["leaf_sha256"], proof,
            role_commitments["AGG_DEVELOPMENT"]["merkle_root"],
        ):
            raise ValueError("DEVELOPMENT_PROOF")
        development_closures.append({
            **record,
            "merkle_index": index,
            "merkle_proof": proof,
        })
        for row in sorted(eligible[closure_id], key=lambda value: value["orientation"]):
            development_rows.append({
                **row,
                "dataset_role": "AGG_DEVELOPMENT",
                "allocation_closure_content_sha256": record[
                    "closure_content_sha256"
                ],
            })
    write_rows(
        output_dir / "development/DEVELOPMENT_CLOSURES.jsonl",
        development_closures,
    )
    write_rows(
        output_dir / "development/DEVELOPMENT_ROWS.jsonl",
        development_rows,
    )
    development_release = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.development_release.v1",
        "role": "AGG_DEVELOPMENT",
        "closures": 256,
        "rows": 512,
        "role_salt_hex": secrets["role_salts"]["AGG_DEVELOPMENT"],
        "merkle_root": role_commitments["AGG_DEVELOPMENT"]["merkle_root"],
        "release_count": 1,
        "future_role_release_count": 0,
        "protected_release_count": 0,
    }
    write_json(output_dir / "development/DEVELOPMENT_RELEASE.json", development_release)
    write_json(output_dir / "ROLE_COMMITMENTS.json", {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.role_commitments.v1",
        "full_allocation_root": full_root,
        "roles": role_commitments,
    })
    manifest = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.manifest.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "source_tree": source_tree,
        "commitment_commit": commitment_commit,
        "seed_commitment_sha256": file_sha(commitment_path),
        "source_population_sha256": config["lineage"]["public_source_sha256"],
        "excluded_car_d0_sha256": config["lineage"]["closed_car_d0_sha256"],
        "eligible_closures": len(eligible),
        "excluded_closures": len(old),
        "full_allocation_root": full_root,
        "full_allocation_sealed_sha256": file_sha(sealed_path),
        "development_release_sha256": file_sha(
            output_dir / "development/DEVELOPMENT_RELEASE.json"
        ),
        "development_closures_sha256": file_sha(
            output_dir / "development/DEVELOPMENT_CLOSURES.jsonl"
        ),
        "development_rows_sha256": file_sha(
            output_dir / "development/DEVELOPMENT_ROWS.jsonl"
        ),
        "role_counts": Counter(assignments.values()),
        "balance": balance,
        "authoritative_allocation_count": 1,
        "released_roles": ["AGG_DEVELOPMENT"],
        "sealed_roles": [
            "AGG_CALIBRATION", "AGG_VALIDATION", "PUBLIC_RESERVE"
        ],
        "future_role_identity_count_in_public_plaintext": 0,
        "protected_identity_count_in_public_plaintext": 0,
    }
    write_json(output_dir / "ALLOCATION_MANIFEST.json", manifest)
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.official_result.v1",
        "terminal": config["terminals"]["ready"],
        "full_allocation_root": full_root,
        "role_counts": Counter(assignments.values()),
        "development_released_closures": 256,
        "calibration_released_closures": 0,
        "validation_released_closures": 0,
        "reserve_released_closures": 0,
        "protected_released_closures": 0,
        "balance_gates_pass": True,
        "commitment_chronology_valid": True,
        "agg_d0_authorized": True,
        "agg_cal1_authorized": False,
        "agg_v1_authorized": False,
        "protected_release_authorized": False,
    }
    write_json(output_dir / "OFFICIAL_RESULT.json", result)
    write_json(output_dir / "TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_car_agg_r0.terminal.v1",
        "terminal": result["terminal"],
        "status": "PUBLIC_ALLOCATION_COMMITTED_DEVELOPMENT_RELEASED_ONLY",
        "authoritative_allocation_count": 1,
        "released_roles": ["AGG_DEVELOPMENT"],
        "next_boundary": "AGG_D0_PUBLIC_IMPLEMENTATION",
    })
    return result


def release_role(output_dir: Path, role: str) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if role not in config["release_policy"]["roles_releasable_in_r0"]:
        raise PermissionError("ROLE_NOT_AUTHORIZED_IN_R0")
    if role != "AGG_DEVELOPMENT":
        raise PermissionError("FUTURE_ROLE_DENIED")
    return read_json(output_dir / "development/DEVELOPMENT_RELEASE.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-commitment")
    create.add_argument("--secret-path", type=Path, required=True)
    create.add_argument("--commitment-path", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    run = sub.add_parser("allocate")
    run.add_argument("--secret-path", type=Path, required=True)
    run.add_argument("--commitment-path", type=Path, required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    release = sub.add_parser("release")
    release.add_argument("--output-dir", type=Path, required=True)
    release.add_argument("--role", required=True)
    args = parser.parse_args()
    try:
        if args.command == "create-commitment":
            result = create_secrets(
                args.secret_path, args.commitment_path, args.source_commit
            )
        elif args.command == "allocate":
            result = allocate(
                args.secret_path, args.commitment_path, args.commitment_commit,
                args.source_commit, args.output_dir,
            )
        else:
            result = release_role(args.output_dir, args.role)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
