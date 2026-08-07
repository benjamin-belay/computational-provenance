#!/usr/bin/env python3
"""Minimal protected-role wrapper for frozen SCB-M1 receipt qualification."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import secrets
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rec-q1/CONFIG.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE_SPEC:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


D0 = load_module("csw_am_scb_m1_rec_d0_frozen_for_q1", ROOT / "tools/csw_am_scb_m1_rec_d0.py")
D0_REF = load_module(
    "csw_am_scb_m1_rec_d0_reference_frozen_for_q1",
    ROOT / "tools/csw_am_scb_m1_rec_d0_reference.py",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
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
    result = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line, object_pairs_hook=strict_pairs)
        if type(value) is not dict:
            raise ValueError("ROW_REQUIRED:" + path.name)
        result.append(value)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_frozen_D0(config: dict[str, Any]) -> None:
    binding = config["D0_binding"]
    for key in ("production", "reference"):
        path = ROOT / binding[f"{key}_path"]
        if (
            file_sha(path) != binding[f"{key}_sha256"]
            or subprocess.check_output(
                ["git", "rev-parse", f"{binding['source_commit']}:{binding[f'{key}_path']}"],
                cwd=ROOT, text=True,
            ).strip() != binding[f"{key}_blob"]
            or git_show(binding["source_commit"], binding[f"{key}_path"]) != path.read_bytes()
        ):
            raise ValueError("D0_IMPLEMENTATION_HASH:" + key)
    attack_path = ROOT / binding["attack_manifest_path"]
    if (
        file_sha(attack_path) != binding["attack_manifest_sha256"]
        or len(read_json(attack_path)["families"]) != binding["attack_families"]
    ):
        raise ValueError("D0_ATTACK_MANIFEST")
    audit = ROOT / binding["main_audit_path"]
    if (
        file_sha(audit) != binding["main_audit_sha256"]
        or read_json(audit)["main_verdict"] != binding["main_terminal"]
    ):
        raise ValueError("D0_AUDIT")


def validate_source(source_commit: str) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config["identity"] != "CSW-AM-SCB-M1-REC-Q1" or len(source_commit) != 40:
        raise ValueError("SOURCE_IDENTITY")
    for relative in config["source_file_paths"]:
        if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    validate_frozen_D0(config)
    return config


def qconfig(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(D0.read_json(D0.CONFIG_PATH))
    role = config["protected_role"]
    value["identity"] = config["identity"]
    value["public_population"]["closures"] = role["closures"]
    value["public_population"]["oriented_rows"] = role["oriented_rows"]
    value["public_population"]["execution_packages"] = role["execution_packages"]
    value["public_population"]["receipt_envelopes"] = role["receipt_envelopes"]
    value["public_population"]["attack_base_packages"] = role["execution_packages"]
    value["public_population"]["attack_base_natural"] = role["oriented_rows"]
    value["public_population"]["attack_base_alias"] = role["oriented_rows"]
    value["key_lifecycle"].update(config["key_lifecycle"])
    return value


def qattacks(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(D0.read_json(D0.ATTACK_PATH))
    value["schema"] = "exp33a.csw_am_scb_m1_rec_q1.attack_assignment.v1"
    value["identity"] = "SCB_M1_REC_Q1_FULL_CARTESIAN_FROZEN_ATTACKS_V1"
    value["expected_attack_rows"] = config["protected_role"]["attack_rows"]
    value["attacks_per_base_package"] = config["protected_role"]["attacks_per_execution"]
    return value


def create_authorization(source_commit: str, output: Path) -> dict[str, Any]:
    config = validate_source(source_commit)
    role = config["protected_role"]
    if output.exists():
        raise FileExistsError("AUTHORIZATION_EXISTS")
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.authorization.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "D0_source_commit": config["D0_binding"]["source_commit"],
        "D0_production_blob": config["D0_binding"]["production_blob"],
        "D0_reference_blob": config["D0_binding"]["reference_blob"],
        "D0_attack_manifest_sha256": config["D0_binding"]["attack_manifest_sha256"],
        "role": role["identity"],
        "closures": role["closures"],
        "oriented_rows": role["oriented_rows"],
        "execution_packages": role["execution_packages"],
        "receipt_envelopes": role["receipt_envelopes"],
        "attack_rows": role["attack_rows"],
        "merkle_root": role["merkle_root"],
        "protected_identities_released": 0,
        "key_committed": False,
        "REC_C1_authorized": False,
        "carrier_authorized": False,
    }
    write_json(output, value)
    return value


def validate_authorization(
    path: Path, commit: str, source_commit: str, config: dict[str, Any],
) -> dict[str, Any]:
    relative = str(path.resolve().relative_to(ROOT))
    if git_show(commit, relative) != path.read_bytes():
        raise ValueError("AUTHORIZATION_COMMIT")
    value = read_json(path)
    role = config["protected_role"]
    if (
        value["source_commit"] != source_commit
        or value["role"] != role["identity"]
        or value["closures"] != role["closures"]
        or value["attack_rows"] != role["attack_rows"]
        or value["merkle_root"] != role["merkle_root"]
        or value["protected_identities_released"] != 0
    ):
        raise ValueError("AUTHORIZATION_CONTENT")
    return value


def create_key(
    source_commit: str, authorization: Path, authorization_commit: str,
    key_path: Path, commitment_path: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    auth = validate_authorization(
        authorization, authorization_commit, source_commit, config
    )
    if key_path.exists() or commitment_path.exists():
        raise FileExistsError("KEY_OR_COMMITMENT_EXISTS")
    key = secrets.token_bytes(config["key_lifecycle"]["master_key_bytes"])
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("xb") as handle:
        handle.write(key)
        handle.flush()
        os.fsync(handle.fileno())
    qc = qconfig(config)
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.key_commitment.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "authorization_commit": authorization_commit,
        "role": auth["role"],
        "key_commitment": D0.key_commitment(key, qc),
        "run_id": D0.run_id(key, qc),
        "key_bytes": len(key),
        "key_revealed": False,
        "second_key": False,
    }
    write_json(commitment_path, value)
    return value


def validate_key(
    key: bytes, commitment: Path, commitment_commit: str,
    source_commit: str, authorization_commit: str, config: dict[str, Any],
) -> dict[str, Any]:
    relative = str(commitment.resolve().relative_to(ROOT))
    if git_show(commitment_commit, relative) != commitment.read_bytes():
        raise ValueError("KEY_COMMITMENT_COMMIT")
    value = read_json(commitment)
    qc = qconfig(config)
    if (
        value["source_commit"] != source_commit
        or value["authorization_commit"] != authorization_commit
        or value["key_commitment"] != D0.key_commitment(key, qc)
        or value["run_id"] != D0.run_id(key, qc)
        or value["key_revealed"] is not False
    ):
        raise ValueError("KEY_COMMITMENT_CONTENT")
    return value


def release_rows(
    allocation: Path, manifest: Path, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role = config["protected_role"]
    if (
        file_sha(allocation) != role["full_allocation_sha256"]
        or file_sha(manifest) != role["sealed_manifest_sha256"]
    ):
        raise ValueError("SEALED_ALLOCATION_BINDING")
    sealed_manifest = read_json(manifest)
    closure_rows = [
        row for row in read_rows(allocation) if row["role"] == role["identity"]
    ]
    if (
        len(closure_rows) != role["closures"]
        or len({row["closure_id"] for row in closure_rows}) != role["closures"]
        or D0.S0.merkle_root(
            "EXP33A|CSW-AM-SCB-M1-REC-S0|" + role["identity"], closure_rows
        ) != role["merkle_root"]
        or sealed_manifest["role_commitments"][role["identity"]]["merkle_root"]
        != role["merkle_root"]
    ):
        raise ValueError("ROLE_MEMBERSHIP")
    oriented = D0.M1.protected_rows(closure_rows, role["identity"])
    if (
        len(oriented) != role["oriented_rows"]
        or len({row["example_id"] for row in oriented}) != role["oriented_rows"]
    ):
        raise ValueError("ORIENTED_DENOMINATOR")
    return closure_rows, oriented


def execute(
    source_commit: str, authorization: Path, authorization_commit: str,
    key_path: Path, commitment: Path, commitment_commit: str,
    allocation: Path, manifest: Path, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    validate_authorization(
        authorization, authorization_commit, source_commit, config
    )
    key = key_path.read_bytes()
    key_binding = validate_key(
        key, commitment, commitment_commit, source_commit,
        authorization_commit, config,
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    closure_rows, rows = release_rows(allocation, manifest, config)
    qc, qa = qconfig(config), qattacks(config)
    protocol = D0.read_json(D0.S0_PROTOCOL_PATH)
    model = D0.M1.load_model(D0.M1.read_json(D0.M1.CONFIG_PATH))
    primitives = []
    packages = []
    with torch.no_grad():
        values = D0.M1.tensors(rows)
        natural = model.run_natural(values)
        targets = torch.tensor(
            [(row["natural"]["z2"] + 8) % 16 for row in rows],
            dtype=torch.long,
        )
        alias = model.run_with_z2(values, targets)
        natural_answers = natural["answer_logits"].argmax(-1)
        alias_answers = alias["answer_logits"].argmax(-1)
        for index, row in enumerate(rows):
            predicted = int(natural["z2_selected"][index])
            observations = (
                ("NATURAL", {
                    "z2_consumed": int(natural["z2_consumed"][index]),
                    "z3_selected": int(natural["z3_selected"][index]),
                    "z3_consumed": int(natural["z3_consumed"][index]),
                    "answer": int(natural_answers[index]),
                }, predicted),
                ("ALIAS_Z2", {
                    "z2_consumed": int(alias["z2_consumed"][index]),
                    "z3_selected": int(alias["z3_selected"][index]),
                    "z3_consumed": int(alias["z3_consumed"][index]),
                    "answer": int(alias_answers[index]),
                }, int(targets[index])),
            )
            for role, observed, target in observations:
                primitive, package = D0.receipt_package(
                    key, protocol, qc, row, role, observed, predicted, target
                )
                primitives.append(primitive)
                packages.append(package)
    role = config["protected_role"]
    if len(primitives) != role["execution_packages"]:
        raise ValueError("EXECUTION_DENOMINATOR")
    attacks = D0.construct_attacks(
        key, primitives, packages, protocol, qc, qa
    )
    if len(attacks) != role["attack_rows"]:
        raise ValueError("ATTACK_DENOMINATOR")
    output.mkdir(parents=True)
    write_rows(output / "RELEASED_CLOSURES.jsonl", closure_rows)
    write_rows(output / "INPUT_ROWS.jsonl", rows)
    write_rows(output / "EXECUTION_ROWS.jsonl", primitives)
    write_rows(output / "RECEIPT_PACKAGES.jsonl", packages)
    write_rows(output / "ATTACK_INPUTS.jsonl", attacks)
    hashes = {
        name: file_sha(output / name)
        for name in (
            "RELEASED_CLOSURES.jsonl", "INPUT_ROWS.jsonl",
            "EXECUTION_ROWS.jsonl", "RECEIPT_PACKAGES.jsonl",
            "ATTACK_INPUTS.jsonl",
        )
    }
    write_json(output / "RAW_MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.raw_manifest.v1",
        "source_commit": source_commit,
        "authorization_commit": authorization_commit,
        "key_commitment_commit": commitment_commit,
        "key_commitment": key_binding["key_commitment"],
        "run_id": key_binding["run_id"],
        "role": role["identity"],
        "role_merkle_root": role["merkle_root"],
        "closures": len(closure_rows),
        "oriented_rows": len(rows),
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "attack_inputs": len(attacks),
        "files": hashes,
        "key_revealed": False,
        "other_role_releases": config["zero_other_role_access"],
    })
    write_json(output / "RAW_TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.raw_terminal.v1",
        "terminal": "CSW_AM_SCB_M1_REC_Q1_RAW_EVIDENCE_COMMITTED_PENDING_REVEAL",
        "qualification_decision_pending": True,
        "REC_C1_authorized": False,
        "carrier_authorized": False,
    })
    return {
        "terminal": "CSW_AM_SCB_M1_REC_Q1_RAW_EVIDENCE_COMMITTED_PENDING_REVEAL",
        "closures": len(closure_rows),
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "attacks": len(attacks),
        "key_revealed": False,
    }


def reveal(
    source_commit: str, authorization: Path, authorization_commit: str,
    key_path: Path, commitment: Path, commitment_commit: str,
    raw_manifest: Path, raw_commit: str, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    validate_authorization(
        authorization, authorization_commit, source_commit, config
    )
    key = key_path.read_bytes()
    key_binding = validate_key(
        key, commitment, commitment_commit, source_commit,
        authorization_commit, config,
    )
    relative = str(raw_manifest.resolve().relative_to(ROOT))
    if git_show(raw_commit, relative) != raw_manifest.read_bytes():
        raise ValueError("RAW_COMMIT_BINDING")
    raw = read_json(raw_manifest)
    if (
        raw["source_commit"] != source_commit
        or raw["authorization_commit"] != authorization_commit
        or raw["key_commitment"] != key_binding["key_commitment"]
        or raw["key_revealed"] is not False
    ):
        raise ValueError("RAW_MANIFEST_BINDING")
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.key_reveal.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "authorization_commit": authorization_commit,
        "key_commitment_commit": commitment_commit,
        "raw_evidence_commit": raw_commit,
        "raw_manifest_sha256": file_sha(raw_manifest),
        "key_hex": key.hex(),
        "key_commitment": key_binding["key_commitment"],
        "run_id": key_binding["run_id"],
        "qualification_key": True,
    }
    if D0.key_commitment(bytes.fromhex(value["key_hex"]), qconfig(config)) != value["key_commitment"]:
        raise ValueError("KEY_REVEAL_RECONSTRUCTION")
    write_json(output, value)
    return {key: value[key] for key in value if key != "key_hex"}


def evaluate(
    source_commit: str, raw: Path, raw_commit: str,
    reveal_path: Path, reveal_commit: str, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    for path, commit in (
        (raw / "RAW_MANIFEST.json", raw_commit),
        (reveal_path, reveal_commit),
    ):
        relative = str(path.resolve().relative_to(ROOT))
        if git_show(commit, relative) != path.read_bytes():
            raise ValueError("EVIDENCE_COMMIT_BINDING:" + path.name)
    raw_manifest = read_json(raw / "RAW_MANIFEST.json")
    reveal_value = read_json(reveal_path)
    if (
        reveal_value["raw_evidence_commit"] != raw_commit
        or reveal_value["raw_manifest_sha256"] != file_sha(raw / "RAW_MANIFEST.json")
        or reveal_value["key_commitment"] != raw_manifest["key_commitment"]
    ):
        raise ValueError("REVEAL_RAW_BINDING")
    for name, expected_hash in raw_manifest["files"].items():
        if file_sha(raw / name) != expected_hash:
            raise ValueError("RAW_FILE_HASH:" + name)
    key = bytes.fromhex(reveal_value["key_hex"])
    protocol = D0.read_json(D0.S0_PROTOCOL_PATH)
    qc, qa = qconfig(config), qattacks(config)
    primitives = read_rows(raw / "EXECUTION_ROWS.jsonl")
    package_rows = read_rows(raw / "RECEIPT_PACKAGES.jsonl")
    attack_rows = read_rows(raw / "ATTACK_INPUTS.jsonl")
    packages = {row["package_id"]: row for row in package_rows}
    primitive_index = {row["package_id"]: row for row in primitives}
    execution_results = []
    for package in package_rows:
        primitive = primitive_index[package["package_id"]]
        expected = {
            field: primitive[field]
            for field in ("run_id", "context_id", "execution_id", "nonce")
        }
        expected["checkpoint_sha256"] = qc["lineage"]["M1_checkpoint_sha256"]
        execution_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.execution_verdict.v1",
            "package_id": package["package_id"],
            "role": package["role"],
            "decisions": D0.evaluate_bundle(
                key, package["receipts"], expected, protocol
            ),
        })
    attack_results = []
    expected_matches = 0
    for attack in attack_rows:
        reference = (
            packages[attack["reference_package_id"]]["receipts"]
            if attack["reference_package_id"] else None
        )
        decisions = D0.evaluate_bundle(
            key, D0.resolve_attack(attack, packages),
            attack["expected_binding"], protocol, reference,
        )
        expected = qa["families"][attack["family"]]["expected"]
        matches = all(decisions[field] == value for field, value in expected.items())
        expected_matches += int(matches)
        attack_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "family": attack["family"],
            "decisions": decisions,
            "expected": expected,
            "matches_expected": matches,
        })
    reference_result = D0_REF.evaluate_all(
        key, primitives, package_rows, attack_rows, protocol, qa, qc
    )
    normalized = {"executions": execution_results, "attacks": attack_results}
    production_digest = digest(normalized)
    agreement = production_digest == reference_result["result_digest"]
    genuine_required = (
        "exact_z2_event", "abstract_z2_state", "exact_z3_event",
        "abstract_z3_state", "chain_authenticator",
        "registered_transition_consistency",
        "registered_answer_consistency", "chain_consistency",
    )
    genuine_passes = sum(
        all(row["decisions"][field] == "ACCEPT" for field in genuine_required)
        for row in execution_results
    )
    role = config["protected_role"]
    denominators = (
        len(execution_results) == role["execution_packages"]
        and len(attack_rows) == role["attack_rows"]
        and raw_manifest["closures"] == role["closures"]
    )
    genuine_ok = genuine_passes == role["execution_packages"]
    attacks_ok = expected_matches == role["attack_rows"]
    zero_other = all(value == 0 for value in config["zero_other_role_access"].values())
    passed = (
        denominators and genuine_ok and attacks_ok and agreement
        and reference_result["passed"] and zero_other
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    output.mkdir(parents=True)
    write_rows(output / "EXECUTION_VERDICTS.jsonl", execution_results)
    write_rows(output / "ATTACK_VERDICTS.jsonl", attack_results)
    write_json(output / "REFERENCE_RESULT.json", reference_result)
    write_json(output / "METRICS.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.metrics.v1",
        "closures": raw_manifest["closures"],
        "oriented_rows": raw_manifest["oriented_rows"],
        "execution_packages": len(execution_results),
        "receipt_envelopes": len(execution_results) * 5,
        "genuine_packages_passing_all_gates": genuine_passes,
        "attack_rows": len(attack_rows),
        "attack_expected_matches": expected_matches,
        "production_result_digest": production_digest,
        "reference_result_digest": reference_result["result_digest"],
        "production_reference_canonical_agreement": agreement,
        "zero_other_role_access": zero_other,
        "passed": passed,
    })
    if not denominators:
        terminal_name = config["terminals"]["evidence_incomplete"]
    elif not genuine_ok:
        terminal_name = config["terminals"]["scientific_gate_failed"]
    elif not attacks_ok:
        terminal_name = config["terminals"]["attack_failed"]
    elif not agreement or not reference_result["passed"]:
        terminal_name = config["terminals"]["reference_disagreement"]
    else:
        terminal_name = config["terminals"]["valid"]
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.terminal.v1",
        "terminal": terminal_name,
        "passed": passed,
        "qualification_evidence_only": True,
        "receipt_final_confirmation": False,
        "REC_C1_authorized": passed,
        "carrier_authorized": False,
        "other_role_releases": config["zero_other_role_access"],
    }
    write_json(output / "TERMINAL.json", terminal)
    write_json(output / "EVALUATION_BINDING.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.evaluation_binding.v1",
        "source_commit": source_commit,
        "raw_evidence_commit": raw_commit,
        "key_reveal_commit": reveal_commit,
        "execution_verdicts_sha256": file_sha(output / "EXECUTION_VERDICTS.jsonl"),
        "attack_verdicts_sha256": file_sha(output / "ATTACK_VERDICTS.jsonl"),
        "reference_result_sha256": file_sha(output / "REFERENCE_RESULT.json"),
        "metrics_sha256": file_sha(output / "METRICS.json"),
    })
    return terminal


def preflight() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_frozen_D0(config)
    qa = qattacks(config)
    checks = {
        "D0_attack_families_unchanged": set(qa["families"])
        == set(D0.read_json(D0.ATTACK_PATH)["families"]),
        "new_attack_families_zero": config["D0_binding"]["new_attack_families"] == 0,
        "full_cartesian_attacks": qa["expected_attack_rows"]
        == config["protected_role"]["execution_packages"] * len(qa["families"]),
        "exact_role_denominators": (
            config["protected_role"]["closures"] == 96
            and config["protected_role"]["oriented_rows"] == 192
            and config["protected_role"]["execution_packages"] == 384
            and config["protected_role"]["receipt_envelopes"] == 1920
            and config["protected_role"]["attack_rows"] == 6912
        ),
        "zero_other_role_access": all(
            value == 0 for value in config["zero_other_role_access"].values()
        ),
        "no_carrier_authorization": config["authorizations"]["carrier"] is False,
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_q1.preflight.v1",
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    auth = sub.add_parser("authorize")
    auth.add_argument("--source-commit", required=True)
    auth.add_argument("--output", type=Path, required=True)
    key = sub.add_parser("create-key")
    key.add_argument("--source-commit", required=True)
    key.add_argument("--authorization", type=Path, required=True)
    key.add_argument("--authorization-commit", required=True)
    key.add_argument("--key-out", type=Path, required=True)
    key.add_argument("--commitment-out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--authorization-commit", required=True)
    run.add_argument("--key", type=Path, required=True)
    run.add_argument("--commitment", type=Path, required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--sealed-allocation", type=Path, required=True)
    run.add_argument("--sealed-manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    reveal_parser = sub.add_parser("reveal")
    reveal_parser.add_argument("--source-commit", required=True)
    reveal_parser.add_argument("--authorization", type=Path, required=True)
    reveal_parser.add_argument("--authorization-commit", required=True)
    reveal_parser.add_argument("--key", type=Path, required=True)
    reveal_parser.add_argument("--commitment", type=Path, required=True)
    reveal_parser.add_argument("--commitment-commit", required=True)
    reveal_parser.add_argument("--raw-manifest", type=Path, required=True)
    reveal_parser.add_argument("--raw-commit", required=True)
    reveal_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--source-commit", required=True)
    evaluate_parser.add_argument("--raw", type=Path, required=True)
    evaluate_parser.add_argument("--raw-commit", required=True)
    evaluate_parser.add_argument("--reveal", type=Path, required=True)
    evaluate_parser.add_argument("--reveal-commit", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight()
        elif args.command == "authorize":
            result = create_authorization(args.source_commit, args.output)
        elif args.command == "create-key":
            result = create_key(
                args.source_commit, args.authorization, args.authorization_commit,
                args.key_out, args.commitment_out,
            )
        elif args.command == "execute":
            result = execute(
                args.source_commit, args.authorization, args.authorization_commit,
                args.key, args.commitment, args.commitment_commit,
                args.sealed_allocation, args.sealed_manifest, args.output,
            )
        elif args.command == "reveal":
            result = reveal(
                args.source_commit, args.authorization, args.authorization_commit,
                args.key, args.commitment, args.commitment_commit,
                args.raw_manifest, args.raw_commit, args.output,
            )
        else:
            result = evaluate(
                args.source_commit, args.raw, args.raw_commit,
                args.reveal, args.reveal_commit, args.output,
            )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
