#!/usr/bin/env python3
"""Materialize the static closure of the incoherent AGG-D0 lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol/csw-am-scb-m1-car-agg-d0-closure/CLOSURE_SPEC.json"
SOURCE_FILES = (
    "protocol/csw-am-scb-m1-car-agg-d0-closure/CLOSURE_SPEC.json",
    "tools/csw_am_scb_m1_car_agg_d0_closure.py",
    "tools/csw_am_scb_m1_car_agg_d0_closure_reference.py",
    "tests/test_csw_am_scb_m1_car_agg_d0_closure.py",
)


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


def validate_source_commit(source_commit: str) -> dict[str, str]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("SOURCE_COMMIT")
    digests: dict[str, str] = {}
    for relative in SOURCE_FILES:
        local = (ROOT / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", source_commit + ":" + relative], cwd=ROOT
        )
        if local != committed:
            raise ValueError("SOURCE_BYTES:" + relative)
        digests[relative] = sha(local)
    return digests


def validate_predecessor(spec: dict[str, Any]) -> dict[str, Any]:
    predecessor = spec["predecessor"]
    commit = predecessor["evidence_commit"]
    tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", commit], cwd=ROOT, text=True
    ).strip()
    if tree != predecessor["evidence_tree"]:
        raise ValueError("PREDECESSOR_TREE")
    for relative, expected in spec["source_bindings"].items():
        local = ROOT / relative
        if file_sha(local) != expected:
            raise ValueError("SOURCE_HASH:" + relative)
        committed = subprocess.check_output(
            ["git", "show", commit + ":" + relative], cwd=ROOT
        )
        if committed != local.read_bytes():
            raise ValueError("PREDECESSOR_BYTES:" + relative)

    base = ROOT / "artifacts/dev_only/csw-am-scb-m1-car-agg-d0"
    audit = read_json(base / "MAIN_AUDIT.json")
    terminal = read_json(base / "TERMINAL.json")
    official = read_json(base / "OFFICIAL_RESULT.json")
    reference = read_json(base / "REFERENCE_RESULT.json")
    preflight = read_json(base / "LITERAL_PREFLIGHT.json")
    expected = spec["expected_accounting"]
    if (
        audit["main_verdict"] != predecessor["scientific_terminal"]
        or terminal["terminal"] != predecessor["scientific_terminal"]
        or official["terminal"] != predecessor["scientific_terminal"]
        or reference["terminal"] != predecessor["scientific_terminal"]
        or terminal["status"] != predecessor["status"]
        or reference["valid"] is not True
    ):
        raise ValueError("PREDECESSOR_TERMINAL")
    if audit["access"] != {
        "calibration_closures": expected["calibration_closures"],
        "development_fixture_closures": expected["development_fixture_closures"],
        "protected_closures": expected["protected_closures"],
        "reserve_closures": expected["reserve_closures"],
        "validation_closures": expected["validation_closures"],
    }:
        raise ValueError("ACCESS_ACCOUNTING")
    if (
        audit["execution"]["complete_development_outputs_generated"] != 0
        or audit["execution"]["complete_development_run_started"] is not False
        or official["complete_development_closures_executed"]
        != expected["complete_development_closures"]
        or audit["thresholds"] != {"M": "NOT_YET_INSTANTIATED", "T": "NOT_YET_INSTANTIATED"}
    ):
        raise ValueError("EXECUTION_ACCOUNTING")
    return preflight


def reconstruct_divergence(spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / (
        "artifacts/dev_only/csw-am-scb-m1-car-agg-d0/"
        "PRIMITIVE_DRAW_COMPARISONS.jsonl"
    )
    rows = read_rows(path)
    expected = spec["expected_accounting"]
    if len(rows) != expected["primitive_draw_comparisons"]:
        raise ValueError("DRAW_COUNT")
    steps = [row["model_step"] for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError("DRAW_STEP_SEQUENCE")
    closure_ids = {row["closure_id"] for row in rows}
    draws = {row["draw"] for row in rows}
    if len(closure_ids) != 1 or draws != {0}:
        raise ValueError("FIXTURE_IDENTITY")
    token_divergences = [
        row["model_step"]
        for row in rows
        if row["natural_selected_token"] != row["alias_selected_token"]
    ]
    uniform_divergences = [
        row["model_step"]
        for row in rows
        if row["natural_base_uniform"] != row["alias_base_uniform"]
    ]
    if (
        min(token_divergences) != expected["first_selected_token_divergence_step"]
        or min(uniform_divergences) != expected["first_primitive_draw_divergence_step"]
        or any(row["uniform_equal"] != (
            row["natural_base_uniform"] == row["alias_base_uniform"]
        ) for row in rows)
    ):
        raise ValueError("DIVERGENCE_RECONSTRUCTION")
    return {
        "closure_id": next(iter(closure_ids)),
        "operation_count": len(rows),
        "first_selected_token_divergence_step": min(token_divergences),
        "first_primitive_draw_divergence_step": min(uniform_divergences),
        "draws_equal_before_token_divergence": all(
            row["uniform_equal"]
            for row in rows
            if row["model_step"] <= min(token_divergences)
        ),
        "draws_diverge_immediately_after_history_diverges": (
            min(uniform_divergences) == min(token_divergences) + 1
        ),
    }


def claims_matrix() -> list[dict[str, str]]:
    return [
        {"claim": "Receipt and authority path in literal fixture", "status": "OPERATIONAL"},
        {"claim": "Frozen sampler and primitive-draw synchronization jointly coherent", "status": "VALID_NEGATIVE"},
        {"claim": "Aggregate implementation completed", "status": "NO"},
        {"claim": "Aggregate detector performance", "status": "UNTESTED"},
        {"claim": "Threshold T", "status": "NOT_INSTANTIATED"},
        {"claim": "Margin M", "status": "NOT_INSTANTIATED"},
        {"claim": "Output-side provenance", "status": "NOT_ESTABLISHED"},
        {"claim": "Position-indexed successor", "status": "NOT_AUTHORIZED"},
    ]


def resource_status() -> dict[str, Any]:
    return {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.resources.v1",
        "consumed": {
            "public_development_fixture_closures": 1,
            "complete_public_development_closures": 0,
        },
        "untouched": {
            "calibration_closures": 512,
            "validation_closures": 512,
            "public_reserve_closures": 640,
            "protected_qualification_closures": 128,
            "protected_confirmation_closures": 256,
        },
        "release_authorizations": {
            "AGG_CAL1": False,
            "AGG_V1": False,
            "AGG_Q1": False,
            "AGG_C1": False,
            "RNG2": False,
        },
    }


def rng2_boundary() -> dict[str, Any]:
    return {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.rng2_boundary.v1",
        "status": "DRAFT_NON_AUTHORIZING",
        "scientific_question": (
            "Can a newly qualified position-indexed common-random-number channel "
            "support reliable receipt-bound output provenance?"
        ),
        "relation_to_closed_lineage": "NEW_CHANNEL_NOT_PATCH_OR_REPAIR",
        "requirements_before_execution": [
            "EXPLICIT_PROGRAMME_LEVEL_OUTPUT_CARRIAGE_DECISION",
            "STATIC_POSITION_INDEXED_SAMPLER_PROTOCOL",
            "FIXED_49_MODEL_STEP_DRAW_SCHEDULE",
            "CANONICAL_PRF_INPUT_AND_UNIFORM_CONVERSION",
            "TOKEN_ORDER_PRECISION_SOFTMAX_AND_BOUNDARY_RULES",
            "RANDOMNESS_CONTEXT_SEPARATE_FROM_STATE_ROLE_AND_RECEIPT_IDENTITY",
            "SEQUENCE_LENGTH_AND_NONE_SEMANTICS",
            "DISTRIBUTIONAL_CORRECTNESS_TESTS",
            "FRESH_ALLOCATION_OR_EXPLICIT_ROLE_MIGRATION",
            "GENERIC_CHANNEL_QUALIFICATION_BEFORE_RECEIPT_REBINDING",
        ],
        "prohibitions": [
            "NO_AGG_CAL1",
            "NO_AUTOMATIC_RNG2_AUTHORIZATION",
            "NO_SILENT_ROLE_MIGRATION",
            "NO_PROTECTED_RELEASE",
            "NO_INHERITANCE_OF_A0_Q1_POWER_OR_ERROR_CLAIMS",
        ],
        "successor_authorized": False,
        "role_migration_authorized": False,
        "new_stochastic_execution_authorized": False,
        "protected_release_authorized": False,
    }


def materialize(source_commit: str, output: Path) -> None:
    if output.exists():
        raise ValueError("OUTPUT_BOUNDARY")
    spec = read_json(SPEC_PATH)
    source_digests = validate_source_commit(source_commit)
    preflight = validate_predecessor(spec)
    divergence = reconstruct_divergence(spec)
    output.mkdir(parents=True)
    closure = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.record.v1",
        "identity": spec["identity"],
        "source_commit": source_commit,
        "predecessor_evidence_commit": spec["predecessor"]["evidence_commit"],
        "predecessor_scientific_terminal_preserved": spec["predecessor"]["scientific_terminal"],
        "scientific_status": "PROTOCOL_INCOHERENT",
        "reason": (
            "The frozen history-conditioned sampler makes post-divergence primitive "
            "draws history-dependent, while the frozen paired-control contract "
            "requires those primitive draws to remain identical."
        ),
        "diagnostic_fixture": divergence,
        "receipt_and_authority_path": "OPERATIONAL_IN_LITERAL_FIXTURE",
        "aggregate_implementation": "NOT_COMPLETED",
        "aggregate_detector_performance": "UNTESTED",
        "threshold_T": "NOT_YET_INSTANTIATED",
        "margin_M": "NOT_YET_INSTANTIATED",
        "output_side_provenance": "NOT_ESTABLISHED",
        "AGG_CAL1": "PROHIBITED",
        "automatic_successor": "PROHIBITED",
    }
    claims = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.claims.v1",
        "claims": claims_matrix(),
    }
    resources = resource_status()
    successor = rng2_boundary()
    write_json(output / "CLOSURE.json", closure)
    write_json(output / "CLAIMS_MATRIX.json", claims)
    write_json(output / "RESOURCE_STATUS.json", resources)
    write_json(output / "RNG2_SUCCESSOR_BOUNDARY.json", successor)
    official = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.official.v1",
        "source_commit": source_commit,
        "source_digests": source_digests,
        "predecessor_fixture_payload_equal": preflight["semantic_payload_equal"],
        "predecessor_fixture_stream_seed_equal": preflight["paired_stream_seed_equal"],
        "divergence": divergence,
        "closure_rules_satisfied": all(spec["closure_rules"].values()),
        "AGG_CAL1_authorized": False,
        "RNG2_authorized": False,
        "passed": True,
        "terminal": spec["closure_terminal"],
    }
    write_json(output / "OFFICIAL_RESULT.json", official)
    manifest = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.manifest.v1",
        "source_commit": source_commit,
        "files": {
            path.name: file_sha(path)
            for path in sorted(output.iterdir())
        },
    }
    write_json(output / "MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_commit, args.output)


if __name__ == "__main__":
    main()
