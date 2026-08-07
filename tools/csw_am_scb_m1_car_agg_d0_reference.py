#!/usr/bin/env python3
"""Independent reconstruction of the aggregate-carrier D0 invariant preflight."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-car-agg-d0/CONFIG.json"


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A0 = load("a0_for_agg_d0_ref", "tools/csw_a0_stochastic_channel.py")
M1 = load("m1_for_agg_d0_ref", "tools/csw_am_scb_m1.py")
OLD = load("old_car_d0_for_agg_d0_ref", "tools/csw_am_scb_m1_car_d0.py")


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
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def state_authority(root: bytes, category: int, spec: dict[str, Any]) -> bytes:
    fixed = spec["verified_claim_projection"]["fixed_values"]
    projection = {
        "carrier_protocol_version": fixed["carrier_protocol_version"],
        "state_ontology_id": fixed["state_ontology_id"],
        "model_domain": fixed["model_domain"],
        "checkpoint_sha256": fixed["checkpoint_sha256"],
        "state_type": fixed["state_type"],
        "category": category,
    }
    return hmac.new(
        root,
        spec["derivation"]["domain"].encode() + b"\x00" + canonical(projection),
        hashlib.sha256,
    ).digest()


def paired_nonce(closure_id: str, draw: int) -> bytes:
    return hashlib.sha256(
        b"SCB_M1_CAR_PAIRED_NONCE_V1\x00"
        + closure_id.encode() + b"\x00" + draw.to_bytes(4, "big")
    ).digest()


def compact(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_step": operation["model_step"],
        "kind": operation["kind"],
        "field": operation.get("field"),
        "uniform": operation["uniform"],
        "selected_token": operation["selected_token"],
    }


def reconstruct(
    source_commit: str, evidence_dir: Path, output_path: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if source_commit != "TEST_ONLY":
        for relative in config["source_paths"]:
            if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
                raise ValueError("SOURCE_BINDING:" + relative)
    diagnostic = read_json(evidence_dir / "LITERAL_PREFLIGHT.json")
    retained = read_rows(evidence_dir / "PRIMITIVE_DRAW_COMPARISONS.jsonl")
    development = read_rows(ROOT / config["lineage"]["development_rows_path"])
    closure_id = min({row["closure_id"] for row in development})
    row = next(
        row for row in development
        if row["closure_id"] == closure_id
        and row["orientation"] == "LOWER_C_ORIENTATION"
    )
    model = M1.load_model(M1.read_json(M1.CONFIG_PATH))
    surface = OLD.load_surface(OLD.read_json(OLD.CONFIG_PATH))
    values = M1.tensors([row])
    with torch.no_grad():
        natural = model.run_natural(values)
        alias_category = (row["natural"]["z2"] + 8) % 16
        alias = model.run_with_z2(
            values, torch.tensor([alias_category], dtype=torch.long)
        )
    natural_category = int(natural["z2_consumed"][0])
    alias_consumed = int(alias["z2_consumed"][0])
    root = hashlib.sha256(
        b"EXP33A|CAR-AGG-D0|PUBLIC-FIXTURE-CARRIER|V1"
    ).digest()
    spec = read_json(ROOT / config["lineage"]["authority_protocol_path"])
    payload = OLD.CAR.surface_payload(
        row["prompt_text"].encode(), row["natural"]["answer"],
        read_json(ROOT / config["lineage"]["realiser_boundary_path"]),
    )
    operations = []
    for category in (natural_category, alias_consumed):
        generated = A0.generate(
            surface, payload, stream_seed=paired_nonce(closure_id, 0),
            target_key=state_authority(root, category, spec),
            carrier_enabled=True, bias=config["channel"]["bias"],
            retain_operations=True,
        )
        operations.append([compact(item) for item in generated["operations"]])
    reconstructed = []
    for left, right in zip(*operations):
        reconstructed.append({
            "closure_id": closure_id,
            "draw": 0,
            "model_step": left["model_step"],
            "kind": left["kind"],
            "field": left["field"],
            "natural_base_uniform": left["uniform"],
            "alias_base_uniform": right["uniform"],
            "uniform_equal": left["uniform"] == right["uniform"],
            "natural_selected_token": left["selected_token"],
            "alias_selected_token": right["selected_token"],
        })
    aligned = all(row["uniform_equal"] for row in reconstructed)
    first_mismatch = next(
        (row["model_step"] for row in reconstructed if not row["uniform_equal"]),
        None,
    )
    gates = {
        "source_commit": diagnostic["source_commit"] == source_commit,
        "fixture_selection": diagnostic["fixture_closure_id"] == closure_id,
        "complete_retained_comparison": retained == reconstructed,
        "operation_count": len(reconstructed) == 49,
        "same_stream_seed": diagnostic["paired_stream_seed_equal"] is True,
        "same_payload": diagnostic["semantic_payload_equal"] is True,
        "primitive_alignment_reconstructed": (
            diagnostic["primitive_draws_aligned"] == aligned
        ),
        "first_mismatch_reconstructed": (
            diagnostic["first_mismatched_model_step"] == first_mismatch
        ),
        "thresholds_uninstantiated": (
            diagnostic["threshold_T"] == diagnostic["threshold_M"]
            == "NOT_YET_INSTANTIATED"
        ),
        "no_future_access": (
            read_json(evidence_dir / "OFFICIAL_RESULT.json")[
                "calibration_closures_accessed"
            ] == 0
        ),
    }
    expected_terminal = (
        config["terminals"]["ready"] if aligned
        else config["terminals"]["randomness_incoherent"]
    )
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0.reference_result.v1",
        "terminal": expected_terminal,
        "valid": all(gates.values()),
        "gates": gates,
        "primitive_draws_aligned": aligned,
        "first_mismatched_model_step": first_mismatch,
        "complete_development_run_authorized": aligned,
        "AGG_CAL1_authorized": aligned,
        "future_role_access": 0,
        "protected_role_access": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical(result) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if audit_path:
        official = read_json(evidence_dir / "OFFICIAL_RESULT.json")
        terminal = read_json(evidence_dir / "TERMINAL.json")
        audit = {
            "schema": "exp33a.csw_am_scb_m1_car_agg_d0.main_audit.v1",
            "identity": config["identity"],
            "main_verdict": expected_terminal,
            "status": terminal["status"],
            "source_commit": source_commit,
            "evidence": {
                "official_result_sha256": file_sha(
                    evidence_dir / "OFFICIAL_RESULT.json"
                ),
                "reference_result_sha256": file_sha(output_path),
                "literal_preflight_sha256": file_sha(
                    evidence_dir / "LITERAL_PREFLIGHT.json"
                ),
                "primitive_draw_comparisons_sha256": file_sha(
                    evidence_dir / "PRIMITIVE_DRAW_COMPARISONS.jsonl"
                ),
                "official_reference_terminal_agreement": (
                    official["terminal"] == result["terminal"]
                    == terminal["terminal"]
                ),
            },
            "finding": {
                "same_payload": True,
                "same_stream_seed": True,
                "history_conditioned_sampler": config["channel"]["sampler"],
                "primitive_draws_aligned": aligned,
                "first_mismatched_model_step": first_mismatch,
                "position_only_substitution_permitted": False,
                "scientific_interpretation": (
                    "FROZEN_REALISER_AND_PAIRED_PRIMITIVE_RANDOMNESS_"
                    "REQUIREMENT_CANNOT_BOTH_HOLD"
                ),
            },
            "execution": {
                "literal_fixture_closures": 1,
                "complete_development_run_started": False,
                "complete_development_outputs_generated": 0,
            },
            "thresholds": {
                "T": "NOT_YET_INSTANTIATED",
                "M": "NOT_YET_INSTANTIATED",
            },
            "access": {
                "development_fixture_closures": 1,
                "calibration_closures": 0,
                "validation_closures": 0,
                "reserve_closures": 0,
                "protected_closures": 0,
            },
            "authorizations": {
                "AGG_CAL1": False,
                "AGG_V1": False,
                "AGG_Q1": False,
                "AGG_C1": False,
                "protected_release": False,
                "automatic_successor": False,
            },
            "next_boundary": "STOP_AGGREGATE_CARRIER_LINEAGE",
        }
        with audit_path.open("xb") as handle:
            handle.write(canonical(audit) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    try:
        result = reconstruct(
            args.source_commit, args.evidence_dir, args.output, args.audit_output
        )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
