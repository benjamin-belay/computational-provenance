#!/usr/bin/env python3
"""Threshold-independent aggregate carrier implementation preflight."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol/csw-am-scb-m1-car-agg-d0"
CONFIG_PATH = PROTOCOL_DIR / "CONFIG.json"
SCHEMA_PATH = PROTOCOL_DIR / "EVIDENCE_SCHEMA.json"
TEST_COMMIT = "TEST_ONLY"


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


OLD = load("car_d0_for_agg_d0", "tools/csw_am_scb_m1_car_d0.py")
A0 = OLD.A0
CAR = OLD.CAR
REC = OLD.REC
M1 = OLD.M1


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
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


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


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_source(source_commit: str) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config["identity"] != "CSW-AM-SCB-M1-CAR-AGG-D0":
        raise ValueError("IDENTITY")
    if source_commit != TEST_COMMIT:
        if len(source_commit) != 40:
            raise ValueError("SOURCE_COMMIT")
        for relative in config["source_paths"]:
            if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
                raise ValueError("SOURCE_BYTES:" + relative)
    lineage = config["lineage"]
    bindings = (
        ("r0_audit_path", "r0_audit_sha256"),
        ("development_release_path", "development_release_sha256"),
        ("development_closures_path", "development_closures_sha256"),
        ("development_rows_path", "development_rows_sha256"),
        ("agg_s0_detector_path", "agg_s0_detector_sha256"),
        ("car_d0_production_path", "car_d0_production_sha256"),
        ("a0_production_path", "a0_production_sha256"),
        ("m1_production_path", "m1_production_sha256"),
        ("receipt_production_path", "receipt_production_sha256"),
        ("authority_protocol_path", "authority_protocol_sha256"),
        ("realiser_boundary_path", "realiser_boundary_sha256"),
    )
    for path_key, hash_key in bindings:
        if file_sha(ROOT / lineage[path_key]) != lineage[hash_key]:
            raise ValueError("LINEAGE_HASH:" + path_key)
    if read_json(ROOT / lineage["r0_audit_path"])["main_verdict"] != config[
        "lineage"
    ]["r0_terminal"]:
        raise ValueError("R0_TERMINAL")
    return config


def development_population(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = read_rows(ROOT / config["lineage"]["development_rows_path"])
    closures: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        closures.setdefault(row["closure_id"], []).append(row)
    if (
        len(rows) != config["development_population"]["oriented_rows"]
        or len(closures) != config["development_population"]["closures"]
    ):
        raise ValueError("DEVELOPMENT_DENOMINATOR")
    for closure_id, members in closures.items():
        if (
            len(members) != 2
            or {row["orientation"] for row in members}
            != {"LOWER_C_ORIENTATION", "UPPER_C_ORIENTATION"}
            or any(row["closure_id"] != closure_id for row in members)
        ):
            raise ValueError("CLOSURE_INTEGRITY:" + closure_id)
    return closures


def normalized(green_count: int, eligible: int) -> float:
    if type(green_count) is not int or type(eligible) is not int or eligible <= 0:
        raise ValueError("NORMALIZATION_INPUT")
    if green_count < 0 or green_count > eligible:
        raise ValueError("GREEN_COUNT")
    return (green_count - eligible / 2) / math.sqrt(eligible / 4)


def aggregate_side(evidence: list[list[float]]) -> list[float]:
    if len(evidence) != 8 or any(len(row) != 16 for row in evidence):
        raise ValueError("AGGREGATE_SHAPE")
    return [
        sum(row[category] for row in evidence) / math.sqrt(8)
        for category in range(16)
    ]


def threshold_independent_side(
    closure_id: str, side: str, authenticated: int, scores: list[float]
) -> dict[str, Any]:
    if len(scores) != 16 or not 0 <= authenticated < 16:
        raise ValueError("SIDE_SCORE_SHAPE")
    strongest_wrong = max(
        (score, category)
        for category, score in enumerate(scores)
        if category != authenticated
    )
    rank = 1 + sum(score > scores[authenticated] for score in scores)
    return {
        "closure_id": closure_id,
        "side": side,
        "sixteen_aggregate_scores": scores,
        "authenticated_authority": authenticated,
        "authenticated_rank": rank,
        "strongest_wrong_authority": strongest_wrong[1],
        "strongest_wrong_score": strongest_wrong[0],
        "raw_top_two_margin": scores[authenticated] - strongest_wrong[0],
        "absolute_threshold_T": "NOT_YET_INSTANTIATED",
        "margin_threshold_M": "NOT_YET_INSTANTIATED",
        "absolute_threshold_decision": "NOT_YET_INSTANTIATED",
        "margin_decision": "NOT_YET_INSTANTIATED",
    }


def strict_quantile(values: list[float], alpha: float) -> float:
    if not values or not 0 < alpha < 1:
        raise ValueError("QUANTILE_INPUT")
    ordered = sorted(values)
    index = math.ceil((len(ordered) + 1) * (1 - alpha)) - 1
    return ordered[min(index, len(ordered) - 1)]


def derive_thresholds(
    carrier_off_max16: list[float], wrong_claim_gaps: list[float],
    absolute_alpha: float = 0.005, confusion_alpha: float = 0.005,
) -> dict[str, float]:
    return {
        "T": strict_quantile(carrier_off_max16, absolute_alpha),
        "M": max(0.0, strict_quantile(wrong_claim_gaps, confusion_alpha)),
    }


def receipt_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(REC.read_json(REC.CONFIG_PATH))
    value["identity"] = config["identity"]
    value["key_lifecycle"]["commitment_domain"] = (
        "EXP33A|SCB-M1|CAR-AGG-D0|FIXTURE-RECEIPT|V1"
    )
    value["key_lifecycle"]["run_id_domain"] = (
        "EXP33A|SCB-M1|CAR-AGG-D0|FIXTURE-RUN|V1"
    )
    value["key_lifecycle"]["nonce_domain"] = (
        "EXP33A|SCB-M1|CAR-AGG-D0|FIXTURE-NONCE|V1"
    )
    value["public_population"]["execution_packages"] = 2
    value["lineage"]["M1_checkpoint_sha256"] = config["lineage"][
        "m1_checkpoint_sha256"
    ]
    return value


def expected_binding(primitive: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": primitive["run_id"],
        "context_id": primitive["context_id"],
        "execution_id": primitive["execution_id"],
        "nonce": primitive["nonce"],
        "checkpoint_sha256": config["lineage"]["m1_checkpoint_sha256"],
    }


def fixture_keys() -> tuple[bytes, bytes]:
    return (
        hashlib.sha256(b"EXP33A|CAR-AGG-D0|PUBLIC-FIXTURE-RECEIPT|V1").digest(),
        hashlib.sha256(b"EXP33A|CAR-AGG-D0|PUBLIC-FIXTURE-CARRIER|V1").digest(),
    )


def compact_operation(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_step": operation["model_step"],
        "kind": operation["kind"],
        "field": operation.get("field"),
        "uniform": operation["uniform"],
        "selected_token": operation["selected_token"],
    }


def diagnose(source_commit: str, output_dir: Path) -> dict[str, Any]:
    config = validate_source(source_commit)
    if output_dir.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    closures = development_population(config)
    closure_id = min(closures)
    row = next(
        row for row in closures[closure_id]
        if row["orientation"] == "LOWER_C_ORIENTATION"
    )
    model = M1.load_model(M1.read_json(M1.CONFIG_PATH))
    surface = OLD.load_surface(OLD.read_json(OLD.CONFIG_PATH))
    values = M1.tensors([row])
    with torch.no_grad():
        natural = model.run_natural(values)
        target = torch.tensor(
            [(row["natural"]["z2"] + 8) % 16], dtype=torch.long
        )
        alias = model.run_with_z2(values, target)
    predicted = int(natural["z2_selected"][0])
    observations = (
        ("NATURAL", {
            "z2_consumed": int(natural["z2_consumed"][0]),
            "z3_selected": int(natural["z3_selected"][0]),
            "z3_consumed": int(natural["z3_consumed"][0]),
            "answer": int(natural["answer_logits"].argmax(-1)[0]),
        }, predicted),
        ("ALIAS_Z2", {
            "z2_consumed": int(alias["z2_consumed"][0]),
            "z3_selected": int(alias["z3_selected"][0]),
            "z3_consumed": int(alias["z3_consumed"][0]),
            "answer": int(alias["answer_logits"].argmax(-1)[0]),
        }, int(target[0])),
    )
    receipt_key, carrier_root = fixture_keys()
    rconfig = receipt_config(config)
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    authority_spec = CAR.read_json(CAR.AUTHORITY_PATH)
    payload = CAR.surface_payload(
        row["prompt_text"].encode(), row["natural"]["answer"],
        CAR.read_json(CAR.BOUNDARY_PATH),
    )
    stream_seed = CAR.paired_nonce(closure_id, 0)
    executions = []
    operations: dict[str, list[dict[str, Any]]] = {}
    for side, observed, target_z2 in observations:
        primitive, package = REC.receipt_package(
            receipt_key, protocol, rconfig, row, side,
            observed, predicted, target_z2,
        )
        decisions = REC.evaluate_bundle(
            receipt_key, package["receipts"],
            expected_binding(primitive, config), protocol,
        )
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        authority = CAR.derive_authority(
            carrier_root, category, decisions["abstract_z2_state"], authority_spec
        )
        if authority is None:
            raise ValueError("RECEIPT_GATING")
        generated = A0.generate(
            surface, payload, stream_seed=stream_seed, target_key=authority,
            carrier_enabled=True, bias=config["channel"]["bias"],
            retain_operations=True,
        )
        operations[side] = [compact_operation(item) for item in generated["operations"]]
        executions.append({
            "side": side,
            "consumed_z2": observed["z2_consumed"],
            "consumed_z3": observed["z3_consumed"],
            "answer": observed["answer"],
            "receipt_decisions": decisions,
            "authority_sha256": sha(authority),
            "semantic_payload_sha256": sha(canonical(payload)),
            "stream_seed_sha256": sha(stream_seed),
            "literal_output": generated["literal_output"],
            "output_sha256": generated["output_sha256"],
            "eligible_positions": generated["eligible_positions"],
            "operation_count": len(generated["operations"]),
        })
    if len(operations["NATURAL"]) != len(operations["ALIAS_Z2"]):
        raise ValueError("OPERATION_COUNT")
    comparisons = []
    for natural_op, alias_op in zip(
        operations["NATURAL"], operations["ALIAS_Z2"]
    ):
        if natural_op["model_step"] != alias_op["model_step"]:
            raise ValueError("MODEL_STEP")
        comparisons.append({
            "closure_id": closure_id,
            "draw": 0,
            "model_step": natural_op["model_step"],
            "kind": natural_op["kind"],
            "field": natural_op["field"],
            "natural_base_uniform": natural_op["uniform"],
            "alias_base_uniform": alias_op["uniform"],
            "uniform_equal": natural_op["uniform"] == alias_op["uniform"],
            "natural_selected_token": natural_op["selected_token"],
            "alias_selected_token": alias_op["selected_token"],
        })
    aligned = all(row["uniform_equal"] for row in comparisons)
    first_mismatch = next(
        (row["model_step"] for row in comparisons if not row["uniform_equal"]),
        None,
    )
    output_dir.mkdir(parents=True)
    write_rows(output_dir / "PRIMITIVE_DRAW_COMPARISONS.jsonl", comparisons)
    diagnostic = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0.diagnostic.v1",
        "source_commit": source_commit,
        "fixture_closure_id": closure_id,
        "fixture_selection": config["development_population"][
            "selection_for_literal_preflight"
        ],
        "development_closure_count": len(closures),
        "executions": executions,
        "paired_stream_seed_equal": (
            executions[0]["stream_seed_sha256"]
            == executions[1]["stream_seed_sha256"]
        ),
        "semantic_payload_equal": (
            executions[0]["semantic_payload_sha256"]
            == executions[1]["semantic_payload_sha256"]
        ),
        "primitive_draw_comparisons": len(comparisons),
        "primitive_draws_aligned": aligned,
        "first_mismatched_model_step": first_mismatch,
        "frozen_sampler": config["channel"]["sampler"],
        "position_only_randomness_substitution": False,
        "threshold_T": "NOT_YET_INSTANTIATED",
        "threshold_M": "NOT_YET_INSTANTIATED",
    }
    write_json(output_dir / "LITERAL_PREFLIGHT.json", diagnostic)
    gates = {
        "complete_development_release_bound": len(closures) == 256,
        "receipt_verification_precedes_authority": all(
            all(value == "ACCEPT" for value in item["receipt_decisions"].values())
            for item in executions
        ),
        "semantic_payload_equal": diagnostic["semantic_payload_equal"],
        "paired_stream_seed_equal": diagnostic["paired_stream_seed_equal"],
        "all_operations_compared": len(comparisons) == 49,
        "paired_primitive_random_draw_alignment": aligned,
        "thresholds_uninstantiated": (
            diagnostic["threshold_T"] == diagnostic["threshold_M"]
            == "NOT_YET_INSTANTIATED"
        ),
        "no_sampler_substitution": not diagnostic[
            "position_only_randomness_substitution"
        ],
        "no_future_role_access": True,
    }
    terminal = (
        config["terminals"]["ready"] if all(gates.values())
        else config["terminals"]["randomness_incoherent"]
        if (
            all(value for key, value in gates.items()
                if key != "paired_primitive_random_draw_alignment")
            and not gates["paired_primitive_random_draw_alignment"]
        )
        else config["terminals"]["invalid"]
    )
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0.official_result.v1",
        "terminal": terminal,
        "gates": gates,
        "literal_fixture_closures": 1,
        "complete_development_run_started": False,
        "complete_development_closures_executed": 0,
        "complete_development_outputs_generated": 0,
        "calibration_closures_accessed": 0,
        "validation_closures_accessed": 0,
        "reserve_closures_accessed": 0,
        "protected_closures_accessed": 0,
        "AGG_CAL1_authorized": terminal == config["terminals"]["ready"],
        "AGG_V1_authorized": False,
        "protected_release_authorized": False,
    }
    write_json(output_dir / "OFFICIAL_RESULT.json", result)
    write_json(output_dir / "TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0.terminal.v1",
        "terminal": terminal,
        "status": (
            "IMPLEMENTATION_READY" if terminal == config["terminals"]["ready"]
            else "FROZEN_SAMPLER_AND_PAIRED_RANDOMNESS_CONTRACT_INCOHERENT"
        ),
        "calibration_release_authorized": False,
        "next_boundary": (
            "SEPARATE_AGG_CAL1_AUTHORIZATION_REQUEST"
            if terminal == config["terminals"]["ready"]
            else "STOP_AGGREGATE_CARRIER_LINEAGE"
        ),
    })
    write_json(output_dir / "EXECUTABLE_MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0.executable_manifest.v1",
        "source_commit": source_commit,
        "source_files": {
            relative: file_sha(ROOT / relative)
            for relative in config["source_paths"]
        },
        "frozen_dependencies": {
            key.removesuffix("_path"): {
                "path": value,
                "sha256": config["lineage"][key.removesuffix("_path") + "_sha256"],
            }
            for key, value in config["lineage"].items()
            if key.endswith("_path")
            and key.removesuffix("_path") + "_sha256" in config["lineage"]
        },
        "calibration_code_frozen": True,
        "thresholds_instantiated": False,
        "future_role_access": 0,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = diagnose(args.source_commit, args.output_dir)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
