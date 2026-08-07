#!/usr/bin/env python3
"""Independent reconstruction of aggregate-carrier static feasibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "protocol/csw-am-scb-m1-car-agg-s0"


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def stable(value: Any) -> Any:
    if type(value) is float:
        return round(value, 12)
    if type(value) is list:
        return [stable(item) for item in value]
    if type(value) is dict:
        return {key: stable(item) for key, item in value.items()}
    return value


def q_strict(data: list[float], error: float) -> float:
    ordered = sorted(data)
    location = math.ceil((len(ordered) + 1) * (1 - error)) - 1
    return ordered[min(location, len(ordered) - 1)]


def z_value(score: int, positions: int) -> float:
    return (score - 0.5 * positions) / math.sqrt(0.25 * positions)


def combine(left: list[list[float]], right: list[list[float]]) -> list[float]:
    result = []
    for authority in range(16):
        total = 0.0
        for output in left:
            total += output[authority]
        for output in right:
            total += output[authority]
        result.append(total / math.sqrt(8))
    return result


def all_pairs(blocks: list[list[list[float]]]) -> list[list[float]]:
    result = []
    for left in blocks:
        for right in blocks:
            result.append(combine(left, right))
    return result


def claimant_accepts(
    vector: list[float], claimant: int, absolute: float, margin: float,
) -> bool:
    rival = max(value for index, value in enumerate(vector) if index != claimant)
    return (
        vector[claimant] > absolute
        and vector[claimant] > rival
        and vector[claimant] - rival > margin
    )


def wrong_gap(vector: list[float], truth: int) -> float:
    candidates = []
    for claimant in range(16):
        if claimant == truth:
            continue
        rival = max(value for index, value in enumerate(vector) if index != claimant)
        candidates.append(vector[claimant] - rival)
    return max(candidates)


def projection(config: dict[str, Any]) -> dict[str, Any]:
    summaries = read_lines(ROOT / config["lineage"]["D0_output_summaries_path"])
    decisions = [
        row for row in read_lines(ROOT / config["lineage"]["D0_detector_rows_path"])
        if row["authority_family"] == "STATE"
    ]
    scores: dict[str, list[float | None]] = defaultdict(lambda: [None] * 16)
    for row in decisions:
        scores[row["output_id"]][row["authority_category"]] = z_value(
            row["score"], row["eligible_count"]
        )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row["condition"] not in ("STATE", "OFF"):
            continue
        if row["condition"] == "OFF" and row["side"] != "NATURAL":
            continue
        vector = scores[row["output_id"]]
        if any(value is None for value in vector):
            raise ValueError("SCORE_VECTOR")
        groups[(row["condition"], row["closure_id"], row["side"])].append({
            "draw": row["draw"],
            "category": row["source_z2_category"],
            "vector": [float(value) for value in vector],
        })
    state = {
        "NATURAL": [[] for _ in range(16)],
        "ALIAS_Z2": [[] for _ in range(16)],
    }
    off = []
    for (condition, _, side), items in groups.items():
        items.sort(key=lambda item: item["draw"])
        block = [item["vector"] for item in items]
        if len(block) != 4:
            raise ValueError("BLOCK")
        if condition == "OFF":
            off.append(block)
        else:
            state[side][items[0]["category"]].append(block)
    off8 = all_pairs(off)
    threshold = q_strict(
        [max(vector) for vector in off8],
        config["feasibility"]["carrier_off_alpha"],
    )
    expanded = {"NATURAL": [], "ALIAS_Z2": []}
    wrong = []
    for side in expanded:
        for category in range(16):
            for vector in all_pairs(state[side][category]):
                expanded[side].append((category, vector))
                wrong.append(wrong_gap(vector, category))
    raw_margin = q_strict(wrong, config["feasibility"]["wrong_confusion_alpha"])
    margin = max(0.0, raw_margin)
    off_false = 0
    for vector in off8:
        ordered = sorted(vector, reverse=True)
        if ordered[0] > threshold and ordered[0] - ordered[1] > margin:
            off_false += 1
    side_rows = {}
    for side, items in expanded.items():
        correct = sum(
            claimant_accepts(vector, truth, threshold, margin)
            for truth, vector in items
        )
        wrong_accepts = 0
        for truth, vector in items:
            for claimant in range(16):
                if claimant != truth and claimant_accepts(
                    vector, claimant, threshold, margin
                ):
                    wrong_accepts += 1
        side_rows[side] = {
            "synthetic_block_pairs": len(items),
            "authenticated_accepts": correct,
            "projected_side_power": correct / len(items),
            "wrong_claim_accepts": wrong_accepts,
            "wrong_claim_rate_per_side": wrong_accepts / len(items),
        }
    return {
        "planning_blocks": {
            "carrier_off_four_output_blocks": len(off),
            "carrier_off_eight_output_block_pairs": len(off8),
            "state_eight_output_block_pairs": sum(
                item["synthetic_block_pairs"] for item in side_rows.values()
            ),
        },
        "diagnostic_threshold_T": threshold,
        "diagnostic_raw_margin_quantile": raw_margin,
        "diagnostic_margin_M": margin,
        "carrier_off_full_rule_false_accepts": off_false,
        "carrier_off_full_rule_false_accept_rate": off_false / len(off8),
        "sides": side_rows,
        "projected_pair_power": (
            side_rows["NATURAL"]["projected_side_power"]
            * side_rows["ALIAS_Z2"]["projected_side_power"]
        ),
        "projection_role": (
            "PLANNING_ONLY_THRESHOLDS_RECALIBRATED_ON_FUTURE_PUBLIC_CALIBRATION"
        ),
    }


def cumulative(k: int, n: int, p: float) -> float:
    total = 0.0
    for events in range(k + 1):
        total += math.comb(n, events) * p ** events * (1 - p) ** (n - events)
    return total


def upper_limit(events: int, units: int, confidence: float) -> float:
    lo, hi = 0.0, 1.0
    target = 1 - confidence
    for _ in range(100):
        point = (lo + hi) / 2
        if cumulative(events, units, point) > target:
            lo = point
        else:
            hi = point
    return (lo + hi) / 2


def capacity(config: dict[str, Any]) -> dict[str, Any]:
    source_path = ROOT / config["public_roles"]["source_path"]
    if hash_file(source_path) != config["public_roles"]["source_sha256"]:
        raise ValueError("SOURCE")
    excluded = {
        row["closure_id"]
        for row in read_lines(ROOT / config["lineage"]["D0_public_manifest_path"])
    }
    closures = {}
    for row in read_lines(source_path):
        if row["orientation"] == "LOWER_C_ORIENTATION":
            closures[row["closure_id"]] = row
    remaining = [
        row for identifier, row in closures.items() if identifier not in excluded
    ]
    census: dict[tuple[int, int, int], int] = defaultdict(int)
    for row in remaining:
        census[(
            row["natural"]["z2"] % 8,
            row["natural"]["answer"],
            row["template_id"],
        )] += 1
    required = sum(config["public_roles"][name] for name in (
        "AGG_DEVELOPMENT_closures", "AGG_CALIBRATION_closures",
        "AGG_VALIDATION_closures", "remaining_public_reserve_closures",
    ))
    minimum = config["public_roles"]["minimum_each_role_per_joint_cell"] * 3
    return {
        "source_closures": len(closures),
        "excluded_closed_D0_closures": len(excluded),
        "remaining_closures": len(remaining),
        "reachable_cells": len(census),
        "minimum_remaining_support_per_cell": min(census.values()),
        "maximum_remaining_support_per_cell": max(census.values()),
        "required_total": required,
        "required_minimum_per_cell_across_nonreserve_roles": minimum,
        "capacity_ready": (
            len(remaining) == required
            and len(census) == config["public_roles"]["reachable_joint_cells"]
            and min(census.values()) >= minimum
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read(BASE / "CONFIG.json")
        detector = read(BASE / "DETECTOR_PROTOCOL.json")
        allocation = read(BASE / "PUBLIC_ALLOCATION_PROTOCOL.json")
        lineage = config["lineage"]
        for name in (
            "CAR2_S0_audit", "D0_detector_rows",
            "D0_output_summaries", "D0_public_manifest",
        ):
            if hash_file(ROOT / lineage[f"{name}_path"]) != lineage[f"{name}_sha256"]:
                raise ValueError("HASH:" + name)
        projected = projection(config)
        support = capacity(config)
        validation = config["validation_error_gate"]
        upper = upper_limit(
            validation["maximum_familywise_false_accept_events"],
            validation["closures"],
            validation["one_sided_confidence"],
        )
        gates = {
            "single_detector": detector["detector_identity"]
            == "AGGREGATE_ZSCORE_SUM_MAX16_V1",
            "dependence_aware_calibration": (
                config["feasibility"]["calibration_unit"] == "COMPLETE_CLOSURE"
                and not config["feasibility"][
                    "dependence_independence_assumption_required"
                ]
            ),
            "joint_error_budget": (
                config["feasibility"]["carrier_off_alpha"]
                + config["feasibility"]["wrong_confusion_alpha"]
                <= config["feasibility"]["combined_familywise_alpha_max"]
            ),
            "projected_power": projected["projected_pair_power"]
            >= config["feasibility"]["projected_pair_power_minimum"],
            "public_capacity": support["capacity_ready"],
            "validation_tail_resolution": upper
            <= validation["maximum_allowed_upper_bound"],
            "full_denominator": config["full_denominator"][
                "computation_fidelity_failure_is_integrated_failure"
            ],
            "zero_new_scientific_work": (
                config["static_work"]["new_model_generations"] == 0
                and config["static_work"]["new_detector_scores"] == 0
                and config["static_work"]["protected_releases"] == 0
            ),
            "no_fallback": not config["feasibility"]["fallback_detector_allowed"],
        }
        ready = all(gates.values())
        result = stable({
            "schema": "exp33a.csw_am_scb_m1_car_agg_s0.reference_result.v1",
            "projection": projected,
            "public_capacity": support,
            "validation_one_sided_upper_bound": upper,
            "gates": gates,
            "terminal": (
                config["terminals"]["ready"]
                if ready else config["terminals"]["invalid"]
            ),
            "AGG_D0_authorized": ready,
            "AGG_V1_authorized": False,
            "protected_release_authorized": False,
            "identities_materialized": False,
            "new_model_generations": 0,
            "new_detector_scores": 0,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode(result) + b"\n")
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
