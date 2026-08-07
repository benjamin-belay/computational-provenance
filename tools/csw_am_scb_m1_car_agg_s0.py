#!/usr/bin/env python3
"""Static dependence-aware aggregate-carrier protocol qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol/csw-am-scb-m1-car-agg-s0"
CONFIG_PATH = PROTOCOL_DIR / "CONFIG.json"
DETECTOR_PATH = PROTOCOL_DIR / "DETECTOR_PROTOCOL.json"
ALLOCATION_PATH = PROTOCOL_DIR / "PUBLIC_ALLOCATION_PROTOCOL.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    path.write_bytes(canonical(value) + b"\n")


def stable(value: Any) -> Any:
    if type(value) is float:
        return round(value, 12)
    if type(value) is list:
        return [stable(item) for item in value]
    if type(value) is dict:
        return {key: stable(item) for key, item in value.items()}
    return value


def validate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    detector, allocation = read_json(DETECTOR_PATH), read_json(ALLOCATION_PATH)
    lineage = config["lineage"]
    for name in (
        "CAR2_S0_audit", "D0_detector_rows",
        "D0_output_summaries", "D0_public_manifest",
    ):
        path = ROOT / lineage[f"{name}_path"]
        if file_sha(path) != lineage[f"{name}_sha256"]:
            raise ValueError("LINEAGE_HASH:" + name)
    audit = read_json(ROOT / lineage["CAR2_S0_audit_path"])
    if (
        audit["main_verdict"] != lineage["CAR2_S0_terminal"]
        or detector["outputs_per_side"] != config["future_counts"]["state_outputs_per_side"]
        or sum(role["closures"] for role in allocation["roles"])
        != config["public_roles"]["source_closures"]
        - config["public_roles"]["closed_D0_closures"]
        or config["static_work"]["new_model_generations"] != 0
        or config["static_work"]["new_detector_scores"] != 0
        or config["static_work"]["protected_releases"] != 0
    ):
        raise ValueError("STATIC_PROTOCOL_BINDING")
    return config, detector, allocation


def strict_quantile(values: list[float], alpha: float) -> float:
    ordered = sorted(values)
    index = math.ceil((len(ordered) + 1) * (1 - alpha)) - 1
    return ordered[min(len(ordered) - 1, index)]


def normalized(score: int, eligible: int) -> float:
    if type(eligible) is not int or eligible <= 0:
        raise ValueError("ELIGIBLE")
    return (score - eligible / 2) / math.sqrt(eligible / 4)


def output_vectors(
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summaries = read_rows(ROOT / config["lineage"]["D0_output_summaries_path"])
    detector_rows = [
        row for row in read_rows(ROOT / config["lineage"]["D0_detector_rows_path"])
        if row["authority_family"] == "STATE"
    ]
    vector: dict[str, list[float | None]] = defaultdict(lambda: [None] * 16)
    for row in detector_rows:
        vector[row["output_id"]][row["authority_category"]] = normalized(
            row["score"], row["eligible_count"]
        )
    outputs = {}
    for row in summaries:
        if row["condition"] not in ("STATE", "OFF"):
            continue
        values = vector[row["output_id"]]
        if any(value is None for value in values):
            raise ValueError("VECTOR")
        outputs[row["output_id"]] = {
            **row,
            "vector": [float(value) for value in values],
        }
    return outputs, summaries


def four_output_blocks(
    outputs: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[list[list[float]]]], list[list[list[float]]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in outputs.values():
        if row["condition"] == "OFF" and row["side"] != "NATURAL":
            continue
        grouped[(row["condition"], row["closure_id"], row["side"])].append(row)
    state = {
        "NATURAL": [[] for _ in range(16)],
        "ALIAS_Z2": [[] for _ in range(16)],
    }
    off = []
    for (condition, _, side), rows in grouped.items():
        rows.sort(key=lambda row: row["draw"])
        if len(rows) != 4:
            raise ValueError("BLOCK_SIZE")
        block = [row["vector"] for row in rows]
        if condition == "OFF":
            off.append(block)
        else:
            category = rows[0]["source_z2_category"]
            state[side][category].append(block)
    return state, off


def aggregate(block_a: list[list[float]], block_b: list[list[float]]) -> list[float]:
    return [
        sum(row[authority] for row in block_a + block_b) / math.sqrt(8)
        for authority in range(16)
    ]


def paired_blocks(blocks: list[list[list[float]]]) -> list[list[float]]:
    return [aggregate(first, second) for first in blocks for second in blocks]


def wrong_claim_gap(scores: list[float], authenticated: int) -> float:
    values = []
    for candidate in range(16):
        if candidate == authenticated:
            continue
        strongest_other = max(
            score for index, score in enumerate(scores) if index != candidate
        )
        values.append(scores[candidate] - strongest_other)
    return max(values)


def accepted(scores: list[float], authenticated: int, threshold: float, margin: float) -> bool:
    strongest_wrong = max(
        score for index, score in enumerate(scores) if index != authenticated
    )
    return (
        scores[authenticated] > threshold
        and scores[authenticated] > strongest_wrong
        and scores[authenticated] - strongest_wrong > margin
    )


def calibration_and_power(config: dict[str, Any]) -> dict[str, Any]:
    outputs, _ = output_vectors(config)
    state, off = four_output_blocks(outputs)
    off_eight = paired_blocks(off)
    absolute_alpha = config["feasibility"]["carrier_off_alpha"]
    confusion_alpha = config["feasibility"]["wrong_confusion_alpha"]
    threshold = strict_quantile(
        [max(scores) for scores in off_eight], absolute_alpha
    )
    state_eight: dict[str, list[tuple[int, list[float]]]] = {
        "NATURAL": [], "ALIAS_Z2": []
    }
    wrong_gaps = []
    for side in state_eight:
        for category, blocks in enumerate(state[side]):
            for scores in paired_blocks(blocks):
                state_eight[side].append((category, scores))
                wrong_gaps.append(wrong_claim_gap(scores, category))
    raw_margin_quantile = strict_quantile(wrong_gaps, confusion_alpha)
    margin = max(0.0, raw_margin_quantile)
    off_false = sum(
        max(scores) > threshold
        and (
            sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]
            > margin
        )
        for scores in off_eight
    )
    sides = {}
    for side, values in state_eight.items():
        authentic_accepts = sum(
            accepted(scores, category, threshold, margin)
            for category, scores in values
        )
        wrong_accepts = 0
        for category, scores in values:
            for candidate in range(16):
                if candidate != category and accepted(
                    scores, candidate, threshold, margin
                ):
                    wrong_accepts += 1
        sides[side] = {
            "synthetic_block_pairs": len(values),
            "authenticated_accepts": authentic_accepts,
            "projected_side_power": authentic_accepts / len(values),
            "wrong_claim_accepts": wrong_accepts,
            "wrong_claim_rate_per_side": wrong_accepts / len(values),
        }
    pair_power = (
        sides["NATURAL"]["projected_side_power"]
        * sides["ALIAS_Z2"]["projected_side_power"]
    )
    return stable({
        "planning_blocks": {
            "carrier_off_four_output_blocks": len(off),
            "carrier_off_eight_output_block_pairs": len(off_eight),
            "state_eight_output_block_pairs": sum(
                value["synthetic_block_pairs"] for value in sides.values()
            ),
        },
        "diagnostic_threshold_T": threshold,
        "diagnostic_raw_margin_quantile": raw_margin_quantile,
        "diagnostic_margin_M": margin,
        "carrier_off_full_rule_false_accepts": off_false,
        "carrier_off_full_rule_false_accept_rate": off_false / len(off_eight),
        "sides": sides,
        "projected_pair_power": pair_power,
        "projection_role": "PLANNING_ONLY_THRESHOLDS_RECALIBRATED_ON_FUTURE_PUBLIC_CALIBRATION",
    })


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(
        math.comb(n, value) * p ** value * (1 - p) ** (n - value)
        for value in range(k + 1)
    )


def one_sided_upper(k: int, n: int, confidence: float) -> float:
    low, high = 0.0, 1.0
    target = 1 - confidence
    for _ in range(100):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def capacity(config: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / config["public_roles"]["source_path"]
    if file_sha(source) != config["public_roles"]["source_sha256"]:
        raise ValueError("PUBLIC_SOURCE")
    old = {
        row["closure_id"] for row in read_rows(
            ROOT / config["lineage"]["D0_public_manifest_path"]
        )
    }
    closures: dict[str, dict[str, Any]] = {}
    for row in read_rows(source):
        if row["orientation"] == "LOWER_C_ORIENTATION":
            closures[row["closure_id"]] = row
    remaining = [row for cid, row in closures.items() if cid not in old]
    cells: dict[tuple[int, int, int], int] = defaultdict(int)
    for row in remaining:
        cells[(
            row["natural"]["z2"] % 8,
            row["natural"]["answer"],
            row["template_id"],
        )] += 1
    required_total = sum(config["public_roles"][key] for key in (
        "AGG_DEVELOPMENT_closures", "AGG_CALIBRATION_closures",
        "AGG_VALIDATION_closures", "remaining_public_reserve_closures",
    ))
    required_minimum = (
        config["public_roles"]["minimum_each_role_per_joint_cell"] * 3
    )
    return stable({
        "source_closures": len(closures),
        "excluded_closed_D0_closures": len(old),
        "remaining_closures": len(remaining),
        "reachable_cells": len(cells),
        "minimum_remaining_support_per_cell": min(cells.values()),
        "maximum_remaining_support_per_cell": max(cells.values()),
        "required_total": required_total,
        "required_minimum_per_cell_across_nonreserve_roles": required_minimum,
        "capacity_ready": (
            len(remaining) == required_total
            and len(cells) == config["public_roles"]["reachable_joint_cells"]
            and min(cells.values()) >= required_minimum
        ),
    })


def evaluate() -> dict[str, Any]:
    config, detector, allocation = validate()
    projection = calibration_and_power(config)
    support = capacity(config)
    validation = config["validation_error_gate"]
    upper = one_sided_upper(
        validation["maximum_familywise_false_accept_events"],
        validation["closures"],
        validation["one_sided_confidence"],
    )
    gates = {
        "single_detector": detector["detector_identity"]
        == "AGGREGATE_ZSCORE_SUM_MAX16_V1",
        "dependence_aware_calibration": (
            config["feasibility"]["calibration_unit"] == "COMPLETE_CLOSURE"
            and not config["feasibility"]["dependence_independence_assumption_required"]
        ),
        "joint_error_budget": (
            config["feasibility"]["carrier_off_alpha"]
            + config["feasibility"]["wrong_confusion_alpha"]
            <= config["feasibility"]["combined_familywise_alpha_max"]
        ),
        "projected_power": projection["projected_pair_power"]
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
    return stable({
        "schema": "exp33a.csw_am_scb_m1_car_agg_s0.official_result.v1",
        "projection": projection,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate()
        write_json(args.output, result)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
