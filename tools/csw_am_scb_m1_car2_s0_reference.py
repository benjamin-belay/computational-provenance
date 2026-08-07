#!/usr/bin/env python3
"""Independent static reconstruction of CAR2 pure-multiplicity feasibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-car2-s0/CONFIG.json"


def encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_at_least(
    success: float, ordinary_failure: float, draws: int, required: int,
) -> float:
    value = 0.0
    for successes in range(required, draws + 1):
        value += (
            math.comb(draws, successes)
            * success ** successes
            * ordinary_failure ** (draws - successes)
        )
    return value


def ordinary_binomial(probability: float, draws: int, required: int) -> float:
    return sum(
        math.comb(draws, successes)
        * probability ** successes
        * (1 - probability) ** (draws - successes)
        for successes in range(required, draws + 1)
    )


def side(source: list[dict[str, Any]], name: str) -> dict[str, Any]:
    data = [item for item in source if item["side"] == name]
    total = len(data)
    match = sum(item["matching_decision"] == "ACCEPT" for item in data)
    clean_match = sum(
        item["matching_decision"] == "ACCEPT" and item["wrong_accept_count"] == 0
        for item in data
    )
    clean_miss = sum(
        item["matching_decision"] != "ACCEPT" and item["wrong_accept_count"] == 0
        for item in data
    )
    wrong = total - clean_match - clean_miss
    grouped: dict[str, int] = defaultdict(int)
    for item in data:
        grouped[item["closure_id"]] += int(item["matching_decision"] == "ACCEPT")
    counts = list(grouped.values())
    mean = sum(counts) / len(counts)
    variance = sum((value - mean) ** 2 for value in counts) / len(counts)
    p = mean / 4
    expected_variance = 4 * p * (1 - p)
    icc = (variance / expected_variance - 1) / 3
    return {
        "side": name,
        "observations": total,
        "matching_accepts": match,
        "wrong_present_outputs": wrong,
        "good_matching_outputs": clean_match,
        "good_nonmatching_outputs": clean_miss,
        "matching_probability": match / total,
        "wrong_present_probability": wrong / total,
        "projected_matching_only_side_success": ordinary_binomial(match / total, 8, 6),
        "projected_zero_wrong_side_success": probability_at_least(
            clean_match / total, clean_miss / total, 8, 6
        ),
        "projected_any_wrong_side_probability": 1
        - ((clean_match + clean_miss) / total) ** 8,
        "four_output_matching_indicator_ICC": icc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read(CONFIG)
        lineage = config["lineage"]
        for name in (
            "CAR_D0_closure", "CAR_D0_diagnostics", "CAR_D0_output_summaries"
        ):
            path = ROOT / lineage[f"{name}_path"]
            if file_hash(path) != lineage[f"{name}_sha256"]:
                raise ValueError("BINDING:" + name)
        source = [
            item for item in rows(
                ROOT / lineage["CAR_D0_output_summaries_path"]
            )
            if item["condition"] == "STATE"
        ]
        natural, alias = side(source, "NATURAL"), side(source, "ALIAS_Z2")
        matching = (
            natural["projected_matching_only_side_success"]
            * alias["projected_matching_only_side_success"]
        )
        integrated = (
            natural["projected_zero_wrong_side_success"]
            * alias["projected_zero_wrong_side_success"]
        )
        threshold = config["feasibility_gate"]["projected_pair_success_minimum"]
        gates = {
            "lineage_and_static_boundary": True,
            "single_capacity_axis": config["candidate"]["capacity_axes_changed"]
            == ["STATE_OUTPUT_MULTIPLICITY"],
            "proportional_matching_rule": (
                config["candidate"]["state_outputs_per_side"] == 8
                and config["candidate"]["matching_outputs_required"] == 6
            ),
            "matching_only_projection_above_pair_target": matching >= threshold,
            "integrated_zero_wrong_projection_above_pair_target": integrated
            >= threshold,
            "protected_boundary": config["static_inputs"]["protected_releases"] == 0,
        }
        ready = all(gates.values())
        result = {
            "schema": "exp33a.csw_am_scb_m1_car2_s0.reference_result.v1",
            "candidate": config["candidate"]["identity"],
            "natural": natural,
            "alias": alias,
            "projected_matching_only_pair_success": matching,
            "projected_integrated_zero_wrong_pair_success": integrated,
            "retained_pair_threshold": threshold,
            "wrong_constraint_cost": matching - integrated,
            "gates": gates,
            "terminal": (
                config["terminals"]["pure_multiplicity_ready"]
                if ready else config["terminals"]["pure_multiplicity_insufficient"]
            ),
            "CAR2_D0_authorized": ready,
            "aggregate_detector_redesign_authorized": False,
            "protected_release_authorized": False,
            "new_model_generations": 0,
            "new_detector_scores": 0,
            "interpretation": (
                "MATCHING_ONLY_POWER_IS_ADEQUATE_BUT_PURE_MULTIPLICITY_WITH_THE_"
                "UNCHANGED_ZERO_WRONG_RULE_IS_NOT_JUSTIFIED"
                if matching >= threshold and integrated < threshold
                else "SEE_GATES"
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode(result) + b"\n")
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
