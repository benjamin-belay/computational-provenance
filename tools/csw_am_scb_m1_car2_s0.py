#!/usr/bin/env python3
"""Static feasibility evaluator for the pure-multiplicity CAR2 proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-car2-s0/CONFIG.json"
PROTOCOL_PATH = ROOT / "protocol/csw-am-scb-m1-car2-s0/PURE_MULTIPLICITY_PROTOCOL.json"


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


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    config, protocol = read_json(CONFIG_PATH), read_json(PROTOCOL_PATH)
    lineage = config["lineage"]
    for name in ("CAR_D0_closure", "CAR_D0_diagnostics", "CAR_D0_output_summaries"):
        path = ROOT / lineage[f"{name}_path"]
        if file_sha(path) != lineage[f"{name}_sha256"]:
            raise ValueError("LINEAGE_HASH:" + name)
    closure = read_json(ROOT / lineage["CAR_D0_closure_path"])
    if (
        closure["terminal"] != lineage["CAR_D0_closure_terminal"]
        or closure["preserved_scientific_terminal"]
        != lineage["preserved_D0_scientific_terminal"]
        or config["static_inputs"]["new_model_generations"] != 0
        or config["static_inputs"]["new_detector_scores"] != 0
        or config["static_inputs"]["protected_releases"] != 0
        or protocol["side_acceptance"]["planned_outputs"]
        != config["candidate"]["state_outputs_per_side"]
        or protocol["side_acceptance"]["matching_detector_accepts_required"]
        != config["candidate"]["matching_outputs_required"]
    ):
        raise ValueError("PROTOCOL_BINDING")
    return config, protocol


def choose(n: int, k: int) -> int:
    return math.comb(n, k)


def binomial_tail(probability: float, n: int, minimum: int) -> float:
    return sum(
        choose(n, value)
        * probability ** value
        * (1 - probability) ** (n - value)
        for value in range(minimum, n + 1)
    )


def zero_wrong_matching_tail(
    good_matching_probability: float,
    good_nonmatching_probability: float,
    n: int,
    minimum: int,
) -> float:
    return sum(
        choose(n, value)
        * good_matching_probability ** value
        * good_nonmatching_probability ** (n - value)
        for value in range(minimum, n + 1)
    )


def intraclass_correlation(counts: list[int], trials: int) -> float:
    mean = sum(counts) / len(counts)
    variance = sum((value - mean) ** 2 for value in counts) / len(counts)
    p = mean / trials
    binomial_variance = trials * p * (1 - p)
    if binomial_variance == 0 or trials == 1:
        return 0.0
    return (variance / binomial_variance - 1) / (trials - 1)


def side_statistics(
    rows: list[dict[str, Any]], side: str, n: int, minimum: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row["side"] == side]
    good_matching = sum(
        row["matching_decision"] == "ACCEPT" and row["wrong_accept_count"] == 0
        for row in selected
    )
    good_nonmatching = sum(
        row["matching_decision"] != "ACCEPT" and row["wrong_accept_count"] == 0
        for row in selected
    )
    wrong_present = sum(row["wrong_accept_count"] > 0 for row in selected)
    matching = sum(row["matching_decision"] == "ACCEPT" for row in selected)
    grouped: dict[str, int] = defaultdict(int)
    for row in selected:
        grouped[row["closure_id"]] += int(row["matching_decision"] == "ACCEPT")
    total = len(selected)
    p_match = matching / total
    p_good_match = good_matching / total
    p_good_nonmatch = good_nonmatching / total
    matching_only = binomial_tail(p_match, n, minimum)
    integrated = zero_wrong_matching_tail(
        p_good_match, p_good_nonmatch, n, minimum
    )
    return {
        "side": side,
        "observations": total,
        "matching_accepts": matching,
        "wrong_present_outputs": wrong_present,
        "good_matching_outputs": good_matching,
        "good_nonmatching_outputs": good_nonmatching,
        "matching_probability": p_match,
        "wrong_present_probability": wrong_present / total,
        "projected_matching_only_side_success": matching_only,
        "projected_zero_wrong_side_success": integrated,
        "projected_any_wrong_side_probability": 1
        - ((good_matching + good_nonmatching) / total) ** n,
        "four_output_matching_indicator_ICC": intraclass_correlation(
            list(grouped.values()), 4
        ),
    }


def evaluate() -> dict[str, Any]:
    config, _ = validate()
    rows = [
        row for row in read_rows(
            ROOT / config["lineage"]["CAR_D0_output_summaries_path"]
        )
        if row["condition"] == "STATE"
    ]
    n = config["candidate"]["state_outputs_per_side"]
    minimum = config["candidate"]["matching_outputs_required"]
    natural = side_statistics(rows, "NATURAL", n, minimum)
    alias = side_statistics(rows, "ALIAS_Z2", n, minimum)
    matching_pair = (
        natural["projected_matching_only_side_success"]
        * alias["projected_matching_only_side_success"]
    )
    integrated_pair = (
        natural["projected_zero_wrong_side_success"]
        * alias["projected_zero_wrong_side_success"]
    )
    threshold = config["feasibility_gate"]["projected_pair_success_minimum"]
    gates = {
        "lineage_and_static_boundary": True,
        "single_capacity_axis": config["candidate"]["capacity_axes_changed"]
        == ["STATE_OUTPUT_MULTIPLICITY"],
        "proportional_matching_rule": n == 8 and minimum == 6,
        "matching_only_projection_above_pair_target": matching_pair >= threshold,
        "integrated_zero_wrong_projection_above_pair_target": integrated_pair
        >= threshold,
        "protected_boundary": config["static_inputs"]["protected_releases"] == 0,
    }
    ready = all(gates.values())
    terminal = (
        config["terminals"]["pure_multiplicity_ready"]
        if ready
        else config["terminals"]["pure_multiplicity_insufficient"]
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_car2_s0.official_result.v1",
        "candidate": config["candidate"]["identity"],
        "natural": natural,
        "alias": alias,
        "projected_matching_only_pair_success": matching_pair,
        "projected_integrated_zero_wrong_pair_success": integrated_pair,
        "retained_pair_threshold": threshold,
        "wrong_constraint_cost": matching_pair - integrated_pair,
        "gates": gates,
        "terminal": terminal,
        "CAR2_D0_authorized": ready,
        "aggregate_detector_redesign_authorized": False,
        "protected_release_authorized": False,
        "new_model_generations": 0,
        "new_detector_scores": 0,
        "interpretation": (
            "MATCHING_ONLY_POWER_IS_ADEQUATE_BUT_PURE_MULTIPLICITY_WITH_THE_"
            "UNCHANGED_ZERO_WRONG_RULE_IS_NOT_JUSTIFIED"
            if matching_pair >= threshold and integrated_pair < threshold
            else "SEE_GATES"
        ),
    }


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
