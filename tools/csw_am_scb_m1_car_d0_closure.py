#!/usr/bin/env python3
"""Read-only closure and failure diagnostics for the frozen CAR-D0 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol/csw-am-scb-m1-car-d0-closure/CLOSURE_SPEC.json"


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


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    spec = read_json(SPEC_PATH)
    binding = spec["source_bindings"]
    for name in (
        "D0_main_audit", "D0_terminal", "production_result",
        "public_rows", "execution_rows", "raw_outputs", "output_summaries",
    ):
        path = ROOT / binding[f"{name}_path"]
        if file_sha(path) != binding[f"{name}_sha256"]:
            raise ValueError("SOURCE_HASH:" + name)
    audit = read_json(ROOT / binding["D0_main_audit_path"])
    terminal = read_json(ROOT / binding["D0_terminal_path"])
    production = read_json(ROOT / binding["production_result_path"])
    if (
        audit["main_verdict"] != binding["D0_main_verdict"]
        or terminal["terminal"] != binding["D0_scientific_terminal"]
        or terminal["passed"] is not False
        or production["passed"] is not False
        or production["gates"]["closure_pair_switch"] is not False
        or sum(not value for value in production["gates"].values()) != 1
    ):
        raise ValueError("D0_TERMINAL_BINDING")
    return spec, production


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def distribution(values: list[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def quantile(fraction: float) -> float | int:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": quantile(0.25),
        "median": quantile(0.5),
        "q75": quantile(0.75),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def diagnostic() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec, production = validate()
    binding = spec["source_bindings"]
    public_rows = read_rows(ROOT / binding["public_rows_path"])
    execution_rows = read_rows(ROOT / binding["execution_rows_path"])
    raw_outputs = read_rows(ROOT / binding["raw_outputs_path"])
    summaries = read_rows(ROOT / binding["output_summaries_path"])
    public = {row["closure_id"]: row for row in public_rows}
    execution = {
        (row["closure_id"], row["role"]): row for row in execution_rows
    }
    raw = {row["output_id"]: row for row in raw_outputs}
    state = [row for row in summaries if row["condition"] == "STATE"]
    side_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state:
        side_groups[(row["closure_id"], row["side"])].append(row)
    side_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rule = spec["frozen_side_success_rule"]
    for key, values in side_groups.items():
        values = sorted(values, key=lambda row: row["draw"])
        matching_accepts = sum(row["matching_decision"] == "ACCEPT" for row in values)
        wrong_accepts = sum(row["wrong_accept_count"] for row in values)
        side_rows[key] = {
            "passed": (
                matching_accepts >= rule["minimum_matching_accepts"]
                and wrong_accepts <= rule["maximum_wrong_authority_accepts_across_side"]
            ),
            "matching_accepts": matching_accepts,
            "wrong_accepts": wrong_accepts,
            "matching_margins": [
                row["matching_score"] - row["max_wrong_score"] for row in values
            ],
            "matching_scores": [row["matching_score"] for row in values],
            "max_wrong_scores": [row["max_wrong_score"] for row in values],
            "eligible_positions": [raw[row["output_id"]]["eligible_positions"] for row in values],
            "source_z2_category": values[0]["source_z2_category"],
        }
    failed, closure_rows = [], []
    classifications = Counter()
    for closure_id, row in sorted(public.items()):
        natural = side_rows[(closure_id, "NATURAL")]
        alias = side_rows[(closure_id, "ALIAS_Z2")]
        natural_execution = execution[(closure_id, "NATURAL")]
        alias_execution = execution[(closure_id, "ALIAS_Z2")]
        registered_natural_z2 = row["natural"]["z2"]
        registered_alias_z2 = (registered_natural_z2 + 8) % 16
        registered_alias_z3 = (5 * registered_alias_z2 + row["inputs"]["d"]) % 16
        if natural["passed"] and alias["passed"]:
            classification = "BOTH_SIDES_PASS"
        elif natural["passed"]:
            classification = "NATURAL_ONLY_PASS"
        elif alias["passed"]:
            classification = "ALIAS_ONLY_PASS"
        else:
            classification = "NEITHER_SIDE_PASS"
        classifications[classification] += 1
        item = {
            "closure_id": closure_id,
            "classification": classification,
            "registered_natural_z2": registered_natural_z2,
            "registered_alias_z2": registered_alias_z2,
            "consumed_natural_z2": natural["source_z2_category"],
            "consumed_alias_z2": alias["source_z2_category"],
            "registered_alias_pair": [
                min(registered_natural_z2, registered_alias_z2),
                max(registered_natural_z2, registered_alias_z2),
            ],
            "consumed_state_pair": [
                min(natural["source_z2_category"], alias["source_z2_category"]),
                max(natural["source_z2_category"], alias["source_z2_category"]),
            ],
            "answer": row["natural"]["answer"],
            "template_id": row["template_id"],
            "registered_execution_fidelity": {
                "natural_z2": natural_execution["consumed_z2"] == registered_natural_z2,
                "natural_z3": natural_execution["consumed_z3"] == row["natural"]["z3"],
                "natural_answer": natural_execution["answer"] == row["natural"]["answer"],
                "alias_z2": alias_execution["consumed_z2"] == registered_alias_z2,
                "alias_z3": alias_execution["consumed_z3"] == registered_alias_z3,
                "alias_answer": alias_execution["answer"] == row["natural"]["answer"],
                "natural_alias_answer_matched": (
                    natural_execution["answer"]
                    == alias_execution["answer"]
                    == row["natural"]["answer"]
                ),
            },
            "natural_side": natural,
            "alias_side": alias,
        }
        closure_rows.append(item)
        if classification != "BOTH_SIDES_PASS":
            failed.append(item)

    def stratify(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in closure_rows:
            value = item[field]
            key = "-".join(map(str, value)) if type(value) is list else str(value)
            groups[key].append(item)
        result = {}
        for key, values in sorted(groups.items()):
            both = sum(item["classification"] == "BOTH_SIDES_PASS" for item in values)
            result[key] = {
                "closures": len(values),
                "both_sides_pass": both,
                "pair_success_rate": rate(both, len(values)),
                "natural_side_pass": sum(item["natural_side"]["passed"] for item in values),
                "alias_side_pass": sum(item["alias_side"]["passed"] for item in values),
                "failed": len(values) - both,
            }
        return result

    natural_success = sum(item["natural_side"]["passed"] for item in closure_rows)
    alias_success = sum(item["alias_side"]["passed"] for item in closure_rows)
    both_success = classifications["BOTH_SIDES_PASS"]
    neither = classifications["NEITHER_SIDE_PASS"]
    n = len(closure_rows)
    p_natural, p_alias = rate(natural_success, n), rate(alias_success, n)
    expected_independent_both = p_natural * p_alias
    p11 = rate(both_success, n)
    covariance = p11 - expected_independent_both
    denominator = math.sqrt(
        p_natural * (1 - p_natural) * p_alias * (1 - p_alias)
    )
    phi = covariance / denominator if denominator else 0.0
    all_margins = [
        margin
        for item in closure_rows
        for side in ("natural_side", "alias_side")
        for margin in item[side]["matching_margins"]
    ]
    failed_margins = [
        margin
        for item in failed
        for side in ("natural_side", "alias_side")
        for margin in item[side]["matching_margins"]
    ]
    diagnostics = {
        "schema": "exp33a.csw_am_scb_m1_car_d0_closure.diagnostics.v1",
        "source_terminal_preserved": spec["source_bindings"]["D0_scientific_terminal"],
        "denominators": {
            "closures": n,
            "sides": len(side_rows),
            "state_outputs": len(state),
            "failed_closures": len(failed),
        },
        "pair_classification": dict(sorted(classifications.items())),
        "side_census": {
            "natural_pass": natural_success,
            "alias_pass": alias_success,
            "total_pass": natural_success + alias_success,
            "total_sides": 2 * n,
        },
        "paired_dependence": {
            "natural_success_rate": p_natural,
            "alias_success_rate": p_alias,
            "observed_both_success_rate": p11,
            "expected_both_rate_under_independent_sides": expected_independent_both,
            "observed_minus_independence": covariance,
            "phi_correlation": phi,
            "neither_side_pass": neither,
            "interpretation": "DESCRIPTIVE_ONLY_NOT_A_REVISED_GATE",
        },
        "stratification": {
            "registered_natural_z2": stratify("registered_natural_z2"),
            "registered_alias_pair": stratify("registered_alias_pair"),
            "template": stratify("template_id"),
            "answer": stratify("answer"),
        },
        "registered_execution_fidelity": {
            field: sum(
                item["registered_execution_fidelity"][field] for item in closure_rows
            )
            for field in (
                "natural_z2", "natural_z3", "natural_answer",
                "alias_z2", "alias_z3", "alias_answer",
                "natural_alias_answer_matched",
            )
        },
        "registered_execution_fidelity_failures": [
            {
                "closure_id": item["closure_id"],
                "checks": item["registered_execution_fidelity"],
                "registered_natural_z2": item["registered_natural_z2"],
                "consumed_natural_z2": item["consumed_natural_z2"],
                "registered_alias_z2": item["registered_alias_z2"],
                "consumed_alias_z2": item["consumed_alias_z2"],
            }
            for item in closure_rows
            if not all(item["registered_execution_fidelity"].values())
        ],
        "matching_output_pass_count_distribution": {
            "natural": dict(sorted(Counter(
                item["natural_side"]["matching_accepts"] for item in closure_rows
            ).items())),
            "alias": dict(sorted(Counter(
                item["alias_side"]["matching_accepts"] for item in closure_rows
            ).items())),
            "failed_closure_sides": dict(sorted(Counter(
                item[side]["matching_accepts"]
                for item in failed
                for side in ("natural_side", "alias_side")
            ).items())),
        },
        "eligible_positions": distribution([
            position
            for item in closure_rows
            for side in ("natural_side", "alias_side")
            for position in item[side]["eligible_positions"]
        ]),
        "matching_margin": {
            "all_state_outputs": distribution(all_margins),
            "outputs_in_failed_closures": distribution(failed_margins),
        },
        "maximum_wrong_state_score": {
            "all_state_outputs": distribution([
                score
                for item in closure_rows
                for side in ("natural_side", "alias_side")
                for score in item[side]["max_wrong_scores"]
            ]),
            "outputs_in_failed_closures": distribution([
                score
                for item in failed
                for side in ("natural_side", "alias_side")
                for score in item[side]["max_wrong_scores"]
            ]),
        },
        "binding_checks": {
            "production_pair_successes": production["metric_core"][
                "closure_pair_switch_successes"
            ],
            "diagnostic_pair_successes": both_success,
            "production_side_successes": production["metric_core"]["side_successes"],
            "diagnostic_side_successes": natural_success + alias_success,
            "only_frozen_pair_gate_failed": (
                sum(not value for value in production["gates"].values()) == 1
                and production["gates"]["closure_pair_switch"] is False
            ),
        },
        "scientific_interpretation": {
            "integration_valid": True,
            "aggregate_authenticated_state_signal": True,
            "pair_level_reliability_sufficient": False,
            "implementation_failure": False,
            "total_carrier_negative": False,
            "output_side_provenance_established": False,
        },
    }
    if (
        diagnostics["pair_classification"] != {
            "ALIAS_ONLY_PASS": classifications["ALIAS_ONLY_PASS"],
            "BOTH_SIDES_PASS": 88,
            "NATURAL_ONLY_PASS": classifications["NATURAL_ONLY_PASS"],
            "NEITHER_SIDE_PASS": 3,
        }
        or len(failed) != 40
        or natural_success + alias_success != 213
        or diagnostics["binding_checks"]["production_pair_successes"] != both_success
        or diagnostics["registered_execution_fidelity"]["natural_alias_answer_matched"]
        != 126
    ):
        raise ValueError("DIAGNOSTIC_RECONSTRUCTION")
    return diagnostics, failed


def close(output: Path) -> dict[str, Any]:
    spec, _ = validate()
    diagnostics, failed = diagnostic()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in (
        "PUBLIC_FAILURE_DIAGNOSTICS.json", "FAILED_CLOSURES.jsonl", "CLOSURE.json"
    )):
        raise FileExistsError("CLOSURE_OUTPUT_EXISTS")
    write_json(output / "PUBLIC_FAILURE_DIAGNOSTICS.json", diagnostics)
    write_rows(output / "FAILED_CLOSURES.jsonl", failed)
    closure = {
        "schema": "exp33a.csw_am_scb_m1_car_d0_closure.record.v1",
        "identity": spec["identity"],
        "terminal": spec["closure_terminal"],
        "preserved_scientific_terminal": spec["source_bindings"]["D0_scientific_terminal"],
        "preserved_main_verdict": spec["source_bindings"]["D0_main_verdict"],
        "diagnostic_sha256": file_sha(output / "PUBLIC_FAILURE_DIAGNOSTICS.json"),
        "failed_closures_sha256": file_sha(output / "FAILED_CLOSURES.jsonl"),
        "pair_result": {
            "passed": 88,
            "registered": 128,
            "observed_rate": 0.6875,
            "required_rate": 0.75,
            "valid_failure": True,
        },
        "registered_execution_fidelity": {
            "natural_correct": diagnostics["registered_execution_fidelity"]["natural_answer"],
            "alias_correct": diagnostics["registered_execution_fidelity"]["alias_answer"],
            "answer_matched_pairs": diagnostics["registered_execution_fidelity"][
                "natural_alias_answer_matched"
            ],
            "registered": diagnostics["denominators"]["closures"],
            "incorrect_natural_executions_retained": len(
                diagnostics["registered_execution_fidelity_failures"]
            ),
        },
        "protected_status": {
            "CAR_Q1_releases": 0,
            "CAR_C1_releases": 0,
            "protected_identities_sealed": True,
        },
        "authorizations": {
            "CAR_Q1": False,
            "CAR_C1": False,
            "automatic_successor": False,
            "separate_future_programme_may_be_requested": True,
        },
        "claims": spec["interpretation_boundary"],
        "closure_complete": True,
    }
    write_json(output / "CLOSURE.json", closure)
    return closure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = close(args.output)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
