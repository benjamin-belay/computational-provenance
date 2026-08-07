#!/usr/bin/env python3
"""Independent case reconstruction and adjudication for CSW-AM-SCB-M1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1/CONFIG.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("DUPLICATE_KEY:" + key)
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, object_pairs_hook=strict_pairs)
        for line in path.read_bytes().splitlines()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def task(inputs: dict[str, int]) -> dict[str, int]:
    z1 = (inputs["a"] + inputs["b"]) % 16
    z2 = (z1 + inputs["c"]) % 16
    z3 = (5 * z2 + inputs["d"]) % 16
    return {"z1": z1, "z2": z2, "z3": z3, "answer": z3 % 8}


def expected_target(
    family: str, truth: dict[str, int], inputs: dict[str, int],
    natural_observed: dict[str, int],
) -> tuple[int | None, int, int]:
    if family == "ALIAS_Z2":
        target = (truth["z2"] + 8) % 16
    elif family == "NON_ALIAS_Z2":
        target = (truth["z2"] + 1) % 16
    elif family == "WRONG_Z2":
        target = (truth["z2"] + 3) % 16
    elif family == "SAME_Z2":
        target = truth["z2"]
    elif family == "SHAM":
        target = natural_observed["z2_selected"]
    elif family == "DIRECT_Z3":
        target_z3 = (truth["z3"] + 1) % 16
        return None, target_z3, target_z3 % 8
    else:
        raise ValueError("UNKNOWN_FAMILY:" + family)
    target_z3 = (5 * target + inputs["d"]) % 16
    return target, target_z3, target_z3 % 8


def accuracy(values: list[bool]) -> float:
    if not values:
        raise ValueError("EMPTY_DENOMINATOR")
    return sum(values) / len(values)


def verify_and_rebuild_cases(
    input_rows: list[dict[str, Any]], cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in input_rows}
    if len(by_id) != len(input_rows):
        raise ValueError("DUPLICATE_INPUT")
    expected_case_keys = {
        (row["example_id"], family)
        for row in input_rows for family in config["families"]
    }
    actual_case_keys = {(row["example_id"], row["family"]) for row in cases}
    if actual_case_keys != expected_case_keys or len(cases) != len(expected_case_keys):
        raise ValueError("CASE_CLOSURE")
    natural_by_id = {
        row["example_id"]: row
        for row in cases if row["family"] == "NATURAL"
    }
    rebuilt = []
    for case in cases:
        source = by_id[case["example_id"]]
        truth = task(source["inputs"])
        if (
            source["natural"] != truth
            or case["registered_natural"] != truth
            or case["inputs"] != source["inputs"]
            or case["closure_id"] != source["closure_id"]
            or case["orientation"] != source["orientation"]
            or case["template_id"] != source["template_id"]
        ):
            raise ValueError("CASE_BINDING")
        natural_observed = natural_by_id[case["example_id"]]["natural_observed"]
        natural_predicates = {
            "natural_z2": natural_observed["z2_selected"] == truth["z2"],
            "natural_z3": natural_observed["z3_selected"] == truth["z3"],
            "natural_answer": natural_observed["answer"] == truth["answer"],
            "selected_equals_consumed": (
                natural_observed["z2_selected"] == natural_observed["z2_consumed"]
                and natural_observed["z3_selected"] == natural_observed["z3_consumed"]
            ),
        }
        natural_joint = all(
            natural_predicates[key]
            for key in ("natural_z2", "natural_z3", "natural_answer")
        )
        natural_predicates["natural_joint"] = natural_joint
        if case["family"] == "NATURAL":
            predicates = natural_predicates
            target_z2 = target_z3 = expected_answer = None
        else:
            target_z2, target_z3, expected_answer = expected_target(
                case["family"], truth, case["inputs"], natural_observed
            )
            observed = case["observed"]
            target_consumed = (
                observed["z2_consumed"] == target_z2
                if target_z2 is not None
                else observed["z3_consumed"] == target_z3
            )
            generated_mapping = (
                observed.get("z3_selected", target_z3) == target_z3
                and observed["answer"] == expected_answer
            )
            trace_identity = None
            if case["family"] == "SHAM":
                trace_identity = (
                    observed["z3_selected"] == natural_observed["z3_selected"]
                    and observed["answer"] == natural_observed["answer"]
                )
            elif case["family"] == "SAME_Z2":
                trace_identity = (
                    not natural_joint
                    or (
                        observed["z3_selected"] == natural_observed["z3_selected"]
                        and observed["answer"] == natural_observed["answer"]
                    )
                )
            predicates = {
                "natural_joint": natural_joint,
                "target_consumed": target_consumed,
                "generated_mapping": generated_mapping,
                "trace_identity": trace_identity,
                "joint_success": natural_joint and target_consumed and generated_mapping,
                "conditional_transport": target_consumed and generated_mapping,
            }
        if (
            case["target_z2"] != target_z2
            or case["target_z3"] != target_z3
            or (
                case["family"] != "NATURAL"
                and case["expected_answer"] != expected_answer
            )
            or case["predicates"] != predicates
        ):
            raise ValueError("CASE_DERIVATION:" + case["family"])
        rebuilt.append({**case, "predicates": predicates})
    return rebuilt


def subgroup(
    rows: list[dict[str, Any]], predicate: str, minimum: int,
) -> tuple[float, list[dict[str, Any]]]:
    dimensions = {
        "natural_z2": [row["registered_natural"]["z2"] for row in rows],
        "answer": [row["registered_natural"]["answer"] for row in rows],
        "template": [row["template_id"] for row in rows],
        "orientation": [row["orientation"] for row in rows],
    }
    output = []
    for dimension, values in dimensions.items():
        for value, denominator in sorted(Counter(values).items(), key=lambda x: str(x[0])):
            if denominator < minimum:
                continue
            selected = [
                row["predicates"][predicate]
                for row, item in zip(rows, values) if item == value
            ]
            output.append({
                "family": dimension, "value": str(value),
                "correct": sum(selected), "denominator": denominator,
                "accuracy": accuracy(selected),
            })
    return min(item["accuracy"] for item in output), output


def aggregate(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_family = {
        family: [row for row in cases if row["family"] == family]
        for family in config["families"]
    }
    natural = by_family["NATURAL"]
    gates = config["gates"]
    natural_values = {
        name: accuracy([row["predicates"][name] for row in natural])
        for name in ("natural_z2", "natural_z3", "natural_answer", "natural_joint")
    }
    natural_min, natural_groups = subgroup(
        natural, "natural_joint", gates["minimum_powered_subgroup_denominator"]
    )
    family_values = {}
    for family, rows in by_family.items():
        if family == "NATURAL":
            continue
        conditional = [row for row in rows if row["predicates"]["natural_joint"]]
        minimum, groups = subgroup(
            rows, "generated_mapping",
            gates["minimum_powered_subgroup_denominator"],
        )
        family_values[family] = {
            "cases": len(rows),
            "target_consumption_accuracy": accuracy([
                row["predicates"]["target_consumed"] for row in rows
            ]),
            "generated_mapping_accuracy": accuracy([
                row["predicates"]["generated_mapping"] for row in rows
            ]),
            "minimum_powered_subgroup_accuracy": minimum,
            "subgroups": groups,
            "complete_denominator_joint_success": accuracy([
                row["predicates"]["joint_success"] for row in rows
            ]),
            "conditional_denominator": len(conditional),
            "conditional_transport_accuracy": accuracy([
                row["predicates"]["conditional_transport"] for row in conditional
            ]),
            "trace_identity_accuracy": (
                accuracy([row["predicates"]["trace_identity"] for row in rows])
                if family == "SHAM" else
                accuracy([
                    row["predicates"]["trace_identity"] for row in conditional
                ]) if family == "SAME_Z2" else None
            ),
        }
    predicates = {
        "complete_denominator": all(
            len(by_family[family]) == len(natural) for family in config["families"]
        ),
        "natural_answer":
            natural_values["natural_answer"] >= gates["natural_answer_accuracy"],
        "natural_z2": natural_values["natural_z2"] >= gates["natural_z2_accuracy"],
        "natural_z3": natural_values["natural_z3"] >= gates["natural_z3_accuracy"],
        "natural_subgroups":
            natural_min >= gates["minimum_powered_subgroup_accuracy"],
        "exact_target_consumption": all(
            row["target_consumption_accuracy"] == gates["target_consumption_accuracy"]
            for row in family_values.values()
        ),
        "generated_mapping": all(
            row["generated_mapping_accuracy"] >= gates["generated_mapping_accuracy"]
            and row["minimum_powered_subgroup_accuracy"]
            >= gates["minimum_powered_subgroup_accuracy"]
            for row in family_values.values()
        ),
        "conditional_transport": all(
            row["conditional_transport_accuracy"]
            >= gates["conditional_transport_accuracy"]
            for row in family_values.values()
        ),
        "complete_joint": all(
            row["complete_denominator_joint_success"]
            >= gates["complete_denominator_joint_success"]
            for row in family_values.values()
        ),
        "sham_identity":
            family_values["SHAM"]["trace_identity_accuracy"]
            == gates["sham_trace_identity"],
        "same_state_identity":
            family_values["SAME_Z2"]["trace_identity_accuracy"]
            == gates["sham_trace_identity"],
    }
    return {
        "schema": "exp33a.csw_am_scb_m1.metrics.v1",
        "cases": len(cases), "examples": len(natural),
        "natural": {
            **natural_values,
            "minimum_powered_subgroup_accuracy": natural_min,
            "subgroups": natural_groups,
        },
        "families": family_values,
        "predicates": predicates,
        "passed": all(predicates.values()),
    }


def equal_metrics(left: Any, right: Any, path: str = "") -> None:
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("METRIC_KEYS:" + path)
        for key in left:
            equal_metrics(left[key], right[key], path + "/" + key)
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError("METRIC_LIST:" + path)
        for index, (a, b) in enumerate(zip(left, right)):
            equal_metrics(a, b, path + f"/{index}")
    elif isinstance(left, float):
        if not isinstance(right, (int, float)) or not math.isclose(
            left, right, abs_tol=1e-15
        ):
            raise ValueError("METRIC_FLOAT:" + path)
    elif left != right:
        raise ValueError("METRIC_VALUE:" + path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["D0", "C1"], required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = read_json(CONFIG)
        inputs = read_rows(args.evidence / "INPUT_ROWS.jsonl")
        cases = read_rows(args.evidence / "CASES.jsonl")
        rebuilt_cases = verify_and_rebuild_cases(inputs, cases, config)
        rebuilt_metrics = aggregate(rebuilt_cases, config)
        equal_metrics(read_json(args.evidence / "METRICS.json"), rebuilt_metrics)
        terminal = read_json(args.evidence / "TERMINAL.json")
        expected_terminal = (
            config["terminals"][f"{args.phase}_pass"]
            if rebuilt_metrics["passed"]
            else config["terminals"][f"{args.phase}_fail"]
        )
        if terminal["terminal"] != expected_terminal:
            raise ValueError("TERMINAL_MISMATCH")
        result = {
            "schema": "exp33a.csw_am_scb_m1.reference.v1",
            "phase": args.phase, "terminal": expected_terminal,
            "examples_reconstructed": len(inputs),
            "cases_reconstructed": len(cases),
            "metrics_agree": True, "terminal_agrees": True,
        }
        write_json(args.evidence / "REFERENCE_RESULT.json", result)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
