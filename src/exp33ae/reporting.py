"""Independent raw-ledger metric reconstruction for audit tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .artifacts import write_json
from .evaluate import read_jsonl
from .spec import NAMESPACE


def _wilson(correct: int, total: int):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def _metric(rows, predicate):
    correct = sum(bool(row["completed"]) and predicate(row) for row in rows)
    total = len(rows)
    return {"correct": correct, "total": total, "rate": correct / total, "wilson95": _wilson(correct, total)}


def run_metric_reconstruction(root: Path):
    evaluation = json.loads((root / "results/dev_only/development-evaluation.json").read_text())
    ledger = read_jsonl(root / "registries/dev_only/evaluation-call-ledger.jsonl")
    output_rows = []
    for result in evaluation["model_results"]:
        kind, seed = result["model_kind"], result["model_seed"]
        calls = [row for row in ledger if row["model_kind"] == kind and row["model_seed"] == seed]
        groups = {
            (family, pair_kind): [
                row for row in calls if row["control_family"] == family and row["pair_kind"] == pair_kind
            ]
            for family, pair_kind in {
                (row["control_family"], row["pair_kind"]) for row in calls
            }
        }
        metrics = {}
        causal = groups[("causal_candidate", "different_state")]
        metrics["different_state_iia"] = _metric(causal, lambda r: r["intervened_prediction"] == r["expected_joint"])
        metrics["state_transport"] = _metric(causal, lambda r: r["decoded_intervened_z2"] == r["source_z2"])
        metrics["descendant_correctness"] = _metric(causal, lambda r: r["intervened_z3"] == r["expected_z3"])
        metrics["non_descendant_invariance"] = _metric(causal, lambda r: r["intervened_n3"] == r["expected_n3"])
        same = groups[("causal_candidate", "same_state")]
        metrics["same_state_preservation"] = _metric(same, lambda r: r["intervened_prediction"] == r["base_joint"])
        same_answer = groups[("causal_candidate", "same_answer_different_state")]
        metrics["same_answer_different_state_iia"] = _metric(
            same_answer, lambda r: r["intervened_prediction"] == r["expected_joint"]
        )
        for family in (
            "matched_random_direction", "matched_orthogonal_direction",
            "matched_random_location", "matched_random_layer",
        ):
            metrics[f"{family}_iia"] = _metric(
                groups[(family, "different_state")],
                lambda r: r["intervened_prediction"] == r["expected_joint"],
            )
        metrics["call_completion"] = _metric(calls, lambda r: r["completed"] is True)
        metrics["base_competence_all_calls"] = _metric(calls, lambda r: r["base_prediction"] == r["base_joint"])
        metrics["source_competence_all_calls"] = _metric(calls, lambda r: r["source_prediction"] == r["source_joint"])
        for axis, value in result["competence"]["by_axis"].items():
            metrics[f"task_accuracy:{axis}"] = {
                "correct": value["correct"], "total": value["total"], "rate": value["accuracy"],
                "wilson95": _wilson(value["correct"], value["total"]),
            }
        metrics["decode_accuracy"] = {
            "correct": result["decode"]["correct"], "total": result["decode"]["total"],
            "rate": result["decode"]["accuracy"],
            "wilson95": _wilson(result["decode"]["correct"], result["decode"]["total"]),
        }
        for name, value in metrics.items():
            output_rows.append({"model_kind": kind, "model_seed": seed, "metric": name, **value})

    report = {
        "schema": "exp33ae.metric_reconstruction.v1", "namespace": NAMESPACE,
        "claim_bearing": False, "unit_of_analysis": "call for intervention rates; model seed for replication",
        "correlation_disclosure": "calls share models and some source/base pair identities; Wilson intervals are descriptive, not independent-replicate claims",
        "rows": output_rows,
        "confirmation_artifacts_created": 0, "confirmation_artifacts_accessed": 0,
    }
    write_json(root / "results/dev_only/metric-reconstruction.json", report)
    path = root / "results/dev_only/metric-reconstruction.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_kind", "model_seed", "metric", "correct", "total", "rate", "wilson95_low", "wilson95_high"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow({
                **{key: row[key] for key in ("model_kind", "model_seed", "metric", "correct", "total", "rate")},
                "wilson95_low": row["wilson95"][0], "wilson95_high": row["wilson95"][1],
            })
    return report
