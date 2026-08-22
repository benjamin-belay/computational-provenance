"""Re-evaluate the frozen multi-seed computation and causal-transport claims."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import torch

from .feedforward import CausalStrictBottleneck
from .transformer import EngineeredCausalTransformer


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("NATURAL", "ALIAS_Z2", "NON_ALIAS_Z2", "WRONG_Z2", "SAME_Z2", "SHAM", "DIRECT_Z3")


def _read_rows() -> list[dict[str, Any]]:
    with gzip.open(ROOT / "data/evaluation/robustness.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_models(architecture: str) -> tuple[dict[str, Any], list[tuple[str, torch.nn.Module]]]:
    if architecture == "feedforward":
        protocol = json.loads((ROOT / "protocols/feedforward_robustness.json").read_text())
        model_class = CausalStrictBottleneck
        paths = sorted((ROOT / "checkpoints/robustness/feedforward").glob("*.pt"))
        expected_count, expected_step, expected_parameters = 5, 5000, 12648
    elif architecture == "transformer":
        protocol = json.loads((ROOT / "protocols/transformer_robustness.json").read_text())
        model_class = EngineeredCausalTransformer
        paths = sorted((ROOT / "checkpoints/robustness/transformer").glob("*.pt"))
        expected_count, expected_step, expected_parameters = 3, 12000, 546088
    else:
        raise ValueError("architecture must be feedforward or transformer")
    if len(paths) != expected_count:
        raise ValueError(f"expected {expected_count} {architecture} checkpoints")
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("step") != expected_step or payload.get("parameter_count") != expected_parameters:
            raise ValueError(f"checkpoint identity: {path.name}")
        model = model_class()
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append((payload["model_id"], model.eval()))
    return protocol, models


def _values(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([[row["inputs"][key] for key in ("a", "b", "c", "d")] for row in rows], dtype=torch.int64)


def _ratio(values: torch.Tensor) -> float:
    return float(values.to(torch.float64).mean())


def _minimum_subgroup(rows: list[dict[str, Any]], success: list[bool], minimum: int) -> float:
    groups: dict[tuple[str, Any], list[bool]] = {}
    for row, passed in zip(rows, success):
        values = {"template": row["template_id"], "z2": row["natural"]["z2"],
                  "z3": row["natural"]["z3"], "answer": row["natural"]["answer"],
                  "orientation": row.get("orientation")}
        for name, value in values.items():
            if value is not None:
                groups.setdefault((name, value), []).append(passed)
    powered = [sum(values) / len(values) for values in groups.values() if len(values) >= minimum]
    return min(powered)


@torch.no_grad()
def _evaluate_model(model: torch.nn.Module, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    counts = {family: {key: 0 for key in ("target", "mapping", "transport", "joint", "identity")} for family in FAMILIES}
    total = {family: 0 for family in FAMILIES}
    natural_correct_total = 0
    natural_z2_total = natural_z3_total = natural_answer_total = 0
    natural_success: list[bool] = []
    mapping_success: dict[str, list[bool]] = {family: [] for family in FAMILIES[1:]}
    selected_equals_consumed = True
    emitted_z2: set[int] = set()
    emitted_z3: set[int] = set()
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        values = _values(block)
        truth_z2 = torch.tensor([row["natural"]["z2"] for row in block])
        truth_z3 = torch.tensor([row["natural"]["z3"] for row in block])
        truth_answer = torch.tensor([row["natural"]["answer"] for row in block])
        natural = model.run_natural(values)
        nz2 = natural["z2_selected"]
        nz3 = natural["z3_selected"]
        nanswer = natural["answer_logits"].argmax(-1)
        natural_correct = (nz2 == truth_z2) & (nz3 == truth_z3) & (nanswer == truth_answer)
        natural_success.extend(natural_correct.tolist())
        selected_equals_consumed = selected_equals_consumed and bool(torch.all(nz2 == natural["z2_consumed"])) and bool(torch.all(nz3 == natural["z3_consumed"]))
        emitted_z2.update(nz2.tolist())
        emitted_z3.update(nz3.tolist())
        natural_z2_total += int((nz2 == truth_z2).sum())
        natural_z3_total += int((nz3 == truth_z3).sum())
        natural_answer_total += int((nanswer == truth_answer).sum())
        natural_correct_total += int(natural_correct.sum())
        total["NATURAL"] += len(block)
        counts["NATURAL"]["joint"] += int(natural_correct.sum())

        definitions = {
            "ALIAS_Z2": ((truth_z2 + 8) % 16, (truth_z3 + 8) % 16),
            "NON_ALIAS_Z2": ((truth_z2 + 1) % 16, (5 * ((truth_z2 + 1) % 16) + values[:, 3]) % 16),
            "WRONG_Z2": ((truth_z2 + 3) % 16, (5 * ((truth_z2 + 3) % 16) + values[:, 3]) % 16),
            "SAME_Z2": (truth_z2, truth_z3),
            "SHAM": (nz2, (5 * nz2 + values[:, 3]) % 16),
        }
        for family, (target_z2, target_z3) in definitions.items():
            output = model.run_with_z2(values, target_z2)
            observed_z3 = output["z3_selected"]
            observed_answer = output["answer_logits"].argmax(-1)
            target = output["z2_consumed"] == target_z2
            mapping = observed_z3 == target_z3
            mapping_success[family].extend(mapping.tolist())
            transport = mapping & (observed_answer == target_z3 % 8)
            joint = natural_correct & target & transport
            identity = (observed_z3 == nz3) & (observed_answer == nanswer)
            if family == "SAME_Z2":
                identity = (~natural_correct) | identity
            total[family] += len(block)
            for key, value in (("target", target), ("mapping", mapping), ("transport", transport), ("joint", joint), ("identity", identity)):
                counts[family][key] += int(value.sum())

        target_z3 = (truth_z3 + 1) % 16
        output = model.run_with_z3(target_z3)
        observed_answer = output["answer_logits"].argmax(-1)
        target = output["z3_consumed"] == target_z3
        mapping = target
        mapping_success["DIRECT_Z3"].extend(mapping.tolist())
        transport = mapping & (observed_answer == target_z3 % 8)
        joint = natural_correct & target & transport
        total["DIRECT_Z3"] += len(block)
        for key, value in (("target", target), ("mapping", mapping), ("transport", transport), ("joint", joint)):
            counts["DIRECT_Z3"][key] += int(value.sum())

    n = len(rows)
    natural_metrics = {
        "z2_accuracy": natural_z2_total / n,
        "z3_accuracy": natural_z3_total / n,
        "answer_accuracy": natural_answer_total / n,
        "joint_accuracy": natural_correct_total / n,
        "minimum_powered_subgroup_accuracy": _minimum_subgroup(rows, natural_success, 16),
        "selected_equals_consumed": selected_equals_consumed,
        "all_z2_categories_emitted": emitted_z2 == set(range(16)),
        "all_z3_categories_emitted": emitted_z3 == set(range(16)),
    }
    families = {}
    for family in FAMILIES[1:]:
        denominator = total[family]
        families[family] = {
            "denominator": denominator,
            "target_consumption_accuracy": counts[family]["target"] / denominator,
            "generated_mapping_accuracy": counts[family]["mapping"] / denominator,
            "conditional_transport_accuracy": counts[family]["transport"] / denominator,
            "complete_denominator_joint_success": counts[family]["joint"] / denominator,
            "minimum_powered_subgroup_accuracy": _minimum_subgroup(rows, mapping_success[family], 16),
            "trace_identity_accuracy": counts[family]["identity"] / denominator if family in ("SHAM", "SAME_Z2") else None,
        }
    return {"examples": n, "cases": n * len(FAMILIES), "natural": natural_metrics, "families": families}


def _passes(architecture: str, metrics: dict[str, Any], protocol: dict[str, Any]) -> bool:
    if architecture == "feedforward":
        computation = protocol["gates"]
        causal = computation
    else:
        computation = protocol["computation"]["gates"]
        causal = protocol["causal"]["gates"]
    natural = metrics["natural"]
    def gate(primary: str, fallback: str) -> float:
        return computation[primary] if primary in computation else computation[fallback]
    checks = [
        natural["z2_accuracy"] >= gate("natural_z2_accuracy", "z2_accuracy"),
        natural["z3_accuracy"] >= gate("natural_z3_accuracy", "z3_accuracy"),
        natural["answer_accuracy"] >= gate("natural_answer_accuracy", "answer_accuracy"),
        natural["minimum_powered_subgroup_accuracy"] >= computation["minimum_powered_subgroup_accuracy"],
        natural["selected_equals_consumed"],
        natural["all_z2_categories_emitted"],
        natural["all_z3_categories_emitted"],
    ]
    for family, values in metrics["families"].items():
        checks.extend((
            values["target_consumption_accuracy"] >= causal["target_consumption_accuracy"],
            values["generated_mapping_accuracy"] >= causal["generated_mapping_accuracy"],
            values["minimum_powered_subgroup_accuracy"] >= computation["minimum_powered_subgroup_accuracy"],
            values["conditional_transport_accuracy"] >= causal["conditional_transport_accuracy"],
            values["complete_denominator_joint_success"] >= causal["complete_denominator_joint_success"],
        ))
        if family == "SHAM":
            checks.append(values["trace_identity_accuracy"] >= causal["sham_trace_identity"])
        if family == "SAME_Z2":
            checks.append(values["trace_identity_accuracy"] >= causal.get("same_state_trace_identity", causal["sham_trace_identity"]))
    return all(checks)


def reproduce_robustness(architecture: str, *, limit: int | None = None) -> dict[str, Any]:
    """Run every registered intervention family on every released held-out row."""
    protocol, models = _load_models(architecture)
    rows = _read_rows()
    if limit is not None:
        rows = rows[:limit]
    results = []
    for model_id, model in models:
        metrics = _evaluate_model(model, rows, batch_size=512)
        results.append({"model_id": model_id, **metrics, "passed": _passes(architecture, metrics, protocol)})
    expected = 5 if architecture == "feedforward" else 3
    return {
        "architecture": architecture,
        "held_out_rows_per_model": len(rows),
        "causal_cases_per_model": len(rows) * len(FAMILIES),
        "models": results,
        "models_passed": sum(row["passed"] for row in results),
        "expected_models": expected,
        "all_models_passed": len(results) == expected and all(row["passed"] for row in results),
    }
