"""Reconstruct the paper's selected-model receipt and carrier experiments."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import torch

from .adjudication import accepted, rows_sha
from .carrier import score_literal
from .evidence import ROOT, _rows
from .feedforward import CausalStrictBottleneck
from .receipts import evaluate_bundle
from .robustness import _evaluate_model


def reproduce_causal_confirmation() -> dict[str, Any]:
    directory = ROOT / "evidence/components/causal_confirmation"
    rows = list(_rows(directory / "INPUT_ROWS.jsonl.gz"))
    payload = torch.load(ROOT / "checkpoints/feedforward.pt", map_location="cpu", weights_only=False)
    model = CausalStrictBottleneck()
    model.load_state_dict(payload["state_dict"], strict=True)
    metrics = _evaluate_model(model.eval(), rows, 512)
    return {"examples": metrics["examples"], "cases": metrics["cases"],
            "all_seven_families_passed": metrics["natural"]["joint_accuracy"] == 1.0
            and all(values["complete_denominator_joint_success"] == 1.0 for values in metrics["families"].values()),
            "metrics": metrics}


def _carrier_rows(component: str, primitive_schema: str, aggregate_schema: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[Any, list[float]], dict[Any, int], dict[str, bool]]:
    directory = ROOT / "evidence/components" / component
    primitive, grouped, authority = [], defaultdict(list), {}
    semantic: dict[str, bool] = defaultdict(lambda: True)
    for row in _rows(directory / "RAW_OUTPUTS.jsonl.gz"):
        parsed, counts, normalized = score_literal(row["literal_output"])
        semantic[row["closure_id"]] &= parsed["values"] == {field: row["semantic_payload"][field] for field in ("z1", "z2", "z3", "answer")}
        primitive.append({"schema": primitive_schema, "closure_id": row["closure_id"], "side": row["side"],
                          "condition": row["condition"], "output_index": row["output_index"],
                          "embedded_authority": row["embedded_authority"], "eligible_positions": 24,
                          "green_counts": counts, "normalized_scores": normalized})
        grouped[(row["closure_id"], row["side"], row["condition"])].append(normalized)
        authority[(row["closure_id"], row["side"])] = row["embedded_authority"]
    aggregates, lookup = [], {}
    for (closure_id, side, condition), vectors in sorted(grouped.items()):
        scores = [math.fsum(vector[category] for vector in vectors) / math.sqrt(len(vectors)) for category in range(16)]
        aggregates.append({"schema": aggregate_schema, "closure_id": closure_id, "side": side,
                           "condition": condition, "embedded_authority": authority[(closure_id, side)], "scores": scores})
        lookup[(closure_id, side, condition)] = scores
    return primitive, aggregates, lookup, authority, semantic


def reproduce_carrier_calibration() -> dict[str, Any]:
    primitive, aggregates, lookup, authority, _ = _carrier_rows(
        "carrier_calibration", "exp33a.csw_am_scb_m1_rng2_cal1.primitive_score.v1",
        "exp33a.csw_am_scb_m1_rng2_cal1.aggregate.v1")
    statistics = []
    for closure_id in sorted({key[0] for key in lookup}):
        off_max, off_side, off_category = max((lookup[(closure_id, side, "CARRIER_OFF")][category], side, category)
                                               for side in (0, 1) for category in range(16))
        confusion, details = [], []
        for side in (0, 1):
            scores, target = lookup[(closure_id, side, "AUTHORITY_ON")], authority[(closure_id, side)]
            wrong = [category for category in range(16) if category != target]
            confusion.extend((scores[candidate] - max(scores[other] for other in range(16) if other != candidate), side, candidate) for candidate in wrong)
            strongest = max(wrong, key=lambda category: (scores[category], category))
            details.append({"side": side, "embedded_authority": target, "authenticated_score": scores[target],
                            "strongest_wrong_authority": strongest, "strongest_wrong_score": scores[strongest],
                            "authenticated_rank": 1 + sum(value > scores[target] for value in scores),
                            "authenticated_top_two_margin": scores[target] - scores[strongest]})
        confusion_max, confusion_side, confusion_category = max(confusion)
        statistics.append({"schema": "exp33a.csw_am_scb_m1_rng2_cal1.closure_statistic.v1", "closure_id": closure_id,
                           "carrier_off_maximum": off_max, "carrier_off_source_side": off_side,
                           "carrier_off_source_authority": off_category, "wrong_confusion_maximum": confusion_max,
                           "wrong_confusion_source_side": confusion_side, "wrong_confusion_source_authority": confusion_category,
                           "authority_on_side_details": details})
    threshold = max(row["carrier_off_maximum"] for row in statistics)
    margin = max(0.0, max(row["wrong_confusion_maximum"] for row in statistics))
    directory = ROOT / "evidence/components/carrier_calibration"
    exact = (rows_sha(primitive) == rows_sha(_rows(directory / "PRIMITIVE_SCORES.jsonl.gz"))
             and rows_sha(aggregates) == rows_sha(_rows(directory / "AGGREGATE_SCORES.jsonl.gz"))
             and rows_sha(statistics) == rows_sha(_rows(directory / "CLOSURE_STATISTICS.jsonl.gz")))
    return {"closures": len(statistics), "outputs": len(primitive), "T": threshold, "M": margin,
            "exact_reconstruction": exact, "passed": exact and threshold == 4.041451884327381 and margin == 0.0}


def reproduce_carrier_validation() -> dict[str, Any]:
    primitive, aggregates, lookup, authority, semantic = _carrier_rows(
        "carrier_validation", "exp33a.csw_am_scb_m1_rng2_v1.primitive.v1",
        "exp33a.csw_am_scb_m1_rng2_v1.aggregate.v1")
    comp = {}
    for row in _rows(ROOT / "evidence/components/carrier_validation/VALIDATION_ROWS.jsonl.gz"):
        inputs = row["inputs"]
        z1, z2 = (inputs["a"] + inputs["b"]) % 16, (inputs["a"] + inputs["b"] + inputs["c"]) % 16
        z3 = (5 * z2 + inputs["d"]) % 16
        truth = {"z1": z1, "z2": z2, "z3": z3, "answer": z3 % 8}
        comp[row["closure_id"]] = comp.get(row["closure_id"], True) and truth == row["natural"] and truth["answer"] == row["visible_answer"]
    config = {"carrier": {"T": 4.041451884327381, "M": 0.0}}
    decisions = []
    for closure_id in sorted({key[0] for key in lookup}):
        sides, wrong_event, off_event = [], False, False
        for side in (0, 1):
            scores, target = lookup[(closure_id, side, "AUTHORITY_ON")], authority[(closure_id, side)]
            wrong = max((candidate for candidate in range(16) if candidate != target), key=lambda candidate: (scores[candidate], candidate))
            sides.append({"side": side, "label": "NATURAL" if side == 0 else "ALIAS", "authenticated_authority": target,
                          "authenticated_score": scores[target], "strongest_wrong_authority": wrong,
                          "strongest_wrong_score": scores[wrong], "margin": scores[target] - scores[wrong],
                          "unique_rank_first": scores[target] == max(scores) and scores.count(max(scores)) == 1,
                          "detector_accept": accepted(scores, target, config)})
            wrong_event |= any(accepted(scores, candidate, config) for candidate in range(16) if candidate != target)
            off_event |= any(accepted(lookup[(closure_id, side, "CARRIER_OFF")], candidate, config) for candidate in range(16))
        decisions.append({"schema": "exp33a.csw_am_scb_m1_rng2_v1.closure_decision.v1", "closure_id": closure_id,
                          "computation_fidelity": comp[closure_id], "semantic_validity": semantic[closure_id], "sides": sides,
                          "carrier_off_familywise_event": off_event, "wrong_confusion_familywise_event": wrong_event,
                          "closure_pair_success": comp[closure_id] and semantic[closure_id] and all(row["detector_accept"] for row in sides)})
    directory = ROOT / "evidence/components/carrier_validation"
    exact = (rows_sha(primitive) == rows_sha(_rows(directory / "PRIMITIVE_SCORES.jsonl.gz"))
             and rows_sha(aggregates) == rows_sha(_rows(directory / "AGGREGATE_SCORES.jsonl.gz"))
             and rows_sha(decisions) == rows_sha(_rows(directory / "CLOSURE_DECISIONS.jsonl.gz")))
    observed = {"closures": len(decisions), "outputs": len(primitive), "natural_side_success": sum(row["sides"][0]["detector_accept"] for row in decisions),
                "alias_side_success": sum(row["sides"][1]["detector_accept"] for row in decisions),
                "closure_pair_success": sum(row["closure_pair_success"] for row in decisions),
                "carrier_off_events": sum(row["carrier_off_familywise_event"] for row in decisions),
                "wrong_confusion_events": sum(row["wrong_confusion_familywise_event"] for row in decisions)}
    passed = exact and observed == {"closures": 512, "outputs": 16384, "natural_side_success": 512,
                                   "alias_side_success": 512, "closure_pair_success": 512,
                                   "carrier_off_events": 1, "wrong_confusion_events": 0}
    return {"observed": observed, "exact_reconstruction": exact, "passed": passed}


def reproduce_receipt_confirmation() -> dict[str, Any]:
    directory = ROOT / "evidence/components/receipt_confirmation"
    protocol = json.loads((ROOT / "protocols/feedforward_component_receipts.json").read_text())
    master = bytes.fromhex(json.loads((directory / "KEY_REVEAL.json").read_text())["key_hex"])
    executions = {row["package_id"]: row for row in _rows(directory / "EXECUTION_ROWS.jsonl.gz")}
    packages = {row["package_id"]: row for row in _rows(directory / "RECEIPT_PACKAGES.jsonl.gz")}
    genuine = 0
    for package_id, package in packages.items():
        row = executions[package_id]
        expected = {key: row[key] for key in ("run_id", "context_id", "execution_id", "nonce")}
        expected["checkpoint_sha256"] = protocol["state_ontology"]["checkpoint_sha256"]
        genuine += all(value == "ACCEPT" for value in evaluate_bundle(master, package["receipts"], expected, protocol).values())
    detected = attacks = 0
    for attack in _rows(directory / "ATTACK_INPUTS.jsonl.gz"):
        attacks += 1
        bundle = {}
        for component, package_id in attack["presented_receipt_sources"].items():
            bundle[component] = packages[package_id]["receipts"][component] if package_id else None
        bundle.update(attack["inline_receipts"])
        decisions = evaluate_bundle(master, bundle, attack["expected_binding"], protocol)
        detected += not all(value == "ACCEPT" for value in decisions.values())
    return {"execution_packages": len(packages), "receipt_envelopes": 5 * len(packages), "attacks": attacks,
            "genuine_packages_passing": genuine, "attacks_detected": detected,
            "passed": genuine == 512 and attacks == detected == 9216}


def reproduce_components() -> dict[str, Any]:
    results = {"causal_confirmation": reproduce_causal_confirmation(),
               "receipt_confirmation": reproduce_receipt_confirmation(),
               "carrier_calibration": reproduce_carrier_calibration(),
               "carrier_validation": reproduce_carrier_validation()}
    return {"results": results, "all_passed": all(result.get("passed", result.get("all_seven_families_passed")) for result in results.values())}
