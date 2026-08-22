"""Independent reconstruction of the four end-to-end paper evaluations."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .carrier import score_literal
from .evidence import ROOT, _rows
from .receipts import abstract_decision, evaluate_bundle, exact_decision


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rows_sha(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical(row) + b"\n")
    return digest.hexdigest()


def combine(*values: str) -> str:
    for decision in ("REJECT", "ABSTAIN", "INDETERMINATE"):
        if decision in values:
            return decision
    return "ACCEPT"


def integrated(exact: str, abstract: str, chain: str, carrier: str) -> dict[str, str]:
    return {
        "exact_execution_provenance": combine(exact, abstract, chain, carrier),
        "abstract_state_provenance": combine(abstract, carrier),
        "causal_output_provenance": combine(abstract, chain, carrier),
    }


def accepted(scores: list[float], category: int, config: dict[str, Any]) -> bool:
    wrong = max(scores[index] for index in range(16) if index != category)
    largest = max(scores)
    return (scores[category] > config["carrier"]["T"] and scores[category] == largest
            and scores.count(largest) == 1 and scores[category] - wrong > config["carrier"]["M"])


def expected(primitive: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return {**{key: primitive[key] for key in ("run_id", "context_id", "execution_id", "nonce")},
            "checkpoint_sha256": checkpoint}


def reconstruct_evaluation(architecture: str, split: str) -> dict[str, Any]:
    """Recompute scores, receipt decisions, attacks, closures, and scientific gates."""
    directory = ROOT / "evidence" / "full" / architecture / split
    config = json.loads((ROOT / "protocols" / f"{architecture}_{split}.json").read_text())
    protocol = json.loads((ROOT / "protocols" / f"{architecture}_public_receipts.json").read_text())
    master = bytes.fromhex(json.loads((directory / "KEY_REVEAL.json").read_text())["receipt_key_hex"])
    checkpoint = protocol["state_ontology"]["checkpoint_sha256"]
    primitive_rows = list(_rows(directory / "EXECUTION_ROWS.jsonl.gz"))
    package_rows = list(_rows(directory / "RECEIPT_PACKAGES.jsonl.gz"))
    output_rows = list(_rows(directory / "OUTPUTS.jsonl.gz"))
    attack_rows = list(_rows(directory / "ATTACK_INPUTS.jsonl.gz"))
    primitives = {row["package_id"]: row for row in primitive_rows}
    packages = {row["package_id"]: row for row in package_rows}

    scores: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for row in output_rows:
        parsed, counts, normalized = score_literal(row["literal_output"])
        if parsed["values"] != {field: row["semantic_payload"][field] for field in ("z1", "z2", "z3", "answer")}:
            raise ValueError(f"semantic payload mismatch: {architecture}/{split}/{row['package_id']}")
        scores.append({"schema": "exp33a.csw_am_scb_m1_rng2_int_d0.primitive_score.v1",
                       "package_id": row["package_id"], "condition": row["condition"],
                       "output_index": row["output_index"], "green_counts": counts,
                       "normalized_scores": normalized})
        grouped[(row["package_id"], row["condition"])].append(normalized)
    aggregates = {key: [math.fsum(vector[category] for vector in vectors) / math.sqrt(len(vectors))
                        for category in range(16)] for key, vectors in grouped.items()}
    aggregate_rows = [{"schema": "exp33a.csw_am_scb_m1_rng2_int_v1.aggregate.v1",
                       "package_id": package_id, "condition": condition, "scores": values}
                      for (package_id, condition), values in sorted(aggregates.items())]

    genuine: list[dict[str, Any]] = []
    for package in package_rows:
        primitive = primitives[package["package_id"]]
        decisions = evaluate_bundle(master, package["receipts"], expected(primitive, checkpoint), protocol)
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        carrier = "ACCEPT" if accepted(aggregates[(package["package_id"], "STATE")], category, config) else "INDETERMINATE"
        genuine.append({"schema": "exp33a.csw_am_scb_m1_rng2_int_v1.genuine.v1",
                        "closure_id": primitive["closure_id"], "package_id": package["package_id"],
                        "role": primitive["role"], "receipt": decisions, "carrier": carrier,
                        "integrated": integrated(decisions["exact_z2_event"], decisions["abstract_z2_state"],
                                                 decisions["chain_consistency"], carrier)})

    attack_verdicts: list[dict[str, Any]] = []
    for attack in attack_rows:
        output_primitive = primitives[attack["output_package_id"]]
        binding = expected(output_primitive, checkpoint)
        exact_package = packages.get(attack["candidate_exact_package_id"])
        abstract_package = packages.get(attack["abstract_package_id"])
        chain_package = packages.get(attack["chain_package_id"])
        exact_receipt = exact_package["receipts"]["exact_z2"] if exact_package else None
        exact = exact_decision(master, exact_receipt, "EXACT_Z2", protocol, binding)
        if attack["inline_abstract_z2"] is not None:
            abstract_receipt = attack["inline_abstract_z2"]
            abstract_exact = packages[attack["output_package_id"]]["receipts"]["exact_z2"]
        elif abstract_package:
            abstract_receipt = abstract_package["receipts"]["abstract_z2"]
            abstract_exact = abstract_package["receipts"]["exact_z2"]
        else:
            abstract_receipt = abstract_exact = None
        abstract = abstract_decision(master, abstract_receipt, abstract_exact, "ABSTRACT_Z2", "EXACT_Z2", protocol)
        if attack["inline_chain"] is not None:
            bundle = json.loads(json.dumps(packages[attack["output_package_id"]]["receipts"]))
            bundle["chain"] = attack["inline_chain"]
        elif chain_package:
            bundle = chain_package["receipts"]
        else:
            bundle = {name: None for name in ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")}
        chain = evaluate_bundle(master, bundle, binding, protocol)["chain_consistency"]
        category = abstract_receipt["preimage"].get("category") if abstract == "ACCEPT" and abstract_receipt else None
        if abstract == "ABSTAIN":
            carrier = "INDETERMINATE"
        elif abstract == "REJECT" or category is None:
            carrier = "REJECT"
        else:
            carrier = "ACCEPT" if accepted(aggregates[(attack["output_package_id"], "STATE")], category, config) else "REJECT"
        attack_verdicts.append({"schema": "exp33a.csw_am_scb_m1_rng2_int_v1.attack_verdict.v1",
                                "attack_id": attack["attack_id"], "closure_id": attack["closure_id"],
                                "family": attack["family"], "exact": exact, "abstract": abstract,
                                "chain": chain, "carrier": carrier,
                                "integrated": integrated(exact, abstract, chain, carrier)})

    by_closure_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_closure_genuine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primitive_rows:
        by_closure_primitive[row["closure_id"]].append(row)
    for row in genuine:
        by_closure_genuine[row["closure_id"]].append(row)
    closures: list[dict[str, Any]] = []
    for closure_id in sorted(by_closure_primitive):
        source = by_closure_primitive[closure_id][0]
        natural, inputs = source["registered_natural"], source["inputs"]
        alias_z2 = (natural["z2"] + 8) % 16
        truth = {"NATURAL": natural, "ALIAS_Z2": {"z2": alias_z2,
                 "z3": (5 * alias_z2 + inputs["d"]) % 16}}
        truth["ALIAS_Z2"]["answer"] = truth["ALIAS_Z2"]["z3"] % 8
        computation_ok = all(row["consumed_z2"] == truth[row["role"]]["z2"]
                             and row["generated_z3"] == row["consumed_z3"] == truth[row["role"]]["z3"]
                             and row["answer"] == truth[row["role"]]["answer"]
                             for row in by_closure_primitive[closure_id])
        sides_ok = all(all(value == "ACCEPT" for value in verdict["receipt"].values())
                       and verdict["carrier"] == "ACCEPT"
                       and verdict["integrated"]["exact_execution_provenance"] == "ACCEPT"
                       and verdict["integrated"]["abstract_state_provenance"] == "ACCEPT"
                       for verdict in by_closure_genuine[closure_id])
        off = wrong = False
        for verdict in by_closure_genuine[closure_id]:
            package_id = verdict["package_id"]
            off = off or any(accepted(aggregates[(package_id, "CARRIER_OFF")], candidate, config) for candidate in range(16))
            category = primitives[package_id]["consumed_z2"]
            wrong = wrong or any(accepted(aggregates[(package_id, "STATE")], candidate, config)
                                 for candidate in range(16) if candidate != category)
        closures.append({"schema": "exp33a.csw_am_scb_m1_rng2_int_v1.closure.v1",
                         "closure_id": closure_id, "computation_fidelity": computation_ok,
                         "receipt_carrier_pair": sides_ok, "carrier_off_event": off,
                         "wrong_confusion_event": wrong,
                         "complete_success": computation_ok and sides_ok and not off and not wrong})

    derived = {"primitive_sha256": rows_sha(scores), "aggregates_sha256": rows_sha(aggregate_rows),
               "genuine_sha256": rows_sha(genuine), "attacks_sha256": rows_sha(attack_verdicts),
               "closures_sha256": rows_sha(closures), "outputs": len(output_rows),
               "execution_packages": len(genuine), "attacks": len(attack_verdicts), "closures": len(closures)}
    stored = {"primitive_sha256": rows_sha(_rows(directory / "PRIMITIVE_SCORES.jsonl.gz")),
              "aggregates_sha256": rows_sha(_rows(directory / "AGGREGATE_SCORES.jsonl.gz")),
              "genuine_sha256": rows_sha(_rows(directory / "GENUINE_VERDICTS.jsonl.gz")),
              "attacks_sha256": rows_sha(_rows(directory / "ATTACK_VERDICTS.jsonl.gz")),
              "closures_sha256": rows_sha(_rows(directory / "CLOSURE_DECISIONS.jsonl.gz"))}
    exact_reconstruction = all(derived[key] == value for key, value in stored.items())
    complete = sum(row["complete_success"] for row in closures)
    attack_rules = config["attack_expectations"]
    attack_failures = sum(any(row["integrated"].get(channel) != decision for channel, decision in attack_rules[row["family"]].items())
                          for row in attack_verdicts)
    gates = {"complete_closures": complete >= config["gates"]["minimum_complete_closure_successes"],
             "carrier_off": sum(row["carrier_off_event"] for row in closures) <= config["gates"]["maximum_carrier_off_regression_events"],
             "wrong_confusion": sum(row["wrong_confusion_event"] for row in closures) <= config["gates"]["maximum_wrong_confusion_events"],
             "attacks": attack_failures <= config["gates"]["maximum_attack_failures"],
             "exact_reconstruction": exact_reconstruction}
    return {"architecture": architecture, "split": split, "counts": {"closures": len(closures),
            "complete_closures": complete, "attacks": len(attack_verdicts), "attack_failures": attack_failures,
            "outputs": len(output_rows)}, "derived": derived, "gates": gates, "passed": all(gates.values())}


def reconstruct_all() -> dict[str, Any]:
    evaluations = [reconstruct_evaluation(architecture, split)
                   for architecture in ("feedforward", "transformer") for split in ("public", "protected")]
    return {"evaluations": evaluations, "all_four_passed": all(row["passed"] for row in evaluations)}
