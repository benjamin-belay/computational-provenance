#!/usr/bin/env python3
"""Independent reconstruction of one-shot receipt-bound integration validation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-int-v1/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REC = module("int_v1_receipt_reference", "tools/csw_am_scb_m1_rec_d0_reference.py")
CAL = module("int_v1_carrier_reference", "tools/csw_am_scb_m1_rng2_cal1_reference.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rows_sha(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical(row) + b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    return (
        scores[category] > config["carrier"]["T"]
        and scores[category] == max(scores)
        and scores.count(max(scores)) == 1
        and scores[category] - wrong > config["carrier"]["M"]
    )


def expected(primitive: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return {
        **{key: primitive[key] for key in ("run_id", "context_id", "execution_id", "nonce")},
        "checkpoint_sha256": checkpoint,
    }


def reconstruct(stage: Path, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    official = read_json(stage / "production/OFFICIAL_RESULT.json")
    key = bytes.fromhex(read_json(stage / "reveal/KEY_REVEAL.json")["receipt_key_hex"])
    protocol = read_json(ROOT / "protocol/csw-am-scb-m1-rec-s0/RECEIPT_PROTOCOL.json")
    receipt_config = read_json(ROOT / "protocol/csw-am-scb-m1-rec-d0/CONFIG.json")
    checkpoint = receipt_config["lineage"]["M1_checkpoint_sha256"]
    input_rows = read_rows(stage / "release/REGISTERED_EXECUTION_ROWS.jsonl")
    primitive_rows = read_rows(stage / "raw/EXECUTION_ROWS.jsonl")
    package_rows = read_rows(stage / "raw/RECEIPT_PACKAGES.jsonl")
    output_rows = read_rows(stage / "raw/OUTPUTS.jsonl")
    attack_rows = read_rows(stage / "raw/ATTACK_INPUTS.jsonl")
    primitives = {row["package_id"]: row for row in primitive_rows}
    packages = {row["package_id"]: row for row in package_rows}

    scores, grouped = [], defaultdict(list)
    for row in output_rows:
        parsed, counts, normalized = CAL.score(row["literal_output"])
        expected_values = {key: row["semantic_payload"][key] for key in CAL.FIELDS}
        if parsed["values"] != expected_values or len(parsed["eligible"]) != 24:
            raise ValueError("SEMANTIC")
        scores.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.primitive_score.v1",
            "package_id": row["package_id"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        grouped[(row["package_id"], row["condition"])].append(normalized)
    aggregates = {
        key: [
            math.fsum(vector[category] for vector in vectors) / math.sqrt(8)
            for category in range(16)
        ]
        for key, vectors in grouped.items()
    }
    aggregate_rows = [
        {
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.aggregate.v1",
            "package_id": package_id,
            "condition": condition,
            "scores": values,
        }
        for (package_id, condition), values in sorted(aggregates.items())
    ]

    genuine = []
    for package in package_rows:
        primitive = primitives[package["package_id"]]
        decisions = REC.evaluate_bundle(
            key, package["receipts"], expected(primitive, checkpoint), protocol
        )
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        carrier = (
            "ACCEPT" if accepted(
                aggregates[(package["package_id"], "STATE")], category, config
            ) else "INDETERMINATE"
        )
        genuine.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.genuine.v1",
            "closure_id": primitive["closure_id"],
            "package_id": package["package_id"],
            "role": primitive["role"],
            "receipt": decisions,
            "carrier": carrier,
            "integrated": integrated(
                decisions["exact_z2_event"], decisions["abstract_z2_state"],
                decisions["chain_consistency"], carrier,
            ),
        })

    attack_verdicts = []
    for attack in attack_rows:
        output_primitive = primitives[attack["output_package_id"]]
        binding = expected(output_primitive, checkpoint)
        exact_package = packages.get(attack["candidate_exact_package_id"])
        abstract_package = packages.get(attack["abstract_package_id"])
        chain_package = packages.get(attack["chain_package_id"])
        exact_receipt = exact_package["receipts"]["exact_z2"] if exact_package else None
        exact = REC.exact_decision(key, exact_receipt, "EXACT_Z2", protocol, binding)
        if attack["inline_abstract_z2"] is not None:
            abstract_receipt = attack["inline_abstract_z2"]
            abstract_exact = packages[attack["output_package_id"]]["receipts"]["exact_z2"]
        elif abstract_package:
            abstract_receipt = abstract_package["receipts"]["abstract_z2"]
            abstract_exact = abstract_package["receipts"]["exact_z2"]
        else:
            abstract_receipt = abstract_exact = None
        abstract = REC.abstract_decision(
            key, abstract_receipt, abstract_exact,
            "ABSTRACT_Z2", "EXACT_Z2", protocol,
        )
        if attack["inline_chain"] is not None:
            bundle = json.loads(json.dumps(packages[attack["output_package_id"]]["receipts"]))
            bundle["chain"] = attack["inline_chain"]
        elif chain_package:
            bundle = chain_package["receipts"]
        else:
            bundle = {
                name: None for name in
                ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")
            }
        chain = REC.evaluate_bundle(key, bundle, binding, protocol)["chain_consistency"]
        category = (
            abstract_receipt["preimage"].get("category")
            if abstract == "ACCEPT" and abstract_receipt else None
        )
        if abstract == "ABSTAIN":
            carrier = "INDETERMINATE"
        elif abstract == "REJECT" or category is None:
            carrier = "REJECT"
        else:
            carrier = (
                "ACCEPT" if accepted(
                    aggregates[(attack["output_package_id"], "STATE")],
                    category, config,
                ) else "REJECT"
            )
        attack_verdicts.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "closure_id": attack["closure_id"],
            "family": attack["family"],
            "exact": exact,
            "abstract": abstract,
            "chain": chain,
            "carrier": carrier,
            "integrated": integrated(exact, abstract, chain, carrier),
        })

    input_by_closure = {row["closure_id"]: row for row in input_rows}
    by_closure_primitive, by_closure_genuine = defaultdict(list), defaultdict(list)
    for row in primitive_rows:
        by_closure_primitive[row["closure_id"]].append(row)
    for row in genuine:
        by_closure_genuine[row["closure_id"]].append(row)
    closures = []
    for closure_id in sorted(input_by_closure):
        row = input_by_closure[closure_id]
        alias_z2 = (row["natural"]["z2"] + 8) % 16
        truth = {
            "NATURAL": row["natural"],
            "ALIAS_Z2": {
                "z2": alias_z2,
                "z3": (5 * alias_z2 + row["inputs"]["d"]) % 16,
            },
        }
        truth["ALIAS_Z2"]["answer"] = truth["ALIAS_Z2"]["z3"] % 8
        computation_ok = all(
            primitive["consumed_z2"] == truth[primitive["role"]]["z2"]
            and primitive["generated_z3"] == primitive["consumed_z3"]
            == truth[primitive["role"]]["z3"]
            and primitive["answer"] == truth[primitive["role"]]["answer"]
            for primitive in by_closure_primitive[closure_id]
        )
        sides_ok = all(
            all(value == "ACCEPT" for value in verdict["receipt"].values())
            and verdict["carrier"] == "ACCEPT"
            and verdict["integrated"]["exact_execution_provenance"] == "ACCEPT"
            and verdict["integrated"]["abstract_state_provenance"] == "ACCEPT"
            for verdict in by_closure_genuine[closure_id]
        )
        off = wrong = False
        for verdict in by_closure_genuine[closure_id]:
            package_id = verdict["package_id"]
            off_values = aggregates[(package_id, "CARRIER_OFF")]
            off = off or any(accepted(off_values, candidate, config) for candidate in range(16))
            state_values = aggregates[(package_id, "STATE")]
            category = primitives[package_id]["consumed_z2"]
            wrong = wrong or any(
                accepted(state_values, candidate, config)
                for candidate in range(16) if candidate != category
            )
        closures.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.closure.v1",
            "closure_id": closure_id,
            "computation_fidelity": computation_ok,
            "receipt_carrier_pair": sides_ok,
            "carrier_off_event": off,
            "wrong_confusion_event": wrong,
            "complete_success": computation_ok and sides_ok and not off and not wrong,
        })

    reconstructed = {
        "primitive_sha256": rows_sha(scores),
        "aggregates_sha256": rows_sha(aggregate_rows),
        "genuine_sha256": rows_sha(genuine),
        "attacks_sha256": rows_sha(attack_verdicts),
        "closures_sha256": rows_sha(closures),
        "outputs": len(output_rows),
        "execution_packages": len(genuine),
        "attacks": len(attack_verdicts),
        "closures": len(closures),
    }
    gates = {
        name: reconstructed[name + "_sha256"] == official["evidence"][name + "_sha256"]
        for name in ("primitive", "aggregates", "genuine", "attacks", "closures")
    }
    gates["counts"] = (
        reconstructed["outputs"] == config["denominators"]["outputs"]
        and reconstructed["execution_packages"] == config["denominators"]["execution_packages"]
        and reconstructed["attacks"] == config["denominators"]["attack_packages"]
        and reconstructed["closures"] == config["denominators"]["closures"]
    )
    gates["official"] = official["passed"] is True
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.reference.v1",
        "identity": config["identity"],
        "reconstructed": reconstructed,
        "gates": gates,
        "passed": all(gates.values()),
    }
    write_json(output / "REFERENCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(
        args.stage.resolve(), args.output.resolve()
    ), sort_keys=True))


if __name__ == "__main__":
    main()
