#!/usr/bin/env python3
"""Independent reconstruction of the public RNG2 receipt integration rehearsal."""
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
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-int-d0/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REC_REF = module("int_d0_receipt_reference", "tools/csw_am_scb_m1_rec_d0_reference.py")
CAL_REF = module("int_d0_carrier_reference", "tools/csw_am_scb_m1_rng2_cal1_reference.py")


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
    other = max(scores[index] for index in range(16) if index != category)
    return (
        scores[category] > config["carrier"]["T"]
        and scores[category] == max(scores)
        and scores.count(max(scores)) == 1
        and scores[category] - other > config["carrier"]["M"]
    )


def expected(primitive: dict[str, Any], checkpoint: str) -> dict[str, Any]:
    return {
        **{key: primitive[key] for key in ("run_id", "context_id", "execution_id", "nonce")},
        "checkpoint_sha256": checkpoint,
    }


def reconstruct(stage: Path, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    official = read_json(stage / "production/OFFICIAL_RESULT.json")
    reveal = read_json(stage / "reveal/KEY_REVEAL.json")
    key = bytes.fromhex(reveal["receipt_key_hex"])
    protocol = read_json(ROOT / config["bindings"]["receipt_protocol_path"])
    receipt_config = read_json(ROOT / "protocol/csw-am-scb-m1-rec-d0/CONFIG.json")
    checkpoint = receipt_config["lineage"]["M1_checkpoint_sha256"]
    primitive_rows = read_rows(stage / "raw/EXECUTION_ROWS.jsonl")
    package_rows = read_rows(stage / "raw/RECEIPT_PACKAGES.jsonl")
    outputs = read_rows(stage / "raw/OUTPUTS.jsonl")
    attacks = read_rows(stage / "raw/ATTACK_INPUTS.jsonl")
    primitives = {row["package_id"]: row for row in primitive_rows}
    packages = {row["package_id"]: row for row in package_rows}

    score_rows, grouped = [], defaultdict(list)
    for row in outputs:
        parsed, counts, normalized = CAL_REF.score(row["literal_output"])
        expected_values = {
            key: row["semantic_payload"][key] for key in CAL_REF.FIELDS
        }
        if parsed["values"] != expected_values or len(parsed["eligible"]) != 24:
            raise ValueError("SEMANTIC")
        score_rows.append({
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

    genuine = []
    for package in package_rows:
        primitive = primitives[package["package_id"]]
        decisions = REC_REF.evaluate_bundle(
            key, package["receipts"], expected(primitive, checkpoint), protocol
        )
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        carrier = (
            "ACCEPT"
            if accepted(aggregates[(package["package_id"], "STATE")], category, config)
            else "INDETERMINATE"
        )
        genuine.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.genuine_verdict.v1",
            "package_id": package["package_id"],
            "receipt": decisions,
            "carrier": carrier,
            "integrated": integrated(
                decisions["exact_z2_event"], decisions["abstract_z2_state"],
                decisions["chain_consistency"], carrier,
            ),
        })

    attack_verdicts = []
    for attack in attacks:
        output_primitive = primitives[attack["output_package_id"]]
        binding = expected(output_primitive, checkpoint)
        exact_package = packages.get(attack["candidate_exact_package_id"])
        abstract_package = packages.get(attack["abstract_package_id"])
        chain_package = packages.get(attack["chain_package_id"])
        exact_receipt = exact_package["receipts"]["exact_z2"] if exact_package else None
        exact = REC_REF.exact_decision(
            key, exact_receipt, "EXACT_Z2", protocol, binding
        )
        if attack["inline_abstract_z2"] is not None:
            abstract_receipt = attack["inline_abstract_z2"]
            abstract_exact = packages[attack["output_package_id"]]["receipts"]["exact_z2"]
        elif abstract_package:
            abstract_receipt = abstract_package["receipts"]["abstract_z2"]
            abstract_exact = abstract_package["receipts"]["exact_z2"]
        else:
            abstract_receipt = abstract_exact = None
        abstract = REC_REF.abstract_decision(
            key, abstract_receipt, abstract_exact,
            "ABSTRACT_Z2", "EXACT_Z2", protocol,
        )
        if attack["inline_chain"] is not None:
            chain_bundle = json.loads(json.dumps(packages[attack["output_package_id"]]["receipts"]))
            chain_bundle["chain"] = attack["inline_chain"]
        elif chain_package:
            chain_bundle = chain_package["receipts"]
        else:
            chain_bundle = {
                name: None for name in
                ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")
            }
        chain = REC_REF.evaluate_bundle(
            key, chain_bundle, binding, protocol
        )["chain_consistency"]
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
                "ACCEPT"
                if accepted(
                    aggregates[(attack["output_package_id"], "STATE")],
                    category, config,
                )
                else "REJECT"
            )
        attack_verdicts.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "family": attack["family"],
            "exact": exact,
            "abstract": abstract,
            "chain": chain,
            "carrier": carrier,
            "integrated": integrated(exact, abstract, chain, carrier),
        })

    reconstructed = {
        "primitive_sha256": rows_sha(score_rows),
        "genuine_sha256": rows_sha(genuine),
        "attacks_sha256": rows_sha(attack_verdicts),
        "genuine_count": len(genuine),
        "attack_count": len(attack_verdicts),
        "output_count": len(outputs),
        "all_genuine_accept": all(
            row["carrier"] == "ACCEPT"
            and row["integrated"]["exact_execution_provenance"] == "ACCEPT"
            for row in genuine
        ),
    }
    gates = {
        "primitive": reconstructed["primitive_sha256"] == official["evidence"]["primitive_sha256"],
        "genuine": reconstructed["genuine_sha256"] == official["evidence"]["genuine_sha256"],
        "attacks": reconstructed["attacks_sha256"] == official["evidence"]["attacks_sha256"],
        "counts": (
            reconstructed["genuine_count"] == official["observed"]["execution_packages"]
            and reconstructed["attack_count"] == official["observed"]["attack_packages"]
            and reconstructed["output_count"] == official["observed"]["outputs"]
        ),
        "genuine_accept": reconstructed["all_genuine_accept"] is True,
        "official": official["passed"] is True,
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.reference.v1",
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
    print(json.dumps(reconstruct(args.stage, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
