#!/usr/bin/env python3
"""One-shot public receipt-bound RNG2 integration validation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-int-v1/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


D0 = module("int_v1_frozen_d0", "tools/csw_am_scb_m1_rng2_int_d0.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


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


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def key_commitment(key: bytes, domain: str) -> str:
    return sha(domain.encode() + b"\x00" + key)


def build_attacks(
    primitives: list[dict[str, Any]], packages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    package_by_id = {row["package_id"]: row for row in packages}
    natural = [row for row in primitives if row["role"] == "NATURAL"]
    alias_by_context = {
        row["context_id"]: row for row in primitives if row["role"] == "ALIAS_Z2"
    }
    by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in natural:
        by_state[row["consumed_z2"]].append(row)
    result = []
    for base in natural:
        alias = alias_by_context[base["context_id"]]
        same = next(
            row for row in by_state[base["consumed_z2"]]
            if row["package_id"] != base["package_id"]
        )
        other = next(
            row for row in natural if row["consumed_z2"] != base["consumed_z2"]
        )
        malformed = copy.deepcopy(package_by_id[base["package_id"]]["receipts"]["abstract_z2"])
        malformed["tag"] = "0" * len(malformed["tag"])
        bad_chain = copy.deepcopy(package_by_id[base["package_id"]]["receipts"]["chain"])
        bad_chain["tag"] = "0" * len(bad_chain["tag"])
        fixtures = [
            ("SAME_STATE_DIFFERENT_EXACT", same, same, base, None, None),
            ("NATURAL_ALIAS_CROSS", alias, alias, alias, None, None),
            ("CORRECT_ABSTRACT_WRONG_EXACT", alias, base, base, None, None),
            ("WRONG_STATE_RECEIPT", base, alias, base, None, None),
            ("REPLAY_WRONG_CONTEXT", other, other, other, None, None),
            ("MISSING_RECEIPT", None, None, None, None, None),
            ("MALFORMED_ABSTRACT_RECEIPT", base, None, base, malformed, None),
            ("INVALID_CHAIN", base, base, None, None, bad_chain),
        ]
        for family, exact, abstract, chain, inline_abstract, inline_chain in fixtures:
            result.append({
                "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.attack.v1",
                "attack_id": sha(canonical([family, base["package_id"]])),
                "closure_id": base["closure_id"],
                "family": family,
                "output_package_id": base["package_id"],
                "candidate_exact_package_id": exact["package_id"] if exact else None,
                "abstract_package_id": abstract["package_id"] if abstract else None,
                "chain_package_id": chain["package_id"] if chain else None,
                "inline_abstract_z2": inline_abstract,
                "inline_chain": inline_chain,
            })
    return result


def execute(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    release = read_json(stage / "release/RELEASE.json")
    commitments = read_json(stage / "release/KEY_COMMITMENTS.json")
    rows = read_rows(stage / "release/REGISTERED_EXECUTION_ROWS.jsonl")
    receipt_key = Path(config["controller"]["receipt_key_path"]).read_bytes()
    sampler_key = Path(config["controller"]["sampler_key_path"]).read_bytes()
    if (
        release["closures"] != config["denominators"]["closures"]
        or len(rows) != config["denominators"]["registered_execution_rows"]
        or key_commitment(receipt_key, config["controller"]["receipt_commitment_domain"])
        != commitments["receipt_key_commitment"]
        or key_commitment(sampler_key, config["controller"]["sampler_commitment_domain"])
        != commitments["sampler_key_commitment"]
    ):
        raise ValueError("RELEASE_OR_KEY")
    rec_config = D0.REC.read_json(D0.REC.CONFIG_PATH)
    protocol = D0.REC.read_json(D0.REC.S0_PROTOCOL_PATH)
    model = D0.REC.M1.load_model(D0.REC.M1.read_json(D0.REC.M1.CONFIG_PATH))
    values = D0.REC.M1.tensors(rows)
    with torch.no_grad():
        natural = model.run_natural(values)
        targets = torch.tensor([(row["natural"]["z2"] + 8) % 16 for row in rows])
        alias = model.run_with_z2(values, targets)
        natural_answers = natural["answer_logits"].argmax(-1)
        alias_answers = alias["answer_logits"].argmax(-1)
    primitives, packages = [], []
    for index, row in enumerate(rows):
        predicted = int(natural["z2_selected"][index])
        observations = (
            ("NATURAL", {
                "z2_consumed": int(natural["z2_consumed"][index]),
                "z3_selected": int(natural["z3_selected"][index]),
                "z3_consumed": int(natural["z3_consumed"][index]),
                "answer": int(natural_answers[index]),
            }, predicted),
            ("ALIAS_Z2", {
                "z2_consumed": int(alias["z2_consumed"][index]),
                "z3_selected": int(alias["z3_selected"][index]),
                "z3_consumed": int(alias["z3_consumed"][index]),
                "answer": int(alias_answers[index]),
            }, int(targets[index])),
        )
        for role, observed, target in observations:
            primitive, package = D0.REC.receipt_package(
                receipt_key, protocol, rec_config, row, role, observed, predicted, target
            )
            primitive["closure_id"] = row["closure_id"]
            primitives.append(primitive)
            packages.append(package)
    jobs, authorizations = D0.make_jobs(
        rows, primitives, packages, receipt_key, protocol, rec_config
    )
    outputs = D0.generate_jobs(jobs, sampler_key, config)
    attacks = build_attacks(primitives, packages)
    raw = stage / "raw"
    raw.mkdir(parents=True)
    write_rows(raw / "EXECUTION_ROWS.jsonl", primitives)
    write_rows(raw / "RECEIPT_PACKAGES.jsonl", packages)
    write_rows(raw / "GENERATION_AUTHORIZATIONS.jsonl", authorizations)
    write_rows(raw / "OUTPUTS.jsonl", outputs)
    write_rows(raw / "ATTACK_INPUTS.jsonl", attacks)
    names = (
        "EXECUTION_ROWS.jsonl", "RECEIPT_PACKAGES.jsonl",
        "GENERATION_AUTHORIZATIONS.jsonl", "OUTPUTS.jsonl", "ATTACK_INPUTS.jsonl",
    )
    den = config["denominators"]
    observed = {
        "execution_packages": len(primitives),
        "receipt_envelopes": len(packages) * 5,
        "generation_authorizations": len(authorizations),
        "outputs": len(outputs),
        "model_steps": sum(len(row["draws"]) for row in outputs),
        "eligible_positions": len(outputs) * 24,
        "primitive_authority_scores": len(outputs) * 16,
        "aggregate_authority_scores": len(packages) * 5 * 16,
        "attack_packages": len(attacks),
    }
    if any(observed[key] != den[key] for key in observed):
        raise ValueError("DENOMINATOR")
    manifest = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.raw_manifest.v1",
        "identity": config["identity"],
        "observed": observed,
        "files": {name: file_sha(raw / name) for name in names},
        "runtime_verification_before_authority": all(
            row["verification_preceded_authority"] for row in authorizations
        ),
        "scores_computed": False,
        "keys_revealed": False,
        "protected_access": 0,
    }
    write_json(raw / "RAW_EVIDENCE_COMMITMENT.json", manifest)
    return manifest


def reveal(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    raw = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    if raw["scores_computed"] or raw["keys_revealed"]:
        raise ValueError("CHRONOLOGY")
    receipt = Path(config["controller"]["receipt_key_path"]).read_bytes()
    sampler = Path(config["controller"]["sampler_key_path"]).read_bytes()
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.key_reveal.v1",
        "identity": config["identity"],
        "receipt_key_hex": receipt.hex(),
        "sampler_key_hex": sampler.hex(),
        "raw_commitment_sha256": file_sha(stage / "raw/RAW_EVIDENCE_COMMITMENT.json"),
        "reveal_after_raw_commitment": True,
    }
    write_json(stage / "reveal/KEY_REVEAL.json", result)
    return {key: value for key, value in result.items() if not key.endswith("_hex")}


EXPECTED_ATTACKS = {
    "SAME_STATE_DIFFERENT_EXACT": ("REJECT", "ACCEPT", "ACCEPT", "ACCEPT"),
    "NATURAL_ALIAS_CROSS": ("REJECT", "ACCEPT", "REJECT", "REJECT"),
    "CORRECT_ABSTRACT_WRONG_EXACT": ("REJECT", "ACCEPT", "ACCEPT", "ACCEPT"),
    "WRONG_STATE_RECEIPT": ("ACCEPT", "ACCEPT", "ACCEPT", "REJECT"),
    "REPLAY_WRONG_CONTEXT": ("REJECT", "ACCEPT", "REJECT", "REJECT"),
    "MISSING_RECEIPT": ("ABSTAIN", "ABSTAIN", "ABSTAIN", "INDETERMINATE"),
    "MALFORMED_ABSTRACT_RECEIPT": ("ACCEPT", "REJECT", "ACCEPT", "REJECT"),
    "INVALID_CHAIN": ("ACCEPT", "ACCEPT", "REJECT", "ACCEPT"),
}


def evaluate(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    raw = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    reveal_value = read_json(stage / "reveal/KEY_REVEAL.json")
    if reveal_value["raw_commitment_sha256"] != file_sha(
        stage / "raw/RAW_EVIDENCE_COMMITMENT.json"
    ):
        raise ValueError("CHRONOLOGY")
    for name, digest in raw["files"].items():
        if file_sha(stage / "raw" / name) != digest:
            raise ValueError("RAW_FILE")
    receipt_key = bytes.fromhex(reveal_value["receipt_key_hex"])
    protocol = D0.REC.read_json(D0.REC.S0_PROTOCOL_PATH)
    rec_config = D0.REC.read_json(D0.REC.CONFIG_PATH)
    input_rows = read_rows(stage / "release/REGISTERED_EXECUTION_ROWS.jsonl")
    primitive_rows = read_rows(stage / "raw/EXECUTION_ROWS.jsonl")
    package_rows = read_rows(stage / "raw/RECEIPT_PACKAGES.jsonl")
    outputs = read_rows(stage / "raw/OUTPUTS.jsonl")
    attacks = read_rows(stage / "raw/ATTACK_INPUTS.jsonl")
    primitives = {row["package_id"]: row for row in primitive_rows}
    packages = {row["package_id"]: row for row in package_rows}
    score_rows, aggregates = D0.score_outputs(outputs)
    aggregate_rows = [
        {
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.aggregate.v1",
            "package_id": package_id,
            "condition": condition,
            "scores": scores,
        }
        for (package_id, condition), scores in sorted(aggregates.items())
    ]
    genuine = []
    for package in package_rows:
        primitive = primitives[package["package_id"]]
        decisions = D0.REC.evaluate_bundle(
            receipt_key, package["receipts"],
            D0.expected_binding(primitive, rec_config), protocol,
        )
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        carrier = (
            "ACCEPT"
            if D0.accepted(aggregates[(package["package_id"], "STATE")], category, config)
            else "INDETERMINATE"
        )
        genuine.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.genuine.v1",
            "closure_id": primitive["closure_id"],
            "package_id": package["package_id"],
            "role": primitive["role"],
            "receipt": decisions,
            "carrier": carrier,
            "integrated": D0.integrated(
                decisions["exact_z2_event"], decisions["abstract_z2_state"],
                decisions["chain_consistency"], carrier,
            ),
        })
    attack_verdicts = []
    for attack in attacks:
        exact, abstract, chain, category = D0.receipt_layers(
            receipt_key, attack, primitives, packages, protocol, rec_config
        )
        if abstract == "ABSTAIN":
            carrier = "INDETERMINATE"
        elif abstract == "REJECT" or category is None:
            carrier = "REJECT"
        else:
            carrier = (
                "ACCEPT"
                if D0.accepted(
                    aggregates[(attack["output_package_id"], "STATE")], category, config
                )
                else "REJECT"
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
            "integrated": D0.integrated(exact, abstract, chain, carrier),
        })

    input_by_closure = {row["closure_id"]: row for row in input_rows}
    genuine_by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    primitive_by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in genuine:
        genuine_by_closure[row["closure_id"]].append(row)
    for row in primitive_rows:
        primitive_by_closure[row["closure_id"]].append(row)
    closures = []
    carrier_off_events = wrong_events = 0
    for closure_id in sorted(input_by_closure):
        row = input_by_closure[closure_id]
        expected = {
            "NATURAL": row["natural"],
            "ALIAS_Z2": {
                "z2": (row["natural"]["z2"] + 8) % 16,
                "z3": (5 * ((row["natural"]["z2"] + 8) % 16) + row["inputs"]["d"]) % 16,
            },
        }
        expected["ALIAS_Z2"]["answer"] = expected["ALIAS_Z2"]["z3"] % 8
        computation_ok = True
        for primitive in primitive_by_closure[closure_id]:
            target = expected[primitive["role"]]
            computation_ok = computation_ok and (
                primitive["consumed_z2"] == target["z2"]
                and primitive["generated_z3"] == primitive["consumed_z3"] == target["z3"]
                and primitive["answer"] == target["answer"]
            )
        sides_ok = all(
            all(value == "ACCEPT" for value in verdict["receipt"].values())
            and verdict["carrier"] == "ACCEPT"
            and verdict["integrated"]["exact_execution_provenance"] == "ACCEPT"
            and verdict["integrated"]["abstract_state_provenance"] == "ACCEPT"
            for verdict in genuine_by_closure[closure_id]
        )
        off = False
        wrong = False
        for verdict in genuine_by_closure[closure_id]:
            package_id = verdict["package_id"]
            off_scores = aggregates[(package_id, "CARRIER_OFF")]
            off = off or any(
                D0.accepted(off_scores, candidate, config) for candidate in range(16)
            )
            state_scores = aggregates[(package_id, "STATE")]
            category = primitives[package_id]["consumed_z2"]
            wrong = wrong or any(
                D0.accepted(state_scores, candidate, config)
                for candidate in range(16) if candidate != category
            )
        carrier_off_events += int(off)
        wrong_events += int(wrong)
        closures.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.closure.v1",
            "closure_id": closure_id,
            "computation_fidelity": computation_ok,
            "receipt_carrier_pair": sides_ok,
            "carrier_off_event": off,
            "wrong_confusion_event": wrong,
            "complete_success": computation_ok and sides_ok and not off and not wrong,
        })
    production = stage / "production"
    production.mkdir(parents=True)
    write_rows(production / "PRIMITIVE_SCORES.jsonl", score_rows)
    write_rows(production / "AGGREGATE_SCORES.jsonl", aggregate_rows)
    write_rows(production / "GENUINE_VERDICTS.jsonl", genuine)
    write_rows(production / "ATTACK_VERDICTS.jsonl", attack_verdicts)
    write_rows(production / "CLOSURE_DECISIONS.jsonl", closures)
    complete = sum(row["complete_success"] for row in closures)
    attack_failures = sum(
        (row["exact"], row["abstract"], row["chain"], row["carrier"])
        != EXPECTED_ATTACKS[row["family"]]
        for row in attack_verdicts
    )
    den = config["denominators"]
    gates = {
        "complete_denominator": (
            len(closures) == den["closures"]
            and len(genuine) == den["execution_packages"]
            and len(score_rows) * 16 == den["primitive_authority_scores"]
            and len(aggregate_rows) * 16 == den["aggregate_authority_scores"]
            and len(attack_verdicts) == den["attack_packages"]
        ),
        "computation_fidelity": all(row["computation_fidelity"] for row in closures),
        "complete_pair_success": complete >= config["gates"]["minimum_complete_closure_successes"],
        "attacks": attack_failures == 0,
        "carrier_off": carrier_off_events == 0,
        "wrong_confusion": wrong_events == 0,
        "runtime_order": raw["runtime_verification_before_authority"] is True,
        "audit_chronology": reveal_value["reveal_after_raw_commitment"] is True,
        "thresholds_frozen": config["carrier"]["T"] == 4.041451884327381
        and config["carrier"]["M"] == 0.0,
        "protected_sealed": raw["protected_access"] == 0,
    }
    scientific = all(gates.values())
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.official.v1",
        "identity": config["identity"],
        "gates": gates,
        "observed": {
            **raw["observed"],
            "complete_closure_successes": complete,
            "complete_closure_fraction": complete / len(closures),
            "carrier_off_regression_events": carrier_off_events,
            "wrong_confusion_events": wrong_events,
            "attack_failures": attack_failures,
        },
        "evidence": {
            "primitive_sha256": file_sha(production / "PRIMITIVE_SCORES.jsonl"),
            "aggregates_sha256": file_sha(production / "AGGREGATE_SCORES.jsonl"),
            "genuine_sha256": file_sha(production / "GENUINE_VERDICTS.jsonl"),
            "attacks_sha256": file_sha(production / "ATTACK_VERDICTS.jsonl"),
            "closures_sha256": file_sha(production / "CLOSURE_DECISIONS.jsonl"),
        },
        "passed": scientific,
        "terminal": (
            config["terminals"]["valid"] if scientific
            else config["terminals"]["scientific_failed"]
        ),
        "claim_boundary": {
            "prospective_public_receipt_bound_output_provenance": scientific,
            "protected_confirmation": False,
            "unconstrained_model_provenance": False,
        },
    }
    write_json(production / "OFFICIAL_RESULT.json", result)
    return result


def finalize(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    official = read_json(stage / "production/OFFICIAL_RESULT.json")
    reference = read_json(stage / "reference/REFERENCE_RESULT.json")
    valid = official["passed"] and reference["passed"]
    verdict = config["terminals"]["valid"] if valid else (
        config["terminals"]["scientific_failed"]
        if reference["passed"] else config["terminals"]["invalid"]
    )
    audit = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.main_audit.v1",
        "identity": config["identity"],
        "valid": valid,
        "main_verdict": verdict,
        "gates": official["gates"],
        "observed": official["observed"],
        "independent_agreement": reference["gates"],
        "claim_boundary": official["claim_boundary"],
        "authorizations": {
            "separate_protected_qualification_request": valid,
            "protected_release": False,
            "automatic_successor": False,
        },
        "evidence": {
            **official["evidence"],
            "official_sha256": file_sha(stage / "production/OFFICIAL_RESULT.json"),
            "reference_sha256": file_sha(stage / "reference/REFERENCE_RESULT.json"),
            "raw_commitment_sha256": file_sha(stage / "raw/RAW_EVIDENCE_COMMITMENT.json"),
        },
    }
    write_json(stage / "MAIN_AUDIT.json", audit)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_v1.terminal.v1",
        "terminal": verdict,
        "valid": valid,
        "main_audit_sha256": file_sha(stage / "MAIN_AUDIT.json"),
        "protected_release": 0,
    }
    write_json(stage / "TERMINAL.json", terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("execute", "reveal", "evaluate", "finalize"):
        item = sub.add_parser(command)
        item.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    value = {
        "execute": execute,
        "reveal": reveal,
        "evaluate": evaluate,
        "finalize": finalize,
    }[args.command](args.stage.resolve())
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
