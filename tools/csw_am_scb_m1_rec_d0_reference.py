#!/usr/bin/env python3
"""Independent receipt and attack reconstruction for SCB-M1 REC-D0."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    path = ROOT / "tools/csw_am_scb_m1_rec_s0_reference.py"
    spec = importlib.util.spec_from_file_location("rec_s0_reference_for_d0", path)
    if not spec or not spec.loader:
        raise RuntimeError("REFERENCE_MODULE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def exact_decision(
    key: bytes, receipt: dict[str, Any] | None, kind: str,
    protocol: dict[str, Any], expected: dict[str, Any],
) -> str:
    decision = REF.receipt_decision(key, receipt, kind, protocol)
    if decision != "ACCEPT":
        return decision
    assert receipt is not None
    return (
        "ACCEPT"
        if all(receipt["preimage"].get(field) == value for field, value in expected.items())
        else "REJECT"
    )


def abstract_decision(
    key: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
    abstract_kind: str, exact_kind: str, protocol: dict[str, Any],
) -> str:
    decision = REF.linked(key, abstract, exact, abstract_kind, exact_kind, protocol)
    if decision != "ACCEPT":
        return decision
    assert abstract is not None
    ontology = protocol["state_ontology"]
    fields = abstract["preimage"]
    return (
        "ACCEPT"
        if (
            fields["state_ontology_id"] == ontology["identity"]
            and fields["model_domain"] == ontology["model_domain"]
            and fields["checkpoint_sha256"] == ontology["checkpoint_sha256"]
        )
        else "REJECT"
    )


def compatible(
    key: bytes, candidate: dict[str, Any] | None, candidate_exact: dict[str, Any] | None,
    reference: dict[str, Any], reference_exact: dict[str, Any],
    abstract_kind: str, exact_kind: str, protocol: dict[str, Any],
) -> str:
    decisions = (
        abstract_decision(key, candidate, candidate_exact, abstract_kind, exact_kind, protocol),
        abstract_decision(key, reference, reference_exact, abstract_kind, exact_kind, protocol),
    )
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "REJECT" in decisions:
        return "REJECT"
    assert candidate is not None
    fields = protocol["state_ontology"]["abstract_compatibility_fields"]
    return (
        "ACCEPT"
        if all(candidate["preimage"][field] == reference["preimage"][field] for field in fields)
        else "REJECT"
    )


def evaluate_bundle(
    key: bytes, bundle: dict[str, dict[str, Any] | None],
    expected: dict[str, Any], protocol: dict[str, Any],
    reference: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    ez2, az2 = bundle.get("exact_z2"), bundle.get("abstract_z2")
    ez3, az3, chain = bundle.get("exact_z3"), bundle.get("abstract_z3"), bundle.get("chain")
    exact_z2 = exact_decision(key, ez2, "EXACT_Z2", protocol, expected)
    exact_z3 = exact_decision(key, ez3, "EXACT_Z3", protocol, expected)
    abstract_z2 = abstract_decision(key, az2, ez2, "ABSTRACT_Z2", "EXACT_Z2", protocol)
    abstract_z3 = abstract_decision(key, az3, ez3, "ABSTRACT_Z3", "EXACT_Z3", protocol)
    chain_expected = {
        field: expected[field]
        for field in ("run_id", "context_id", "execution_id", "nonce")
    }
    chain_auth = exact_decision(key, chain, "CHAIN", protocol, chain_expected)
    if ez2 is None or ez3 is None:
        transition = "ABSTAIN"
    elif exact_z2 == "REJECT" or exact_z3 == "REJECT":
        transition = "REJECT"
    else:
        p2, p3 = ez2["preimage"], ez3["preimage"]
        registered = (5 * p2["consumed_z2"] + p3["transition_input_d"]) % 16
        transition = (
            "ACCEPT"
            if p3["consumed_z2"] == p2["consumed_z2"]
            and p3["generated_z3"] == p3["consumed_z3"] == registered
            else "REJECT"
        )
    if chain is None or ez3 is None:
        answer = "ABSTAIN"
    elif chain_auth == "REJECT" or exact_z3 == "REJECT":
        answer = "REJECT"
    else:
        answer = (
            "ACCEPT"
            if chain["preimage"]["answer_digest"]
            == digest(["ANSWER", ez3["preimage"]["consumed_z3"] % 8])
            else "REJECT"
        )
    decisions = (exact_z2, abstract_z2, exact_z3, abstract_z3, chain_auth, transition, answer)
    if "ABSTAIN" in decisions:
        chain_consistency = "ABSTAIN"
    elif "REJECT" in decisions:
        chain_consistency = "REJECT"
    else:
        chain_consistency = REF.chain_decision(key, bundle, protocol)
    result = {
        "exact_z2_event": exact_z2,
        "abstract_z2_state": abstract_z2,
        "exact_z3_event": exact_z3,
        "abstract_z3_state": abstract_z3,
        "chain_authenticator": chain_auth,
        "registered_transition_consistency": transition,
        "registered_answer_consistency": answer,
        "chain_consistency": chain_consistency,
    }
    if reference:
        result["abstract_z2_compatibility"] = compatible(
            key, az2, ez2, reference["abstract_z2"], reference["exact_z2"],
            "ABSTRACT_Z2", "EXACT_Z2", protocol,
        )
        result["abstract_z3_compatibility"] = compatible(
            key, az3, ez3, reference["abstract_z3"], reference["exact_z3"],
            "ABSTRACT_Z3", "EXACT_Z3", protocol,
        )
    return result


def resolve(
    attack: dict[str, Any], packages: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    result = {}
    for component, package_id in attack["presented_receipt_sources"].items():
        result[component] = (
            None if package_id is None
            else packages[package_id]["receipts"][component]
        )
    result.update(attack["inline_receipts"])
    return result


def evaluate_all(
    key: bytes, primitives: list[dict[str, Any]],
    package_rows: list[dict[str, Any]], attacks: list[dict[str, Any]],
    protocol: dict[str, Any], attack_manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    packages = {row["package_id"]: row for row in package_rows}
    primitive_index = {row["package_id"]: row for row in primitives}
    execution_results = []
    for package in package_rows:
        primitive = primitive_index[package["package_id"]]
        expected = {
            field: primitive[field]
            for field in ("run_id", "context_id", "execution_id", "nonce")
        }
        expected["checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
        execution_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.execution_verdict.v1",
            "package_id": package["package_id"],
            "role": package["role"],
            "decisions": evaluate_bundle(
                key, package["receipts"], expected, protocol
            ),
        })
    attack_results = []
    matched = 0
    for attack in attacks:
        reference = (
            packages[attack["reference_package_id"]]["receipts"]
            if attack["reference_package_id"] else None
        )
        decisions = evaluate_bundle(
            key, resolve(attack, packages), attack["expected_binding"],
            protocol, reference,
        )
        expected = attack_manifest["families"][attack["family"]]["expected"]
        correct = all(decisions[field] == value for field, value in expected.items())
        matched += int(correct)
        attack_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "family": attack["family"],
            "decisions": decisions,
            "expected": expected,
            "matches_expected": correct,
        })
    result_digest = digest({
        "executions": execution_results, "attacks": attack_results
    })
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.reference_result.v1",
        "execution_rows": len(execution_results),
        "attack_rows": len(attack_results),
        "attack_expected_matches": matched,
        "result_digest": result_digest,
        "imports_production_D0_logic": False,
        "passed": (
            len(execution_results) == config["public_population"]["execution_packages"]
            and len(attack_results) == attack_manifest["expected_attack_rows"]
            and matched == len(attack_results)
        ),
    }
