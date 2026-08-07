from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


PROD = module("csw_am_scb_m1_rec_d0", ROOT / "tools/csw_am_scb_m1_rec_d0.py")
REF = module(
    "csw_am_scb_m1_rec_d0_reference",
    ROOT / "tools/csw_am_scb_m1_rec_d0_reference.py",
)


def fixture_row() -> dict:
    inputs = {"a": 1, "b": 2, "c": 0, "d": 4}
    return {
        "closure_id": "CLOSURE-A",
        "example_id": "EXAMPLE-A",
        "orientation": "LOWER_C_ORIENTATION",
        "inputs": inputs,
        "natural": {"z1": 3, "z2": 3, "z3": 3, "answer": 3},
    }


def expected(primitive: dict, config: dict) -> dict:
    return {
        **{
            field: primitive[field]
            for field in ("run_id", "context_id", "execution_id", "nonce")
        },
        "checkpoint_sha256": config["lineage"]["M1_checkpoint_sha256"],
    }


def test_literal_preflight_passes_without_protected_work() -> None:
    result = PROD.preflight()
    assert result["passed"] is True
    assert result["attack_families"] == 18
    assert result["domain_count"] == 5
    assert result["protected_releases"] == 0


def test_authentic_incorrect_computation_is_not_erased() -> None:
    config = PROD.read_json(PROD.CONFIG_PATH)
    protocol = PROD.read_json(PROD.S0_PROTOCOL_PATH)
    key = bytes(range(32))
    primitive, package = PROD.receipt_package(
        key, protocol, config, fixture_row(), "NATURAL",
        {
            "z2_consumed": 3,
            "z3_selected": 9,
            "z3_consumed": 9,
            "answer": 1,
        },
        predicted_z2=3,
        target_z2=3,
    )
    decisions = PROD.evaluate_bundle(
        key, package["receipts"], expected(primitive, config), protocol
    )
    assert decisions["exact_z2_event"] == "ACCEPT"
    assert decisions["abstract_z2_state"] == "ACCEPT"
    assert decisions["exact_z3_event"] == "ACCEPT"
    assert decisions["abstract_z3_state"] == "ACCEPT"
    assert decisions["chain_authenticator"] == "ACCEPT"
    assert decisions["registered_transition_consistency"] == "REJECT"
    assert decisions["registered_answer_consistency"] == "ACCEPT"
    assert decisions["chain_consistency"] == "REJECT"


def test_production_and_reference_agree_on_valid_bundle() -> None:
    config = PROD.read_json(PROD.CONFIG_PATH)
    protocol = PROD.read_json(PROD.S0_PROTOCOL_PATH)
    key = bytes(range(32))
    primitive, package = PROD.receipt_package(
        key, protocol, config, fixture_row(), "NATURAL",
        {
            "z2_consumed": 3,
            "z3_selected": 3,
            "z3_consumed": 3,
            "answer": 3,
        },
        predicted_z2=3,
        target_z2=3,
    )
    binding = expected(primitive, config)
    assert PROD.evaluate_bundle(
        key, package["receipts"], binding, protocol
    ) == REF.evaluate_bundle(
        key, package["receipts"], binding, protocol
    )


def test_wrong_domain_and_mutation_are_present_evidence_rejections() -> None:
    config = PROD.read_json(PROD.CONFIG_PATH)
    protocol = PROD.read_json(PROD.S0_PROTOCOL_PATH)
    key = bytes(range(32))
    primitive, package = PROD.receipt_package(
        key, protocol, config, fixture_row(), "NATURAL",
        {
            "z2_consumed": 3,
            "z3_selected": 3,
            "z3_consumed": 3,
            "answer": 3,
        },
        predicted_z2=3,
        target_z2=3,
    )
    wrong = copy.deepcopy(package["receipts"])
    wrong["exact_z2"]["tag"] = PROD.hmac.new(
        PROD.S0.derived_key(key, "EXACT_Z3"),
        PROD.S0.canonical(wrong["exact_z2"]["preimage"]),
        PROD.hashlib.sha256,
    ).hexdigest()
    decisions = PROD.evaluate_bundle(key, wrong, expected(primitive, config), protocol)
    assert decisions["exact_z2_event"] == "REJECT"
    assert decisions["chain_consistency"] == "REJECT"
    missing = copy.deepcopy(package["receipts"])
    missing["exact_z2"] = None
    decisions = PROD.evaluate_bundle(key, missing, expected(primitive, config), protocol)
    assert decisions["exact_z2_event"] == "ABSTAIN"
    assert decisions["abstract_z2_state"] == "ABSTAIN"
    assert decisions["chain_consistency"] == "ABSTAIN"


def test_wrong_ontology_rejects_even_with_valid_authenticator() -> None:
    config = PROD.read_json(PROD.CONFIG_PATH)
    protocol = PROD.read_json(PROD.S0_PROTOCOL_PATH)
    key = bytes(range(32))
    primitive, package = PROD.receipt_package(
        key, protocol, config, fixture_row(), "NATURAL",
        {
            "z2_consumed": 3,
            "z3_selected": 3,
            "z3_consumed": 3,
            "answer": 3,
        },
        predicted_z2=3,
        target_z2=3,
    )
    bundle = copy.deepcopy(package["receipts"])
    bundle["abstract_z2"] = PROD.reissue(
        key, bundle["abstract_z2"],
        {"state_ontology_id": "WRONG_ONTOLOGY"}, protocol,
    )
    decisions = PROD.evaluate_bundle(
        key, bundle, expected(primitive, config), protocol
    )
    assert decisions["abstract_z2_state"] == "REJECT"
    assert decisions["chain_consistency"] == "REJECT"


def test_attack_manifest_is_closed_and_uses_real_inputs() -> None:
    attacks = PROD.read_json(PROD.ATTACK_PATH)
    assert len(attacks["families"]) == attacks["attack_families"] == 18
    assert attacks["expected_attack_rows"] == 256 * 18
    assert attacks["summary_mutation_prohibited"] is True
    assert attacks["concrete_verifier_inputs_required"] is True


def test_d0_does_not_authorize_or_consume_protected_roles() -> None:
    config = PROD.read_json(PROD.CONFIG_PATH)
    assert sum(config["zero_protected_work"].values()) == 0
    assert config["authorizations"]["REC_sealed_confirmation"] is False
    assert config["authorizations"]["carrier"] is False
    assert config["semantic_interpretation"][
        "incorrect_computation_must_remain_authentically_receiptable"
    ] is True
