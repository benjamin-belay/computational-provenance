import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


RUN = load("tools/csw_am_scb_m1_rng2_int_v1.py", "int_v1_test")
CTRL = load("tools/csw_am_scb_m1_rng2_int_v1_controller.py", "int_v1_controller_test")


def config():
    return json.loads(
        (ROOT / "protocol/csw-am-scb-m1-rng2-int-v1/CONFIG.json").read_bytes()
    )


def test_frozen_denominators_reconcile():
    cfg = config()
    den = cfg["denominators"]
    assert den["execution_packages"] == den["closures"] * 2
    assert den["receipt_envelopes"] == den["execution_packages"] * 5
    assert den["outputs"] == (
        den["execution_packages"] * len(cfg["carrier"]["conditions"]) * 8
    )
    assert den["model_steps"] == den["outputs"] * 49
    assert den["primitive_authority_scores"] == den["outputs"] * 16
    assert den["aggregate_authority_scores"] == (
        den["execution_packages"] * len(cfg["carrier"]["conditions"]) * 16
    )
    assert den["attack_packages"] == den["closures"] * len(cfg["attacks"])


def test_frozen_scientific_boundaries():
    cfg = config()
    assert cfg["carrier"]["T"] == 4.041451884327381
    assert cfg["carrier"]["M"] == 0.0
    assert cfg["gates"]["minimum_complete_closure_successes"] == 96
    assert cfg["gates"]["maximum_carrier_off_regression_events"] == 0
    assert cfg["gates"]["maximum_wrong_confusion_events"] == 0
    assert cfg["authorizations"]["protected_release"] is False


def test_exact_and_abstract_decisions_remain_distinct():
    assert RUN.D0.integrated("REJECT", "ACCEPT", "ACCEPT", "ACCEPT") == {
        "exact_execution_provenance": "REJECT",
        "abstract_state_provenance": "ACCEPT",
        "causal_output_provenance": "ACCEPT",
    }


def test_key_commitments_are_domain_separated():
    cfg = config()
    key = bytes(range(32))
    assert CTRL.key_commitment(
        key, cfg["controller"]["receipt_commitment_domain"]
    ) != CTRL.key_commitment(
        key, cfg["controller"]["sampler_commitment_domain"]
    )
