from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "csw_am_scb_m1", ROOT / "tools/csw_am_scb_m1.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_symbolic_target_rules_have_registered_relations() -> None:
    for z2 in range(16):
        for d in range(16):
            natural_z3 = (5 * z2 + d) % 16
            natural_answer = natural_z3 % 8
            alias, _ = MODULE.target_for("ALIAS_Z2", z2, natural_z3)
            non_alias, _ = MODULE.target_for("NON_ALIAS_Z2", z2, natural_z3)
            wrong, _ = MODULE.target_for("WRONG_Z2", z2, natural_z3)
            same, _ = MODULE.target_for("SAME_Z2", z2, natural_z3)
            assert ((5 * alias + d) % 16) % 8 == natural_answer
            assert ((5 * non_alias + d) % 16) % 8 != natural_answer
            assert ((5 * wrong + d) % 16) % 8 != natural_answer
            assert ((5 * same + d) % 16) % 8 == natural_answer


def test_hard_category_validation_is_fail_closed() -> None:
    assert MODULE.hard_one_hot(torch.tensor([0, 15])).shape == (2, 16)
    with pytest.raises(ValueError):
        MODULE.hard_one_hot(torch.tensor([-1]))
    with pytest.raises(ValueError):
        MODULE.hard_one_hot(torch.tensor([16]))
    with pytest.raises(ValueError):
        MODULE.hard_one_hot(torch.tensor([[1]]))
    with pytest.raises(ValueError):
        MODULE.hard_one_hot(torch.tensor([1.0]))
    with pytest.raises(ValueError):
        MODULE.validate_one_hot(torch.zeros(1, 16))
    invalid = torch.zeros(1, 16)
    invalid[0, :2] = 1
    with pytest.raises(ValueError):
        MODULE.validate_one_hot(invalid)


def test_model_entry_points_exclude_unregistered_inputs() -> None:
    assert list(inspect.signature(MODULE.CausalStrictBottleneck.transition).parameters) == [
        "self", "hard_z2", "d_values"
    ]
    assert list(inspect.signature(MODULE.CausalStrictBottleneck.answer).parameters) == [
        "self", "hard_z3"
    ]
    assert list(inspect.signature(MODULE.CausalStrictBottleneck.run_with_z2).parameters) == [
        "self", "values", "target_z2"
    ]


def test_loaded_checkpoint_is_exact_frozen_model_1() -> None:
    config = MODULE.read_json(MODULE.CONFIG_PATH)
    model = MODULE.load_model(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 12648


def test_z2_override_is_exactly_consumed_and_z3_is_model_generated() -> None:
    config = MODULE.read_json(MODULE.CONFIG_PATH)
    model = MODULE.load_model(config)
    values = torch.tensor([[1, 2, 3, 4], [8, 7, 6, 5]])
    targets = torch.tensor([9, 2], dtype=torch.long)
    result = model.run_with_z2(values, targets)
    assert torch.equal(result["z2_consumed"], targets)
    assert torch.equal(result["z3_selected"], result["z3_consumed"])
    assert result["z3_logits"].shape == (2, 16)


def test_direct_z3_control_is_exactly_consumed() -> None:
    config = MODULE.read_json(MODULE.CONFIG_PATH)
    model = MODULE.load_model(config)
    target = torch.tensor([0, 7, 15], dtype=torch.long)
    result = model.run_with_z3(target)
    assert torch.equal(result["z3_consumed"], target)
    assert result["answer_logits"].shape == (3, 8)


def test_new_programme_preserves_t1_negative_and_zero_s0_work() -> None:
    config = MODULE.read_json(MODULE.CONFIG_PATH)
    closure = MODULE.read_json(MODULE.T1_CLOSURE_PATH)
    assert config["lineage"]["SCB_T1_three_seed_result"] == "VALID_HETEROGENEITY_2_OF_3"
    assert config["lineage"]["SCB_T1_result_superseded"] is False
    assert closure["registered_denominator"]["models_passing"] == 2
    assert closure["registered_denominator"]["strict_models"] == 3
    assert config["chronology"]["S0_interventions"] == 0
    assert config["chronology"]["receipts_authorized"] is False
    assert config["chronology"]["carrier_operations_authorized"] is False
