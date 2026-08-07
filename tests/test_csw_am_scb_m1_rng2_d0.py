import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D0 = load("rng2_d0_test", "tools/csw_am_scb_m1_rng2_d0.py")
REF = load("rng2_ref_test", "tools/csw_am_scb_m1_rng2_reference.py")


def base_index():
    return D0.index_for("0" * 64, "1" * 64, 0, 0)


def test_production_reference_uniform_exact_agreement():
    key = bytes(range(32))
    assert D0.position_uniform(key, base_index()) == REF.reconstruct_uniform(key, base_index())


def test_uniform_is_history_and_role_independent_by_closed_schema():
    key = bytes(range(32))
    for forbidden in ("history", "role", "z2", "receipt_id", "exact_execution_id"):
        index = base_index()
        index[forbidden] = "changed"
        with pytest.raises(ValueError):
            D0.position_uniform(key, index)
        with pytest.raises(ValueError):
            REF.reconstruct_uniform(key, index)


def test_permitted_index_fields_change_preimage():
    key = bytes(range(32))
    base = D0.position_uniform(key, base_index())
    variants = []
    for field, value in (
        ("pairing_context_id", "2" * 64),
        ("closure_id", "3" * 64),
        ("output_index", 1),
        ("model_step", 1),
    ):
        index = base_index()
        index[field] = value
        variants.append(D0.position_uniform(key, index))
    assert all(row["canonical_preimage_hex"] != base["canonical_preimage_hex"] for row in variants)
    assert all(row["uniform_bytes_hex"] != base["uniform_bytes_hex"] for row in variants)


def test_inverse_cdf_boundary_and_nonunit_weights():
    observed = D0.inverse_cdf([2.0, 2.0], 0.5)
    assert observed["selected_index"] == 0
    assert observed["normalized_probabilities"] == [0.5, 0.5]
    assert REF.inverse_cdf([2.0, 2.0], 0.5)["selected_index"] == 0


@pytest.mark.parametrize(
    "weights",
    ([0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0]),
)
def test_inverse_cdf_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        D0.inverse_cdf(weights, 0.5)
    with pytest.raises(ValueError):
        REF.inverse_cdf(weights, 0.5)


def test_development_selection_has_frozen_support():
    config = D0.read_json(D0.CONFIG_PATH)
    closures, rows = D0.select_development(config)
    assert len(closures) == 64
    assert len(rows) == 128
    assert {(row["natural"]["z2"], row["template_id"]) for row in rows} == {
        (z2, template) for z2 in range(16) for template in range(4)
    }
    assert {row["visible_answer"] for row in rows} == set(range(8))

