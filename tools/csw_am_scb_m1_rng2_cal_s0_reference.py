#!/usr/bin/env python3
"""Independent static reconstruction for RNG2-CAL-S0."""
from __future__ import annotations

from fractions import Fraction
from typing import Any


def evaluate(
    config: dict[str, Any],
    calibration: dict[str, Any],
    denominators: dict[str, Any],
    chronology: dict[str, Any],
) -> dict[str, Any]:
    n = config["calibration_role"]["closures"]
    per_component = Fraction(1, n + 1)
    combined = 2 * per_component
    counts = denominators["calibration"]
    conditions = calibration["conditions_per_closure"]
    calculated = {
        "oriented_rows": 2 * n,
        "generic_sides": 2 * n,
        "authority_on_outputs": n * 2 * conditions["authority_on_outputs_per_side"],
        "carrier_off_outputs": n * 2 * conditions["carrier_off_outputs_per_side"],
        "total_outputs": n * conditions["outputs_per_closure"],
        "total_model_step_operations": n * conditions["outputs_per_closure"] * 49,
        "total_eligible_positions": n * conditions["outputs_per_closure"] * 24,
        "per_output_authority_scores": n * conditions["outputs_per_closure"] * 16,
        "side_condition_aggregate_vectors": n * 2 * 2,
        "aggregate_authority_scores": n * 2 * 2 * 16,
        "carrier_off_closure_null_statistics": n,
        "wrong_confusion_closure_null_statistics": n,
    }
    checks = {
        "calibration_unit_is_closure": (
            config["familywise_control"]["calibration_unit"]
            == "COMPLETE_DEPENDENCY_CLOSURE"
        ),
        "role_count": n == 512 and config["calibration_role"]["oriented_rows"] == 1024,
        "output_design": (
            conditions["generic_sides"] == 2
            and conditions["authority_on_outputs_per_side"] == 8
            and conditions["carrier_off_outputs_per_side"] == 8
            and conditions["outputs_per_closure"] == 32
        ),
        "denominators_reconstruct": all(counts[key] == value for key, value in calculated.items()),
        "strict_maximum_rule": (
            calibration["threshold_derivation"]["absolute_threshold_T"] == "MAX_C N_C"
            and calibration["threshold_derivation"]["margin_threshold_M"]
            == "MAX(0, MAX_C C_C)"
            and calibration["threshold_derivation"]["acceptance_inequalities_are_strict"]
            is True
        ),
        "dependence_absorbed_by_maxima": (
            calibration["finite_sample_guarantee"]["within_closure_dependence"]
            == "ABSORBED_BY_CLOSURE_LEVEL_MAXIMA"
            and config["familywise_control"]["within_closure_independence_assumed"] is False
            and config["familywise_control"]["between_output_independence_assumed"] is False
            and config["familywise_control"]["between_authority_independence_assumed"] is False
        ),
        "component_bound": per_component == Fraction(1, 513),
        "combined_bound": combined == Fraction(2, 513) and combined < Fraction(1, 100),
        "configured_bound_matches": (
            abs(config["familywise_control"]["component_tail_probability"] - float(per_component))
            < 1e-18
            and abs(
                config["familywise_control"]["combined_familywise_probability"]
                - float(combined)
            ) < 1e-18
        ),
        "support_is_mechanical": (
            denominators["adequate_support_predicate"]["exact_role_membership"] is True
            and denominators["adequate_support_predicate"]["exact_closure_count"] == 512
            and denominators["adequate_support_predicate"]["exact_oriented_row_count"]
            == 1024
            and all(
                denominators["adequate_support_predicate"][key] is True
                for key in (
                    "all_registered_outputs_present",
                    "all_outputs_semantically_valid",
                    "all_outputs_parse_valid",
                    "all_outputs_have_exactly_24_eligible_positions",
                    "all_16_candidate_scores_finite_per_output",
                    "all_512_carrier_off_null_statistics_finite",
                    "all_512_wrong_confusion_statistics_finite",
                    "T_exists_as_finite_maximum",
                    "M_exists_as_finite_nonnegative_maximum",
                )
            )
        ),
        "raw_commit_before_analysis": (
            set(chronology["raw_commit_precedes"]) >= {
                "CALIBRATION_KEY_REVEAL",
                "PRELIMINARY_AGGREGATE_SCORE_INSPECTION",
                "THRESHOLD_CALCULATION",
                "MARGIN_CALCULATION",
                "ORDERING_NULL_STATISTICS",
                "EXCLUDING_ANY_CALIBRATION_ROW",
            }
            and chronology["required_CAL1_chronology"].index(
                "COMMIT_COMPLETE_RAW_PRIMITIVE_EVIDENCE"
            )
            < chronology["required_CAL1_chronology"].index(
                "REVEAL_FRESH_CALIBRATION_SAMPLER_KEY"
            )
            < chronology["required_CAL1_chronology"].index(
                "COMPUTE_ALL_PER_OUTPUT_NORMALIZED_EVIDENCE"
            )
        ),
        "complete_reference_chain": len(chronology["independent_reconstruction"]) == 13,
        "no_auto_release": (
            config["authorizations"]["CAL1_identity_release"] is False
            and config["authorizations"]["RNG2_V1_release"] is False
            and config["authorizations"]["protected_release"] is False
        ),
    }
    return {
        "checks": checks,
        "calculated_denominators": calculated,
        "error_control": {
            "component_fraction": "1/513",
            "component_probability": float(per_component),
            "combined_fraction": "2/513",
            "combined_probability": float(combined),
            "registered_maximum": 0.01,
        },
        "passed": all(checks.values()),
    }
