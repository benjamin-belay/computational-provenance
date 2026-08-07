#!/usr/bin/env python3
"""Static qualification of the RNG2 generic-channel calibration protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol/csw-am-scb-m1-rng2-cal-s0"
CONFIG_PATH = PROTOCOL / "CONFIG.json"
ARTIFACT = ROOT / "artifacts/dev_only/csw-am-scb-m1-rng2-cal-s0"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REF = module(
    "rng2_cal_s0_reference",
    ROOT / "tools/csw_am_scb_m1_rng2_cal_s0_reference.py",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(path: Path, commit: str) -> bool:
    relative = str(path.relative_to(ROOT))
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    ) == path.read_bytes()


def binding_checks(config: dict[str, Any]) -> dict[str, bool]:
    bindings = config["bindings"]
    checks = {}
    for name in (
        "RNG2_R0_main_audit", "RNG2_R0_terminal", "role_migration",
        "exposure_status", "RNG2_D0_main_audit", "RNG2_D0_terminal",
    ):
        checks[name + "_sha256"] = (
            file_sha(ROOT / bindings[f"{name}_path"]) == bindings[f"{name}_sha256"]
        )
    r0 = read_json(ROOT / bindings["RNG2_R0_main_audit_path"])
    d0 = read_json(ROOT / bindings["RNG2_D0_main_audit_path"])
    r0_terminal = read_json(ROOT / bindings["RNG2_R0_terminal_path"])
    d0_terminal = read_json(ROOT / bindings["RNG2_D0_terminal_path"])
    checks["R0_ready"] = (
        r0["main_verdict"] == bindings["RNG2_R0_terminal"]
        and r0_terminal["terminal"] == bindings["RNG2_R0_terminal"]
    )
    checks["D0_ready"] = (
        d0["main_verdict"] == bindings["RNG2_D0_terminal"]
        and d0["passed"] is True
        and d0_terminal["terminal"] == bindings["RNG2_D0_terminal"]
        and d0_terminal["passed"] is True
    )
    checks["D0_did_not_release_CAL1"] = (
        d0["claim_boundary"]["RNG2_CAL1_release_authorized"] is False
        and d0_terminal["CAL1_release_authorized"] is False
    )
    return checks


def role_checks(config: dict[str, Any]) -> dict[str, bool]:
    bindings = read_json(ROOT / config["bindings"]["role_migration_path"])
    exposure = read_json(ROOT / config["bindings"]["exposure_status_path"])
    candidates = [
        row for row in bindings["bindings"]
        if row["new_role"] == "RNG2_CALIBRATION"
    ]
    role = config["calibration_role"]
    row = candidates[0] if len(candidates) == 1 else {}
    status = exposure["roles"].get("RNG2_CALIBRATION", {})
    return {
        "single_role_binding": len(candidates) == 1,
        "closure_count": row.get("closures") == role["closures"] == 512,
        "oriented_row_count": row.get("rows") == role["oriented_rows"] == 1024,
        "membership_root": (
            row.get("inherited_membership_root") == role["inherited_membership_root"]
        ),
        "binding_digest": row.get("binding_sha256") == role["binding_sha256"],
        "sealed_unseen": (
            row.get("exposure") == "SEALED_UNSEEN"
            and row.get("prior_execution") == "NONE"
            and status.get("exposure") == "SEALED_UNSEEN"
            and status.get("prior_execution") == "NONE"
        ),
        "no_release_in_S0": (
            row.get("new_release") is False
            and status.get("new_identity_release") is False
            and role["identities_materialized_in_S0"] is False
        ),
    }


def error_proof_checks(
    config: dict[str, Any], proof: dict[str, Any],
) -> dict[str, bool]:
    n = config["calibration_role"]["closures"]
    component = Fraction(1, n + 1)
    combined = component * 2
    rule = proof["strict_maximum_rule"]
    return {
        "closure_count": proof["calibration_closures"] == n == 512,
        "component_probability": (
            rule["one_component_exceedance_probability"] == float(component)
        ),
        "combined_probability": rule["union_bound_probability"] == float(combined),
        "familywise_budget": rule["registered_familywise_budget"] == 0.01,
        "safety_margin": rule["safety_margin"] == 0.01 - float(combined),
        "strict_ties_reject": (
            proof["tie_handling"]["future_acceptance_requires_strict_exceedance"]
            is True
            and proof["tie_handling"]["ties_at_T_or_M_reject"] is True
        ),
        "within_closure_dependence_absorbed": (
            proof["dependence_handling"]["no_output_or_score_is_COUNTED_AS_AN_INDEPENDENT_CALIBRATION_UNIT"]
            is True
        ),
        "combined_bound_below_budget": combined < Fraction(1, 100),
    }


def production_evaluate(
    config: dict[str, Any],
    calibration: dict[str, Any],
    denominators: dict[str, Any],
    chronology: dict[str, Any],
) -> dict[str, Any]:
    n = config["calibration_role"]["closures"]
    component = Fraction(1, n + 1)
    union = component * 2
    condition = calibration["conditions_per_closure"]
    counts = denominators["calibration"]
    calculated = {
        "oriented_rows": n * 2,
        "generic_sides": n * 2,
        "authority_on_outputs": n * 2 * 8,
        "carrier_off_outputs": n * 2 * 8,
        "total_outputs": n * 32,
        "total_model_step_operations": n * 32 * 49,
        "total_eligible_positions": n * 32 * 24,
        "per_output_authority_scores": n * 32 * 16,
        "side_condition_aggregate_vectors": n * 2 * 2,
        "aggregate_authority_scores": n * 2 * 2 * 16,
        "carrier_off_closure_null_statistics": n,
        "wrong_confusion_closure_null_statistics": n,
    }
    checks = {
        "complete_closure_unit": (
            config["familywise_control"]["calibration_unit"]
            == "COMPLETE_DEPENDENCY_CLOSURE"
        ),
        "two_side_design": (
            condition["generic_sides"] == 2
            and condition["semantic_payload"]
            == "BYTE_IDENTICAL_ACROSS_THE_TWO_SIDES"
        ),
        "eight_on_eight_off": (
            condition["authority_on_outputs_per_side"] == 8
            and condition["carrier_off_outputs_per_side"] == 8
            and condition["outputs_per_closure"] == 32
        ),
        "all_denominators_exact": all(
            counts[key] == value for key, value in calculated.items()
        ),
        "carrier_off_maximum_defined": (
            calibration["closure_level_null_statistics"]["carrier_off"]["formula"]
            == "MAX_OVER_2_SIDES_AND_16_AUTHORITIES S(C,S,OFF,K)"
        ),
        "wrong_confusion_maximum_defined": (
            calibration["closure_level_null_statistics"]["wrong_authority_confusion"][
                "values_per_closure_before_max"
            ] == 30
        ),
        "strict_thresholds": (
            calibration["threshold_derivation"]["absolute_threshold_T"] == "MAX_C N_C"
            and calibration["threshold_derivation"]["margin_threshold_M"]
            == "MAX(0, MAX_C C_C)"
            and calibration["threshold_derivation"]["acceptance_inequalities_are_strict"]
            is True
        ),
        "component_error": component == Fraction(1, 513),
        "union_error": union == Fraction(2, 513) and union < Fraction(1, 100),
        "no_independence_assumption": not any(
            config["familywise_control"][field]
            for field in (
                "within_closure_independence_assumed",
                "between_output_independence_assumed",
                "between_authority_independence_assumed",
            )
        ),
        "mechanical_support": (
            denominators["adequate_support_predicate"]["exact_closure_count"] == 512
            and denominators["adequate_support_predicate"][
                "all_registered_outputs_present"
            ] is True
            and denominators["adequate_support_predicate"][
                "all_outputs_have_exactly_24_eligible_positions"
            ] is True
        ),
        "raw_before_scores": (
            chronology["required_CAL1_chronology"].index(
                "COMMIT_COMPLETE_RAW_PRIMITIVE_EVIDENCE"
            )
            < chronology["required_CAL1_chronology"].index(
                "COMPUTE_ALL_PER_OUTPUT_NORMALIZED_EVIDENCE"
            )
        ),
        "validation_not_authorized": (
            config["authorizations"]["RNG2_V1_release"] is False
            and chronology["prohibitions"]["validation_auto_release"] is True
        ),
    }
    return {
        "checks": checks,
        "calculated_denominators": calculated,
        "error_control": {
            "component_fraction": "1/513",
            "component_probability": float(component),
            "combined_fraction": "2/513",
            "combined_probability": float(union),
            "registered_maximum": 0.01,
        },
        "passed": all(checks.values()),
    }


def run(source_commit: str) -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = read_json(CONFIG_PATH)
    calibration = read_json(PROTOCOL / "CALIBRATION_PROTOCOL.json")
    error_proof = read_json(PROTOCOL / "ERROR_CONTROL_PROOF.json")
    denominators = read_json(PROTOCOL / "DENOMINATORS_AND_SUPPORT.json")
    chronology = read_json(PROTOCOL / "CHRONOLOGY_AND_EVIDENCE.json")
    source_paths = [
        CONFIG_PATH,
        PROTOCOL / "CALIBRATION_PROTOCOL.json",
        PROTOCOL / "ERROR_CONTROL_PROOF.json",
        PROTOCOL / "DENOMINATORS_AND_SUPPORT.json",
        PROTOCOL / "CHRONOLOGY_AND_EVIDENCE.json",
        ROOT / "tools/csw_am_scb_m1_rng2_cal_s0.py",
        ROOT / "tools/csw_am_scb_m1_rng2_cal_s0_reference.py",
    ]
    source_bound = all(committed(path, source_commit) for path in source_paths)
    bindings = binding_checks(config)
    roles = role_checks(config)
    proof_checks = error_proof_checks(config, error_proof)
    production = production_evaluate(config, calibration, denominators, chronology)
    reference = REF.evaluate(config, calibration, denominators, chronology)
    agreement = (
        production["calculated_denominators"] == reference["calculated_denominators"]
        and production["error_control"] == reference["error_control"]
        and production["passed"] is True
        and reference["passed"] is True
    )
    gates = {
        "source_commit_binding": source_bound,
        "lineage_bindings": all(bindings.values()),
        "sealed_role_binding": all(roles.values()),
        "error_control_proof": all(proof_checks.values()),
        "production_protocol": production["passed"],
        "reference_protocol": reference["passed"],
        "production_reference_agreement": agreement,
        "zero_scientific_work": all(
            value == 0 for value in config["static_work"].values()
        ),
    }
    passed = all(gates.values())
    terminal = (
        config["terminals"]["ready"]
        if passed else config["terminals"]["implementation_invalid"]
    )
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal_s0.result.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "passed": passed,
        "terminal": terminal,
        "gates": gates,
        "binding_checks": bindings,
        "role_checks": roles,
        "error_proof_checks": proof_checks,
        "production": production,
        "reference": reference,
        "protocol_digests": {
            str(path.relative_to(ROOT)): file_sha(path) for path in source_paths[:5]
        },
        "claim_boundary": {
            "calibration_protocol_ready": passed,
            "T_instantiated": False,
            "M_instantiated": False,
            "generic_channel_power_confirmed": False,
            "RNG2_CAL1_identity_release_authorized": False,
            "separate_CAL1_request_permitted": passed,
            "RNG2_V1_authorized": False,
            "protected_release_authorized": False,
        },
    }
    write_json(ARTIFACT / "OFFICIAL_RESULT.json", result)
    write_json(ARTIFACT / "REFERENCE_RESULT.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal_s0.reference_result.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "passed": reference["passed"] and agreement,
        "terminal": terminal,
        "reference": reference,
        "result_core_sha256": digest({
            "calculated_denominators": reference["calculated_denominators"],
            "error_control": reference["error_control"],
            "terminal": terminal,
        }),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(canonical(run(args.source_commit)).decode())


if __name__ == "__main__":
    main()
