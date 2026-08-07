#!/usr/bin/env python3
"""Independent closure audit for RNG2-CAL-S0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/dev_only/csw-am-scb-m1-rng2-cal-s0"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError(path.name)
    return value


def committed(commit: str, path: Path) -> bool:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path.relative_to(ROOT)}"], cwd=ROOT
    ) == path.read_bytes()


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit(source_commit: str, evidence_commit: str) -> dict[str, Any]:
    official_path = ARTIFACT / "OFFICIAL_RESULT.json"
    reference_path = ARTIFACT / "REFERENCE_RESULT.json"
    official = read_json(official_path)
    reference = read_json(reference_path)
    production = official["production"]
    reconstructed = {
        "calibration_closures": 512,
        "outputs": 512 * 2 * 2 * 8,
        "model_steps": 512 * 2 * 2 * 8 * 49,
        "eligible_positions": 512 * 2 * 2 * 8 * 24,
        "primitive_authority_scores": 512 * 2 * 2 * 8 * 16,
        "aggregate_authority_scores": 512 * 2 * 2 * 16,
        "component_bound": Fraction(1, 513),
        "combined_bound": Fraction(2, 513),
    }
    checks = {
        "source_commit": official["source_commit"] == source_commit,
        "evidence_commit": (
            committed(evidence_commit, official_path)
            and committed(evidence_commit, reference_path)
        ),
        "official_passed": official["passed"] is True,
        "reference_passed": reference["passed"] is True,
        "terminal_agreement": (
            official["terminal"] == reference["terminal"]
            == "MAIN_CSW_AM_SCB_M1_RNG2_CAL_S0_PROTOCOL_READY"
        ),
        "all_official_gates": all(official["gates"].values()),
        "lineage_bindings": all(official["binding_checks"].values()),
        "sealed_role": all(official["role_checks"].values()),
        "error_proof": all(official["error_proof_checks"].values()),
        "production_reference_denominators": (
            production["calculated_denominators"]
            == official["reference"]["calculated_denominators"]
        ),
        "production_reference_error_control": (
            production["error_control"] == official["reference"]["error_control"]
        ),
        "output_denominator": (
            production["calculated_denominators"]["total_outputs"]
            == reconstructed["outputs"]
        ),
        "model_step_denominator": (
            production["calculated_denominators"]["total_model_step_operations"]
            == reconstructed["model_steps"]
        ),
        "eligible_position_denominator": (
            production["calculated_denominators"]["total_eligible_positions"]
            == reconstructed["eligible_positions"]
        ),
        "primitive_score_denominator": (
            production["calculated_denominators"]["per_output_authority_scores"]
            == reconstructed["primitive_authority_scores"]
        ),
        "aggregate_score_denominator": (
            production["calculated_denominators"]["aggregate_authority_scores"]
            == reconstructed["aggregate_authority_scores"]
        ),
        "finite_sample_bound": (
            production["error_control"]["component_fraction"] == "1/513"
            and production["error_control"]["combined_fraction"] == "2/513"
            and reconstructed["combined_bound"] < Fraction(1, 100)
        ),
        "no_threshold_instantiation": (
            official["claim_boundary"]["T_instantiated"] is False
            and official["claim_boundary"]["M_instantiated"] is False
        ),
        "no_release": (
            official["claim_boundary"]["RNG2_CAL1_identity_release_authorized"] is False
            and official["claim_boundary"]["RNG2_V1_authorized"] is False
            and official["claim_boundary"]["protected_release_authorized"] is False
        ),
    }
    passed = all(checks.values())
    terminal = (
        "MAIN_CSW_AM_SCB_M1_RNG2_CAL_S0_PROTOCOL_READY"
        if passed else "MAIN_CSW_AM_SCB_M1_RNG2_CAL_S0_AUDIT_FAILED"
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal_s0.main_audit.v1",
        "identity": "CSW-AM-SCB-M1-RNG2-CAL-S0",
        "source_commit": source_commit,
        "evidence_commit": evidence_commit,
        "passed": passed,
        "main_verdict": terminal,
        "checks": checks,
        "reconstructed": {
            key: str(value) if isinstance(value, Fraction) else value
            for key, value in reconstructed.items()
        },
        "evidence_bindings": {
            "official_result_sha256": file_sha(official_path),
            "reference_result_sha256": file_sha(reference_path),
        },
        "claim_boundary": {
            "calibration_protocol_ready": passed,
            "T_and_M_instantiated": False,
            "CAL1_identity_release_authorized": False,
            "separate_CAL1_authorization_permitted": passed,
            "V1_authorized": False,
            "protected_release_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evidence-commit", required=True)
    args = parser.parse_args()
    result = audit(args.source_commit, args.evidence_commit)
    write_json(ARTIFACT / "MAIN_AUDIT.json", result)
    write_json(ARTIFACT / "TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal_s0.terminal.v1",
        "identity": result["identity"],
        "passed": result["passed"],
        "terminal": result["main_verdict"],
        "main_audit_sha256": file_sha(ARTIFACT / "MAIN_AUDIT.json"),
        "T_and_M_instantiated": False,
        "CAL1_identity_release_authorized": False,
        "separate_CAL1_authorization_permitted": result["passed"],
        "V1_authorized": False,
    })
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
