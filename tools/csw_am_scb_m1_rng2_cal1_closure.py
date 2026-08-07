#!/usr/bin/env python3
"""Main audit and terminal closure for RNG2-CAL1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-cal1/CONFIG.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open("rb"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    release = stage / "release"
    raw = stage / "raw"
    reveal = stage / "reveal"
    production = stage / "production"
    reference = stage / "reference"
    authorization = read_json(release / "AUTHORIZATION.json")
    key_commitment = read_json(release / "KEY_COMMITMENT.json")
    release_record = read_json(release / "RELEASE.json")
    raw_record = read_json(raw / "RAW_EVIDENCE_COMMITMENT.json")
    reveal_record = read_json(reveal / "SAMPLER_KEY_REVEAL.json")
    official = read_json(production / "OFFICIAL_RESULT.json")
    independent = read_json(reference / "REFERENCE_RESULT.json")
    v1 = read_json(production / "FROZEN_V1_CONFIGURATION.json")
    exposure = read_json(
        ROOT / config["bindings"]["exposure_status_path"]
    )
    checks = {
        "source_bindings": all(authorization["binding_checks"].values()),
        "authorized_role": (
            authorization["authorized_role"] == "RNG2_CALIBRATION"
            and authorization["authorized_closures"] == 512
            and authorization["authorized_root"]
            == "7518f095e22dafe7c7bd5fb7baaf05818e77c0c823aab4b851fb01ef79af311a"
        ),
        "one_release": (
            authorization["release_number"] == 1
            and release_record["release_number"] == 1
        ),
        "release_denominator": (
            line_count(release / "CALIBRATION_CLOSURES.jsonl") == 512
            and line_count(release / "CALIBRATION_ROWS.jsonl") == 1024
        ),
        "key_committed_before_release": (
            key_commitment["committed_before_release"] is True
            and key_commitment["key_revealed"] is False
            and release_record["key_commitment_sha256"]
            == file_sha(release / "KEY_COMMITMENT.json")
        ),
        "raw_denominator": line_count(raw / "RAW_OUTPUTS.jsonl") == 16384,
        "raw_commitment": (
            file_sha(raw / "RAW_OUTPUTS.jsonl") == raw_record["raw_output_sha256"]
            and raw_record["outputs"] == 16384
            and raw_record["model_steps"] == 802816
            and raw_record["score_fields_present"] is False
            and raw_record["threshold_fields_present"] is False
        ),
        "reveal_after_raw_commitment": (
            reveal_record["reveal_after_raw_commitment"] is True
            and reveal_record["raw_evidence_sha256"] == raw_record["raw_output_sha256"]
            and reveal_record["raw_commitment_sha256"]
            == file_sha(raw / "RAW_EVIDENCE_COMMITMENT.json")
            and reveal_record["key_commitment"] == key_commitment["key_commitment"]
        ),
        "primitive_denominator": (
            line_count(production / "PRIMITIVE_SCORES.jsonl") == 16384
            and official["observed"]["primitive_authority_scores"] == 262144
        ),
        "aggregate_denominator": (
            line_count(production / "AGGREGATE_SCORES.jsonl") == 2048
            and official["observed"]["aggregate_authority_scores"] == 32768
        ),
        "closure_statistic_denominator": (
            line_count(production / "CLOSURE_STATISTICS.jsonl") == 512
            and official["observed"]["carrier_off_closure_statistics"] == 512
            and official["observed"]["wrong_confusion_closure_statistics"] == 512
        ),
        "thresholds_finite": (
            math.isfinite(official["T"]) and math.isfinite(official["M"])
            and official["M"] >= 0.0
        ),
        "thresholds_strict_and_ties_reject": (
            v1["ties"] == "REJECT"
            and all("STRICTLY_GREATER_THAN" in rule or "UNIQUELY_RANKS_FIRST" in rule
                    for rule in v1["side_acceptance"])
        ),
        "production_valid": (
            official["passed"] is True
            and official["terminal"] == config["terminals"]["valid"]
        ),
        "reference_valid": (
            independent["passed"] is True
            and independent["terminal"] == config["terminals"]["valid"]
            and all(independent["gates"].values())
        ),
        "complete_result_agreement": (
            independent["reconstructed"]["T"] == official["T"]
            and independent["reconstructed"]["M"] == official["M"]
            and independent["reconstructed"]["T_source"] == official["T_source"]
            and independent["reconstructed"]["M_source"] == official["M_source"]
            and independent["reconstructed"]["primitive_scores_sha256"]
            == official["evidence"]["primitive_scores_sha256"]
            and independent["reconstructed"]["aggregate_scores_sha256"]
            == official["evidence"]["aggregate_scores_sha256"]
            and independent["reconstructed"]["closure_statistics_sha256"]
            == official["evidence"]["closure_statistics_sha256"]
        ),
        "V1_frozen_not_released": (
            v1["validation_release_authorized"] is False
            and official["claim_boundary"]["RNG2_V1_release_authorized"] is False
        ),
        "future_boundaries": (
            exposure["roles"]["RNG2_VALIDATION"]["exposure"] == "SEALED_UNSEEN"
            and exposure["roles"]["RNG2_PUBLIC_RESERVE"]["exposure"] == "SEALED_UNSEEN"
            and release_record["validation_release"] == 0
            and release_record["reserve_release"] == 0
            and release_record["protected_release"] == 0
        ),
        "claim_boundary": (
            official["claim_boundary"]["thresholds_instantiated"] is True
            and official["claim_boundary"]["matching_power_established"] is False
            and official["claim_boundary"]["generic_channel_validated"] is False
            and official["claim_boundary"]["output_side_provenance"] is False
        ),
    }
    passed = all(checks.values())
    terminal = (
        config["terminals"]["valid"]
        if passed else config["terminals"]["reference_disagreement"]
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.main_audit.v1",
        "identity": config["identity"],
        "checks": checks,
        "passed": passed,
        "main_verdict": terminal,
        "thresholds": {
            "T": official["T"],
            "M": official["M"],
            "T_source": official["T_source"],
            "M_source": official["M_source"],
        },
        "denominators": official["observed"],
        "chronology": official["chronology"],
        "evidence": {
            **official["evidence"],
            "official_result_sha256": file_sha(production / "OFFICIAL_RESULT.json"),
            "reference_result_sha256": file_sha(reference / "REFERENCE_RESULT.json"),
            "raw_evidence_sha256": raw_record["raw_output_sha256"],
        },
        "authorizations": {
            "separate_RNG2_V1_request": passed,
            "RNG2_V1_release": False,
            "reserve_release": False,
            "protected_release": False,
            "automatic_successor": False,
        },
        "claim_boundary": official["claim_boundary"],
    }


def close(stage: Path) -> dict[str, Any]:
    main_path = stage / "MAIN_AUDIT.json"
    terminal_path = stage / "TERMINAL.json"
    if main_path.exists() or terminal_path.exists():
        raise FileExistsError("CLOSURE_BOUNDARY")
    result = audit(stage)
    write_json(main_path, result)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.terminal.v1",
        "terminal": result["main_verdict"],
        "passed": result["passed"],
        "T": result["thresholds"]["T"],
        "M": result["thresholds"]["M"],
        "main_audit_sha256": file_sha(main_path),
        "RNG2_V1_release_authorized": False,
        "separate_RNG2_V1_request_permitted": result["passed"],
        "reserve_release": 0,
        "protected_release": 0,
    }
    write_json(terminal_path, terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(close(args.stage), sort_keys=True))


if __name__ == "__main__":
    main()
