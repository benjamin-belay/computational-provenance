#!/usr/bin/env python3
"""Corrected lifecycle runner and final adjudicator for RNG2-CCAL1."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-ccal1/CONFIG.json"
SKELETON = ROOT / "protocol/csw-am-scb-m1-rng2-ccal1/V1_SKELETON.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ENGINE = module("rng2_ccal1_engine", "tools/csw_am_scb_m1_rng2_cal1.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


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


def resolve(path: Path) -> Path:
    value = path.expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def commitment(config: dict[str, Any], key: bytes) -> str:
    return hmac.new(
        config["controller"]["key_commitment_domain"].encode(),
        key, hashlib.sha256,
    ).hexdigest()


def generate(key_argument: Path, release_argument: Path, output_argument: Path) -> dict[str, Any]:
    key_path, release, output = map(
        resolve, (key_argument, release_argument, output_argument)
    )
    if output.exists():
        raise FileExistsError("RAW_OUTPUT_BOUNDARY")
    config = read_json(CONFIG)
    key_record = read_json(key_path)
    key = bytes.fromhex(key_record["sampler_key_hex"])
    public_commitment = read_json(release / "KEY_COMMITMENT.json")
    release_record = read_json(release / "RELEASE.json")
    if (
        commitment(config, key) != public_commitment["key_commitment"]
        or release_record["closures"] != 512
        or release_record["oriented_rows"] != 1024
        or release_record["membership_root"] != config["role"]["merkle_root"]
    ):
        raise ValueError("KEY_OR_RELEASE")
    rows = read_rows(release / "CALIBRATION_ROWS.jsonl")
    if len(rows) != 1024:
        raise ValueError("ROW_DENOMINATOR")
    outputs = ENGINE.generate_batch(config, rows, key)
    if len(outputs) != 16384:
        raise ValueError("OUTPUT_DENOMINATOR")
    output.mkdir(parents=True)
    raw_path = output / "RAW_OUTPUTS.jsonl"
    write_rows(raw_path, outputs)
    raw_sha = file_sha(raw_path)
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.raw_commitment.v1",
        "identity": config["identity"],
        "raw_evidence_tree": [
            {
                "path": "RAW_OUTPUTS.jsonl",
                "sha256": raw_sha,
                "rows": 16384,
            }
        ],
        "raw_tree_root": sha(
            b"EXP33A|SCB-M1|RNG2-CCAL1|RAW-TREE|V1\x00"
            + canonical([{"path": "RAW_OUTPUTS.jsonl", "sha256": raw_sha, "rows": 16384}])
        ),
        "raw_output_sha256": raw_sha,
        "outputs": 16384,
        "model_steps": 802816,
        "sufficient_sampler_reconstruction_fields": [
            "closure_id", "side", "condition", "output_index",
            "pairing_context_id", "semantic_payload", "literal_output"
        ],
        "derived_score_fields_present": False,
        "committed_before_key_reveal": True,
    }
    write_json(output / "RAW_EVIDENCE_COMMITMENT.json", record)
    return record


def reveal(key_argument: Path, release_argument: Path, raw_argument: Path,
           output_argument: Path) -> dict[str, Any]:
    key_path, release, raw, output = map(
        resolve, (key_argument, release_argument, raw_argument, output_argument)
    )
    if output.exists():
        raise FileExistsError("REVEAL_BOUNDARY")
    config = read_json(CONFIG)
    key = bytes.fromhex(read_json(key_path)["sampler_key_hex"])
    public = read_json(release / "KEY_COMMITMENT.json")
    evidence = read_json(raw / "RAW_EVIDENCE_COMMITMENT.json")
    if (
        commitment(config, key) != public["key_commitment"]
        or file_sha(raw / "RAW_OUTPUTS.jsonl") != evidence["raw_output_sha256"]
        or evidence["committed_before_key_reveal"] is not True
    ):
        raise PermissionError("RAW_COMMITMENT_REQUIRED")
    output.mkdir(parents=True)
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.key_reveal.v1",
        "identity": config["identity"],
        "sampler_key_hex": key.hex(),
        "key_commitment": public["key_commitment"],
        "raw_evidence_tree_root": evidence["raw_tree_root"],
        "raw_evidence_sha256": evidence["raw_output_sha256"],
        "raw_commitment_sha256": file_sha(raw / "RAW_EVIDENCE_COMMITMENT.json"),
        "reveal_after_raw_commitment": True,
    }
    write_json(output / "SAMPLER_KEY_REVEAL.json", record)
    return record


def derive(raw_argument: Path, reveal_argument: Path, release_argument: Path,
           output_argument: Path) -> dict[str, Any]:
    raw, reveal_path, release, output = map(
        resolve, (raw_argument, reveal_argument, release_argument, output_argument)
    )
    if not (reveal_path / "SAMPLER_KEY_REVEAL.json").exists():
        raise PermissionError("KEY_REVEAL_REQUIRED")
    return ENGINE.derive(raw, reveal_path, release, output)


def finalize(stage_argument: Path) -> dict[str, Any]:
    stage = resolve(stage_argument)
    main_path, terminal_path = stage / "MAIN_AUDIT.json", stage / "TERMINAL.json"
    if main_path.exists() or terminal_path.exists():
        raise FileExistsError("FINAL_BOUNDARY")
    config = read_json(CONFIG)
    release = read_json(stage / "release/RELEASE.json")
    raw = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    reveal_record = read_json(stage / "reveal/SAMPLER_KEY_REVEAL.json")
    production = read_json(stage / "production/OFFICIAL_RESULT.json")
    reference = read_json(stage / "reference/REFERENCE_RESULT.json")
    skeleton = read_json(SKELETON)
    complete = {
        "release": release["closures"] == 512 and release["oriented_rows"] == 1024,
        "raw": raw["outputs"] == 16384 and raw["model_steps"] == 802816,
        "commitment": (
            file_sha(stage / "raw/RAW_OUTPUTS.jsonl") == raw["raw_output_sha256"]
            and reveal_record["raw_evidence_tree_root"] == raw["raw_tree_root"]
        ),
        "chronology": (
            raw["committed_before_key_reveal"] is True
            and reveal_record["reveal_after_raw_commitment"] is True
        ),
        "production": production["passed"] is True,
        "reference": reference["passed"] is True and all(reference["gates"].values()),
        "intermediate_agreement": (
            reference["reconstructed"]["primitive_scores_sha256"]
            == production["evidence"]["primitive_scores_sha256"]
            and reference["reconstructed"]["aggregate_scores_sha256"]
            == production["evidence"]["aggregate_scores_sha256"]
            and reference["reconstructed"]["closure_statistics_sha256"]
            == production["evidence"]["closure_statistics_sha256"]
        ),
        "threshold_agreement": (
            reference["reconstructed"]["T"] == production["T"]
            and reference["reconstructed"]["M"] == production["M"]
            and reference["reconstructed"]["T_source"] == production["T_source"]
            and reference["reconstructed"]["M_source"] == production["M_source"]
        ),
        "thresholds_finite": (
            math.isfinite(production["T"])
            and math.isfinite(production["M"]) and production["M"] >= 0.0
        ),
        "denominators": (
            production["observed"]["raw_outputs"] == 16384
            and production["observed"]["primitive_authority_scores"] == 262144
            and production["observed"]["aggregate_authority_scores"] == 32768
            and production["observed"]["carrier_off_closure_statistics"] == 512
            and production["observed"]["wrong_confusion_closure_statistics"] == 512
        ),
        "future_roles_sealed": (
            release["validation_release"] == 0
            and release["remaining_reserve_release"] == 0
            and release["protected_release"] == 0
        ),
    }
    passed = all(complete.values())
    v1 = {
        **skeleton,
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.frozen_config.v1",
        "T": production["T"],
        "M": production["M"],
        "calibration_terminal": config["terminals"]["valid"] if passed else None,
        "validation_release_authorized": False,
    }
    write_json(stage / "FROZEN_V1_CONFIGURATION.json", v1)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.main_audit.v1",
        "identity": config["identity"],
        "checks": complete,
        "passed": passed,
        "main_verdict": (
            config["terminals"]["valid"] if passed
            else config["terminals"]["reference_disagreement"]
        ),
        "thresholds": {
            "T": production["T"], "M": production["M"],
            "T_source": production["T_source"],
            "M_source": production["M_source"],
        },
        "denominators": production["observed"],
        "evidence": {
            "raw_tree_root": raw["raw_tree_root"],
            "production_result_sha256": file_sha(stage / "production/OFFICIAL_RESULT.json"),
            "reference_result_sha256": file_sha(stage / "reference/REFERENCE_RESULT.json"),
            "primitive_scores_sha256": production["evidence"]["primitive_scores_sha256"],
            "aggregate_scores_sha256": production["evidence"]["aggregate_scores_sha256"],
            "closure_statistics_sha256": production["evidence"]["closure_statistics_sha256"],
            "V1_configuration_sha256": file_sha(stage / "FROZEN_V1_CONFIGURATION.json"),
        },
        "claim_boundary": {
            "thresholds_instantiated": passed,
            "matching_power_established": False,
            "pair_reliability_established": False,
            "generic_channel_validated": False,
            "receipt_bound_provenance": False,
            "output_side_provenance": False,
        },
        "authorizations": {
            "separate_RNG2_V1_request": passed,
            "RNG2_V1_release": False,
            "remaining_reserve_release": False,
            "protected_release": False,
            "second_clean_calibration": False,
            "automatic_successor": False,
        },
    }
    write_json(main_path, result)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal1.terminal.v1",
        "terminal": result["main_verdict"],
        "passed": passed,
        "T": production["T"] if passed else None,
        "M": production["M"] if passed else None,
        "main_audit_sha256": file_sha(main_path),
        "V1_configuration_sha256": file_sha(stage / "FROZEN_V1_CONFIGURATION.json"),
        "RNG2_V1_release_authorized": False,
        "separate_RNG2_V1_request_permitted": passed,
        "remaining_reserve_release": 0,
        "protected_release": 0,
        "second_clean_calibration_authorized": False,
    }
    write_json(terminal_path, terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--key", type=Path, required=True)
    gen.add_argument("--release", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    rev = sub.add_parser("reveal")
    rev.add_argument("--key", type=Path, required=True)
    rev.add_argument("--release", type=Path, required=True)
    rev.add_argument("--raw", type=Path, required=True)
    rev.add_argument("--output", type=Path, required=True)
    der = sub.add_parser("derive")
    der.add_argument("--raw", type=Path, required=True)
    der.add_argument("--reveal", type=Path, required=True)
    der.add_argument("--release", type=Path, required=True)
    der.add_argument("--output", type=Path, required=True)
    fin = sub.add_parser("finalize")
    fin.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        value = generate(args.key, args.release, args.output)
    elif args.command == "reveal":
        value = reveal(args.key, args.release, args.raw, args.output)
    elif args.command == "derive":
        value = derive(args.raw, args.reveal, args.release, args.output)
    else:
        value = finalize(args.stage)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
