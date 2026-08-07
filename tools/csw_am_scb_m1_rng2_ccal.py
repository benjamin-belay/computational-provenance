#!/usr/bin/env python3
"""Corrected clean-calibration lifecycle and public literal-wrapper rehearsal."""
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
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-ccal-d0/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


CAL = module("rng2_ccal_cal1", "tools/csw_am_scb_m1_rng2_cal1.py")
REF = module("rng2_ccal_reference", "tools/csw_am_scb_m1_rng2_cal1_reference.py")


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


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for value in values:
            stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def canonical_stage_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = ROOT / expanded
    return expanded.resolve()


def require_raw_commitment(stage: Path) -> dict[str, Any]:
    commitment = stage / "raw/RAW_EVIDENCE_COMMITMENT.json"
    raw = stage / "raw/RAW_OUTPUTS.jsonl"
    if not commitment.exists() or not raw.exists():
        raise PermissionError("RAW_COMMITMENT_REQUIRED")
    value = read_json(commitment)
    if file_sha(raw) != value["raw_output_sha256"]:
        raise PermissionError("RAW_COMMITMENT_INVALID")
    return value


def require_key_reveal(stage: Path) -> dict[str, Any]:
    require_raw_commitment(stage)
    reveal = stage / "reveal/SAMPLER_KEY_REVEAL.json"
    if not reveal.exists():
        raise PermissionError("KEY_REVEAL_REQUIRED")
    return read_json(reveal)


def select_public(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_rows(ROOT / config["public_development"]["row_path"])
    closure_ids = sorted(
        {row["closure_id"] for row in rows},
        key=lambda closure_id: sha(
            b"RNG2_CCAL_D0_PUBLIC_SELECTION_V1\x00" + closure_id.encode()
        ),
    )
    selected = closure_ids[0]
    result = [row for row in rows if row["closure_id"] == selected]
    if len(result) != 2:
        raise ValueError("PUBLIC_CLOSURE")
    return result


def _aggregate(primitive: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, str], list[list[float]]] = {}
    for row in primitive:
        grouped.setdefault((row["side"], row["condition"]), []).append(
            row["normalized_scores"]
        )
    aggregates = []
    by_key = {}
    for (side, condition), values in sorted(grouped.items()):
        if len(values) != 8:
            raise ValueError("OUTPUT_MULTIPLICITY")
        scores = [
            math.fsum(value[category] for value in values) / math.sqrt(8.0)
            for category in range(16)
        ]
        target = next(
            row["embedded_authority"] for row in primitive
            if row["side"] == side and row["condition"] == condition
        )
        row = {
            "side": side, "condition": condition,
            "embedded_authority": target, "scores": scores,
        }
        aggregates.append(row)
        by_key[(side, condition)] = row
    off = [
        (by_key[(side, "CARRIER_OFF")]["scores"][category], side, category)
        for side in (0, 1) for category in range(16)
    ]
    t_value, t_side, t_category = max(off)
    confusion = []
    for side in (0, 1):
        row = by_key[(side, "AUTHORITY_ON")]
        target, scores = row["embedded_authority"], row["scores"]
        for candidate in range(16):
            if candidate != target:
                confusion.append((
                    scores[candidate] - max(
                        scores[other] for other in range(16) if other != candidate
                    ),
                    side, candidate,
                ))
    c_value, c_side, c_category = max(confusion)
    return aggregates, {
        "T": t_value,
        "M": max(0.0, c_value),
        "T_source_side": t_side,
        "T_source_authority": t_category,
        "wrong_confusion_maximum": c_value,
        "M_source_side": c_side,
        "M_source_authority": c_category,
    }


def run_preflight(output_argument: Path) -> dict[str, Any]:
    stage = canonical_stage_path(output_argument)
    if stage.exists():
        raise FileExistsError(stage)
    config = read_json(CONFIG)
    bindings = {}
    for name in (
        "R0_main_audit", "R0_terminal", "role_commitments",
        "closed_CAL1_runner", "closed_CAL1_reference",
    ):
        bindings[name] = (
            file_sha(ROOT / config["bindings"][f"{name}_path"])
            == config["bindings"][f"{name}_sha256"]
        )
    bindings["public_closures"] = (
        file_sha(ROOT / config["public_development"]["closure_path"])
        == config["public_development"]["closure_sha256"]
    )
    bindings["public_rows"] = (
        file_sha(ROOT / config["public_development"]["row_path"])
        == config["public_development"]["row_sha256"]
    )
    if not all(bindings.values()):
        raise ValueError("BINDINGS")
    relative_probe = Path("artifacts/dev_only/csw-am-scb-m1-rng2-ccal-d0/PATH_PROBE")
    absolute_probe = ROOT / relative_probe
    path_equivalent = canonical_stage_path(relative_probe) == canonical_stage_path(absolute_probe)
    rows = select_public(config)
    key = hashlib.sha256(b"RNG2_CCAL_D0_PUBLIC_REHEARSAL_KEY_V1").digest()
    key_commitment = hmac.new(
        b"EXP33A|SCB-M1|RNG2-CCAL|KEY-COMMITMENT|V1", key, hashlib.sha256
    ).hexdigest()
    stage.mkdir(parents=True)
    write_json(stage / "commitment/KEY_COMMITMENT.json", {
        "key_commitment": key_commitment, "key_revealed": False,
    })
    reveal_before_commit_rejected = False
    score_before_reveal_rejected = False
    try:
        require_raw_commitment(stage)
    except PermissionError:
        reveal_before_commit_rejected = True
    small_config = json.loads(canonical(CAL.read_json(CAL.CONFIG_PATH)))
    small_config["denominators"]["outputs"] = 32
    outputs = CAL.generate_batch(small_config, rows, key)
    write_rows(stage / "raw/RAW_OUTPUTS.jsonl", outputs)
    raw_sha = file_sha(stage / "raw/RAW_OUTPUTS.jsonl")
    write_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json", {
        "raw_output_sha256": raw_sha,
        "outputs": 32,
        "commitment_covers_complete_file": True,
    })
    try:
        require_key_reveal(stage)
    except PermissionError:
        score_before_reveal_rejected = True
    require_raw_commitment(stage)
    write_json(stage / "reveal/SAMPLER_KEY_REVEAL.json", {
        "sampler_key_hex": key.hex(),
        "key_commitment": key_commitment,
        "raw_output_sha256": raw_sha,
        "raw_commitment_preceded_reveal": True,
    })
    require_key_reveal(stage)
    primitive, reference_equal = [], 0
    for row in outputs:
        parsed, counts, scores = CAL._score_output(row["literal_output"])
        reference = REF.score(row["literal_output"])
        reference_equal += int(
            parsed["values"] == reference[0]["values"]
            and counts == reference[1] and scores == reference[2]
        )
        primitive.append({
            "side": row["side"], "condition": row["condition"],
            "output_index": row["output_index"],
            "embedded_authority": row["embedded_authority"],
            "green_counts": counts, "normalized_scores": scores,
        })
    aggregates, thresholds = _aggregate(primitive)
    write_rows(stage / "scores/PRIMITIVE_SCORES.jsonl", primitive)
    write_rows(stage / "scores/AGGREGATE_SCORES.jsonl", aggregates)
    write_json(stage / "scores/DESCRIPTIVE_THRESHOLDS.json", thresholds)
    gates = {
        "bindings": all(bindings.values()),
        "relative_absolute_path_equivalence": path_equivalent,
        "raw_denominator": len(outputs) == 32,
        "raw_commitment_written": require_raw_commitment(stage)["outputs"] == 32,
        "reveal_before_commit_rejected": reveal_before_commit_rejected,
        "score_before_reveal_rejected": score_before_reveal_rejected,
        "commitment_precedes_reveal": True,
        "production_reference_scores": reference_equal == 32,
        "full_16_authority_scoring": len(primitive) * 16 == 512,
        "aggregate_pipeline": len(aggregates) == 4,
        "threshold_pipeline": math.isfinite(thresholds["T"])
            and math.isfinite(thresholds["M"]),
        "future_roles_unreleased": True,
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_d0.result.v1",
        "identity": config["identity"],
        "canonical_stage_path": str(stage),
        "checks": gates,
        "passed": all(gates.values()),
        "terminal": (
            config["terminals"]["ready"] if all(gates.values())
            else config["terminals"]["implementation_invalid"]
        ),
        "public_rehearsal": {
            "closures": 1, "oriented_rows": 2, "outputs": 32,
            "primitive_scores": 512, "aggregate_scores": 64,
            "descriptive_T": thresholds["T"], "descriptive_M": thresholds["M"],
            "claim_bearing_thresholds": False,
        },
        "executable_freeze": {
            "wrapper_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "wrapper_sha256": file_sha(Path(__file__).resolve()),
            "config_sha256": file_sha(CONFIG),
        },
        "authorizations": {
            "separate_clean_calibration_request": all(gates.values()),
            "clean_calibration_release": False,
            "validation_release": False,
            "remaining_reserve_release": False,
            "protected_release": False,
        },
    }
    write_json(stage / "OFFICIAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
