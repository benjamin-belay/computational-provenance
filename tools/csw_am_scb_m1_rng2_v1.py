#!/usr/bin/env python3
"""Frozen generation, scoring, and final adjudication for RNG2-V1."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-v1/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ENGINE = module("rng2_v1_engine", "tools/csw_am_scb_m1_rng2_cal1.py")
TASK = module("rng2_v1_task", "tools/csw_am_scb_m1_reference.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
    return (path if path.is_absolute() else ROOT / path).resolve()


def key_commitment(config: dict[str, Any], key: bytes) -> str:
    return hmac.new(
        config["controller"]["key_commitment_domain"].encode(), key, hashlib.sha256
    ).hexdigest()


def generate(key_arg: Path, release_arg: Path, output_arg: Path) -> dict[str, Any]:
    key_path, release, output = map(resolve, (key_arg, release_arg, output_arg))
    if output.exists():
        raise FileExistsError("RAW_OUTPUT_BOUNDARY")
    config = read_json(CONFIG)
    key = bytes.fromhex(read_json(key_path)["sampler_key_hex"])
    committed = read_json(release / "KEY_COMMITMENT.json")
    released = read_json(release / "RELEASE.json")
    if (
        key_commitment(config, key) != committed["key_commitment"]
        or released["closures"] != 512
        or released["oriented_rows"] != 1024
        or released["membership_root"] != config["role"]["membership_root"]
    ):
        raise ValueError("KEY_OR_RELEASE")
    rows = read_rows(release / "VALIDATION_ROWS.jsonl")
    if len(rows) != 1024:
        raise ValueError("ROW_DENOMINATOR")
    engine_config = {
        **config,
        "surface_channel": {
            **config["surface_channel"],
            "checkpoint_path": config["surface_channel"]["checkpoint_path"],
        },
    }
    outputs = ENGINE.generate_batch(engine_config, rows, key)
    if len(outputs) != 16384:
        raise ValueError("OUTPUT_DENOMINATOR")
    output.mkdir(parents=True)
    raw_path = output / "RAW_OUTPUTS.jsonl"
    write_rows(raw_path, outputs)
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.raw_commitment.v1",
        "identity": config["identity"],
        "raw_output_sha256": file_sha(raw_path),
        "outputs": 16384,
        "model_steps": 802816,
        "derived_score_fields_present": False,
        "committed_before_key_reveal": True,
    }
    record["raw_tree_root"] = sha(
        b"EXP33A|SCB-M1|RNG2-V1|RAW\x00" + canonical(record)
    )
    write_json(output / "RAW_EVIDENCE_COMMITMENT.json", record)
    return record


def reveal(key_arg: Path, release_arg: Path, raw_arg: Path, output_arg: Path) -> dict[str, Any]:
    key_path, release, raw, output = map(resolve, (key_arg, release_arg, raw_arg, output_arg))
    if output.exists():
        raise FileExistsError("REVEAL_BOUNDARY")
    config = read_json(CONFIG)
    key = bytes.fromhex(read_json(key_path)["sampler_key_hex"])
    public = read_json(release / "KEY_COMMITMENT.json")
    evidence = read_json(raw / "RAW_EVIDENCE_COMMITMENT.json")
    if (
        key_commitment(config, key) != public["key_commitment"]
        or file_sha(raw / "RAW_OUTPUTS.jsonl") != evidence["raw_output_sha256"]
        or evidence["committed_before_key_reveal"] is not True
    ):
        raise PermissionError("RAW_COMMITMENT_REQUIRED")
    output.mkdir(parents=True)
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.key_reveal.v1",
        "identity": config["identity"],
        "sampler_key_hex": key.hex(),
        "key_commitment": public["key_commitment"],
        "raw_evidence_sha256": evidence["raw_output_sha256"],
        "raw_tree_root": evidence["raw_tree_root"],
        "raw_commitment_sha256": file_sha(raw / "RAW_EVIDENCE_COMMITMENT.json"),
        "reveal_after_raw_commitment": True,
    }
    write_json(output / "SAMPLER_KEY_REVEAL.json", record)
    return record


def computation(rows: list[dict[str, Any]]) -> dict[str, bool]:
    result = {}
    for row in rows:
        truth = TASK.task(row["inputs"])
        result[row["closure_id"]] = result.get(row["closure_id"], True) and (
            truth == row["natural"] and truth["answer"] == row["visible_answer"]
        )
    return result


def _accepted(scores: list[float], authority: int, threshold_t: float, threshold_m: float) -> bool:
    strongest_wrong = max(scores[index] for index in range(16) if index != authority)
    unique_first = scores.count(max(scores)) == 1 and scores[authority] == max(scores)
    return (
        scores[authority] > threshold_t
        and unique_first
        and scores[authority] - strongest_wrong > threshold_m
    )


def score(raw_arg: Path, reveal_arg: Path, release_arg: Path, output_arg: Path) -> dict[str, Any]:
    raw, reveal_dir, release, output = map(
        resolve, (raw_arg, reveal_arg, release_arg, output_arg)
    )
    if output.exists():
        raise FileExistsError("SCORE_BOUNDARY")
    config = read_json(CONFIG)
    raw_commit = read_json(raw / "RAW_EVIDENCE_COMMITMENT.json")
    revealed = read_json(reveal_dir / "SAMPLER_KEY_REVEAL.json")
    if (
        file_sha(raw / "RAW_OUTPUTS.jsonl") != raw_commit["raw_output_sha256"]
        or revealed["raw_evidence_sha256"] != raw_commit["raw_output_sha256"]
    ):
        raise ValueError("RAW_REVEAL_BINDING")
    outputs = read_rows(raw / "RAW_OUTPUTS.jsonl")
    released_rows = read_rows(release / "VALIDATION_ROWS.jsonl")
    if len(outputs) != 16384 or len(released_rows) != 1024:
        raise ValueError("DENOMINATOR")
    comp = computation(released_rows)
    primitive = []
    grouped: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    semantic: dict[str, bool] = defaultdict(lambda: True)
    for row in outputs:
        parsed, counts, normalized = ENGINE._score_output(row["literal_output"])
        expected = {key: row["semantic_payload"][key] for key in ("z1", "z2", "z3", "answer")}
        valid = parsed["values"] == expected and parsed["eligible_count"] == 24
        semantic[row["closure_id"]] = semantic[row["closure_id"]] and valid
        primitive.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.primitive.v1",
            "closure_id": row["closure_id"],
            "side": row["side"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "embedded_authority": row["embedded_authority"],
            "eligible_positions": parsed["eligible_count"],
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        grouped[(row["closure_id"], row["side"], row["condition"])].append(normalized)
    aggregates, by_closure, embedded = [], defaultdict(dict), {}
    for (closure_id, side, condition), vectors in sorted(grouped.items()):
        if len(vectors) != 8:
            raise ValueError("EIGHT_OUTPUTS")
        scores = [
            math.fsum(vector[category] for vector in vectors) / math.sqrt(8.0)
            for category in range(16)
        ]
        source = next(
            row for row in primitive
            if row["closure_id"] == closure_id and row["side"] == side
            and row["condition"] == condition
        )
        embedded[(closure_id, side)] = source["embedded_authority"]
        by_closure[closure_id][(side, condition)] = scores
        aggregates.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.aggregate.v1",
            "closure_id": closure_id,
            "side": side,
            "condition": condition,
            "embedded_authority": source["embedded_authority"],
            "scores": scores,
        })
    threshold_t, threshold_m = config["thresholds"]["T"], config["thresholds"]["M"]
    decisions = []
    for closure_id in sorted(by_closure):
        side_rows = []
        wrong_event = False
        carrier_event = False
        for side in (0, 1):
            scores = by_closure[closure_id][(side, "AUTHORITY_ON")]
            target = embedded[(closure_id, side)]
            strongest_wrong = max(
                (index for index in range(16) if index != target),
                key=lambda index: (scores[index], index),
            )
            accepted = _accepted(scores, target, threshold_t, threshold_m)
            wrong_event = wrong_event or any(
                _accepted(scores, candidate, threshold_t, threshold_m)
                for candidate in range(16) if candidate != target
            )
            off_scores = by_closure[closure_id][(side, "CARRIER_OFF")]
            carrier_event = carrier_event or any(
                _accepted(off_scores, candidate, threshold_t, threshold_m)
                for candidate in range(16)
            )
            side_rows.append({
                "side": side,
                "label": "NATURAL" if side == 0 else "ALIAS",
                "authenticated_authority": target,
                "authenticated_score": scores[target],
                "strongest_wrong_authority": strongest_wrong,
                "strongest_wrong_score": scores[strongest_wrong],
                "margin": scores[target] - scores[strongest_wrong],
                "unique_rank_first": scores.count(max(scores)) == 1 and scores[target] == max(scores),
                "detector_accept": accepted,
            })
        full_pair = comp[closure_id] and semantic[closure_id] and all(
            row["detector_accept"] for row in side_rows
        )
        decisions.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.closure_decision.v1",
            "closure_id": closure_id,
            "computation_fidelity": comp[closure_id],
            "semantic_validity": semantic[closure_id],
            "sides": side_rows,
            "carrier_off_familywise_event": carrier_event,
            "wrong_confusion_familywise_event": wrong_event,
            "closure_pair_success": full_pair,
        })
    if len(decisions) != 512:
        raise ValueError("CLOSURE_DENOMINATOR")
    observed = {
        "closures": 512,
        "raw_outputs": len(outputs),
        "model_steps": len(outputs) * 49,
        "eligible_positions": sum(row["eligible_positions"] for row in primitive),
        "primitive_authority_scores": len(primitive) * 16,
        "aggregate_authority_scores": len(aggregates) * 16,
        "computation_fidelity_pass": sum(row["computation_fidelity"] for row in decisions),
        "semantic_valid_closures": sum(row["semantic_validity"] for row in decisions),
        "natural_side_success": sum(row["sides"][0]["detector_accept"] for row in decisions),
        "alias_side_success": sum(row["sides"][1]["detector_accept"] for row in decisions),
        "closure_pair_success": sum(row["closure_pair_success"] for row in decisions),
        "carrier_off_familywise_events": sum(row["carrier_off_familywise_event"] for row in decisions),
        "wrong_confusion_familywise_events": sum(row["wrong_confusion_familywise_event"] for row in decisions),
    }
    gates = {
        "complete_denominator": (
            observed["raw_outputs"] == 16384
            and observed["eligible_positions"] == 393216
            and observed["primitive_authority_scores"] == 262144
            and observed["aggregate_authority_scores"] == 32768
        ),
        "computation_fidelity": observed["computation_fidelity_pass"] == 512,
        "semantic_validity": observed["semantic_valid_closures"] == 512,
        "natural_side_power": observed["natural_side_success"] / 512 >= 0.8,
        "alias_side_power": observed["alias_side_success"] / 512 >= 0.8,
        "closure_pair_reliability": observed["closure_pair_success"] / 512 >= 0.75,
        "carrier_off_familywise": observed["carrier_off_familywise_events"] <= 1,
        "wrong_confusion_familywise": observed["wrong_confusion_familywise_events"] <= 1,
    }
    output.mkdir(parents=True)
    write_rows(output / "PRIMITIVE_SCORES.jsonl", primitive)
    write_rows(output / "AGGREGATE_SCORES.jsonl", aggregates)
    write_rows(output / "CLOSURE_DECISIONS.jsonl", decisions)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.official.v1",
        "identity": config["identity"],
        "thresholds": config["thresholds"],
        "observed": observed,
        "gates": gates,
        "scientific_pass": all(gates.values()),
        "terminal": (
            config["terminals"]["valid"] if all(gates.values())
            else config["terminals"]["scientific_failed"]
        ),
        "evidence": {
            "primitive_sha256": file_sha(output / "PRIMITIVE_SCORES.jsonl"),
            "aggregate_sha256": file_sha(output / "AGGREGATE_SCORES.jsonl"),
            "decisions_sha256": file_sha(output / "CLOSURE_DECISIONS.jsonl"),
        },
        "chronology": {
            "raw_commitment_preceded_key_reveal": True,
            "raw_commitment_sha256": file_sha(raw / "RAW_EVIDENCE_COMMITMENT.json"),
            "key_reveal_sha256": file_sha(reveal_dir / "SAMPLER_KEY_REVEAL.json"),
        },
        "claim_boundary": {
            "generic_channel_validated": all(gates.values()),
            "receipt_binding": False,
            "output_side_provenance": False,
            "protected_evidence": False,
        },
    }
    write_json(output / "OFFICIAL_RESULT.json", result)
    return result


def finalize(stage_arg: Path) -> dict[str, Any]:
    stage = resolve(stage_arg)
    config = read_json(CONFIG)
    production = read_json(stage / "production/OFFICIAL_RESULT.json")
    reference = read_json(stage / "reference/REFERENCE_RESULT.json")
    release = read_json(stage / "release/RELEASE.json")
    raw = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    reveal_record = read_json(stage / "reveal/SAMPLER_KEY_REVEAL.json")
    checks = {
        "release": release["closures"] == 512 and release["oriented_rows"] == 1024,
        "chronology": (
            raw["committed_before_key_reveal"] is True
            and reveal_record["reveal_after_raw_commitment"] is True
            and reveal_record["raw_tree_root"] == raw["raw_tree_root"]
        ),
        "production_complete": production["gates"]["complete_denominator"],
        "reference_complete": reference["passed"] is True,
        "full_agreement": reference["reconstructed"] == {
            "observed": production["observed"],
            "gates": production["gates"],
            "terminal": production["terminal"],
            "primitive_sha256": production["evidence"]["primitive_sha256"],
            "aggregate_sha256": production["evidence"]["aggregate_sha256"],
            "decisions_sha256": production["evidence"]["decisions_sha256"],
        },
        "future_boundaries": (
            release["reserve_release"] == 0
            and release["receipt_integration_release"] == 0
            and release["protected_release"] == 0
        ),
    }
    valid = all(checks.values())
    verdict = (
        production["terminal"] if valid
        else config["terminals"]["implementation_invalid"]
    )
    audit = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.main_audit.v1",
        "identity": config["identity"],
        "checks": checks,
        "valid": valid,
        "scientific_pass": valid and production["scientific_pass"],
        "main_verdict": verdict,
        "thresholds": production["thresholds"],
        "observed": production["observed"],
        "gates": production["gates"],
        "evidence": {
            **production["evidence"],
            "raw_tree_root": raw["raw_tree_root"],
            "production_result_sha256": file_sha(stage / "production/OFFICIAL_RESULT.json"),
            "reference_result_sha256": file_sha(stage / "reference/REFERENCE_RESULT.json"),
        },
        "claim_boundary": production["claim_boundary"],
        "authorizations": {
            "separate_receipt_integration_request": valid and production["scientific_pass"],
            "receipt_integration_release": False,
            "reserve_release": False,
            "protected_release": False,
            "V1_rerun": False,
            "automatic_successor": False,
        },
    }
    write_json(stage / "MAIN_AUDIT.json", audit)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.terminal.v1",
        "terminal": verdict,
        "valid": valid,
        "scientific_pass": audit["scientific_pass"],
        "main_audit_sha256": file_sha(stage / "MAIN_AUDIT.json"),
        "receipt_integration_release": 0,
        "protected_release": 0,
        "rerun_authorized": False,
    }
    write_json(stage / "TERMINAL.json", terminal)
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
    scr = sub.add_parser("score")
    scr.add_argument("--raw", type=Path, required=True)
    scr.add_argument("--reveal", type=Path, required=True)
    scr.add_argument("--release", type=Path, required=True)
    scr.add_argument("--output", type=Path, required=True)
    fin = sub.add_parser("finalize")
    fin.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        value = generate(args.key, args.release, args.output)
    elif args.command == "reveal":
        value = reveal(args.key, args.release, args.raw, args.output)
    elif args.command == "score":
        value = score(args.raw, args.reveal, args.release, args.output)
    else:
        value = finalize(args.stage)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
