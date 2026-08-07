#!/usr/bin/env python3
"""Generation, reveal, and production calibration for RNG2-CAL1."""
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
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rng2-cal1/CONFIG.json"
DEFAULT_OUTPUT = ROOT / "artifacts/dev_only/csw-am-scb-m1-rng2-cal1"
SQRT6 = math.sqrt(6.0)
SQRT8 = math.sqrt(8.0)


def module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A0 = module("rng2_cal1_a0", "tools/csw_a0_stochastic_channel.py")
CAR = module("rng2_cal1_car", "tools/csw_am_scb_m1_car_s0.py")
D0 = module("rng2_cal1_d0", "tools/csw_am_scb_m1_rng2_d0.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("JSON_OBJECT:" + str(path))
    return value


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


def authority_key(category: int) -> bytes:
    if type(category) is not int or not 0 <= category < 16:
        raise ValueError("AUTHORITY_CATEGORY")
    return hmac.new(
        sha(b"RNG2_D0_GENERIC_AUTHORITY_ROOT").encode(),
        b"RNG2_D0_GENERIC_STATE_AUTHORITY_V1\x00" + category.to_bytes(1, "big"),
        hashlib.sha256,
    ).digest()


def key_commitment(config: dict[str, Any], key: bytes) -> str:
    return hmac.new(
        config["controller"]["key_commitment_domain"].encode(),
        key,
        hashlib.sha256,
    ).hexdigest()


def _jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_closure[row["closure_id"]].append(row)
    jobs = []
    for closure_id in sorted(by_closure):
        closure_rows = sorted(by_closure[closure_id], key=lambda row: row["orientation"])
        if (
            len(closure_rows) != 2
            or {row["natural"]["z2"] for row in closure_rows}.__len__() != 2
            or closure_rows[0]["visible_answer"] != closure_rows[1]["visible_answer"]
        ):
            raise ValueError("CLOSURE_SIDES")
        payload = CAR.surface_payload(
            closure_rows[0]["prompt_text"].encode(),
            closure_rows[0]["visible_answer"],
            {"surface_semantic_payload": {"schema": "SCB_M1_SURFACE_SEMANTIC_V1"}},
        )
        for side_index, row in enumerate(closure_rows):
            for condition in ("AUTHORITY_ON", "CARRIER_OFF"):
                pairing = sha(
                    b"RNG2_CAL1_PAIRING_CONTEXT_V1\x00"
                    + closure_id.encode() + b"\x00" + condition.encode()
                )
                for output_index in range(8):
                    jobs.append({
                        "closure_id": closure_id,
                        "side": side_index,
                        "orientation": row["orientation"],
                        "embedded_authority": row["natural"]["z2"],
                        "condition": condition,
                        "output_index": output_index,
                        "pairing_context_id": pairing,
                        "semantic_payload": payload,
                        "public_tokens": [],
                        "order": None,
                    })
    return jobs


def _candidate_rows(
    jobs: list[dict[str, Any]], step: int
) -> tuple[list[list[str]], bool]:
    if step == 0:
        return [list(("<ORDER_A>", "<ORDER_B>")) for _ in jobs], False
    position = (step - 1) % 12
    clause = (step - 1) // 12
    static: dict[int, tuple[str, ...]] = {
        0: A0.GROUPS[0], 2: A0.GROUPS[1], 3: A0.GROUPS[2],
        4: A0.GROUPS[3], 5: A0.GROUPS[4], 9: A0.GROUPS[5],
    }
    if position in static:
        return [list(static[position]) for _ in jobs], True
    if position == 11:
        return [list(A0.OPTIONALS) for _ in jobs], False
    fixed = {1: "the", 7: "is", 10: "."}
    if position in fixed:
        return [[fixed[position]] for _ in jobs], False
    candidates = []
    for job in jobs:
        field = job["order"][clause]
        candidates.append([
            field if position == 6 else str(job["semantic_payload"][field])
        ])
    return candidates, False


def _uniforms(
    jobs: list[dict[str, Any]], key: bytes, step: int
) -> torch.Tensor:
    values = [
        D0.position_uniform(
            key,
            D0.index_for(
                job["pairing_context_id"], job["closure_id"],
                job["output_index"], step,
            ),
        )["uniform"]
        for job in jobs
    ]
    return torch.tensor(values, dtype=torch.float64)


def generate_batch(
    config: dict[str, Any], rows: list[dict[str, Any]], sampler_key: bytes
) -> list[dict[str, Any]]:
    jobs = _jobs(rows)
    if len(jobs) != config["denominators"]["outputs"]:
        raise ValueError("JOB_DENOMINATOR")
    model = A0.load_checkpoint(ROOT / config["surface_channel"]["checkpoint_path"])[0]
    model.eval()
    prompts = torch.tensor(
        [A0.encode(A0.prompt_tokens(job["semantic_payload"])) for job in jobs],
        dtype=torch.long,
    )
    with torch.no_grad():
        states, hidden = model.gru(model.embedding(prompts))
        logits = model.output(states[:, -1, :])
        for step in range(49):
            candidates, eligible = _candidate_rows(jobs, step)
            width = len(candidates[0])
            if any(len(row) != width for row in candidates):
                raise ValueError("CANDIDATE_WIDTH")
            candidate_ids = torch.tensor(
                [[A0.TO_ID[token] for token in row] for row in candidates],
                dtype=torch.long,
            )
            selected_logits = logits.gather(1, candidate_ids).to(torch.float64)
            if eligible:
                green_masks = []
                for job, candidate_row in zip(jobs, candidates):
                    if job["condition"] == "AUTHORITY_ON":
                        green = A0.green_set(
                            authority_key(job["embedded_authority"]),
                            sum(
                                1 for token in job["public_tokens"]
                                if token in A0.ELIGIBLE_TOKEN_SET
                            ),
                            job["public_tokens"],
                            candidate_row,
                        )
                    else:
                        green = set()
                    green_masks.append([token in green for token in candidate_row])
                selected_logits += (
                    torch.tensor(green_masks, dtype=torch.float64)
                    * config["surface_channel"]["bias"]
                )
            probabilities = torch.softmax(selected_logits, dim=1)
            uniforms = _uniforms(jobs, sampler_key, step)
            cumulative = probabilities.cumsum(dim=1)
            selections = (uniforms[:, None] <= cumulative).to(torch.int64).argmax(dim=1)
            selected_ids = candidate_ids.gather(1, selections[:, None]).squeeze(1)
            selected_tokens = [
                candidates[index][int(selections[index])]
                for index in range(len(jobs))
            ]
            for job, token in zip(jobs, selected_tokens):
                if step == 0:
                    job["order"] = A0.ORDERS[0 if token == "<ORDER_A>" else 1]
                elif token != "<NONE>":
                    job["public_tokens"].append(token)
            states, hidden = model.gru(
                model.embedding(selected_ids[:, None]), hidden
            )
            logits = model.output(states[:, -1, :])
    result = []
    for job in jobs:
        literal = " ".join(job["public_tokens"])
        result.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.raw_output.v1",
            "closure_id": job["closure_id"],
            "side": job["side"],
            "orientation": job["orientation"],
            "condition": job["condition"],
            "output_index": job["output_index"],
            "embedded_authority": job["embedded_authority"],
            "pairing_context_id": job["pairing_context_id"],
            "semantic_payload": job["semantic_payload"],
            "literal_output": literal,
            "literal_output_sha256": sha(literal.encode()),
            "model_steps": 49,
        })
    return result


def generate(
    key_path: Path, release_dir: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("RAW_OUTPUT_BOUNDARY")
    config = read_json(CONFIG_PATH)
    private = read_json(key_path)
    key = bytes.fromhex(private["sampler_key_hex"])
    committed_key = read_json(release_dir / "KEY_COMMITMENT.json")
    if key_commitment(config, key) != committed_key["key_commitment"]:
        raise ValueError("KEY_COMMITMENT")
    release = read_json(release_dir / "RELEASE.json")
    rows = read_rows(release_dir / "CALIBRATION_ROWS.jsonl")
    if (
        release["closures"] != 512
        or release["oriented_rows"] != 1024
        or len(rows) != 1024
    ):
        raise ValueError("RELEASE_DENOMINATOR")
    outputs = generate_batch(config, rows, key)
    output.mkdir(parents=True)
    raw_path = output / "RAW_OUTPUTS.jsonl"
    write_rows(raw_path, outputs)
    raw_digest = file_sha(raw_path)
    commitment_tag = hmac.new(
        config["controller"]["raw_commitment_domain"].encode(),
        bytes.fromhex(raw_digest),
        hashlib.sha256,
    ).hexdigest()
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.raw_commitment.v1",
        "identity": config["identity"],
        "raw_output_path": str(raw_path.relative_to(ROOT)),
        "raw_output_sha256": raw_digest,
        "raw_commitment_tag": commitment_tag,
        "outputs": len(outputs),
        "model_steps": len(outputs) * 49,
        "score_fields_present": False,
        "threshold_fields_present": False,
        "committed_before_key_reveal": True,
    }
    write_json(output / "RAW_EVIDENCE_COMMITMENT.json", record)
    return record


def reveal(key_path: Path, release_dir: Path, raw_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("KEY_REVEAL_BOUNDARY")
    config = read_json(CONFIG_PATH)
    private = read_json(key_path)
    key = bytes.fromhex(private["sampler_key_hex"])
    commitment = read_json(release_dir / "KEY_COMMITMENT.json")
    raw = read_json(raw_dir / "RAW_EVIDENCE_COMMITMENT.json")
    if (
        key_commitment(config, key) != commitment["key_commitment"]
        or file_sha(raw_dir / "RAW_OUTPUTS.jsonl") != raw["raw_output_sha256"]
        or raw["committed_before_key_reveal"] is not True
    ):
        raise ValueError("REVEAL_CHRONOLOGY")
    output.mkdir(parents=True)
    record = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.key_reveal.v1",
        "identity": config["identity"],
        "sampler_key_hex": key.hex(),
        "key_commitment": commitment["key_commitment"],
        "raw_evidence_sha256": raw["raw_output_sha256"],
        "raw_commitment_sha256": file_sha(raw_dir / "RAW_EVIDENCE_COMMITMENT.json"),
        "reveal_after_raw_commitment": True,
    }
    write_json(output / "SAMPLER_KEY_REVEAL.json", record)
    return record


def _score_output(literal: str) -> tuple[dict[str, Any], list[int], list[float]]:
    parsed = A0.parse_output(literal)
    if not parsed["valid"] or parsed["eligible_count"] != 24:
        raise ValueError("SEMANTIC_OR_POSITION_SUPPORT_INVALID")
    counts = [0] * 16
    for item in parsed["eligible"]:
        candidates = A0.GROUPS[item["group_index"]]
        for category in range(16):
            green = A0.green_set(
                authority_key(category), item["eligible_position"],
                item["public_history_before"], candidates,
            )
            counts[category] += int(item["selected_token"] in green)
    normalized = [(count - 12.0) / SQRT6 for count in counts]
    return parsed, counts, normalized


def derive(
    raw_dir: Path, reveal_dir: Path, release_dir: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("ANALYSIS_BOUNDARY")
    config = read_json(CONFIG_PATH)
    raw_commitment = read_json(raw_dir / "RAW_EVIDENCE_COMMITMENT.json")
    reveal_record = read_json(reveal_dir / "SAMPLER_KEY_REVEAL.json")
    if (
        file_sha(raw_dir / "RAW_OUTPUTS.jsonl")
        != raw_commitment["raw_output_sha256"]
        or reveal_record["raw_evidence_sha256"] != raw_commitment["raw_output_sha256"]
    ):
        raise ValueError("RAW_REVEAL_BINDING")
    outputs = read_rows(raw_dir / "RAW_OUTPUTS.jsonl")
    if len(outputs) != config["denominators"]["outputs"]:
        raise ValueError("OUTPUT_DENOMINATOR")
    primitive_rows = []
    aggregates: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    semantic_valid = 0
    for row in outputs:
        parsed, counts, normalized = _score_output(row["literal_output"])
        expected = {
            key: row["semantic_payload"][key] for key in ("z1", "z2", "z3", "answer")
        }
        valid = parsed["values"] == expected
        semantic_valid += int(valid)
        if not valid:
            raise ValueError("SEMANTIC_OR_POSITION_SUPPORT_INVALID")
        primitive_rows.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.primitive_score.v1",
            "closure_id": row["closure_id"],
            "side": row["side"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "embedded_authority": row["embedded_authority"],
            "eligible_positions": parsed["eligible_count"],
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        aggregates[(row["closure_id"], row["side"], row["condition"])].append(
            normalized
        )
    if any(len(values) != 8 for values in aggregates.values()):
        raise ValueError("AGGREGATE_OUTPUT_DENOMINATOR")

    aggregate_rows, closure_statistics = [], []
    by_closure: dict[str, dict[tuple[int, str], list[float]]] = defaultdict(dict)
    embedded: dict[tuple[str, int], int] = {}
    for key, values in sorted(aggregates.items()):
        closure_id, side, condition = key
        scores = [
            math.fsum(row[category] for row in values) / SQRT8
            for category in range(16)
        ]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("NONFINITE_AGGREGATE")
        source = next(
            row for row in primitive_rows
            if row["closure_id"] == closure_id
            and row["side"] == side and row["condition"] == condition
        )
        embedded[(closure_id, side)] = source["embedded_authority"]
        by_closure[closure_id][(side, condition)] = scores
        aggregate_rows.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.aggregate.v1",
            "closure_id": closure_id,
            "side": side,
            "condition": condition,
            "embedded_authority": source["embedded_authority"],
            "scores": scores,
        })
    for closure_id in sorted(by_closure):
        carrier_candidates = [
            (by_closure[closure_id][(side, "CARRIER_OFF")][category], side, category)
            for side in (0, 1) for category in range(16)
        ]
        carrier_max, carrier_side, carrier_category = max(carrier_candidates)
        wrong_candidates = []
        side_details = []
        for side in (0, 1):
            scores = by_closure[closure_id][(side, "AUTHORITY_ON")]
            target = embedded[(closure_id, side)]
            wrong = [category for category in range(16) if category != target]
            for candidate in wrong:
                strongest_other = max(
                    scores[other] for other in range(16) if other != candidate
                )
                wrong_candidates.append(
                    (scores[candidate] - strongest_other, side, candidate)
                )
            max_wrong_category = max(wrong, key=lambda category: (scores[category], category))
            side_details.append({
                "side": side,
                "embedded_authority": target,
                "authenticated_score": scores[target],
                "strongest_wrong_authority": max_wrong_category,
                "strongest_wrong_score": scores[max_wrong_category],
                "authenticated_rank": 1 + sum(
                    value > scores[target] for value in scores
                ),
                "authenticated_top_two_margin":
                    scores[target] - scores[max_wrong_category],
            })
        confusion_max, confusion_side, confusion_category = max(wrong_candidates)
        closure_statistics.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.closure_statistic.v1",
            "closure_id": closure_id,
            "carrier_off_maximum": carrier_max,
            "carrier_off_source_side": carrier_side,
            "carrier_off_source_authority": carrier_category,
            "wrong_confusion_maximum": confusion_max,
            "wrong_confusion_source_side": confusion_side,
            "wrong_confusion_source_authority": confusion_category,
            "authority_on_side_details": side_details,
        })
    if len(closure_statistics) != 512:
        raise ValueError("CLOSURE_STATISTIC_DENOMINATOR")
    t_source = max(
        closure_statistics,
        key=lambda row: (row["carrier_off_maximum"], row["closure_id"]),
    )
    m_source = max(
        closure_statistics,
        key=lambda row: (row["wrong_confusion_maximum"], row["closure_id"]),
    )
    threshold_t = t_source["carrier_off_maximum"]
    threshold_m = max(0.0, m_source["wrong_confusion_maximum"])
    if not math.isfinite(threshold_t) or not math.isfinite(threshold_m):
        raise ValueError("STATISTIC_UNDEFINED")
    output.mkdir(parents=True)
    write_rows(output / "PRIMITIVE_SCORES.jsonl", primitive_rows)
    write_rows(output / "AGGREGATE_SCORES.jsonl", aggregate_rows)
    write_rows(output / "CLOSURE_STATISTICS.jsonl", closure_statistics)
    v1 = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.frozen_config.v1",
        "source_calibration": config["identity"],
        "T": threshold_t,
        "M": threshold_m,
        "score": "(GREEN_COUNT-12)/SQRT(6)",
        "aggregation": "SUM_OF_8_NORMALIZED_OUTPUT_SCORES/SQRT(8)",
        "candidate_authorities": 16,
        "side_acceptance": [
            "AUTHENTICATED_SCORE_STRICTLY_GREATER_THAN_T",
            "AUTHENTICATED_AUTHORITY_UNIQUELY_RANKS_FIRST",
            "AUTHENTICATED_SCORE_MINUS_STRONGEST_WRONG_STRICTLY_GREATER_THAN_M"
        ],
        "ties": "REJECT",
        "validation_role": "RNG2_VALIDATION",
        "validation_closures": 512,
        "validation_release_authorized": False,
    }
    write_json(output / "FROZEN_V1_CONFIGURATION.json", v1)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.official.v1",
        "identity": config["identity"],
        "terminal": config["terminals"]["valid"],
        "passed": True,
        "T": threshold_t,
        "M": threshold_m,
        "T_source": {
            key: t_source[key] for key in (
                "closure_id", "carrier_off_source_side",
                "carrier_off_source_authority", "carrier_off_maximum",
            )
        },
        "M_source": {
            key: m_source[key] for key in (
                "closure_id", "wrong_confusion_source_side",
                "wrong_confusion_source_authority", "wrong_confusion_maximum",
            )
        },
        "denominators": config["denominators"],
        "observed": {
            "raw_outputs": len(outputs),
            "semantic_valid_outputs": semantic_valid,
            "primitive_score_vectors": len(primitive_rows),
            "primitive_authority_scores": len(primitive_rows) * 16,
            "aggregate_vectors": len(aggregate_rows),
            "aggregate_authority_scores": len(aggregate_rows) * 16,
            "carrier_off_closure_statistics": len(closure_statistics),
            "wrong_confusion_closure_statistics": len(closure_statistics),
        },
        "chronology": {
            "key_commitment_sha256": file_sha(release_dir / "KEY_COMMITMENT.json"),
            "release_sha256": file_sha(release_dir / "RELEASE.json"),
            "raw_evidence_sha256": raw_commitment["raw_output_sha256"],
            "raw_commitment_sha256": file_sha(
                raw_dir / "RAW_EVIDENCE_COMMITMENT.json"
            ),
            "key_reveal_sha256": file_sha(
                reveal_dir / "SAMPLER_KEY_REVEAL.json"
            ),
            "raw_commitment_preceded_key_reveal": True,
        },
        "evidence": {
            "primitive_scores_sha256": file_sha(output / "PRIMITIVE_SCORES.jsonl"),
            "aggregate_scores_sha256": file_sha(output / "AGGREGATE_SCORES.jsonl"),
            "closure_statistics_sha256": file_sha(output / "CLOSURE_STATISTICS.jsonl"),
            "V1_configuration_sha256": file_sha(
                output / "FROZEN_V1_CONFIGURATION.json"
            ),
        },
        "claim_boundary": {
            "thresholds_instantiated": True,
            "matching_power_established": False,
            "pair_reliability_established": False,
            "generic_channel_validated": False,
            "receipt_binding": False,
            "output_side_provenance": False,
            "RNG2_V1_release_authorized": False,
            "reserve_release": 0,
            "protected_release": 0,
        },
    }
    write_json(output / "OFFICIAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--key", type=Path, required=True)
    generate_parser.add_argument("--release", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    reveal_parser = sub.add_parser("reveal")
    reveal_parser.add_argument("--key", type=Path, required=True)
    reveal_parser.add_argument("--release", type=Path, required=True)
    reveal_parser.add_argument("--raw", type=Path, required=True)
    reveal_parser.add_argument("--output", type=Path, required=True)
    derive_parser = sub.add_parser("derive")
    derive_parser.add_argument("--raw", type=Path, required=True)
    derive_parser.add_argument("--reveal", type=Path, required=True)
    derive_parser.add_argument("--release", type=Path, required=True)
    derive_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        value = generate(args.key, args.release, args.output)
    elif args.command == "reveal":
        value = reveal(args.key, args.release, args.raw, args.output)
    else:
        value = derive(args.raw, args.reveal, args.release, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
