#!/usr/bin/env python3
"""Independent reconstruction of RNG2-V1 from committed raw outputs."""
from __future__ import annotations

import argparse
import hashlib
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


REF = module("rng2_v1_independent_score", "tools/csw_am_scb_m1_rng2_cal1_reference.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rows_sha(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical(row) + b"\n")
    return digest.hexdigest()


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


def task(inputs: dict[str, int]) -> dict[str, int]:
    z1 = (inputs["a"] + inputs["b"]) % 16
    z2 = (z1 + inputs["c"]) % 16
    z3 = (5 * z2 + inputs["d"]) % 16
    return {"z1": z1, "z2": z2, "z3": z3, "answer": z3 % 8}


def accepted(scores: list[float], category: int, threshold_t: float, threshold_m: float) -> bool:
    largest_other = max(scores[index] for index in range(16) if index != category)
    return (
        scores[category] > threshold_t
        and scores[category] == max(scores)
        and scores.count(max(scores)) == 1
        and scores[category] - largest_other > threshold_m
    )


def reconstruct(raw_dir: Path, release_dir: Path, production_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("REFERENCE_BOUNDARY")
    config = read_json(CONFIG)
    official = read_json(production_dir / "OFFICIAL_RESULT.json")
    raw = read_rows(raw_dir / "RAW_OUTPUTS.jsonl")
    released = read_rows(release_dir / "VALIDATION_ROWS.jsonl")
    if len(raw) != 16384 or len(released) != 1024:
        raise ValueError("DENOMINATOR")
    comp: dict[str, bool] = {}
    for row in released:
        truth = task(row["inputs"])
        comp[row["closure_id"]] = comp.get(row["closure_id"], True) and (
            truth == row["natural"] and truth["answer"] == row["visible_answer"]
        )
    primitive = []
    grouped: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    expected_authority = {}
    semantic: dict[str, bool] = defaultdict(lambda: True)
    for row in raw:
        parsed, counts, normalized = REF.score(row["literal_output"])
        expected = {name: row["semantic_payload"][name] for name in REF.FIELDS}
        valid = parsed["values"] == expected and len(parsed["eligible"]) == 24
        semantic[row["closure_id"]] = semantic[row["closure_id"]] and valid
        primitive.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.primitive.v1",
            "closure_id": row["closure_id"],
            "side": row["side"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "embedded_authority": row["embedded_authority"],
            "eligible_positions": len(parsed["eligible"]),
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        grouped[(row["closure_id"], row["side"], row["condition"])].append(normalized)
        expected_authority[(row["closure_id"], row["side"])] = row["embedded_authority"]
    aggregates, by_closure = [], defaultdict(dict)
    for (closure_id, side, condition), vectors in sorted(grouped.items()):
        scores = [
            math.fsum(vector[category] for vector in vectors) / math.sqrt(8.0)
            for category in range(16)
        ]
        aggregates.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.aggregate.v1",
            "closure_id": closure_id,
            "side": side,
            "condition": condition,
            "embedded_authority": expected_authority[(closure_id, side)],
            "scores": scores,
        })
        by_closure[closure_id][(side, condition)] = scores
    threshold_t, threshold_m = config["thresholds"]["T"], config["thresholds"]["M"]
    decisions = []
    for closure_id in sorted(by_closure):
        sides, wrong_event, off_event = [], False, False
        for side in (0, 1):
            scores = by_closure[closure_id][(side, "AUTHORITY_ON")]
            target = expected_authority[(closure_id, side)]
            wrong = max(
                (index for index in range(16) if index != target),
                key=lambda index: (scores[index], index),
            )
            sides.append({
                "side": side,
                "label": "NATURAL" if side == 0 else "ALIAS",
                "authenticated_authority": target,
                "authenticated_score": scores[target],
                "strongest_wrong_authority": wrong,
                "strongest_wrong_score": scores[wrong],
                "margin": scores[target] - scores[wrong],
                "unique_rank_first": scores[target] == max(scores) and scores.count(max(scores)) == 1,
                "detector_accept": accepted(scores, target, threshold_t, threshold_m),
            })
            wrong_event = wrong_event or any(
                accepted(scores, candidate, threshold_t, threshold_m)
                for candidate in range(16) if candidate != target
            )
            off = by_closure[closure_id][(side, "CARRIER_OFF")]
            off_event = off_event or any(
                accepted(off, candidate, threshold_t, threshold_m)
                for candidate in range(16)
            )
        decisions.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_v1.closure_decision.v1",
            "closure_id": closure_id,
            "computation_fidelity": comp[closure_id],
            "semantic_validity": semantic[closure_id],
            "sides": sides,
            "carrier_off_familywise_event": off_event,
            "wrong_confusion_familywise_event": wrong_event,
            "closure_pair_success": (
                comp[closure_id] and semantic[closure_id]
                and all(row["detector_accept"] for row in sides)
            ),
        })
    observed = {
        "closures": 512,
        "raw_outputs": len(raw),
        "model_steps": len(raw) * 49,
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
    reconstructed = {
        "observed": observed,
        "gates": gates,
        "terminal": (
            config["terminals"]["valid"] if all(gates.values())
            else config["terminals"]["scientific_failed"]
        ),
        "primitive_sha256": rows_sha(primitive),
        "aggregate_sha256": rows_sha(aggregates),
        "decisions_sha256": rows_sha(decisions),
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_v1.reference.v1",
        "identity": config["identity"],
        "reconstructed": reconstructed,
        "gates": {
            "observed": reconstructed["observed"] == official["observed"],
            "scientific_gates": reconstructed["gates"] == official["gates"],
            "terminal": reconstructed["terminal"] == official["terminal"],
            "primitive": reconstructed["primitive_sha256"] == official["evidence"]["primitive_sha256"],
            "aggregate": reconstructed["aggregate_sha256"] == official["evidence"]["aggregate_sha256"],
            "decisions": reconstructed["decisions_sha256"] == official["evidence"]["decisions_sha256"],
        },
    }
    result["passed"] = all(result["gates"].values())
    output.mkdir(parents=True)
    write_json(output / "REFERENCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(
        args.raw, args.release, args.production, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
