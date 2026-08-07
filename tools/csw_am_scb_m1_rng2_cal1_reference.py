#!/usr/bin/env python3
"""Independent reconstruction of RNG2-CAL1 from raw outputs."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-cal1/CONFIG.json"
SQRT6 = math.sqrt(6.0)
SQRT8 = math.sqrt(8.0)
GROUPS = (
    ("Here", "Now", "Presently", "Directly", "Briefly", "Simply", "Accordingly", "Formally"),
    ("record", "calculation", "derivation", "account", "summary", "analysis", "result", "trace"),
    ("plainly", "explicitly", "carefully", "concisely", "firmly", "notably", "precisely", "transparently"),
    ("shows", "states", "gives", "reports", "records", "yields", "lists", "presents"),
    ("that", "how", "namely", "specifically", "directly-as", "in-form", "as-value", "with-value"),
    ("indeed", "therefore", "accordingly", "thus", "thereby", "consistently", "exactly", "formally-so"),
)
FIELDS = ("z1", "z2", "z3", "answer")
ORDERS = (("z1", "z2", "z3", "answer"), ("z1", "z3", "z2", "answer"))
OPTIONALS = {
    "Additionally-noted", "Separately-recorded", "Explicitly-retained",
    "Carefully-preserved", "Directly-confirmed", "Formally-listed",
    "Briefly-restated", "Transparently-given",
}


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows_sha(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical(row) + b"\n")
    return digest.hexdigest()


def authority(category: int) -> bytes:
    root = sha(b"RNG2_D0_GENERIC_AUTHORITY_ROOT").encode()
    return hmac.new(
        root,
        b"RNG2_D0_GENERIC_STATE_AUTHORITY_V1\x00" + bytes([category]),
        hashlib.sha256,
    ).digest()


def parse(literal: str) -> dict[str, Any]:
    tokens = literal.split()
    cursor = 0
    history: list[str] = []
    eligible = []
    values: dict[str, int] = {}
    order = []
    try:
        for _ in range(4):
            for group_index, group in enumerate(GROUPS):
                token = tokens[cursor]
                if token not in group:
                    raise ValueError("GROUP")
                eligible.append({
                    "eligible_position": len(eligible),
                    "group_index": group_index,
                    "selected_token": token,
                    "public_history_before": list(history),
                })
                history.append(token)
                cursor += 1
                if group_index == 0:
                    if tokens[cursor] != "the":
                        raise ValueError("THE")
                    history.append("the")
                    cursor += 1
                if group_index == 4:
                    field = tokens[cursor]
                    if field not in FIELDS or field in values:
                        raise ValueError("FIELD")
                    order.append(field)
                    history.append(field)
                    cursor += 1
                    if tokens[cursor] != "is":
                        raise ValueError("IS")
                    history.append("is")
                    cursor += 1
                    value = tokens[cursor]
                    if value not in tuple(str(index) for index in range(8)):
                        raise ValueError("VALUE")
                    values[field] = int(value)
                    history.append(value)
                    cursor += 1
            if tokens[cursor] != ".":
                raise ValueError("PERIOD")
            history.append(".")
            cursor += 1
            if cursor < len(tokens) and tokens[cursor] in OPTIONALS:
                history.append(tokens[cursor])
                cursor += 1
        valid = (
            cursor == len(tokens)
            and tuple(order) in ORDERS
            and set(values) == set(FIELDS)
            and values["answer"] == values["z3"]
            and len(eligible) == 24
        )
        return {
            "valid": valid,
            "values": {field: values[field] for field in FIELDS},
            "eligible": eligible,
        }
    except (IndexError, ValueError, KeyError):
        return {"valid": False, "eligible": eligible}


def green(key: bytes, position: int, history: list[str], candidates: tuple[str, ...]) -> set[str]:
    history_digest = hashlib.sha256(canonical(history)).digest()
    ranked = sorted(
        candidates,
        key=lambda token: (
            hmac.new(
                key,
                b"CSW_A0_SYNTHETIC_AUTHORITY_GREENLIST_V1\x00"
                + position.to_bytes(4, "big")
                + history_digest
                + token.encode(),
                hashlib.sha256,
            ).digest(),
            token,
        ),
    )
    return set(ranked[:4])


def score(literal: str) -> tuple[dict[str, Any], list[int], list[float]]:
    parsed = parse(literal)
    if not parsed["valid"]:
        raise ValueError("PARSE")
    counts = [0] * 16
    keys = [authority(category) for category in range(16)]
    for item in parsed["eligible"]:
        candidates = GROUPS[item["group_index"]]
        for category, key in enumerate(keys):
            counts[category] += int(
                item["selected_token"] in green(
                    key, item["eligible_position"],
                    item["public_history_before"], candidates,
                )
            )
    return parsed, counts, [(count - 12.0) / SQRT6 for count in counts]


def reconstruct(raw_dir: Path, production_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config = read_json(CONFIG)
    official = read_json(production_dir / "OFFICIAL_RESULT.json")
    raw = read_rows(raw_dir / "RAW_OUTPUTS.jsonl")
    if len(raw) != 16384:
        raise ValueError("RAW_DENOMINATOR")
    primitive = []
    grouped: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    expected_authority: dict[tuple[str, int], int] = {}
    for row in raw:
        parsed, counts, normalized = score(row["literal_output"])
        expected = {
            name: row["semantic_payload"][name] for name in FIELDS
        }
        if parsed["values"] != expected:
            raise ValueError("SEMANTIC")
        primitive.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.primitive_score.v1",
            "closure_id": row["closure_id"],
            "side": row["side"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "embedded_authority": row["embedded_authority"],
            "eligible_positions": 24,
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        grouped[(row["closure_id"], row["side"], row["condition"])].append(normalized)
        expected_authority[(row["closure_id"], row["side"])] = row["embedded_authority"]
    aggregates = []
    by_closure: dict[str, dict[tuple[int, str], list[float]]] = defaultdict(dict)
    for (closure_id, side, condition), values in sorted(grouped.items()):
        scores = [
            math.fsum(vector[category] for vector in values) / SQRT8
            for category in range(16)
        ]
        target = expected_authority[(closure_id, side)]
        aggregates.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.aggregate.v1",
            "closure_id": closure_id,
            "side": side,
            "condition": condition,
            "embedded_authority": target,
            "scores": scores,
        })
        by_closure[closure_id][(side, condition)] = scores
    statistics = []
    for closure_id in sorted(by_closure):
        off = [
            (by_closure[closure_id][(side, "CARRIER_OFF")][category], side, category)
            for side in (0, 1) for category in range(16)
        ]
        off_max, off_side, off_category = max(off)
        confusion = []
        details = []
        for side in (0, 1):
            scores = by_closure[closure_id][(side, "AUTHORITY_ON")]
            target = expected_authority[(closure_id, side)]
            wrong = [category for category in range(16) if category != target]
            for candidate in wrong:
                confusion.append((
                    scores[candidate] - max(
                        scores[other] for other in range(16) if other != candidate
                    ),
                    side,
                    candidate,
                ))
            strongest = max(wrong, key=lambda category: (scores[category], category))
            details.append({
                "side": side,
                "embedded_authority": target,
                "authenticated_score": scores[target],
                "strongest_wrong_authority": strongest,
                "strongest_wrong_score": scores[strongest],
                "authenticated_rank": 1 + sum(value > scores[target] for value in scores),
                "authenticated_top_two_margin": scores[target] - scores[strongest],
            })
        confusion_max, confusion_side, confusion_category = max(confusion)
        statistics.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_cal1.closure_statistic.v1",
            "closure_id": closure_id,
            "carrier_off_maximum": off_max,
            "carrier_off_source_side": off_side,
            "carrier_off_source_authority": off_category,
            "wrong_confusion_maximum": confusion_max,
            "wrong_confusion_source_side": confusion_side,
            "wrong_confusion_source_authority": confusion_category,
            "authority_on_side_details": details,
        })
    t_row = max(statistics, key=lambda row: (row["carrier_off_maximum"], row["closure_id"]))
    m_row = max(statistics, key=lambda row: (row["wrong_confusion_maximum"], row["closure_id"]))
    t_value = t_row["carrier_off_maximum"]
    m_value = max(0.0, m_row["wrong_confusion_maximum"])
    reconstructed = {
        "primitive_scores_sha256": rows_sha(primitive),
        "aggregate_scores_sha256": rows_sha(aggregates),
        "closure_statistics_sha256": rows_sha(statistics),
        "T": t_value,
        "M": m_value,
        "T_source": {
            key: t_row[key] for key in (
                "closure_id", "carrier_off_source_side",
                "carrier_off_source_authority", "carrier_off_maximum",
            )
        },
        "M_source": {
            key: m_row[key] for key in (
                "closure_id", "wrong_confusion_source_side",
                "wrong_confusion_source_authority", "wrong_confusion_maximum",
            )
        },
    }
    gates = {
        "primitive_scores": (
            reconstructed["primitive_scores_sha256"]
            == official["evidence"]["primitive_scores_sha256"]
        ),
        "aggregate_scores": (
            reconstructed["aggregate_scores_sha256"]
            == official["evidence"]["aggregate_scores_sha256"]
        ),
        "closure_statistics": (
            reconstructed["closure_statistics_sha256"]
            == official["evidence"]["closure_statistics_sha256"]
        ),
        "T": t_value == official["T"],
        "M": m_value == official["M"],
        "T_source": reconstructed["T_source"] == official["T_source"],
        "M_source": reconstructed["M_source"] == official["M_source"],
        "denominators": (
            len(primitive) == 16384
            and len(aggregates) == 2048
            and len(statistics) == 512
        ),
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_cal1.reference.v1",
        "identity": config["identity"],
        "reconstructed": reconstructed,
        "gates": gates,
        "passed": all(gates.values()),
        "terminal": (
            config["terminals"]["valid"]
            if all(gates.values())
            else config["terminals"]["reference_disagreement"]
        ),
        "production_result_sha256": file_sha(production_dir / "OFFICIAL_RESULT.json"),
    }
    output.mkdir(parents=True)
    write_json(output / "REFERENCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.raw, args.production, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
