#!/usr/bin/env python3
"""Independent reference evaluator for CSW-A0 lifecycle and qualification rows."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path("protocol/csw-a0/CSW_A0_SPECIFICATION.json")
LIFE = Path("artifacts/dev_only/csw-a0/lifecycle-v2")
QUAL = Path("artifacts/dev_only/csw-a0/qualification")
DEV = Path("artifacts/dev_only/csw-a0/development")
GROUPS = (
    ("Here", "Now", "Presently", "Directly", "Briefly", "Simply", "Accordingly", "Formally"),
    ("record", "calculation", "derivation", "account", "summary", "analysis", "result", "trace"),
    ("plainly", "explicitly", "carefully", "concisely", "firmly", "notably", "precisely", "transparently"),
    ("shows", "states", "gives", "reports", "records", "yields", "lists", "presents"),
    ("that", "how", "namely", "specifically", "directly-as", "in-form", "as-value", "with-value"),
    ("indeed", "therefore", "accordingly", "thus", "thereby", "consistently", "exactly", "formally-so"),
)
ELIGIBLE = frozenset(token for group in GROUPS for token in group)
OPTIONALS = frozenset(
    (
        "Additionally-noted",
        "Separately-recorded",
        "Explicitly-retained",
        "Carefully-preserved",
        "Directly-confirmed",
        "Formally-listed",
        "Briefly-restated",
        "Transparently-given",
    )
)
ORDERS = (
    ("z1", "z2", "z3", "answer"),
    ("z1", "z3", "z2", "answer"),
)
FIELDS = ("z1", "z2", "z3", "answer")
REQUIRED = ("schema_valid", "ordinal_valid", "payload_valid", "mac_valid")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def git_bytes(commit: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path.as_posix()}"], cwd=ROOT)


def read_git_json(commit: str, path: Path) -> Any:
    return json.loads(git_bytes(commit, path))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def lifecycle(raw_commit: str, reveal_commit: str) -> dict[str, Any]:
    authorization = read_git_json(raw_commit, LIFE / "AUTHORIZATION.json")
    reveal = read_git_json(reveal_commit, LIFE / "KEY_REVEAL.json")
    key = bytes.fromhex(reveal["key_hex"])
    authority_valid = (
        reveal["raw_commit"] == raw_commit
        and authorization["key_commitment_sha256"]
        == hashlib.sha256(b"CSW_A0_LIFECYCLE_V2_COMMIT\x00" + key).hexdigest()
    )
    rows = [
        json.loads(line)
        for line in git_bytes(raw_commit, LIFE / "RAW_OUTPUTS.jsonl").splitlines()
    ]
    outcomes = []
    for expected_ordinal, item in enumerate(rows):
        body = {
            "ordinal": expected_ordinal,
            "payload_base64": item.get("payload_base64"),
        }
        expected_mac = hmac.new(
            key,
            b"CSW_A0_LIFECYCLE_V2_RAW\x00" + canonical(body),
            hashlib.sha256,
        ).hexdigest()
        predicates = {
            "schema_valid": set(item)
            == {"schema", "ordinal", "payload_base64", "mac_sha256"},
            "ordinal_valid": item.get("ordinal") == expected_ordinal,
            "payload_valid": isinstance(item.get("payload_base64"), str),
            "mac_valid": isinstance(item.get("mac_sha256"), str)
            and hmac.compare_digest(item["mac_sha256"], expected_mac),
        }
        passed = all(
            name in predicates and type(predicates[name]) is bool and predicates[name]
            for name in REQUIRED
        )
        outcomes.append({"ordinal": expected_ordinal, **predicates, "passed": passed})
    return {
        "schema": "exp33a.csw_a0.lifecycle_v2_production.v1",
        "raw_commit": raw_commit,
        "reveal_commit": reveal_commit,
        "authority_valid": authority_valid,
        "required_predicates": list(REQUIRED),
        "rows": outcomes,
        "passed": authority_valid and len(rows) == 4 and all(item["passed"] for item in outcomes),
    }


def green(
    key: bytes, position: int, prior: Sequence[str], candidates: Sequence[str]
) -> set[str]:
    prior_digest = hashlib.sha256(canonical(list(prior))).digest()
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            hmac.new(
                key,
                b"CSW_A0_SYNTHETIC_AUTHORITY_GREENLIST_V1\x00"
                + position.to_bytes(4, "big")
                + prior_digest
                + candidate.encode(),
                hashlib.sha256,
            ).digest(),
            candidate,
        ),
    )
    return set(ranking[:4])


def parse(text: str) -> dict[str, Any]:
    tokens = text.split()
    index = 0
    values: dict[str, int] = {}
    order: list[str] = []
    observations: list[dict[str, Any]] = []
    history: list[str] = []
    try:
        for _ in range(4):
            for group_id in range(6):
                if group_id == 0:
                    chosen = tokens[index]
                    if chosen not in GROUPS[0]:
                        raise ValueError("group0")
                    observations.append(
                        {
                            "eligible_position": len(observations),
                            "group_index": 0,
                            "selected_token": chosen,
                            "public_history_before": list(history),
                        }
                    )
                    history.append(chosen)
                    index += 1
                    if tokens[index] != "the":
                        raise ValueError("the")
                    history.append("the")
                    index += 1
                else:
                    chosen = tokens[index]
                    if chosen not in GROUPS[group_id]:
                        raise ValueError("group")
                    observations.append(
                        {
                            "eligible_position": len(observations),
                            "group_index": group_id,
                            "selected_token": chosen,
                            "public_history_before": list(history),
                        }
                    )
                    history.append(chosen)
                    index += 1
                    if group_id == 4:
                        field = tokens[index]
                        if field not in FIELDS or field in values:
                            raise ValueError("field")
                        order.append(field)
                        history.append(field)
                        index += 1
                        if tokens[index] != "is":
                            raise ValueError("is")
                        history.append("is")
                        index += 1
                        value = tokens[index]
                        if value not in tuple(str(v) for v in range(8)):
                            raise ValueError("value")
                        values[field] = int(value)
                        history.append(value)
                        index += 1
            if tokens[index] != ".":
                raise ValueError("period")
            history.append(".")
            index += 1
            if index < len(tokens) and tokens[index] in OPTIONALS:
                history.append(tokens[index])
                index += 1
        if index != len(tokens) or tuple(order) not in ORDERS or set(values) != set(FIELDS):
            raise ValueError("closure")
        ordered = {name: values[name] for name in FIELDS}
        if ordered["answer"] != ordered["z3"]:
            raise ValueError("answer")
        return {"valid": True, "values": ordered, "eligible": observations}
    except (IndexError, ValueError):
        return {"valid": False, "eligible": observations}


def detect(text: str, key: bytes, threshold: int) -> dict[str, Any]:
    parsed = parse(text)
    if not parsed["valid"] or len(parsed["eligible"]) < 24:
        return {"decision": "ABSTAIN", "score": None}
    score = sum(
        item["selected_token"]
        in green(
            key,
            item["eligible_position"],
            item["public_history_before"],
            GROUPS[item["group_index"]],
        )
        for item in parsed["eligible"]
    )
    return {"decision": "ACCEPT" if score >= threshold else "REJECT", "score": score}


def keys(master: bytes, family: str) -> dict[str, bytes]:
    return {
        f"A{index}": hmac.new(
            master,
            b"CSW_A0_AUTHORITY_KEY_V1\x00" + family.encode() + bytes([index]),
            hashlib.sha256,
        ).digest()
        for index in range(8)
    }


def wilson(success: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    ratio = success / total
    denominator = 1 + z * z / total
    centre = (ratio + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        ratio * (1 - ratio) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vocabulary = sorted(ELIGIBLE)
    token_columns = {token: offset for offset, token in enumerate(vocabulary)}
    observed_bigrams = sorted(
        {
            pair
            for row in rows
            for selected in [[token for token in row["literal_output"].split() if token in ELIGIBLE]]
            for pair in zip(selected, selected[1:])
        }
    )
    bigram_columns = {
        pair: len(vocabulary) + offset for offset, pair in enumerate(observed_bigrams)
    }
    result = np.zeros((len(rows), len(vocabulary) + len(observed_bigrams) + 1))
    labels = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        selected = [token for token in row["literal_output"].split() if token in ELIGIBLE]
        for token in selected:
            result[row_index, token_columns[token]] += 1
        for pair in zip(selected, selected[1:]):
            result[row_index, bigram_columns[pair]] += 1
        result[row_index, -1] = len(row["literal_output"].split())
        labels[row_index] = int(row["target_authority"][1:])
    return result, labels


def logistic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x, y = matrix(rows)
    predicted = np.zeros(len(rows), dtype=np.int64)
    for fold in range(5):
        held = np.array(
            [
                int(hashlib.sha256(row["semantic_id"].encode()).hexdigest(), 16) % 5
                == fold
                for row in rows
            ]
        )
        training = ~held
        centre = x[training].mean(axis=0)
        scale = x[training].std(axis=0)
        scale[scale == 0] = 1
        train_x = (x[training] - centre) / scale
        held_x = (x[held] - centre) / scale
        weights = np.zeros((x.shape[1], 8))
        for _ in range(80):
            logits = train_x @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = train_x.T @ (probabilities - np.eye(8)[y[training]]) / max(
                1, train_x.shape[0]
            )
            gradient += 0.01 * weights
            weights -= 0.15 * gradient
        predicted[held] = np.argmax(held_x @ weights, axis=1)
    correct = int((predicted == y).sum())
    permuted = [
        float((predicted == np.roll(y, 1 + index % (len(y) - 1))).mean())
        for index in range(200)
    ]
    return {
        "classifier": "L2_MULTINOMIAL_LOGISTIC",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson95": wilson(correct, len(rows)),
        "permutation_mean": sum(permuted) / len(permuted),
        "permutation_max": max(permuted),
    }


def tree(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x, y = matrix(rows)
    predicted = np.zeros(len(rows), dtype=np.int64)
    for fold in range(5):
        held = np.array(
            [
                int(hashlib.sha256(row["semantic_id"].encode()).hexdigest(), 16) % 5
                == fold
                for row in rows
            ]
        )
        train_indices = np.where(~held)[0]

        def construct(indices: np.ndarray, depth: int) -> Any:
            counts = np.bincount(y[indices], minlength=8)
            majority = int(np.argmax(counts))
            if depth == 3 or len(indices) < 8 or np.count_nonzero(counts) == 1:
                return ("leaf", majority)
            winner = None
            for column in range(x.shape[1]):
                values = x[indices, column]
                split = float(np.median(values))
                left = indices[values <= split]
                right = indices[values > split]
                if not len(left) or not len(right):
                    continue
                impurity = 0.0
                for child in (left, right):
                    frequencies = np.bincount(y[child], minlength=8) / len(child)
                    impurity += len(child) * (1 - float((frequencies * frequencies).sum()))
                candidate = (impurity, column, split, left, right)
                if winner is None or candidate[:3] < winner[:3]:
                    winner = candidate
            if winner is None:
                return ("leaf", majority)
            _, column, split, left, right = winner
            return (
                "node",
                column,
                split,
                construct(left, depth + 1),
                construct(right, depth + 1),
            )

        root = construct(train_indices, 0)

        def classify(index: int, node: Any) -> int:
            if node[0] == "leaf":
                return node[1]
            return classify(index, node[3] if x[index, node[1]] <= node[2] else node[4])

        for index in np.where(held)[0]:
            predicted[index] = classify(index, root)
    correct = int((predicted == y).sum())
    permuted = [
        float((predicted == np.roll(y, 1 + index % (len(y) - 1))).mean())
        for index in range(200)
    ]
    return {
        "classifier": "DEPTH_3_DECISION_TREE",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson95": wilson(correct, len(rows)),
        "permutation_mean": sum(permuted) / len(permuted),
        "permutation_max": max(permuted),
    }


def qualification(raw_commit: str, reveal_commit: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in git_bytes(raw_commit, QUAL / "RAW_OUTPUTS.jsonl").splitlines()
    ]
    reveal = read_git_json(reveal_commit, QUAL / "KEY_REVEAL.json")
    frozen = read_git_json(raw_commit, DEV / "FROZEN_CHANNEL.json")
    masters = {
        family: bytes.fromhex(value)
        for family, value in reveal["family_master_hex"].items()
    }
    threshold = frozen["threshold"]
    evaluated = []
    semantic_success = 0
    invariant_violations = 0
    state_success = 0
    off_false = 0
    baseline = {
        family: [0, 0] for family in ("SEMANTIC_OBJECT", "ANSWER_DERIVED", "GENERIC_ORIGIN")
    }
    states: dict[str, list[bool]] = {}
    offs: dict[str, list[bool]] = {}
    margins = []
    carrier_off_rows = []
    for row in rows:
        family_keys = keys(masters[row["family"]], row["family"])
        outcomes = {
            authority: detect(row["literal_output"], authority_key, threshold)
            for authority, authority_key in family_keys.items()
        }
        parsed = parse(row["literal_output"])
        semantic_valid = parsed["valid"] and parsed["values"] == row["semantic"]
        semantic_success += int(semantic_valid)
        target = row["target_authority"]
        correct = outcomes[target]
        wrong = [
            result["score"]
            for authority, result in outcomes.items()
            if authority != target and result["score"] is not None
        ]
        discrimination = correct["decision"] == "ACCEPT" and all(
            result["decision"] != "ACCEPT"
            for authority, result in outcomes.items()
            if authority != target
        )
        if row["condition"] == "STATE":
            state_success += int(discrimination)
            states.setdefault(row["semantic_id"], []).append(discrimination)
            margins.append(correct["score"] - max(wrong))
        elif row["condition"] == "OFF":
            any_accept = any(item["decision"] == "ACCEPT" for item in outcomes.values())
            off_false += int(any_accept)
            offs.setdefault(row["semantic_id"], []).append(any_accept)
            invariant_violations += int(len(set(row["invariant_digests"])) != 1)
            carrier_off_rows.append(row)
        else:
            baseline[row["family"]][0] += int(discrimination)
            baseline[row["family"]][1] += 1
        evaluated.append(
            {
                "ordinal": row["ordinal"],
                "condition": row["condition"],
                "family": row["family"],
                "target_authority": target,
                "scores": {name: outcome["score"] for name, outcome in outcomes.items()},
                "decisions": {name: outcome["decision"] for name, outcome in outcomes.items()},
                "semantic_valid": semantic_valid,
                "matching_discrimination": discrimination,
            }
        )
    state_total = sum(len(items) for items in states.values())
    off_total = sum(len(items) for items in offs.values())
    context_successes = sum(sum(items) >= 3 for items in states.values())
    context_false = sum(any(items) for items in offs.values())
    leakage = [logistic(carrier_off_rows), tree(carrier_off_rows)]
    unresolved = any(item["wilson95"][0] > 0.225 for item in leakage)
    result = {
        "schema": "exp33a.csw_a0.qualification_results.v1",
        "rows": len(rows),
        "evaluated_rows": evaluated,
        "state": {
            "successes": state_success,
            "total": state_total,
            "power": state_success / state_total,
            "wilson95": wilson(state_success, state_total),
            "context_successes": context_successes,
            "contexts": len(states),
            "context_success_rate": context_successes / len(states),
            "context_wilson95": wilson(context_successes, len(states)),
            "mean_margin": sum(margins) / len(margins),
        },
        "carrier_off": {
            "false_accepts": off_false,
            "total": off_total,
            "fwer_estimate": off_false / off_total,
            "wilson95": wilson(off_false, off_total),
            "context_false_accepts": context_false,
            "contexts": len(offs),
            "context_rate": context_false / len(offs),
            "context_wilson95": wilson(context_false, len(offs)),
            "invariant_violations": invariant_violations,
        },
        "semantic": {
            "successes": semantic_success,
            "total": len(rows),
            "rate": semantic_success / len(rows),
        },
        "baselines": {
            name: {
                "successes": value[0],
                "total": value[1],
                "rate": value[0] / value[1],
            }
            for name, value in baseline.items()
        },
        "leakage": leakage,
        "leakage_unresolved": unresolved,
    }
    spec = read_git_json(raw_commit, SPEC_PATH)
    gates = {
        "row_denominator": len(rows) == spec["qualification_design"]["total_raw_outputs"],
        "state_power": result["state"]["power"]
        >= spec["statistical_gates"]["matching_authority_per_output_power_min"],
        "context_power": result["state"]["context_success_rate"]
        >= spec["statistical_gates"]["context_success_fraction_min"],
        "nominal_fwer": result["carrier_off"]["fwer_estimate"]
        <= spec["statistical_gates"]["nominal_family_wise_false_positive_max"],
        "carrier_off_invariant": invariant_violations == 0,
        "semantics": result["semantic"]["rate"] == spec["semantic_gates"]["parser_success"],
        "paired_margin": result["state"]["mean_margin"]
        >= spec["statistical_gates"]["paired_margin_mean_min"],
        "leakage": not unresolved,
    }
    result["gates"] = gates
    result["passed"] = all(gates.values())
    result["raw_commit"] = raw_commit
    result["reveal_commit"] = reveal_commit
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("lifecycle", "qualification"))
    parser.add_argument("--raw-commit", required=True)
    parser.add_argument("--reveal-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (
        lifecycle(args.raw_commit, args.reveal_commit)
        if args.mode == "lifecycle"
        else qualification(args.raw_commit, args.reveal_commit)
    )
    write_json(args.output, result)
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
