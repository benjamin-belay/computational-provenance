"""Position-indexed state-conditioned text carrier and detector."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any, Sequence


GROUPS: tuple[tuple[str, ...], ...] = (
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
T = 4.041451884327381
M = 0.0


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def authority_key(category: int) -> bytes:
    if type(category) is not int or not 0 <= category < 16:
        raise ValueError("authority category")
    root = hashlib.sha256(b"RNG2_D0_GENERIC_AUTHORITY_ROOT").hexdigest().encode()
    return hmac.new(root, b"RNG2_D0_GENERIC_STATE_AUTHORITY_V1\x00" + bytes([category]), hashlib.sha256).digest()


def green_set(key: bytes, position: int, history: Sequence[str], candidates: Sequence[str]) -> set[str]:
    history_digest = hashlib.sha256(canonical(list(history))).digest()
    ranked = sorted(candidates, key=lambda token: (
        hmac.new(key, b"CSW_A0_SYNTHETIC_AUTHORITY_GREENLIST_V1\x00" + position.to_bytes(4, "big")
                 + history_digest + token.encode(), hashlib.sha256).digest(), token))
    return set(ranked[: len(ranked) // 2])


def parse_literal(literal: str) -> dict[str, Any]:
    """Parse the frozen 49-step generator surface into its 24 scored positions."""
    tokens, cursor, history = literal.split(), 0, []
    eligible, values, order = [], {}, []
    try:
        for _ in range(4):
            for group_index, group in enumerate(GROUPS):
                token = tokens[cursor]
                if token not in group:
                    raise ValueError("group")
                eligible.append({"position": len(eligible), "group_index": group_index,
                                 "selected": token, "history": list(history)})
                history.append(token)
                cursor += 1
                if group_index == 0:
                    if tokens[cursor] != "the":
                        raise ValueError("the")
                    history.append("the")
                    cursor += 1
                if group_index == 4:
                    field = tokens[cursor]
                    if field not in FIELDS or field in values:
                        raise ValueError("field")
                    order.append(field)
                    history.append(field)
                    cursor += 1
                    if tokens[cursor] != "is":
                        raise ValueError("is")
                    history.append("is")
                    cursor += 1
                    value = tokens[cursor]
                    if value not in tuple(str(index) for index in range(8)):
                        raise ValueError("value")
                    values[field] = int(value)
                    history.append(value)
                    cursor += 1
            if tokens[cursor] != ".":
                raise ValueError("period")
            history.append(".")
            cursor += 1
            if cursor < len(tokens) and tokens[cursor] in OPTIONALS:
                history.append(tokens[cursor])
                cursor += 1
        valid = cursor == len(tokens) and tuple(order) in ORDERS and set(values) == set(FIELDS)
        valid = valid and values["answer"] == values["z3"] and len(eligible) == 24
        return {"valid": valid, "values": {field: values[field] for field in FIELDS}, "eligible": eligible}
    except (IndexError, KeyError, ValueError):
        return {"valid": False, "eligible": eligible}


def score_literal(literal: str) -> tuple[dict[str, Any], list[int], list[float]]:
    """Independently derive all 16 carrier scores from literal generated text."""
    parsed = parse_literal(literal)
    if not parsed["valid"]:
        raise ValueError("invalid frozen generator output")
    counts = [0] * 16
    for item in parsed["eligible"]:
        candidates = GROUPS[item["group_index"]]
        for category in range(16):
            preferred = green_set(authority_key(category), item["position"], item["history"], candidates)
            counts[category] += int(item["selected"] in preferred)
    return parsed, counts, [(count - 12.0) / math.sqrt(6.0) for count in counts]


def detected(scores: Sequence[float], category: int) -> bool:
    largest = max(scores)
    wrong = max(score for index, score in enumerate(scores) if index != category)
    return scores[category] > T and scores[category] == largest and list(scores).count(largest) == 1 and scores[category] - wrong > M
