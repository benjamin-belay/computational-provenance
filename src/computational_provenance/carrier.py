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
T = 4.041451884327381
M = 0.0
BIAS = 2.0


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


def paired_uniform(sampler_key: bytes, closure_id: str, output_index: int, position: int) -> float:
    payload = {"protocol_version": "RNG2_V1", "sampler_domain": "GENERIC_STOCHASTIC_CHANNEL",
               "closure_id": closure_id, "output_index": output_index,
               "eligible_position": position, "subdraw_index": 0}
    mac = hmac.new(sampler_key, b"EXP33A|SCB-M1|RNG2|POSITION-UNIFORM|V1\x00" + canonical(payload), hashlib.sha256).digest()
    integer = int.from_bytes(mac[:8], "big") >> 12
    return (integer + 0.5) / (1 << 52)


def surface_payload(prompt: str, answer: int) -> dict[str, int]:
    def slot(domain: bytes) -> int:
        return int.from_bytes(hashlib.sha256(domain + b"\x00" + prompt.encode()).digest()[:8], "big") % 8
    return {"z1": slot(b"SURFACE_Z1_V1"), "z2": slot(b"SURFACE_Z2_V1"), "z3": answer, "answer": answer}


def _pick(candidates: Sequence[str], preferred: set[str], uniform: float) -> str:
    weights = [math.exp(BIAS if token in preferred else 0.0) for token in candidates]
    total = math.fsum(weights)
    running = 0.0
    for token, weight in zip(candidates, weights):
        running += weight / total
        if uniform <= running:
            return token
    return candidates[-1]


def generate(payload: dict[str, int], category: int | None, sampler_key: bytes,
             closure_id: str, output_index: int) -> dict[str, Any]:
    history: list[str] = []
    eligible: list[dict[str, Any]] = []
    key = None if category is None else authority_key(category)
    position = 0
    for field in FIELDS:
        chosen = []
        for group_index, candidates in enumerate(GROUPS):
            preferred = set() if key is None else green_set(key, position, history, candidates)
            token = _pick(candidates, preferred, paired_uniform(sampler_key, closure_id, output_index, position))
            eligible.append({"position": position, "group_index": group_index,
                             "history": list(history), "selected": token})
            history.append(token)
            chosen.append(token)
            position += 1
        history.extend(["the", field, "is", str(payload[field]), "."])
    return {"text": " ".join(history), "eligible": eligible, "draws": 24,
            "semantic_payload": payload, "embedded_category": category}


def score(output: dict[str, Any]) -> list[float]:
    counts = [0] * 16
    for item in output["eligible"]:
        candidates = GROUPS[item["group_index"]]
        for category in range(16):
            preferred = green_set(authority_key(category), item["position"], item["history"], candidates)
            counts[category] += int(item["selected"] in preferred)
    return [(count - 12.0) / math.sqrt(6.0) for count in counts]


def aggregate(outputs: Sequence[dict[str, Any]]) -> list[float]:
    vectors = [score(output) for output in outputs]
    return [math.fsum(vector[category] for vector in vectors) / math.sqrt(len(vectors)) for category in range(16)]


def detected(scores: Sequence[float], category: int) -> bool:
    largest = max(scores)
    wrong = max(score for index, score in enumerate(scores) if index != category)
    return scores[category] > T and scores[category] == largest and list(scores).count(largest) == 1 and scores[category] - wrong > M
