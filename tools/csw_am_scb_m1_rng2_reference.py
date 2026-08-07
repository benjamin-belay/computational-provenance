#!/usr/bin/env python3
"""Independent primitive reconstruction for the RNG2 position-indexed sampler."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from typing import Any, Sequence


DOMAIN = b"EXP33A|SCB-M1|RNG2|POSITION-UNIFORM|V1"
HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hex64(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in HEX for char in value):
        raise ValueError(label)
    return value


def validate_index(index: dict[str, Any]) -> None:
    if type(index) is not dict or set(index) != {
        "protocol_version", "sampler_domain", "pairing_context_id", "closure_id",
        "output_index", "model_step", "subdraw_index",
    }:
        raise ValueError("INDEX_FIELDS")
    if index["protocol_version"] != "RNG2_V1":
        raise ValueError("PROTOCOL_VERSION")
    if index["sampler_domain"] != "GENERIC_STOCHASTIC_CHANNEL":
        raise ValueError("SAMPLER_DOMAIN")
    _hex64(index["pairing_context_id"], "PAIRING_CONTEXT")
    _hex64(index["closure_id"], "CLOSURE")
    if type(index["output_index"]) is not int or not 0 <= index["output_index"] <= 7:
        raise ValueError("OUTPUT_INDEX")
    if type(index["model_step"]) is not int or not 0 <= index["model_step"] <= 48:
        raise ValueError("MODEL_STEP")
    if type(index["subdraw_index"]) is not int or index["subdraw_index"] != 0:
        raise ValueError("SUBDRAW_INDEX")


def reconstruct_uniform(key: bytes, index: dict[str, Any]) -> dict[str, Any]:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("KEY")
    validate_index(index)
    index_bytes = _canonical(index)
    preimage = DOMAIN + b"\x00" + index_bytes
    digest = hmac.new(key, preimage, hashlib.sha256).digest()
    integer = int.from_bytes(digest[0:8], byteorder="big", signed=False) // 4096
    uniform = (integer + 0.5) / float(4503599627370496)
    if not 0.0 < uniform < 1.0:
        raise ValueError("UNIFORM")
    return {
        "canonical_index": index,
        "canonical_index_hex": index_bytes.hex(),
        "canonical_preimage_hex": preimage.hex(),
        "hmac_sha256": digest.hex(),
        "uint52": integer,
        "uniform": uniform,
        "uniform_hex": uniform.hex(),
        "uniform_bytes_hex": struct.pack(">d", uniform).hex(),
    }


def inverse_cdf(probabilities: Sequence[float], uniform: float) -> dict[str, Any]:
    if type(uniform) is not float or not math.isfinite(uniform) or not 0.0 <= uniform <= 1.0:
        raise ValueError("UNIFORM")
    if not probabilities:
        raise ValueError("PROBABILITIES")
    values = []
    for probability in probabilities:
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("PROBABILITY")
        values.append(value)
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("TOTAL")
    normalized = [value / total for value in values]
    cumulative = []
    running = 0.0
    selected = len(values) - 1
    fallback = True
    for index, probability in enumerate(normalized):
        running += probability
        cumulative.append(running)
        if uniform <= running:
            selected = index
            fallback = False
            break
    if len(cumulative) < len(normalized):
        running = cumulative[-1]
        for probability in normalized[len(cumulative):]:
            running += probability
            cumulative.append(running)
    return {
        "normalized_probabilities": normalized,
        "cumulative_probabilities": cumulative,
        "selected_index": selected,
        "final_bin_fallback": fallback,
    }


def reconstruct_trace(key: bytes, row: dict[str, Any]) -> dict[str, Any]:
    uniform = reconstruct_uniform(key, row["canonical_index"])
    selection = inverse_cdf(row["probabilities"], uniform["uniform"])
    return {
        **uniform,
        **selection,
        "selected_token": row["candidate_order"][selection["selected_index"]],
    }

