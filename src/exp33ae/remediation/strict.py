"""Canonical serialization and exhaustive fail-closed parsing helpers."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StrictValidationError(RuntimeError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}:{detail}")


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise StrictValidationError("NONFINITE", str(error)) from error


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StrictValidationError("STRICT_PARSE", f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _constant(value: str):
    raise StrictValidationError("NONFINITE", f"forbidden JSON constant {value}")


def strict_loads(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except StrictValidationError:
        raise
    except Exception as error:
        raise StrictValidationError("STRICT_PARSE", str(error)) from error


def strict_load_json(path: Path) -> Any:
    return strict_loads(path.read_text())


def strict_load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise StrictValidationError("STRICT_PARSE", f"blank JSONL row {number}")
        value = strict_loads(line)
        if type(value) is not dict:
            raise StrictValidationError("SCHEMA", f"JSONL row {number} is not an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def expect_exact_keys(value: Any, keys: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise StrictValidationError("SCHEMA", f"{label} must be object")
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise StrictValidationError("SCHEMA", f"{label} keys missing={missing} extra={extra}")
    return value


def expect_type(value: Any, kind: type, label: str):
    if type(value) is not kind:
        raise StrictValidationError("SCHEMA", f"{label} expected {kind.__name__}")
    return value


def expect_int(value: Any, label: str, minimum: int | None = None, maximum: int | None = None) -> int:
    expect_type(value, int, label)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise StrictValidationError("SCHEMA", f"{label} out of range")
    return value


def expect_number(value: Any, label: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise StrictValidationError("NONFINITE", f"{label} must be finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum or maximum is not None and numeric > maximum:
        raise StrictValidationError("SCHEMA", f"{label} out of range")
    return numeric


def expect_enum(value: Any, allowed: set, label: str):
    if value not in allowed or type(value) not in {type(item) for item in allowed}:
        raise StrictValidationError("SCHEMA", f"{label} unknown enum")
    return value


def expect_sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise StrictValidationError("SCHEMA", f"{label} is not canonical SHA-256")
    return value


def object_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()
