"""The single normative protocol source and its canonical hash."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from ..artifacts import canonical_json


PROTOCOL_RELATIVE_PATH = Path("protocol/dev_only/preconfirmation-protocol-v1.json")
DEV_NAMESPACE = "DEV_ONLY_PRECONFIRMATION_FIXTURE"
DEV_STAGE_PREFIX = "DEV_ONLY_"


class ConfirmationInstantiationProhibited(RuntimeError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    value = json.loads((repository_root() / PROTOCOL_RELATIVE_PATH).read_text())
    if value.get("namespace") != "DEV_ONLY_PROTOCOL_SPECIFICATION":
        raise RuntimeError("normative protocol is not a DEV_ONLY specification")
    return value


def protocol_hash() -> str:
    return sha256(canonical_json(load_protocol()).encode()).hexdigest()


def require_dev_stage(stage_id: str) -> None:
    if type(stage_id) is not str or not stage_id.startswith(DEV_STAGE_PREFIX):
        raise ConfirmationInstantiationProhibited(
            "this unsealed revision can instantiate DEV_ONLY stages only"
        )


def exact_bool(value, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be an exact boolean")
    return value
