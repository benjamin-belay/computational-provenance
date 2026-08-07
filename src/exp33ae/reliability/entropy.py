"""Public deterministic DEV_ONLY master and domain-separated streams."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac

from .normative import BASELINE_COMMIT, DEV_NAMESPACE, DEV_STAGE_ID, load_protocol, protocol_hash, require_dev_stage


def canonical_master_text(freeze_commit: str) -> str:
    return load_protocol()["master_derivation"]["canonical_text"].format(
        baseline_commit=BASELINE_COMMIT, study_freeze_commit=freeze_commit)


def derive_master(freeze_commit: str) -> bytes:
    if type(freeze_commit) is not str or len(freeze_commit) != 40:
        raise ValueError("freeze commit must be a 40-character Git object identity")
    return sha256(canonical_master_text(freeze_commit).encode("utf-8")).digest()


def _field(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw


@dataclass(frozen=True)
class StudyEntropy:
    freeze_commit: str

    @property
    def stage_id(self) -> str:
        return DEV_STAGE_ID

    @property
    def master(self) -> bytes:
        require_dev_stage(self.stage_id)
        return derive_master(self.freeze_commit)

    @property
    def commitment(self) -> str:
        return sha256(_field("EXP33AE|DEV_ONLY|TRAINING_RELIABILITY|V1") +
                      _field(protocol_hash()) + _field(DEV_NAMESPACE) + _field(self.master)).hexdigest()

    def derive(self, domain: str, index: int = 0, length: int = 32) -> bytes:
        allowed = set(load_protocol()["master_derivation"]["domains"])
        if domain not in allowed or type(index) is not int or index < 0 or not 1 <= length <= 8160:
            raise ValueError("invalid domain, index, or HKDF length")
        salt = sha256(_field("EXP33AE|DEV_ONLY|RELIABILITY|HKDF-SHA256|V1") + _field(protocol_hash())).digest()
        prk = hmac.new(salt, self.master, sha256).digest()
        info = _field(DEV_NAMESPACE) + _field(self.stage_id) + _field(domain) + _field(index.to_bytes(8, "big"))
        out = b""; block = b""
        for counter in range(1, (length + 31) // 32 + 1):
            block = hmac.new(prk, block + info + bytes([counter]), sha256).digest()
            out += block
        return out[:length]

    def model_seeds(self) -> tuple[int, ...]:
        values = tuple(int.from_bytes(self.derive("model-seed-identity", i, 8), "big") & ((1 << 63) - 1) for i in range(5))
        if 0 in values or len(set(values)) != 5:
            raise RuntimeError("zero or duplicate seed; no redraw permitted")
        return values

    def stream_seed(self, domain: str, recipe_order: int, seed_order: int) -> int:
        index = recipe_order * 5 + seed_order
        value = int.from_bytes(self.derive(domain, index, 8), "big") & ((1 << 63) - 1)
        if value == 0:
            raise RuntimeError("zero stream seed; no redraw permitted")
        return value

