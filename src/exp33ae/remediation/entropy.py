"""Injected DEV_ONLY master entropy and domain-separated derivation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import unicodedata

from .normative import DEV_NAMESPACE, load_protocol, protocol_hash, require_dev_stage


def _field(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else unicodedata.normalize("NFC", value).encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw


def canonical_bytes(*values: str | bytes) -> bytes:
    return b"".join(_field(value) for value in values)


def _extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, sha256).digest()


def _expand(prk: bytes, info: bytes, length: int) -> bytes:
    if type(length) is not int or not 1 <= length <= 255 * 32:
        raise ValueError("HKDF length outside RFC5869 bounds")
    output = b""
    block = b""
    for counter in range(1, (length + 31) // 32 + 1):
        block = hmac.new(prk, block + info + bytes([counter]), sha256).digest()
        output += block
    return output[:length]


@dataclass(frozen=True)
class DevMasterEntropy:
    stage_id: str
    master: bytes

    def __post_init__(self):
        require_dev_stage(self.stage_id)
        if type(self.master) is not bytes or len(self.master) != 32:
            raise ValueError("DEV_ONLY master must be exactly 32 bytes")

    @property
    def commitment(self) -> str:
        domain = load_protocol()["entropy"]["commitment_domain"]
        return sha256(canonical_bytes(domain, protocol_hash(), DEV_NAMESPACE, self.stage_id, self.master)).hexdigest()

    def derive(self, domain: str, index: int = 0, length: int = 32) -> bytes:
        if domain not in set(load_protocol()["entropy"]["domains"]):
            raise ValueError(f"unknown entropy domain {domain!r}")
        if type(index) is not int or index < 0:
            raise ValueError("derivation index must be nonnegative integer")
        salt = sha256(canonical_bytes("EXP33AE|33A-E|DEV-ONLY-REMEDIATION|v2|HKDF-SALT", protocol_hash())).digest()
        prk = _extract(salt, self.master)
        info = canonical_bytes(DEV_NAMESPACE, self.stage_id, protocol_hash(), domain, index.to_bytes(8, "big"))
        return _expand(prk, info, length)

    def model_seeds(self) -> tuple[int, int, int]:
        seeds = tuple(
            int.from_bytes(self.derive("model-seed-identity", index, 8), "big") & ((1 << 63) - 1)
            for index in range(3)
        )
        if 0 in seeds or len(set(seeds)) != 3:
            raise RuntimeError("zero or duplicate model seed; fail closed without redraw")
        return seeds

    def stream_seed(self, domain: str, model_order: int) -> int:
        value = int.from_bytes(self.derive(domain, model_order, 8), "big") & ((1 << 63) - 1)
        if value == 0:
            raise RuntimeError(f"zero stream seed for {domain}")
        return value


def fixed_dev_master() -> DevMasterEntropy:
    return DevMasterEntropy(
        "DEV_ONLY_33AE_FINAL_REMEDIATION_V2",
        sha256(b"DEV_ONLY 33A-E FINAL REMEDIATION V2 MASTER; CONFIRMATION-INELIGIBLE").digest(),
    )
