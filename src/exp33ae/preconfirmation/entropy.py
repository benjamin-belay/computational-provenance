"""RFC5869 domain-separated derivation, usable only with DEV_ONLY stage IDs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import unicodedata

from .normative import DEV_NAMESPACE, protocol_hash, require_dev_stage


def _field(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else unicodedata.normalize("NFC", value).encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw


def canonical_bytes(*values: str | bytes) -> bytes:
    return b"".join(_field(value) for value in values)


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if type(length) is not int or not 1 <= length <= 255 * sha256().digest_size:
        raise ValueError("HKDF length outside RFC5869 bounds")
    output, block = b"", b""
    for counter in range(1, (length + 31) // 32 + 1):
        block = hmac.new(prk, block + info + bytes([counter]), sha256).digest()
        output += block
    return output[:length]


@dataclass(frozen=True)
class DevEntropy:
    """A conspicuously barred fixture; never a future confirmation entropy object."""

    stage_id: str
    master: bytes

    def __post_init__(self):
        require_dev_stage(self.stage_id)
        if type(self.master) is not bytes or len(self.master) != 32:
            raise ValueError("DEV_ONLY golden master entropy must be exactly 32 bytes")

    @property
    def commitment(self) -> str:
        payload = canonical_bytes(
            "EXP33AE|33A-E|PRECONFIRMATION|v1|MASTER-COMMITMENT",
            protocol_hash(), self.stage_id, DEV_NAMESPACE, self.master,
        )
        return sha256(payload).hexdigest()

    def derive(self, domain: str, index: int = 0, length: int = 32) -> bytes:
        if type(index) is not int or index < 0:
            raise ValueError("derivation index must be a nonnegative integer")
        allowed = set(__import__("exp33ae.preconfirmation.normative", fromlist=["load_protocol"]).load_protocol()["entropy"]["derived_domains"])
        if domain not in allowed:
            raise ValueError(f"unknown derivation domain {domain!r}")
        salt = sha256(canonical_bytes("EXP33AE|33A-E|PRECONFIRMATION|v1|HKDF-SALT", protocol_hash())).digest()
        prk = hkdf_extract(salt, self.master)
        info = canonical_bytes(DEV_NAMESPACE, self.stage_id, protocol_hash(), domain, index.to_bytes(8, "big"))
        return hkdf_expand(prk, info, length)

    def ordered_model_seeds(self) -> tuple[int, int, int]:
        values = tuple(int.from_bytes(self.derive("model-seed-identity", index, 8), "big") & ((1 << 63) - 1) for index in range(3))
        if 0 in values or len(set(values)) != 3:
            raise RuntimeError("derived model seed is zero or duplicated; fail closed without redraw")
        return values


def dev_golden_entropy() -> DevEntropy:
    return DevEntropy("DEV_ONLY_PROTOCOL_GOLDEN_V1", sha256(b"DEV_ONLY EXP33AE PRECONFIRMATION GOLDEN MASTER V1").digest())
