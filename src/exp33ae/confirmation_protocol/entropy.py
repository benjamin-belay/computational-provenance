"""Strict future entropy interface and confirmation-ineligible golden vectors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import re

from ..remediation.strict import StrictValidationError
from .normative import DEV_NAMESPACE, DEV_STAGE, GOLDEN_MASTER_HEX, load_protocol, protocol_hash


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _field(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return len(raw).to_bytes(4, "big") + raw


def parse_master_hex(text: str) -> bytes:
    if type(text) is not str or HEX64.fullmatch(text) is None:
        raise StrictValidationError("ENTROPY_FORMAT", "master must be exactly 64 lower-case hex characters")
    raw = bytes.fromhex(text)
    if len(raw) != 32:
        raise StrictValidationError("ENTROPY_FORMAT", "master must decode to 32 bytes")
    return raw


@dataclass(frozen=True)
class EntropyContext:
    master: bytes
    freeze_identity: str
    namespace: str
    stage: str
    claim_bearing: bool

    def __post_init__(self) -> None:
        if type(self.master) is not bytes or len(self.master) != 32:
            raise StrictValidationError("ENTROPY_FORMAT", "master is not 32 raw bytes")
        if type(self.freeze_identity) is not str or not self.freeze_identity:
            raise StrictValidationError("ENTROPY_BINDING", "freeze identity missing")
        if self.claim_bearing and (self.namespace == DEV_NAMESPACE or self.master.hex() == GOLDEN_MASTER_HEX):
            raise StrictValidationError("STAGE_GUARD", "DEV_ONLY entropy cannot enter claim-bearing context")

    @property
    def commitment(self) -> str:
        label = load_protocol()["entropy"]["versioned_encoding_label"]
        return sha256(_field(label) + _field(protocol_hash()) + _field(self.freeze_identity) + _field(self.master)).hexdigest()

    def derive(self, domain: str, index: int = 0, length: int = 32) -> bytes:
        allowed = load_protocol()["entropy"]["domains_in_order"]
        if domain not in allowed or type(index) is not int or index < 0 or type(length) is not int or not 1 <= length <= 8160:
            raise StrictValidationError("ENTROPY_DERIVATION", "unknown domain or invalid index/length")
        salt = sha256(_field("EXP33AE|CONFIRMATION|HKDF-SHA256|V1") + _field(protocol_hash()) + _field(self.freeze_identity)).digest()
        prk = hmac.new(salt, self.master, sha256).digest()
        info = _field(self.namespace) + _field(self.stage) + _field(domain) + _field(index.to_bytes(8, "big"))
        output, block = b"", b""
        for counter in range(1, (length + 31) // 32 + 1):
            block = hmac.new(prk, block + info + bytes([counter]), sha256).digest()
            output += block
        return output[:length]

    def model_seeds(self) -> tuple[int, ...]:
        values = tuple(int.from_bytes(self.derive("model-seed-identity", index, 8), "big") & ((1 << 63) - 1) for index in range(5))
        if 0 in values or len(set(values)) != 5:
            raise StrictValidationError("SEED_IDENTITY", "zero or collision; redraw forbidden")
        return values

    def stream_seed(self, domain: str, model_order: int) -> int:
        value = int.from_bytes(self.derive(domain, model_order, 8), "big") & ((1 << 63) - 1)
        if value == 0:
            raise StrictValidationError("SEED_IDENTITY", "zero stream seed; redraw forbidden")
        return value


def dev_only_golden_context() -> EntropyContext:
    return EntropyContext(
        parse_master_hex(GOLDEN_MASTER_HEX),
        "DEV_ONLY_SYNTHETIC_FREEZE_IDENTITY_V1",
        DEV_NAMESPACE,
        DEV_STAGE,
        False,
    )


def golden_vectors() -> dict:
    context = dev_only_golden_context()
    vectors = {
        domain: context.derive(domain, 0, 32).hex()
        for domain in load_protocol()["entropy"]["domains_in_order"]
    }
    return {
        "schema": "exp33ae.dev_only_entropy_golden_vectors.v1",
        "namespace": DEV_NAMESPACE,
        "stage": DEV_STAGE,
        "claim_bearing": False,
        "master_hex": GOLDEN_MASTER_HEX,
        "commitment": context.commitment,
        "model_seeds": list(context.model_seeds()),
        "vectors": vectors,
        "confirmation_objects_created": 0,
        "confirmation_objects_accessed": 0,
    }
