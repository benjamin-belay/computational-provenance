"""Minimal HMAC receipt and causal-chain implementation used by the replay."""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def issue(key: bytes, payload: dict[str, Any]) -> dict[str, Any]:
    preimage = canonical(payload)
    return {"payload": payload, "hmac_sha256": hmac.new(key, preimage, hashlib.sha256).hexdigest()}


def verify(key: bytes, receipt: dict[str, Any]) -> bool:
    try:
        expected = hmac.new(key, canonical(receipt["payload"]), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, receipt["hmac_sha256"])
    except (KeyError, TypeError, ValueError):
        return False


def issue_execution(
    key: bytes, *, model_domain: str, checkpoint_sha256: str, closure_id: str,
    role: str, z2: int, z3: int, answer: int,
) -> dict[str, Any]:
    z2_payload = {"schema": "computational_provenance.abstract_z2.v1", "model_domain": model_domain,
                  "checkpoint_sha256": checkpoint_sha256, "closure_id": closure_id,
                  "role": role, "state_type": "Z2", "category": z2}
    z2_receipt = issue(key, z2_payload)
    z3_payload = {"schema": "computational_provenance.abstract_z3.v1", "model_domain": model_domain,
                  "checkpoint_sha256": checkpoint_sha256, "closure_id": closure_id,
                  "role": role, "state_type": "Z3", "category": z3,
                  "parent_receipt_sha256": digest(z2_receipt)}
    z3_receipt = issue(key, z3_payload)
    chain_payload = {"schema": "computational_provenance.chain.v1", "closure_id": closure_id,
                     "role": role, "z2_receipt_sha256": digest(z2_receipt),
                     "z3_receipt_sha256": digest(z3_receipt), "answer": answer}
    return {"z2": z2_receipt, "z3": z3_receipt, "chain": issue(key, chain_payload)}


def verify_execution(key: bytes, bundle: dict[str, Any]) -> bool:
    if not all(verify(key, bundle[name]) for name in ("z2", "z3", "chain")):
        return False
    z2, z3, chain = (bundle[name]["payload"] for name in ("z2", "z3", "chain"))
    return (
        z3["parent_receipt_sha256"] == digest(bundle["z2"])
        and chain["z2_receipt_sha256"] == digest(bundle["z2"])
        and chain["z3_receipt_sha256"] == digest(bundle["z3"])
        and z2["closure_id"] == z3["closure_id"] == chain["closure_id"]
        and z2["role"] == z3["role"] == chain["role"]
    )
