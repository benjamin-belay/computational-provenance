#!/usr/bin/env python3
"""Independent static reconstruction for CSW-AM-SCB-M1-REC-S0."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol/csw-am-scb-m1-rec-s0/RECEIPT_PROTOCOL.json"
SPLIT = ROOT / "protocol/csw-am-scb-m1-rec-s0/SPLIT_SPEC.json"


def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("object required")
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line, object_pairs_hook=pairs) for line in path.read_text().splitlines()]


def encoded(value: Any) -> bytes:
    if isinstance(value, float):
        raise ValueError("floats prohibited")
    if isinstance(value, list):
        for item in value:
            encoded(item)
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("string keys required")
            encoded(item)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def hash_value(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def purpose_key(master: bytes, purpose: str) -> bytes:
    return hmac.new(
        master, encoded(["CSW-AM-SCB-M1-REC", purpose, "V1"]), hashlib.sha256
    ).digest()


def receipt_decision(
    master: bytes, envelope: dict[str, Any] | None, kind: str,
    protocol: dict[str, Any],
) -> str:
    if envelope is None:
        return "ABSTAIN"
    try:
        if set(envelope) != {"schema", "receipt_type", "preimage", "receipt_id", "tag"}:
            return "REJECT"
        preimage = envelope["preimage"]
        definition = protocol["receipt_types"][kind]
        if envelope["receipt_type"] != kind or set(preimage) != set(definition["required_fields"]):
            return "REJECT"
        if preimage["protocol_id"] != protocol["identity"]:
            return "REJECT"
        expected_id = hash_value([definition["domain"], preimage])
        expected_tag = hmac.new(
            purpose_key(master, kind), encoded(preimage), hashlib.sha256
        ).hexdigest()
        if envelope["receipt_id"] != expected_id or not hmac.compare_digest(envelope["tag"], expected_tag):
            return "REJECT"
        for key in ("predicted_z2", "consumed_z2", "intervention_target", "generated_z3", "consumed_z3", "category"):
            if key in preimage and (type(preimage[key]) is not int or not 0 <= preimage[key] < 16):
                return "REJECT"
        return "ACCEPT"
    except (KeyError, TypeError, ValueError):
        return "REJECT"


def linked(
    master: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
    abstract_kind: str, exact_kind: str, protocol: dict[str, Any],
) -> str:
    decisions = (
        receipt_decision(master, abstract, abstract_kind, protocol),
        receipt_decision(master, exact, exact_kind, protocol),
    )
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "REJECT" in decisions:
        return "REJECT"
    assert abstract is not None and exact is not None
    category_key = "consumed_z2" if exact_kind == "EXACT_Z2" else "consumed_z3"
    if (
        abstract["preimage"]["exact_receipt_id"] != exact["receipt_id"]
        or abstract["preimage"]["category"] != exact["preimage"][category_key]
    ):
        return "REJECT"
    return "ACCEPT"


def compatibility(
    master: bytes, left: dict[str, Any], right: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    if (
        linked(master, left["abstract_z2"], left["exact_z2"], "ABSTRACT_Z2", "EXACT_Z2", protocol) != "ACCEPT"
        or linked(master, right["abstract_z2"], right["exact_z2"], "ABSTRACT_Z2", "EXACT_Z2", protocol) != "ACCEPT"
    ):
        return "REJECT"
    fields = protocol["state_ontology"]["abstract_compatibility_fields"]
    return (
        "ACCEPT"
        if all(left["abstract_z2"]["preimage"][key] == right["abstract_z2"]["preimage"][key] for key in fields)
        else "REJECT"
    )


def chain_decision(
    master: bytes, bundle: dict[str, dict[str, Any] | None],
    protocol: dict[str, Any],
) -> str:
    if any(bundle.get(key) is None for key in ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")):
        return "ABSTAIN"
    z2, a2 = bundle["exact_z2"], bundle["abstract_z2"]
    z3, a3, chain = bundle["exact_z3"], bundle["abstract_z3"], bundle["chain"]
    assert z2 and a2 and z3 and a3 and chain
    if (
        linked(master, a2, z2, "ABSTRACT_Z2", "EXACT_Z2", protocol) != "ACCEPT"
        or linked(master, a3, z3, "ABSTRACT_Z3", "EXACT_Z3", protocol) != "ACCEPT"
        or receipt_decision(master, chain, "CHAIN", protocol) != "ACCEPT"
    ):
        return "REJECT"
    zp, tp, cp = z2["preimage"], z3["preimage"], chain["preimage"]
    binding = all(zp[key] == tp[key] == cp[key] for key in ("run_id", "context_id", "execution_id", "nonce"))
    linkage = (
        tp["parent_exact_z2_receipt_id"] == z2["receipt_id"]
        and cp["exact_z2_receipt_id"] == z2["receipt_id"]
        and cp["abstract_z2_receipt_id"] == a2["receipt_id"]
        and cp["exact_z3_receipt_id"] == z3["receipt_id"]
        and cp["abstract_z3_receipt_id"] == a3["receipt_id"]
    )
    order = cp["z2_sequence"] == 10 and cp["z3_sequence"] == 20 and cp["answer_sequence"] == 30
    expected_z3 = (5 * zp["consumed_z2"] + tp["transition_input_d"]) % 16
    transition = tp["consumed_z2"] == zp["consumed_z2"] and tp["generated_z3"] == tp["consumed_z3"] == expected_z3
    answer = cp["answer_digest"] == hash_value(["ANSWER", expected_z3 % 8])
    modules = tp["transition_module_id"] == cp["transition_module_id"] == "SCB_M1_Z3_MODULE_V1"
    return "ACCEPT" if all((binding, linkage, order, transition, answer, modules)) else "REJECT"


def leaf_root(domain: str, values: list[dict[str, Any]]) -> str:
    level = [
        hashlib.sha256(domain.encode() + b"\x00LEAF\x00" + encoded(row)).digest()
        for row in sorted(values, key=lambda row: row["closure_id"])
    ]
    if not level:
        raise ValueError("empty tree")
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(domain.encode() + b"\x00NODE\x00" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def audit_allocation(
    public: Path, sealed: Path, source_allocation: Path,
) -> dict[str, Any]:
    split = load(SPLIT)
    commitments = load(public / "PROTECTED_ROLE_COMMITMENTS.json")
    terminal = load(public / "TERMINAL.json")
    allocation = rows(sealed / "FULL_ALLOCATION.jsonl")
    source = [row for row in rows(source_allocation) if row["role"] == "SCB_UNTOUCHED_RESERVE"]
    identities = [row["closure_id"] for row in allocation]
    counts = Counter(row["role"] for row in allocation)
    roots = {
        role: leaf_root(
            "EXP33A|CSW-AM-SCB-M1-REC-S0|" + role,
            [row for row in allocation if row["role"] == role],
        )
        for role in split["roles"]
    }
    checks = {
        "terminal_ready": terminal["terminal"] == "CSW_AM_SCB_M1_REC_S0_PROTOCOL_AND_ALLOCATION_READY",
        "exact_800": len(allocation) == 800,
        "unique": len(set(identities)) == 800,
        "set_equality": set(identities) == {row["closure_id"] for row in source},
        "quotas": counts == Counter(split["roles"]),
        "roots": all(
            roots[role] == commitments["roles"][role]["merkle_root"]
            for role in split["roles"]
        ),
        "no_release": terminal["protected_identities_released"] == 0,
        "no_carrier": terminal["carrier_authorized"] is False,
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.reference_audit.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "role_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--source-allocation", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_allocation(args.public, args.sealed, args.source_allocation)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
