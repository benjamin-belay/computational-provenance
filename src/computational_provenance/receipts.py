"""Independent validation and adjudication of the frozen experiment receipts."""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def purpose_key(master: bytes, purpose: str) -> bytes:
    return hmac.new(master, canonical(["CSW-AM-SCB-M1-REC", purpose, "V1"]), hashlib.sha256).digest()


def receipt_decision(master: bytes, envelope: dict[str, Any] | None, kind: str,
                     protocol: dict[str, Any]) -> str:
    """Validate one frozen experiment receipt without trusting released verdicts."""
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
        expected_id = digest([definition["domain"], preimage])
        expected_tag = hmac.new(purpose_key(master, kind), canonical(preimage), hashlib.sha256).hexdigest()
        if envelope["receipt_id"] != expected_id or not hmac.compare_digest(envelope["tag"], expected_tag):
            return "REJECT"
        for field in ("predicted_z2", "consumed_z2", "intervention_target", "generated_z3",
                      "consumed_z3", "category"):
            if field in preimage and (type(preimage[field]) is not int or not 0 <= preimage[field] < 16):
                return "REJECT"
        return "ACCEPT"
    except (KeyError, TypeError, ValueError):
        return "REJECT"


def linked(master: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
           abstract_kind: str, exact_kind: str, protocol: dict[str, Any]) -> str:
    decisions = receipt_decision(master, abstract, abstract_kind, protocol), receipt_decision(master, exact, exact_kind, protocol)
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "REJECT" in decisions:
        return "REJECT"
    assert abstract is not None and exact is not None
    category_field = "consumed_z2" if exact_kind == "EXACT_Z2" else "consumed_z3"
    return "ACCEPT" if (abstract["preimage"]["exact_receipt_id"] == exact["receipt_id"]
                         and abstract["preimage"]["category"] == exact["preimage"][category_field]) else "REJECT"


def exact_decision(master: bytes, receipt: dict[str, Any] | None, kind: str,
                   protocol: dict[str, Any], expected: dict[str, Any]) -> str:
    decision = receipt_decision(master, receipt, kind, protocol)
    if decision != "ACCEPT":
        return decision
    assert receipt is not None
    return "ACCEPT" if all(receipt["preimage"].get(field) == value for field, value in expected.items()) else "REJECT"


def abstract_decision(master: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
                      abstract_kind: str, exact_kind: str, protocol: dict[str, Any]) -> str:
    decision = linked(master, abstract, exact, abstract_kind, exact_kind, protocol)
    if decision != "ACCEPT":
        return decision
    assert abstract is not None
    ontology, fields = protocol["state_ontology"], abstract["preimage"]
    return "ACCEPT" if (fields["state_ontology_id"] == ontology["identity"]
                         and fields["model_domain"] == ontology["model_domain"]
                         and fields["checkpoint_sha256"] == ontology["checkpoint_sha256"]) else "REJECT"


def chain_decision(master: bytes, bundle: dict[str, dict[str, Any] | None], protocol: dict[str, Any]) -> str:
    names = ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")
    if any(bundle.get(name) is None for name in names):
        return "ABSTAIN"
    z2, a2, z3, a3, chain = (bundle[name] for name in names)
    assert z2 and a2 and z3 and a3 and chain
    if (linked(master, a2, z2, "ABSTRACT_Z2", "EXACT_Z2", protocol) != "ACCEPT"
            or linked(master, a3, z3, "ABSTRACT_Z3", "EXACT_Z3", protocol) != "ACCEPT"
            or receipt_decision(master, chain, "CHAIN", protocol) != "ACCEPT"):
        return "REJECT"
    zp, tp, cp = z2["preimage"], z3["preimage"], chain["preimage"]
    binding = all(zp[key] == tp[key] == cp[key] for key in ("run_id", "context_id", "execution_id", "nonce"))
    linkage = (tp["parent_exact_z2_receipt_id"] == z2["receipt_id"]
               and cp["exact_z2_receipt_id"] == z2["receipt_id"]
               and cp["abstract_z2_receipt_id"] == a2["receipt_id"]
               and cp["exact_z3_receipt_id"] == z3["receipt_id"]
               and cp["abstract_z3_receipt_id"] == a3["receipt_id"])
    order = cp["z2_sequence"] == 10 and cp["z3_sequence"] == 20 and cp["answer_sequence"] == 30
    expected_z3 = (5 * zp["consumed_z2"] + tp["transition_input_d"]) % 16
    transition = (tp["consumed_z2"] == zp["consumed_z2"]
                  and tp["generated_z3"] == tp["consumed_z3"] == expected_z3)
    answer = cp["answer_digest"] == digest(["ANSWER", expected_z3 % 8])
    expected_module = protocol["state_ontology"]["transition_module_id"]
    modules = tp["transition_module_id"] == cp["transition_module_id"] == expected_module
    return "ACCEPT" if all((binding, linkage, order, transition, answer, modules)) else "REJECT"


def evaluate_bundle(master: bytes, bundle: dict[str, dict[str, Any] | None], expected: dict[str, Any],
                    protocol: dict[str, Any]) -> dict[str, str]:
    ez2, az2 = bundle.get("exact_z2"), bundle.get("abstract_z2")
    ez3, az3, chain = bundle.get("exact_z3"), bundle.get("abstract_z3"), bundle.get("chain")
    exact_z2 = exact_decision(master, ez2, "EXACT_Z2", protocol, expected)
    exact_z3 = exact_decision(master, ez3, "EXACT_Z3", protocol, expected)
    abstract_z2 = abstract_decision(master, az2, ez2, "ABSTRACT_Z2", "EXACT_Z2", protocol)
    abstract_z3 = abstract_decision(master, az3, ez3, "ABSTRACT_Z3", "EXACT_Z3", protocol)
    chain_expected = {field: expected[field] for field in ("run_id", "context_id", "execution_id", "nonce")}
    chain_auth = exact_decision(master, chain, "CHAIN", protocol, chain_expected)
    if ez2 is None or ez3 is None:
        transition = "ABSTAIN"
    elif exact_z2 == "REJECT" or exact_z3 == "REJECT":
        transition = "REJECT"
    else:
        p2, p3 = ez2["preimage"], ez3["preimage"]
        registered = (5 * p2["consumed_z2"] + p3["transition_input_d"]) % 16
        transition = "ACCEPT" if (p3["consumed_z2"] == p2["consumed_z2"]
                                   and p3["generated_z3"] == p3["consumed_z3"] == registered) else "REJECT"
    if chain is None or ez3 is None:
        answer = "ABSTAIN"
    elif chain_auth == "REJECT" or exact_z3 == "REJECT":
        answer = "REJECT"
    else:
        answer = "ACCEPT" if chain["preimage"]["answer_digest"] == digest(["ANSWER", ez3["preimage"]["consumed_z3"] % 8]) else "REJECT"
    decisions = (exact_z2, abstract_z2, exact_z3, abstract_z3, chain_auth, transition, answer)
    consistency = "ABSTAIN" if "ABSTAIN" in decisions else ("REJECT" if "REJECT" in decisions else chain_decision(master, bundle, protocol))
    return {"exact_z2_event": exact_z2, "abstract_z2_state": abstract_z2,
            "exact_z3_event": exact_z3, "abstract_z3_state": abstract_z3,
            "chain_authenticator": chain_auth, "registered_transition_consistency": transition,
            "registered_answer_consistency": answer, "chain_consistency": consistency}
