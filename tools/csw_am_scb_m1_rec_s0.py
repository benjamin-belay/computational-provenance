#!/usr/bin/env python3
"""Static receipt protocol and one-shot sealed reserve allocation for SCB-M1."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "protocol/csw-am-scb-m1-rec-s0"
CONFIG_PATH = PROTOCOL_ROOT / "CONFIG.json"
RECEIPT_PATH = PROTOCOL_ROOT / "RECEIPT_PROTOCOL.json"
SPLIT_PATH = PROTOCOL_ROOT / "SPLIT_SPEC.json"
THREAT_PATH = PROTOCOL_ROOT / "THREAT_MODEL.json"


def _validate_json_value(value: Any) -> None:
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError("INTEGER_OUT_OF_RANGE")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("NON_STRING_KEY")
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError("NON_CANONICAL_TYPE:" + type(value).__name__)


def canonical(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("DUPLICATE_KEY:" + key)
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line, object_pairs_hook=strict_pairs)
        if type(value) is not dict:
            raise ValueError("ROW_REQUIRED:" + path.name)
        result.append(value)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_source(source_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    protocol = read_json(RECEIPT_PATH)
    split = read_json(SPLIT_PATH)
    threat = read_json(THREAT_PATH)
    if (
        config["identity"] != "CSW-AM-SCB-M1-REC-S0"
        or protocol["identity"] != "SCB_M1_EXACT_ABSTRACT_CHAIN_RECEIPTS_V1"
        or split["source_closures"] != 800
        or "malicious receipt issuer" not in threat["issuer_trust_limitation"]
    ):
        raise ValueError("PROTOCOL_IDENTITY_OR_SCOPE")
    if len(source_commit) != 40:
        raise ValueError("SOURCE_COMMIT")
    for relative in config["source_file_paths"]:
        if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    audit = ROOT / config["lineage"]["M1_main_audit_path"]
    if file_sha(audit) != config["lineage"]["M1_main_audit_sha256"]:
        raise ValueError("M1_AUDIT_HASH")
    audit_value = read_json(audit)
    if (
        audit_value["main_verdict"]
        != config["lineage"]["M1_main_terminal"]
        or config["lineage"]["T1_result_superseded"] is not False
    ):
        raise ValueError("M1_LINEAGE_BOUNDARY")
    return config, protocol


def derived_key(master: bytes, purpose: str) -> bytes:
    return hmac.new(
        master,
        canonical(["CSW-AM-SCB-M1-REC", purpose, "V1"]),
        hashlib.sha256,
    ).digest()


def receipt_id(receipt_type: str, preimage: dict[str, Any], protocol: dict[str, Any]) -> str:
    domain = protocol["receipt_types"][receipt_type]["domain"]
    return digest([domain, preimage])


def validate_preimage(
    receipt_type: str, preimage: dict[str, Any], protocol: dict[str, Any],
) -> None:
    if receipt_type not in protocol["receipt_types"]:
        raise ValueError("UNKNOWN_RECEIPT_TYPE")
    required = set(protocol["receipt_types"][receipt_type]["required_fields"])
    if set(preimage) != required:
        raise ValueError("RECEIPT_FIELDS")
    _validate_json_value(preimage)
    if preimage["protocol_id"] != protocol["identity"]:
        raise ValueError("PROTOCOL_PURPOSE")
    if receipt_type in {"EXACT_Z2", "EXACT_Z3"}:
        expected_event = "Z2_CONSUMED" if receipt_type == "EXACT_Z2" else "Z3_CONSUMED"
        expected_sequence = protocol["event_order"][expected_event]
        if preimage["event_type"] != expected_event or preimage["sequence"] != expected_sequence:
            raise ValueError("EVENT_TYPE_OR_SEQUENCE")
    for key in ("predicted_z2", "consumed_z2", "intervention_target", "generated_z3", "consumed_z3", "category"):
        if key in preimage and not (type(preimage[key]) is int and 0 <= preimage[key] <= 15):
            raise ValueError("STATE_CATEGORY")
    if receipt_type == "ABSTRACT_Z2" and preimage["state_type"] != "Z2":
        raise ValueError("STATE_TYPE")
    if receipt_type == "ABSTRACT_Z3" and preimage["state_type"] != "Z3":
        raise ValueError("STATE_TYPE")


def issue_receipt(
    master: bytes, receipt_type: str, preimage: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    validate_preimage(receipt_type, preimage, protocol)
    rid = receipt_id(receipt_type, preimage, protocol)
    tag = hmac.new(derived_key(master, receipt_type), canonical(preimage), hashlib.sha256).hexdigest()
    return {
        "schema": "exp33a.csw_am_scb_m1_rec.receipt_envelope.v1",
        "receipt_type": receipt_type,
        "preimage": preimage,
        "receipt_id": rid,
        "tag": tag,
    }


def verify_receipt(
    master: bytes, envelope: dict[str, Any] | None, receipt_type: str,
    protocol: dict[str, Any], expected: dict[str, Any] | None = None,
) -> str:
    if envelope is None:
        return "ABSTAIN"
    try:
        if set(envelope) != {"schema", "receipt_type", "preimage", "receipt_id", "tag"}:
            return "REJECT"
        if (
            envelope["schema"] != "exp33a.csw_am_scb_m1_rec.receipt_envelope.v1"
            or envelope["receipt_type"] != receipt_type
            or type(envelope["preimage"]) is not dict
        ):
            return "REJECT"
        validate_preimage(receipt_type, envelope["preimage"], protocol)
        if envelope["receipt_id"] != receipt_id(receipt_type, envelope["preimage"], protocol):
            return "REJECT"
        tag = hmac.new(
            derived_key(master, receipt_type),
            canonical(envelope["preimage"]),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(envelope["tag"], tag):
            return "REJECT"
        if expected and any(envelope["preimage"].get(key) != value for key, value in expected.items()):
            return "REJECT"
        return "ACCEPT"
    except (KeyError, TypeError, ValueError):
        return "REJECT"


def verify_abstract_link(
    master: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
    abstract_type: str, exact_type: str, protocol: dict[str, Any],
) -> str:
    abstract_decision = verify_receipt(master, abstract, abstract_type, protocol)
    exact_decision = verify_receipt(master, exact, exact_type, protocol)
    if "ABSTAIN" in (abstract_decision, exact_decision):
        return "ABSTAIN"
    if "REJECT" in (abstract_decision, exact_decision):
        return "REJECT"
    assert abstract is not None and exact is not None
    state_key = "consumed_z2" if exact_type == "EXACT_Z2" else "consumed_z3"
    if (
        abstract["preimage"]["exact_receipt_id"] != exact["receipt_id"]
        or abstract["preimage"]["category"] != exact["preimage"][state_key]
    ):
        return "REJECT"
    return "ACCEPT"


def abstract_compatible(
    master: bytes,
    left_abstract: dict[str, Any] | None,
    left_exact: dict[str, Any] | None,
    right_abstract: dict[str, Any] | None,
    right_exact: dict[str, Any] | None,
    abstract_type: str,
    exact_type: str,
    protocol: dict[str, Any],
) -> str:
    decisions = (
        verify_abstract_link(master, left_abstract, left_exact, abstract_type, exact_type, protocol),
        verify_abstract_link(master, right_abstract, right_exact, abstract_type, exact_type, protocol),
    )
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "REJECT" in decisions:
        return "REJECT"
    assert left_abstract is not None and right_abstract is not None
    fields = protocol["state_ontology"]["abstract_compatibility_fields"]
    return (
        "ACCEPT"
        if all(left_abstract["preimage"][key] == right_abstract["preimage"][key] for key in fields)
        else "REJECT"
    )


def make_receipt_set(
    master: bytes, protocol: dict[str, Any], execution: str, z2: int, d: int,
    role: str = "NATURAL", run: str = "RUN-A", context: str = "CTX-A",
    nonce: str = "NONCE-A",
) -> dict[str, dict[str, Any]]:
    ontology = protocol["state_ontology"]
    z3 = (5 * z2 + d) % 16
    checkpoint = ontology["checkpoint_sha256"]
    exact_z2 = issue_receipt(master, "EXACT_Z2", {
        "schema": "exp33a.csw_am_scb_m1_rec.exact_z2.v1",
        "protocol_id": protocol["identity"],
        "model_id": "MODEL_1",
        "checkpoint_sha256": checkpoint,
        "run_id": run,
        "context_id": context,
        "execution_id": execution,
        "nonce": nonce,
        "sequence": 10,
        "event_type": "Z2_CONSUMED",
        "role": role,
        "predicted_z2": z2 if role == "NATURAL" else (z2 - 8) % 16,
        "consumed_z2": z2,
        "intervention_family": role,
        "intervention_target": z2,
    }, protocol)
    abstract_z2 = issue_receipt(master, "ABSTRACT_Z2", {
        "schema": "exp33a.csw_am_scb_m1_rec.abstract_z2.v1",
        "protocol_id": protocol["identity"],
        "state_ontology_id": ontology["identity"],
        "model_domain": ontology["model_domain"],
        "checkpoint_sha256": checkpoint,
        "state_type": "Z2",
        "category": z2,
        "exact_receipt_id": exact_z2["receipt_id"],
    }, protocol)
    exact_z3 = issue_receipt(master, "EXACT_Z3", {
        "schema": "exp33a.csw_am_scb_m1_rec.exact_z3.v1",
        "protocol_id": protocol["identity"],
        "model_id": "MODEL_1",
        "checkpoint_sha256": checkpoint,
        "run_id": run,
        "context_id": context,
        "execution_id": execution,
        "nonce": nonce,
        "sequence": 20,
        "event_type": "Z3_CONSUMED",
        "parent_exact_z2_receipt_id": exact_z2["receipt_id"],
        "consumed_z2": z2,
        "transition_input_d": d,
        "transition_module_id": "SCB_M1_Z3_MODULE_V1",
        "generated_z3": z3,
        "consumed_z3": z3,
    }, protocol)
    abstract_z3 = issue_receipt(master, "ABSTRACT_Z3", {
        "schema": "exp33a.csw_am_scb_m1_rec.abstract_z3.v1",
        "protocol_id": protocol["identity"],
        "state_ontology_id": ontology["identity"],
        "model_domain": ontology["model_domain"],
        "checkpoint_sha256": checkpoint,
        "state_type": "Z3",
        "category": z3,
        "exact_receipt_id": exact_z3["receipt_id"],
    }, protocol)
    answer_digest = digest(["ANSWER", z3 % 8])
    chain = issue_receipt(master, "CHAIN", {
        "schema": "exp33a.csw_am_scb_m1_rec.chain.v1",
        "protocol_id": protocol["identity"],
        "run_id": run,
        "context_id": context,
        "execution_id": execution,
        "nonce": nonce,
        "exact_z2_receipt_id": exact_z2["receipt_id"],
        "abstract_z2_receipt_id": abstract_z2["receipt_id"],
        "exact_z3_receipt_id": exact_z3["receipt_id"],
        "abstract_z3_receipt_id": abstract_z3["receipt_id"],
        "answer_digest": answer_digest,
        "transition_module_id": "SCB_M1_Z3_MODULE_V1",
        "answer_module_id": "SCB_M1_ANSWER_MODULE_V1",
        "z2_sequence": 10,
        "z3_sequence": 20,
        "answer_sequence": 30,
    }, protocol)
    return {
        "exact_z2": exact_z2, "abstract_z2": abstract_z2,
        "exact_z3": exact_z3, "abstract_z3": abstract_z3, "chain": chain,
    }


def verify_chain(
    master: bytes, receipts: dict[str, dict[str, Any] | None],
    protocol: dict[str, Any],
) -> str:
    required = ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")
    if any(receipts.get(key) is None for key in required):
        return "ABSTAIN"
    assert all(receipts.get(key) is not None for key in required)
    ez2 = receipts["exact_z2"]
    az2 = receipts["abstract_z2"]
    ez3 = receipts["exact_z3"]
    az3 = receipts["abstract_z3"]
    chain = receipts["chain"]
    assert ez2 and az2 and ez3 and az3 and chain
    if (
        verify_abstract_link(master, az2, ez2, "ABSTRACT_Z2", "EXACT_Z2", protocol) != "ACCEPT"
        or verify_abstract_link(master, az3, ez3, "ABSTRACT_Z3", "EXACT_Z3", protocol) != "ACCEPT"
        or verify_receipt(master, chain, "CHAIN", protocol) != "ACCEPT"
    ):
        return "REJECT"
    c, z2, z3 = chain["preimage"], ez2["preimage"], ez3["preimage"]
    same_binding = all(
        z2[key] == z3[key] == c[key]
        for key in ("run_id", "context_id", "execution_id", "nonce")
    )
    linkage = (
        z3["parent_exact_z2_receipt_id"] == ez2["receipt_id"]
        and c["exact_z2_receipt_id"] == ez2["receipt_id"]
        and c["abstract_z2_receipt_id"] == az2["receipt_id"]
        and c["exact_z3_receipt_id"] == ez3["receipt_id"]
        and c["abstract_z3_receipt_id"] == az3["receipt_id"]
    )
    ordering = c["z2_sequence"] == 10 < c["z3_sequence"] == 20 < c["answer_sequence"] == 30
    expected_z3 = (5 * z2["consumed_z2"] + z3["transition_input_d"]) % 16
    mapping = z3["consumed_z2"] == z2["consumed_z2"] and z3["generated_z3"] == z3["consumed_z3"] == expected_z3
    answer = c["answer_digest"] == digest(["ANSWER", expected_z3 % 8])
    modules = (
        z3["transition_module_id"] == c["transition_module_id"] == "SCB_M1_Z3_MODULE_V1"
        and c["answer_module_id"] == "SCB_M1_ANSWER_MODULE_V1"
    )
    return "ACCEPT" if all((same_binding, linkage, ordering, mapping, answer, modules)) else "REJECT"


def preflight_results(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or read_json(RECEIPT_PATH)
    master = bytes(range(32))
    first = make_receipt_set(master, protocol, "EXEC-A", 3, 4)
    same = make_receipt_set(master, protocol, "EXEC-B", 3, 4, context="CTX-B", nonce="NONCE-B")
    different = make_receipt_set(master, protocol, "EXEC-C", 11, 4, role="ALIAS", context="CTX-C")
    crossed = dict(first)
    crossed["exact_z3"] = different["exact_z3"]
    crossed["abstract_z3"] = different["abstract_z3"]
    wrong_answer = json.loads(json.dumps(first))
    wrong_answer["chain"]["preimage"]["answer_digest"] = digest(["ANSWER", 7])
    missing = dict(first)
    missing["chain"] = None
    mutated = json.loads(json.dumps(first["exact_z2"]))
    mutated["preimage"]["consumed_z2"] = 4
    unknown = json.loads(json.dumps(first["exact_z2"]))
    unknown["preimage"]["extra"] = "NO"
    outcomes = {
        "valid_exact_z2": verify_receipt(master, first["exact_z2"], "EXACT_Z2", protocol),
        "valid_abstract_z2_link": verify_abstract_link(
            master, first["abstract_z2"], first["exact_z2"], "ABSTRACT_Z2", "EXACT_Z2", protocol
        ),
        "valid_chain": verify_chain(master, first, protocol),
        "same_state_different_exact": verify_receipt(
            master, same["exact_z2"], "EXACT_Z2", protocol,
            {"execution_id": first["exact_z2"]["preimage"]["execution_id"]},
        ),
        "same_state_abstract_compatible": abstract_compatible(
            master, first["abstract_z2"], first["exact_z2"],
            same["abstract_z2"], same["exact_z2"], "ABSTRACT_Z2", "EXACT_Z2", protocol,
        ),
        "different_state_abstract_incompatible": abstract_compatible(
            master, first["abstract_z2"], first["exact_z2"],
            different["abstract_z2"], different["exact_z2"], "ABSTRACT_Z2", "EXACT_Z2", protocol,
        ),
        "crossed_chain": verify_chain(master, crossed, protocol),
        "wrong_answer_digest": verify_chain(master, wrong_answer, protocol),
        "missing_chain": verify_chain(master, missing, protocol),
        "mutated_exact": verify_receipt(master, mutated, "EXACT_Z2", protocol),
        "unknown_field": verify_receipt(master, unknown, "EXACT_Z2", protocol),
        "missing_exact": verify_receipt(master, None, "EXACT_Z2", protocol),
    }
    expected = {
        "valid_exact_z2": "ACCEPT",
        "valid_abstract_z2_link": "ACCEPT",
        "valid_chain": "ACCEPT",
        "same_state_different_exact": "REJECT",
        "same_state_abstract_compatible": "ACCEPT",
        "different_state_abstract_incompatible": "REJECT",
        "crossed_chain": "REJECT",
        "wrong_answer_digest": "REJECT",
        "missing_chain": "ABSTAIN",
        "mutated_exact": "REJECT",
        "unknown_field": "REJECT",
        "missing_exact": "ABSTAIN",
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.preflight.v1",
        "synthetic_only": True,
        "outcomes": outcomes,
        "expected": expected,
        "passed": outcomes == expected,
    }


def merkle_root(domain: str, rows: list[dict[str, Any]]) -> str:
    nodes = [
        hashlib.sha256(domain.encode() + b"\x00LEAF\x00" + canonical(row)).digest()
        for row in sorted(rows, key=lambda value: value["closure_id"])
    ]
    if not nodes:
        raise ValueError("EMPTY_MERKLE")
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(domain.encode() + b"\x00NODE\x00" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def read_seed(path: Path, size: int = 32) -> bytes:
    value = path.read_bytes()
    if len(value) != size:
        raise ValueError("SEED_LENGTH")
    return value


def seed_commitment(seed: bytes, split: dict[str, Any]) -> str:
    domain = split["allocation"]["commitment_domain"].encode()
    return sha(domain + b"\x00" + seed)


def reserve_rows(
    allocation_path: Path, manifest_path: Path, config: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        file_sha(allocation_path) != config["reserve_binding"]["R0_full_allocation_sha256"]
        or file_sha(manifest_path) != config["reserve_binding"]["R0_sealed_manifest_sha256"]
    ):
        raise ValueError("R0_SEALED_BINDING")
    manifest = read_json(manifest_path)
    rows = [
        row for row in read_rows(allocation_path)
        if row["role"] == config["reserve_binding"]["source_role"]
    ]
    if len(rows) != config["reserve_binding"]["source_closures"]:
        raise ValueError("RESERVE_DENOMINATOR")
    root = merkle_root("EXP33A|CSW-AM-SCB-R0|SCB_UNTOUCHED_RESERVE", rows)
    if (
        root != config["reserve_binding"]["source_merkle_root"]
        or root != manifest["role_commitments"]["SCB_UNTOUCHED_RESERVE"]["merkle_root"]
    ):
        raise ValueError("RESERVE_ROOT")
    if len({row["closure_id"] for row in rows}) != len(rows):
        raise ValueError("DUPLICATE_CLOSURE")
    return rows


def balance_key(row: dict[str, Any], feature: str) -> int:
    if feature == "natural_z2":
        return row["natural"]["z2"]
    if feature == "answer":
        return row["natural"]["answer"]
    if feature == "template_id":
        return row["template_id"]
    raise ValueError("BALANCE_FEATURE")


def allocate_rows(seed: bytes, rows: list[dict[str, Any]], split: dict[str, Any]) -> list[dict[str, Any]]:
    remaining = list(rows)
    result = []
    domains = split["allocation"]
    source_counts = {
        feature: Counter(balance_key(row, feature) for row in rows)
        for feature in split["balance_features"]
    }
    for role in domains["role_order"][:-1]:
        selected: list[dict[str, Any]] = []
        counts = {feature: Counter() for feature in split["balance_features"]}
        quota = split["roles"][role]
        for _ in range(split["roles"][role]):
            def score(row: dict[str, Any]) -> tuple[Any, ...]:
                marginal = sum(
                    Fraction(
                        (counts[feature][balance_key(row, feature)] + 1) * len(rows),
                        source_counts[feature][balance_key(row, feature)] * quota,
                    )
                    for feature in split["balance_features"]
                )
                rank = hmac.new(
                    seed,
                    canonical([domains["row_rank_domain"], role, row["closure_id"]]),
                    hashlib.sha256,
                ).hexdigest()
                return (marginal, rank)
            chosen = min(remaining, key=score)
            remaining.remove(chosen)
            selected.append(chosen)
            for feature in split["balance_features"]:
                counts[feature][balance_key(chosen, feature)] += 1
        for row in selected:
            result.append({
                **row,
                "schema": "exp33a.csw_am_scb_m1_rec_s0.allocation_row.v1",
                "prior_role": row["role"],
                "role": role,
            })
    final_role = domains["role_order"][-1]
    if len(remaining) != split["roles"][final_role]:
        raise ValueError("FINAL_ROLE_DENOMINATOR")
    for row in remaining:
        result.append({
            **row,
            "schema": "exp33a.csw_am_scb_m1_rec_s0.allocation_row.v1",
            "prior_role": row["role"],
            "role": final_role,
        })
    return result


def balance_summary(rows: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    passed = True
    bounds = split["balance_requirements"]
    cardinalities = {"natural_z2": 16, "answer": 8, "template_id": 4}
    maximum_ranges = {
        "natural_z2": bounds["maximum_z2_count_range"],
        "answer": bounds["maximum_answer_count_range"],
        "template_id": bounds["maximum_template_count_range"],
    }
    for role, quota in split["roles"].items():
        role_rows = [row for row in rows if row["role"] == role]
        features = {}
        role_pass = len(role_rows) == quota
        for feature in split["balance_features"]:
            values = Counter(balance_key(row, feature) for row in role_rows)
            full = {str(key): values[key] for key in range(cardinalities[feature])}
            count_range = max(full.values()) - min(full.values())
            feature_pass = all(value > 0 for value in full.values()) and count_range <= maximum_ranges[feature]
            features[feature] = {
                "counts": full,
                "count_range": count_range,
                "maximum_count_range": maximum_ranges[feature],
                "passed": feature_pass,
            }
            role_pass = role_pass and feature_pass
        summaries[role] = {
            "closure_count": len(role_rows),
            "oriented_row_count": len(role_rows) * 2,
            "features": features,
            "passed": role_pass,
        }
        passed = passed and role_pass
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.balance_summary.v1",
        "roles": summaries,
        "passed": passed,
    }


def create_seed(source_commit: str, seed_path: Path, commitment_path: Path) -> dict[str, Any]:
    config, _ = validate_source(source_commit)
    split = read_json(SPLIT_PATH)
    if seed_path.exists() or commitment_path.exists():
        raise FileExistsError("SEED_OR_COMMITMENT_EXISTS")
    seed = secrets.token_bytes(split["allocation"]["seed_bytes"])
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("xb") as handle:
        handle.write(seed)
        handle.flush()
        os.fsync(handle.fileno())
    commitment = {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.seed_commitment.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "seed_commitment": seed_commitment(seed, split),
        "seed_bytes": len(seed),
        "seed_revealed": False,
        "redraws": 0,
        "allocation_generated": False,
    }
    write_json(commitment_path, commitment)
    return commitment


def validate_commitment(
    seed: bytes, commitment_path: Path, commitment_commit: str,
    source_commit: str, split: dict[str, Any],
) -> dict[str, Any]:
    relative = str(commitment_path.resolve().relative_to(ROOT))
    if git_show(commitment_commit, relative) != commitment_path.read_bytes():
        raise ValueError("COMMITMENT_COMMIT_BINDING")
    value = read_json(commitment_path)
    if (
        value["source_commit"] != source_commit
        or value["seed_commitment"] != seed_commitment(seed, split)
        or value["seed_revealed"] is not False
        or value["redraws"] != 0
    ):
        raise ValueError("SEED_COMMITMENT")
    return value


def allocate(
    source_commit: str, commitment_commit: str, seed_path: Path,
    commitment_path: Path, former_allocation: Path, former_manifest: Path,
    public_output: Path, sealed_output: Path,
) -> dict[str, Any]:
    config, protocol = validate_source(source_commit)
    split = read_json(SPLIT_PATH)
    seed = read_seed(seed_path, split["allocation"]["seed_bytes"])
    commitment = validate_commitment(
        seed, commitment_path, commitment_commit, source_commit, split
    )
    if public_output.exists() or sealed_output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    public_r, sealed_r = public_output.resolve(), sealed_output.resolve()
    if public_r in sealed_r.parents or sealed_r in public_r.parents or ROOT in sealed_r.parents:
        raise ValueError("PUBLIC_SEALED_BOUNDARY")
    source_rows = reserve_rows(former_allocation, former_manifest, config)
    allocation = allocate_rows(seed, source_rows, split)
    if (
        len(allocation) != 800
        or {row["closure_id"] for row in allocation} != {row["closure_id"] for row in source_rows}
    ):
        raise ValueError("ALLOCATION_SET")
    balance = balance_summary(allocation, split)
    if not balance["passed"]:
        raise ValueError("ALLOCATION_BALANCE")
    preflight = preflight_results(protocol)
    if not preflight["passed"]:
        raise ValueError("PREFLIGHT")
    by_role = {
        role: [row for row in allocation if row["role"] == role]
        for role in split["roles"]
    }
    roots = {
        role: {
            "closure_count": len(values),
            "oriented_row_count": len(values) * 2,
            "merkle_root": merkle_root(
                "EXP33A|CSW-AM-SCB-M1-REC-S0|" + role, values
            ),
        }
        for role, values in by_role.items()
    }
    public_output.mkdir(parents=True)
    sealed_output.mkdir(parents=True)
    write_json(public_output / "SEED_COMMITMENT.json", commitment)
    write_json(public_output / "PREFLIGHT_RESULTS.json", preflight)
    write_json(public_output / "BALANCE_SUMMARY.json", balance)
    write_json(public_output / "PROTECTED_ROLE_COMMITMENTS.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.protected_commitments.v1",
        "source_reserve_merkle_root": config["reserve_binding"]["source_merkle_root"],
        "seed_commitment": commitment["seed_commitment"],
        "roles": roots,
        "protected_identity_fields_exported": 0,
        "seed_revealed": False,
    })
    write_json(public_output / "ZERO_WORK.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.zero_work.v1",
        **config["zero_work"],
        "synthetic_preflight_receipts": 5,
        "scientific_keys_generated": 0,
        "allocation_seed_reveals": 0,
    })
    write_json(public_output / "TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.terminal.v1",
        "terminal": config["terminals"]["ready"],
        "protocol_ready": True,
        "allocation_ready": True,
        "REC_D0_authorized": True,
        "REC_C1_authorized": False,
        "carrier_authorized": False,
        "protected_identities_released": 0,
        "scientific_receipts_issued": 0,
    })
    protected_bytes = b"".join(path.read_bytes() for path in public_output.iterdir())
    if any(row["closure_id"].encode() in protected_bytes for row in allocation):
        raise ValueError("PROTECTED_IDENTITY_LEAK")
    files = {
        path.name: file_sha(path)
        for path in sorted(public_output.iterdir()) if path.is_file()
    }
    write_json(public_output / "MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.manifest.v1",
        "source_commit": source_commit,
        "commitment_commit": commitment_commit,
        "files": files,
        "complete": True,
    })
    write_rows(sealed_output / "FULL_ALLOCATION.jsonl", allocation)
    write_json(sealed_output / "SEALED_MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_s0.sealed_manifest.v1",
        "source_commit": source_commit,
        "commitment_commit": commitment_commit,
        "source_allocation_sha256": file_sha(former_allocation),
        "source_manifest_sha256": file_sha(former_manifest),
        "seed_commitment": commitment["seed_commitment"],
        "full_allocation_sha256": file_sha(sealed_output / "FULL_ALLOCATION.jsonl"),
        "source_set_digest": digest(sorted(row["closure_id"] for row in source_rows)),
        "allocated_set_digest": digest(sorted(row["closure_id"] for row in allocation)),
        "set_equality": True,
        "role_commitments": roots,
        "seed_reveal_policy": split["seed_reveal_policy"],
        "complete": True,
    })
    return {
        "terminal": config["terminals"]["ready"],
        "closures": len(allocation),
        "role_counts": {role: len(values) for role, values in by_role.items()},
        "protected_identities_released": 0,
        "REC_D0_authorized": True,
        "REC_C1_authorized": False,
        "carrier_authorized": False,
    }


def dry_run_allocation(
    former_allocation: Path, former_manifest: Path, seed_hex: str,
) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    split = read_json(SPLIT_PATH)
    rows = reserve_rows(former_allocation, former_manifest, config)
    allocation = allocate_rows(bytes.fromhex(seed_hex), rows, split)
    return balance_summary(allocation, split)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--former-allocation", type=Path)
    pre.add_argument("--former-manifest", type=Path)
    pre.add_argument("--test-seed-hex", default="00" * 32)
    seed = sub.add_parser("create-seed")
    seed.add_argument("--source-commit", required=True)
    seed.add_argument("--seed-out", type=Path, required=True)
    seed.add_argument("--commitment-out", type=Path, required=True)
    run = sub.add_parser("allocate")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--seed", type=Path, required=True)
    run.add_argument("--commitment", type=Path, required=True)
    run.add_argument("--former-allocation", type=Path, required=True)
    run.add_argument("--former-manifest", type=Path, required=True)
    run.add_argument("--public-output", type=Path, required=True)
    run.add_argument("--sealed-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result: dict[str, Any] = {"receipt": preflight_results()}
            if args.former_allocation and args.former_manifest:
                result["allocation"] = dry_run_allocation(
                    args.former_allocation, args.former_manifest, args.test_seed_hex
                )
                result["passed"] = result["receipt"]["passed"] and result["allocation"]["passed"]
            else:
                result["passed"] = result["receipt"]["passed"]
        elif args.command == "create-seed":
            result = create_seed(
                args.source_commit, args.seed_out, args.commitment_out
            )
        else:
            result = allocate(
                args.source_commit, args.commitment_commit, args.seed,
                args.commitment, args.former_allocation, args.former_manifest,
                args.public_output, args.sealed_output,
            )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
