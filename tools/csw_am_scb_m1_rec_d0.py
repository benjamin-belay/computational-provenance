#!/usr/bin/env python3
"""Public receipt implementation and lifecycle rehearsal for SCB Model 1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib.util
import json
import os
import secrets
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rec-d0/CONFIG.json"
ATTACK_PATH = ROOT / "protocol/csw-am-scb-m1-rec-d0/ATTACK_MANIFEST.json"
S0_PROTOCOL_PATH = ROOT / "protocol/csw-am-scb-m1-rec-s0/RECEIPT_PROTOCOL.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE_SPEC:" + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S0 = load_module("csw_am_scb_m1_rec_s0_frozen", ROOT / "tools/csw_am_scb_m1_rec_s0.py")
M1 = load_module("csw_am_scb_m1_frozen", ROOT / "tools/csw_am_scb_m1.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
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


def write_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_source(source_commit: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    attacks = read_json(ATTACK_PATH)
    protocol = read_json(S0_PROTOCOL_PATH)
    if (
        config["identity"] != "CSW-AM-SCB-M1-REC-D0"
        or attacks["attack_families"] != len(attacks["families"])
        or protocol["identity"] != "SCB_M1_EXACT_ABSTRACT_CHAIN_RECEIPTS_V1"
    ):
        raise ValueError("IDENTITY")
    if len(source_commit) != 40:
        raise ValueError("SOURCE_COMMIT")
    for relative in config["source_file_paths"]:
        if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    s0_audit = ROOT / config["lineage"]["REC_S0_main_audit_path"]
    if (
        file_sha(s0_audit) != config["lineage"]["REC_S0_main_audit_sha256"]
        or read_json(s0_audit)["main_verdict"] != config["lineage"]["REC_S0_main_terminal"]
    ):
        raise ValueError("S0_AUDIT")
    for commit, relative in (
        (config["lineage"]["REC_S0_source_commit"], "tools/csw_am_scb_m1_rec_s0.py"),
        (config["lineage"]["M1_source_commit"], "tools/csw_am_scb_m1.py"),
    ):
        if git_show(commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("FROZEN_DEPENDENCY:" + relative)
    return config, attacks, protocol


def key_commitment(master: bytes, config: dict[str, Any]) -> str:
    return sha(config["key_lifecycle"]["commitment_domain"].encode() + b"\x00" + master)


def run_id(master: bytes, config: dict[str, Any]) -> str:
    return sha(config["key_lifecycle"]["run_id_domain"].encode() + b"\x00" + master)


def execution_nonce(master: bytes, execution_id: str, config: dict[str, Any]) -> str:
    return hmac.new(
        master,
        canonical([config["key_lifecycle"]["nonce_domain"], execution_id]),
        hashlib.sha256,
    ).hexdigest()


def validate_key_commitment(
    key: bytes, commitment_path: Path, commitment_commit: str,
    source_commit: str, config: dict[str, Any],
) -> dict[str, Any]:
    relative = str(commitment_path.resolve().relative_to(ROOT))
    if git_show(commitment_commit, relative) != commitment_path.read_bytes():
        raise ValueError("KEY_COMMITMENT_COMMIT")
    value = read_json(commitment_path)
    if (
        value["source_commit"] != source_commit
        or value["key_commitment"] != key_commitment(key, config)
        or value["run_id"] != run_id(key, config)
        or value["key_revealed"] is not False
    ):
        raise ValueError("KEY_COMMITMENT")
    return value


def create_key(
    source_commit: str, key_path: Path, commitment_path: Path,
) -> dict[str, Any]:
    config, _, _ = validate_source(source_commit)
    if key_path.exists() or commitment_path.exists():
        raise FileExistsError("KEY_OR_COMMITMENT_EXISTS")
    key = secrets.token_bytes(config["key_lifecycle"]["master_key_bytes"])
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("xb") as handle:
        handle.write(key)
        handle.flush()
        os.fsync(handle.fileno())
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.key_commitment.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "key_commitment": key_commitment(key, config),
        "run_id": run_id(key, config),
        "key_bytes": len(key),
        "key_revealed": False,
        "scientific_confirmation_key": False,
        "development_lifecycle_rehearsal": True,
    }
    write_json(commitment_path, value)
    return value


def receipt_package(
    master: bytes, protocol: dict[str, Any], config: dict[str, Any],
    row: dict[str, Any], role: str, observed: dict[str, int],
    predicted_z2: int, target_z2: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = run_id(master, config)
    execution_id = digest([
        "SCB_M1_REC_D0_EXECUTION_V1", run, row["example_id"], role,
    ])
    nonce = execution_nonce(master, execution_id, config)
    ontology = protocol["state_ontology"]
    checkpoint = ontology["checkpoint_sha256"]
    exact_z2 = S0.issue_receipt(master, "EXACT_Z2", {
        "schema": "exp33a.csw_am_scb_m1_rec.exact_z2.v1",
        "protocol_id": protocol["identity"],
        "model_id": "MODEL_1",
        "checkpoint_sha256": checkpoint,
        "run_id": run,
        "context_id": row["example_id"],
        "execution_id": execution_id,
        "nonce": nonce,
        "sequence": 10,
        "event_type": "Z2_CONSUMED",
        "role": role,
        "predicted_z2": predicted_z2,
        "consumed_z2": observed["z2_consumed"],
        "intervention_family": role,
        "intervention_target": target_z2,
    }, protocol)
    abstract_z2 = S0.issue_receipt(master, "ABSTRACT_Z2", {
        "schema": "exp33a.csw_am_scb_m1_rec.abstract_z2.v1",
        "protocol_id": protocol["identity"],
        "state_ontology_id": ontology["identity"],
        "model_domain": ontology["model_domain"],
        "checkpoint_sha256": checkpoint,
        "state_type": "Z2",
        "category": observed["z2_consumed"],
        "exact_receipt_id": exact_z2["receipt_id"],
    }, protocol)
    exact_z3 = S0.issue_receipt(master, "EXACT_Z3", {
        "schema": "exp33a.csw_am_scb_m1_rec.exact_z3.v1",
        "protocol_id": protocol["identity"],
        "model_id": "MODEL_1",
        "checkpoint_sha256": checkpoint,
        "run_id": run,
        "context_id": row["example_id"],
        "execution_id": execution_id,
        "nonce": nonce,
        "sequence": 20,
        "event_type": "Z3_CONSUMED",
        "parent_exact_z2_receipt_id": exact_z2["receipt_id"],
        "consumed_z2": observed["z2_consumed"],
        "transition_input_d": row["inputs"]["d"],
        "transition_module_id": "SCB_M1_Z3_MODULE_V1",
        "generated_z3": observed["z3_selected"],
        "consumed_z3": observed["z3_consumed"],
    }, protocol)
    abstract_z3 = S0.issue_receipt(master, "ABSTRACT_Z3", {
        "schema": "exp33a.csw_am_scb_m1_rec.abstract_z3.v1",
        "protocol_id": protocol["identity"],
        "state_ontology_id": ontology["identity"],
        "model_domain": ontology["model_domain"],
        "checkpoint_sha256": checkpoint,
        "state_type": "Z3",
        "category": observed["z3_consumed"],
        "exact_receipt_id": exact_z3["receipt_id"],
    }, protocol)
    chain = S0.issue_receipt(master, "CHAIN", {
        "schema": "exp33a.csw_am_scb_m1_rec.chain.v1",
        "protocol_id": protocol["identity"],
        "run_id": run,
        "context_id": row["example_id"],
        "execution_id": execution_id,
        "nonce": nonce,
        "exact_z2_receipt_id": exact_z2["receipt_id"],
        "abstract_z2_receipt_id": abstract_z2["receipt_id"],
        "exact_z3_receipt_id": exact_z3["receipt_id"],
        "abstract_z3_receipt_id": abstract_z3["receipt_id"],
        "answer_digest": digest(["ANSWER", observed["answer"]]),
        "transition_module_id": "SCB_M1_Z3_MODULE_V1",
        "answer_module_id": "SCB_M1_ANSWER_MODULE_V1",
        "z2_sequence": 10,
        "z3_sequence": 20,
        "answer_sequence": 30,
    }, protocol)
    primitive = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.execution.v1",
        "package_id": digest(["SCB_M1_REC_D0_PACKAGE_V1", execution_id]),
        "run_id": run,
        "context_id": row["example_id"],
        "closure_id": row["closure_id"],
        "orientation": row["orientation"],
        "execution_id": execution_id,
        "nonce": nonce,
        "role": role,
        "inputs": row["inputs"],
        "registered_natural": row["natural"],
        "predicted_natural_z2": predicted_z2,
        "intervention_target_z2": target_z2,
        "consumed_z2": observed["z2_consumed"],
        "generated_z3": observed["z3_selected"],
        "consumed_z3": observed["z3_consumed"],
        "answer": observed["answer"],
    }
    package = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.receipt_package.v1",
        "package_id": primitive["package_id"],
        "execution_id": execution_id,
        "role": role,
        "receipts": {
            "exact_z2": exact_z2,
            "abstract_z2": abstract_z2,
            "exact_z3": exact_z3,
            "abstract_z3": abstract_z3,
            "chain": chain,
        },
    }
    return primitive, package


def execute_public(
    source_commit: str, commitment_path: Path, commitment_commit: str,
    key_path: Path, output: Path,
) -> dict[str, Any]:
    config, attacks, protocol = validate_source(source_commit)
    key = key_path.read_bytes()
    commitment = validate_key_commitment(
        key, commitment_path, commitment_commit, source_commit, config
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    population = ROOT / config["public_population"]["path"]
    if file_sha(population) != config["public_population"]["sha256"]:
        raise ValueError("PUBLIC_POPULATION_HASH")
    rows = read_rows(population)
    if (
        len(rows) != config["public_population"]["oriented_rows"]
        or len({row["example_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("PUBLIC_DENOMINATOR")
    model = M1.load_model(M1.read_json(M1.CONFIG_PATH))
    primitives: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            values = M1.tensors(batch_rows)
            natural = model.run_natural(values)
            target = torch.tensor(
                [(row["natural"]["z2"] + 8) % 16 for row in batch_rows],
                dtype=torch.long,
            )
            alias = model.run_with_z2(values, target)
            natural_answers = natural["answer_logits"].argmax(-1)
            alias_answers = alias["answer_logits"].argmax(-1)
            for index, row in enumerate(batch_rows):
                predicted = int(natural["z2_selected"][index])
                natural_observed = {
                    "z2_consumed": int(natural["z2_consumed"][index]),
                    "z3_selected": int(natural["z3_selected"][index]),
                    "z3_consumed": int(natural["z3_consumed"][index]),
                    "answer": int(natural_answers[index]),
                }
                alias_observed = {
                    "z2_consumed": int(alias["z2_consumed"][index]),
                    "z3_selected": int(alias["z3_selected"][index]),
                    "z3_consumed": int(alias["z3_consumed"][index]),
                    "answer": int(alias_answers[index]),
                }
                for role, observed, target_z2 in (
                    ("NATURAL", natural_observed, predicted),
                    ("ALIAS_Z2", alias_observed, int(target[index])),
                ):
                    primitive, package = receipt_package(
                        key, protocol, config, row, role, observed, predicted, target_z2
                    )
                    primitives.append(primitive)
                    packages.append(package)
    if (
        len(primitives) != config["public_population"]["execution_packages"]
        or len(packages) != len(primitives)
    ):
        raise ValueError("EXECUTION_PACKAGE_DENOMINATOR")
    attack_rows = construct_attacks(key, primitives, packages, protocol, config, attacks)
    if len(attack_rows) != attacks["expected_attack_rows"]:
        raise ValueError("ATTACK_DENOMINATOR")
    output.mkdir(parents=True)
    write_rows(output / "EXECUTION_ROWS.jsonl", primitives)
    write_rows(output / "RECEIPT_PACKAGES.jsonl", packages)
    write_rows(output / "ATTACK_INPUTS.jsonl", attack_rows)
    write_json(output / "RAW_MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.raw_manifest.v1",
        "source_commit": source_commit,
        "key_commitment_commit": commitment_commit,
        "key_commitment": commitment["key_commitment"],
        "run_id": commitment["run_id"],
        "execution_rows": len(primitives),
        "receipt_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "attack_inputs": len(attack_rows),
        "execution_rows_sha256": file_sha(output / "EXECUTION_ROWS.jsonl"),
        "receipt_packages_sha256": file_sha(output / "RECEIPT_PACKAGES.jsonl"),
        "attack_inputs_sha256": file_sha(output / "ATTACK_INPUTS.jsonl"),
        "key_revealed": False,
        "protected_releases": 0,
        "carrier_operations": 0,
    })
    write_json(output / "RAW_TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.raw_terminal.v1",
        "terminal": "CSW_AM_SCB_M1_REC_D0_RAW_EVIDENCE_COMMITTED_PENDING_REVEAL",
        "evaluation_authorized_after_raw_commit": True,
        "D0_ready": False,
        "REC_qualification_authorized": False,
        "key_revealed": False,
    })
    return {
        "terminal": "CSW_AM_SCB_M1_REC_D0_RAW_EVIDENCE_COMMITTED_PENDING_REVEAL",
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "attack_inputs": len(attack_rows),
        "key_revealed": False,
    }


def reissue(
    key: bytes, envelope: dict[str, Any], updates: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    preimage = copy.deepcopy(envelope["preimage"])
    preimage.update(updates)
    return S0.issue_receipt(key, envelope["receipt_type"], preimage, protocol)


def attack_row(
    family: str, base: dict[str, Any], expected_binding: dict[str, Any],
    sources: dict[str, str | None], inline: dict[str, dict[str, Any]],
    reference_package_id: str | None = None,
) -> dict[str, Any]:
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.attack_input.v1",
        "family": family,
        "base_package_id": base["package_id"],
        "expected_binding": expected_binding,
        "presented_receipt_sources": sources,
        "inline_receipts": inline,
        "reference_package_id": reference_package_id,
    }
    return {"attack_id": digest(["SCB_M1_REC_D0_ATTACK_V1", value]), **value}


def construct_attacks(
    key: bytes, primitives: list[dict[str, Any]], packages: list[dict[str, Any]],
    protocol: dict[str, Any], config: dict[str, Any], attacks: dict[str, Any],
) -> list[dict[str, Any]]:
    package_index = {row["package_id"]: row for row in packages}
    primitive_index = {row["package_id"]: row for row in primitives}
    by_example_role = {
        (row["context_id"], row["role"]): row for row in primitives
    }
    by_z2: dict[int, list[dict[str, Any]]] = {}
    by_z3: dict[int, list[dict[str, Any]]] = {}
    for row in primitives:
        by_z2.setdefault(row["consumed_z2"], []).append(row)
        by_z3.setdefault(row["consumed_z3"], []).append(row)
    for values in (*by_z2.values(), *by_z3.values()):
        values.sort(key=lambda row: row["package_id"])
    bases = []
    for role in ("NATURAL", "ALIAS_Z2"):
        candidates = sorted(
            (row for row in primitives if row["role"] == role),
            key=lambda row: row["package_id"],
        )
        bases.extend(candidates[:config["public_population"]["attack_base_packages"] // 2])
    names = ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")
    result = []
    for base in bases:
        base_package = package_index[base["package_id"]]
        base_receipts = base_package["receipts"]
        default_sources = {name: base["package_id"] for name in names}
        expected = {
            key_name: base[key_name]
            for key_name in ("run_id", "context_id", "execution_id", "nonce")
        }
        expected["checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
        same_z2 = next(row for row in by_z2[base["consumed_z2"]] if row["execution_id"] != base["execution_id"])
        same_z3 = next(row for row in by_z3[base["consumed_z3"]] if row["execution_id"] != base["execution_id"])
        different = next(row for row in primitives if row["consumed_z2"] != base["consumed_z2"])
        counterpart_role = "ALIAS_Z2" if base["role"] == "NATURAL" else "NATURAL"
        counterpart = by_example_role[(base["context_id"], counterpart_role)]
        natural = base if base["role"] == "NATURAL" else counterpart
        alias = counterpart if base["role"] == "NATURAL" else base

        def sources(**changes: str | None) -> dict[str, str | None]:
            return {**default_sources, **changes}

        result.append(attack_row(
            "SAME_Z2_DIFFERENT_EXACT", base, expected,
            sources(exact_z2=same_z2["package_id"], abstract_z2=same_z2["package_id"]),
            {}, base["package_id"],
        ))
        result.append(attack_row(
            "SAME_Z3_DIFFERENT_EXACT", base, expected,
            sources(exact_z3=same_z3["package_id"], abstract_z3=same_z3["package_id"]),
            {}, base["package_id"],
        ))
        result.append(attack_row(
            "DIFFERENT_Z2_DIFFERENT_EXACT", base, expected,
            sources(exact_z2=different["package_id"], abstract_z2=different["package_id"]),
            {}, base["package_id"],
        ))
        result.append(attack_row(
            "NATURAL_Z2_ALIAS_Z3", base, expected,
            {
                "exact_z2": natural["package_id"], "abstract_z2": natural["package_id"],
                "exact_z3": alias["package_id"], "abstract_z3": alias["package_id"],
                "chain": alias["package_id"],
            }, {},
        ))
        result.append(attack_row(
            "ALIAS_Z2_NATURAL_Z3", base, expected,
            {
                "exact_z2": alias["package_id"], "abstract_z2": alias["package_id"],
                "exact_z3": natural["package_id"], "abstract_z3": natural["package_id"],
                "chain": natural["package_id"],
            }, {},
        ))
        reversed_chain = reissue(
            key, base_receipts["chain"],
            {"z2_sequence": 20, "z3_sequence": 10}, protocol,
        )
        result.append(attack_row(
            "REVERSED_EVENT_ORDER", base, expected, default_sources,
            {"chain": reversed_chain},
        ))
        duplicate_chain = reissue(
            key, base_receipts["chain"], {"z3_sequence": 10}, protocol,
        )
        result.append(attack_row(
            "DUPLICATE_EVENT_SEQUENCE", base, expected, default_sources,
            {"chain": duplicate_chain},
        ))
        wrong_parent_z3 = reissue(
            key, base_receipts["exact_z3"],
            {"parent_exact_z2_receipt_id": package_index[same_z2["package_id"]]["receipts"]["exact_z2"]["receipt_id"]},
            protocol,
        )
        wrong_parent_a3 = reissue(
            key, base_receipts["abstract_z3"],
            {"exact_receipt_id": wrong_parent_z3["receipt_id"]}, protocol,
        )
        wrong_parent_chain = reissue(
            key, base_receipts["chain"], {
                "exact_z3_receipt_id": wrong_parent_z3["receipt_id"],
                "abstract_z3_receipt_id": wrong_parent_a3["receipt_id"],
            }, protocol,
        )
        result.append(attack_row(
            "WRONG_PARENT_RECEIPT", base, expected, default_sources, {
                "exact_z3": wrong_parent_z3,
                "abstract_z3": wrong_parent_a3,
                "chain": wrong_parent_chain,
            },
        ))
        wrong_answer = reissue(
            key, base_receipts["chain"],
            {"answer_digest": digest(["ANSWER", (base["answer"] + 1) % 8])},
            protocol,
        )
        result.append(attack_row(
            "WRONG_ANSWER_DIGEST", base, expected, default_sources,
            {"chain": wrong_answer},
        ))
        result.append(attack_row(
            "EXACT_CROSSED_ABSTRACT_COMPATIBLE", base, expected,
            sources(exact_z2=same_z2["package_id"], abstract_z2=same_z2["package_id"]),
            {}, base["package_id"],
        ))
        natural_expected = {
            key_name: alias[key_name]
            for key_name in ("run_id", "context_id", "execution_id", "nonce")
        }
        natural_expected["checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
        result.append(attack_row(
            "NATURAL_PACKAGE_TRANSPLANTED_TO_ALIAS", base, natural_expected,
            {name: natural["package_id"] for name in names}, {},
        ))
        wrong_ontology = reissue(
            key, base_receipts["abstract_z2"],
            {"state_ontology_id": "WRONG_STATE_ONTOLOGY_V9"}, protocol,
        )
        result.append(attack_row(
            "WRONG_STATE_ONTOLOGY", base, expected, default_sources,
            {"abstract_z2": wrong_ontology},
        ))
        wrong_domain = copy.deepcopy(base_receipts["exact_z2"])
        wrong_domain["tag"] = hmac.new(
            S0.derived_key(key, "EXACT_Z3"),
            S0.canonical(wrong_domain["preimage"]),
            hashlib.sha256,
        ).hexdigest()
        result.append(attack_row(
            "WRONG_RECEIPT_DOMAIN", base, expected, default_sources,
            {"exact_z2": wrong_domain},
        ))
        mutation = copy.deepcopy(base_receipts["exact_z2"])
        mutation["preimage"]["consumed_z2"] = (mutation["preimage"]["consumed_z2"] + 1) % 16
        result.append(attack_row(
            "MUTATED_PRESENT_RECEIPT", base, expected, default_sources,
            {"exact_z2": mutation},
        ))
        wrong_context = dict(expected)
        wrong_context["context_id"] = counterpart["context_id"]
        wrong_context["execution_id"] = counterpart["execution_id"]
        wrong_context["nonce"] = counterpart["nonce"]
        result.append(attack_row(
            "WRONG_CONTEXT_BINDING", base, wrong_context, default_sources, {},
        ))
        result.append(attack_row(
            "MISSING_EXACT_Z2", base, expected, sources(exact_z2=None), {},
        ))
        result.append(attack_row(
            "MISSING_ABSTRACT_Z2", base, expected, sources(abstract_z2=None), {},
        ))
        result.append(attack_row(
            "MISSING_CHAIN", base, expected, sources(chain=None), {},
        ))
    if Counter(row["family"] for row in result) != Counter({
        family: len(bases) for family in attacks["families"]
    }):
        raise ValueError("ATTACK_FAMILY_DENOMINATOR")
    return result


def resolve_attack(
    attack: dict[str, Any], packages: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for component, package_id in attack["presented_receipt_sources"].items():
        result[component] = (
            None if package_id is None
            else packages[package_id]["receipts"][component]
        )
    result.update(attack["inline_receipts"])
    return result


def ontology_decision(
    key: bytes, abstract: dict[str, Any] | None, exact: dict[str, Any] | None,
    abstract_type: str, exact_type: str, protocol: dict[str, Any],
) -> str:
    decision = S0.verify_abstract_link(
        key, abstract, exact, abstract_type, exact_type, protocol
    )
    if decision != "ACCEPT":
        return decision
    assert abstract is not None
    expected = protocol["state_ontology"]
    fields = abstract["preimage"]
    return (
        "ACCEPT"
        if (
            fields["state_ontology_id"] == expected["identity"]
            and fields["model_domain"] == expected["model_domain"]
            and fields["checkpoint_sha256"] == expected["checkpoint_sha256"]
        )
        else "REJECT"
    )


def compatible(
    key: bytes, candidate: dict[str, Any] | None, candidate_exact: dict[str, Any] | None,
    reference: dict[str, Any], reference_exact: dict[str, Any],
    abstract_type: str, exact_type: str, protocol: dict[str, Any],
) -> str:
    decisions = (
        ontology_decision(key, candidate, candidate_exact, abstract_type, exact_type, protocol),
        ontology_decision(key, reference, reference_exact, abstract_type, exact_type, protocol),
    )
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "REJECT" in decisions:
        return "REJECT"
    assert candidate is not None
    fields = protocol["state_ontology"]["abstract_compatibility_fields"]
    return (
        "ACCEPT"
        if all(candidate["preimage"][field] == reference["preimage"][field] for field in fields)
        else "REJECT"
    )


def evaluate_bundle(
    key: bytes, bundle: dict[str, dict[str, Any] | None],
    expected_binding: dict[str, Any], protocol: dict[str, Any],
    reference: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    ez2, az2 = bundle.get("exact_z2"), bundle.get("abstract_z2")
    ez3, az3, chain = bundle.get("exact_z3"), bundle.get("abstract_z3"), bundle.get("chain")
    exact_z2 = S0.verify_receipt(key, ez2, "EXACT_Z2", protocol, expected_binding)
    exact_z3 = S0.verify_receipt(key, ez3, "EXACT_Z3", protocol, expected_binding)
    abstract_z2 = ontology_decision(key, az2, ez2, "ABSTRACT_Z2", "EXACT_Z2", protocol)
    abstract_z3 = ontology_decision(key, az3, ez3, "ABSTRACT_Z3", "EXACT_Z3", protocol)
    chain_auth = S0.verify_receipt(key, chain, "CHAIN", protocol, {
        key_name: expected_binding[key_name]
        for key_name in ("run_id", "context_id", "execution_id", "nonce")
    })
    if ez2 is None or ez3 is None:
        transition = "ABSTAIN"
    elif exact_z2 == "REJECT" or exact_z3 == "REJECT":
        transition = "REJECT"
    else:
        z2, z3 = ez2["preimage"], ez3["preimage"]
        expected_z3 = (5 * z2["consumed_z2"] + z3["transition_input_d"]) % 16
        transition = (
            "ACCEPT"
            if (
                z3["consumed_z2"] == z2["consumed_z2"]
                and z3["generated_z3"] == z3["consumed_z3"] == expected_z3
            )
            else "REJECT"
        )
    if chain is None or ez3 is None:
        answer = "ABSTAIN"
    elif chain_auth == "REJECT" or exact_z3 == "REJECT":
        answer = "REJECT"
    else:
        answer = (
            "ACCEPT"
            if chain["preimage"]["answer_digest"] == digest([
                "ANSWER", ez3["preimage"]["consumed_z3"] % 8
            ])
            else "REJECT"
        )
    required = (exact_z2, abstract_z2, exact_z3, abstract_z3, chain_auth, transition, answer)
    if "ABSTAIN" in required:
        chain_consistency = "ABSTAIN"
    elif "REJECT" in required:
        chain_consistency = "REJECT"
    else:
        chain_consistency = S0.verify_chain(key, bundle, protocol)
    result = {
        "exact_z2_event": exact_z2,
        "abstract_z2_state": abstract_z2,
        "exact_z3_event": exact_z3,
        "abstract_z3_state": abstract_z3,
        "chain_authenticator": chain_auth,
        "registered_transition_consistency": transition,
        "registered_answer_consistency": answer,
        "chain_consistency": chain_consistency,
    }
    if reference:
        result["abstract_z2_compatibility"] = compatible(
            key, az2, ez2, reference["abstract_z2"], reference["exact_z2"],
            "ABSTRACT_Z2", "EXACT_Z2", protocol,
        )
        result["abstract_z3_compatibility"] = compatible(
            key, az3, ez3, reference["abstract_z3"], reference["exact_z3"],
            "ABSTRACT_Z3", "EXACT_Z3", protocol,
        )
    return result


def reveal_key(
    source_commit: str, commitment_path: Path, commitment_commit: str,
    key_path: Path, raw_manifest: Path, raw_commit: str, reveal_path: Path,
) -> dict[str, Any]:
    config, _, _ = validate_source(source_commit)
    key = key_path.read_bytes()
    commitment = validate_key_commitment(
        key, commitment_path, commitment_commit, source_commit, config
    )
    relative = str(raw_manifest.resolve().relative_to(ROOT))
    if git_show(raw_commit, relative) != raw_manifest.read_bytes():
        raise ValueError("RAW_COMMIT_BINDING")
    raw = read_json(raw_manifest)
    if (
        raw["source_commit"] != source_commit
        or raw["key_commitment"] != commitment["key_commitment"]
        or raw["key_revealed"] is not False
    ):
        raise ValueError("RAW_MANIFEST")
    value = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.key_reveal.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "key_commitment_commit": commitment_commit,
        "raw_evidence_commit": raw_commit,
        "raw_manifest_sha256": file_sha(raw_manifest),
        "key_hex": key.hex(),
        "key_commitment": commitment["key_commitment"],
        "run_id": commitment["run_id"],
        "development_lifecycle_rehearsal": True,
    }
    if sha(config["key_lifecycle"]["commitment_domain"].encode() + b"\x00" + bytes.fromhex(value["key_hex"])) != value["key_commitment"]:
        raise ValueError("REVEAL_RECONSTRUCTION")
    write_json(reveal_path, value)
    return {key_name: value[key_name] for key_name in value if key_name != "key_hex"}


def evaluate(
    source_commit: str, raw: Path, raw_commit: str,
    reveal_path: Path, reveal_commit: str, output: Path,
) -> dict[str, Any]:
    config, attack_manifest, protocol = validate_source(source_commit)
    for path in (raw / "RAW_MANIFEST.json", reveal_path):
        relative = str(path.resolve().relative_to(ROOT))
        commit = raw_commit if path.name == "RAW_MANIFEST.json" else reveal_commit
        if git_show(commit, relative) != path.read_bytes():
            raise ValueError("COMMIT_BINDING:" + path.name)
    reveal = read_json(reveal_path)
    key = bytes.fromhex(reveal["key_hex"])
    raw_manifest = read_json(raw / "RAW_MANIFEST.json")
    if (
        reveal["raw_evidence_commit"] != raw_commit
        or reveal["raw_manifest_sha256"] != file_sha(raw / "RAW_MANIFEST.json")
        or reveal["key_commitment"] != raw_manifest["key_commitment"]
    ):
        raise ValueError("REVEAL_RAW_LINK")
    for name, field in (
        ("EXECUTION_ROWS.jsonl", "execution_rows_sha256"),
        ("RECEIPT_PACKAGES.jsonl", "receipt_packages_sha256"),
        ("ATTACK_INPUTS.jsonl", "attack_inputs_sha256"),
    ):
        if file_sha(raw / name) != raw_manifest[field]:
            raise ValueError("RAW_FILE_HASH:" + name)
    primitives = read_rows(raw / "EXECUTION_ROWS.jsonl")
    packages_list = read_rows(raw / "RECEIPT_PACKAGES.jsonl")
    attacks = read_rows(raw / "ATTACK_INPUTS.jsonl")
    packages = {row["package_id"]: row for row in packages_list}
    primitive_index = {row["package_id"]: row for row in primitives}
    execution_results = []
    for package in packages_list:
        primitive = primitive_index[package["package_id"]]
        expected = {
            key_name: primitive[key_name]
            for key_name in ("run_id", "context_id", "execution_id", "nonce")
        }
        expected["checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
        decisions = evaluate_bundle(
            key, package["receipts"], expected, protocol
        )
        execution_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.execution_verdict.v1",
            "package_id": package["package_id"],
            "role": package["role"],
            "decisions": decisions,
        })
    attack_results = []
    expected_correct = 0
    for attack in attacks:
        bundle = resolve_attack(attack, packages)
        reference = (
            packages[attack["reference_package_id"]]["receipts"]
            if attack["reference_package_id"] else None
        )
        decisions = evaluate_bundle(
            key, bundle, attack["expected_binding"], protocol, reference
        )
        expected = attack_manifest["families"][attack["family"]]["expected"]
        matches = all(decisions[key_name] == value for key_name, value in expected.items())
        expected_correct += int(matches)
        attack_results.append({
            "schema": "exp33a.csw_am_scb_m1_rec_d0.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "family": attack["family"],
            "decisions": decisions,
            "expected": expected,
            "matches_expected": matches,
        })
    # Independent implementation executes from the same immutable primitives.
    reference_tool = load_module(
        "csw_am_scb_m1_rec_d0_reference_runtime",
        ROOT / "tools/csw_am_scb_m1_rec_d0_reference.py",
    )
    reference = reference_tool.evaluate_all(
        key, primitives, packages_list, attacks, protocol, attack_manifest, config
    )
    production_digest = digest({
        "executions": execution_results, "attacks": attack_results
    })
    agreement = production_digest == reference["result_digest"]
    receipt_keys = ("exact_z2_event", "abstract_z2_state", "exact_z3_event", "abstract_z3_state", "chain_authenticator")
    unmodified_authentic = all(
        all(row["decisions"][key_name] == "ACCEPT" for key_name in receipt_keys)
        for row in execution_results
    )
    transition_counts = Counter(
        row["decisions"]["registered_transition_consistency"]
        for row in execution_results
    )
    answer_counts = Counter(
        row["decisions"]["registered_answer_consistency"]
        for row in execution_results
    )
    passed = (
        len(execution_results) == config["public_population"]["execution_packages"]
        and len(attacks) == attack_manifest["expected_attack_rows"]
        and unmodified_authentic
        and expected_correct == len(attacks)
        and agreement
        and reference["passed"]
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    output.mkdir(parents=True)
    write_rows(output / "EXECUTION_VERDICTS.jsonl", execution_results)
    write_rows(output / "ATTACK_VERDICTS.jsonl", attack_results)
    write_json(output / "REFERENCE_RESULT.json", reference)
    write_json(output / "METRICS.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.metrics.v1",
        "execution_packages": len(execution_results),
        "receipt_envelopes": len(execution_results) * 5,
        "unmodified_authentic_packages": sum(
            all(row["decisions"][key_name] == "ACCEPT" for key_name in receipt_keys)
            for row in execution_results
        ),
        "registered_transition_consistency": dict(sorted(transition_counts.items())),
        "registered_answer_consistency": dict(sorted(answer_counts.items())),
        "attack_rows": len(attacks),
        "attack_expected_verdicts": expected_correct,
        "production_reference_agreement": agreement,
        "passed": passed,
    })
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.terminal.v1",
        "terminal": config["terminals"]["ready"] if passed else config["terminals"]["failed"],
        "passed": passed,
        "development_evidence_only": True,
        "receipt_scientifically_confirmed": False,
        "REC_excluded_qualification_authorized": passed,
        "REC_sealed_confirmation_authorized": False,
        "carrier_authorized": False,
        "protected_releases": 0,
    }
    write_json(output / "TERMINAL.json", terminal)
    write_json(output / "EVALUATION_BINDING.json", {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.evaluation_binding.v1",
        "source_commit": source_commit,
        "raw_evidence_commit": raw_commit,
        "key_reveal_commit": reveal_commit,
        "execution_verdicts_sha256": file_sha(output / "EXECUTION_VERDICTS.jsonl"),
        "attack_verdicts_sha256": file_sha(output / "ATTACK_VERDICTS.jsonl"),
        "reference_result_sha256": file_sha(output / "REFERENCE_RESULT.json"),
        "metrics_sha256": file_sha(output / "METRICS.json"),
    })
    return terminal


def preflight() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    attacks = read_json(ATTACK_PATH)
    protocol = read_json(S0_PROTOCOL_PATH)
    key = bytes(range(32))
    first = S0.make_receipt_set(key, protocol, "EXEC-A", 3, 4)
    expected = {
        "run_id": "RUN-A", "context_id": "CTX-A", "execution_id": "EXEC-A",
        "nonce": "NONCE-A",
        "checkpoint_sha256": protocol["state_ontology"]["checkpoint_sha256"],
    }
    decisions = evaluate_bundle(key, first, expected, protocol)
    roundtrip = all(
        canonical(json.loads(canonical(receipt))) == canonical(receipt)
        for receipt in first.values()
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_rec_d0.preflight.v1",
        "valid_bundle_decisions": decisions,
        "canonical_roundtrip": roundtrip,
        "domain_count": len(protocol["receipt_types"]),
        "attack_families": len(attacks["families"]),
        "protected_releases": sum(config["zero_protected_work"].values()),
        "passed": (
            all(value == "ACCEPT" for value in decisions.values())
            and roundtrip
            and len(attacks["families"]) == 18
            and sum(config["zero_protected_work"].values()) == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    key = sub.add_parser("create-key")
    key.add_argument("--source-commit", required=True)
    key.add_argument("--key-out", type=Path, required=True)
    key.add_argument("--commitment-out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--key", type=Path, required=True)
    run.add_argument("--commitment", type=Path, required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--output", type=Path, required=True)
    reveal = sub.add_parser("reveal")
    reveal.add_argument("--source-commit", required=True)
    reveal.add_argument("--key", type=Path, required=True)
    reveal.add_argument("--commitment", type=Path, required=True)
    reveal.add_argument("--commitment-commit", required=True)
    reveal.add_argument("--raw-manifest", type=Path, required=True)
    reveal.add_argument("--raw-commit", required=True)
    reveal.add_argument("--reveal-out", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--source-commit", required=True)
    evaluate_parser.add_argument("--raw", type=Path, required=True)
    evaluate_parser.add_argument("--raw-commit", required=True)
    evaluate_parser.add_argument("--reveal", type=Path, required=True)
    evaluate_parser.add_argument("--reveal-commit", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight()
        elif args.command == "create-key":
            result = create_key(args.source_commit, args.key_out, args.commitment_out)
        elif args.command == "execute":
            result = execute_public(
                args.source_commit, args.commitment, args.commitment_commit,
                args.key, args.output,
            )
        elif args.command == "reveal":
            result = reveal_key(
                args.source_commit, args.commitment, args.commitment_commit,
                args.key, args.raw_manifest, args.raw_commit, args.reveal_out,
            )
        else:
            result = evaluate(
                args.source_commit, args.raw, args.raw_commit,
                args.reveal, args.reveal_commit, args.output,
            )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
