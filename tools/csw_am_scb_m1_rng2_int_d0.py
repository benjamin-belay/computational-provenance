#!/usr/bin/env python3
"""Public receipt-bound integration implementation and lifecycle rehearsal."""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib.util
import json
import math
import os
import secrets
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-int-d0/CONFIG.json"


def module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REC = module("int_d0_receipts", "tools/csw_am_scb_m1_rec_d0.py")
RNG = module("int_d0_rng", "tools/csw_am_scb_m1_rng2_cal1.py")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(path: Path, commit: str) -> bool:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ) == path.read_bytes()
    except subprocess.CalledProcessError:
        return False


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_closure = {}
    for row in sorted(rows, key=lambda value: (value["closure_id"], value["orientation"])):
        by_closure.setdefault(row["closure_id"], row)
    candidates = list(by_closure.values())
    uncovered = {
        "pair": set(range(8)),
        "answer": set(range(8)),
        "template": set(range(4)),
    }
    selected = []
    while any(uncovered.values()):
        def attributes(row):
            return {
                "pair": row["natural"]["z2"] % 8,
                "answer": row["visible_answer"],
                "template": row["template_id"],
            }

        def gain(row):
            return sum(value in uncovered[key] for key, value in attributes(row).items())

        best = max(candidates, key=lambda row: (gain(row), -int(row["closure_id"], 16)))
        if gain(best) == 0:
            raise ValueError("MARGINAL_COVER")
        selected.append(best)
        candidates.remove(best)
        for key, value in attributes(best).items():
            uncovered[key].discard(value)
    duplicate_state = selected[0]["natural"]["z2"]
    duplicate = next(
        row for row in candidates
        if row["natural"]["z2"] == duplicate_state
        and row["closure_id"] not in {value["closure_id"] for value in selected}
    )
    selected.append(duplicate)
    return selected


def combine(*decisions: str) -> str:
    if "REJECT" in decisions:
        return "REJECT"
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    if "INDETERMINATE" in decisions:
        return "INDETERMINATE"
    return "ACCEPT"


def integrated(exact: str, abstract: str, chain: str, carrier: str) -> dict[str, str]:
    return {
        "exact_execution_provenance": combine(exact, abstract, chain, carrier),
        "abstract_state_provenance": combine(abstract, carrier),
        "causal_output_provenance": combine(abstract, chain, carrier),
    }


def validate_source(source_commit: str) -> dict[str, Any]:
    config = read_json(CONFIG)
    if len(source_commit) != 40:
        raise ValueError("SOURCE_COMMIT")
    for relative in config["source_paths"]:
        if not committed(ROOT / relative, source_commit):
            raise ValueError("SOURCE_BYTES:" + relative)
    for name in (
        "S0_main", "S0_terminal", "S0_config", "receipt_protocol",
        "receipt_production", "receipt_reference", "rng2_production",
        "rng2_reference", "public_rows",
    ):
        if (
            file_sha(ROOT / config["bindings"][name + "_path"])
            != config["bindings"][name + "_sha256"]
        ):
            raise ValueError("BINDING:" + name)
    if (
        read_json(ROOT / config["bindings"]["S0_main_path"])["main_verdict"]
        != config["bindings"]["S0_terminal"]
    ):
        raise ValueError("S0")
    return config


def key_commitment(key: bytes, domain: str) -> str:
    return sha(domain.encode() + b"\x00" + key)


def prepare(source_commit: str, output: Path) -> dict[str, Any]:
    config = validate_source(source_commit)
    receipt_key = Path(config["keys"]["receipt_key_path"])
    sampler_key = Path(config["keys"]["sampler_key_path"])
    if output.exists() or receipt_key.exists() or sampler_key.exists():
        raise FileExistsError("PREPARE_BOUNDARY")
    selected = select_rows(read_rows(ROOT / config["bindings"]["public_rows_path"]))
    if len(selected) != config["fixture"]["expected_input_rows"]:
        raise ValueError("FIXTURE_DENOMINATOR")
    rkey, skey = secrets.token_bytes(32), secrets.token_bytes(32)
    for path, key in ((receipt_key, rkey), (sampler_key, skey)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    output.mkdir(parents=True)
    write_rows(output / "SELECTED_PUBLIC_ROWS.jsonl", selected)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.key_commitments.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "receipt_key_commitment": key_commitment(
            rkey, config["keys"]["receipt_commitment_domain"]
        ),
        "sampler_key_commitment": key_commitment(
            skey, config["keys"]["sampler_commitment_domain"]
        ),
        "selected_rows": len(selected),
        "selected_rows_sha256": file_sha(output / "SELECTED_PUBLIC_ROWS.jsonl"),
        "keys_revealed": False,
        "INT_V1_access": 0,
        "protected_access": 0,
    }
    write_json(output / "KEY_COMMITMENTS.json", result)
    return result


def expected_binding(primitive: dict[str, Any], rec_config: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: primitive[key] for key in ("run_id", "context_id", "execution_id", "nonce")},
        "checkpoint_sha256": rec_config["lineage"]["M1_checkpoint_sha256"],
    }


def state_authority(
    package: dict[str, Any], decisions: dict[str, str], protocol: dict[str, Any],
) -> tuple[bytes | None, dict[str, Any] | None]:
    if (
        decisions["abstract_z2_state"] != "ACCEPT"
        or decisions["chain_consistency"] != "ACCEPT"
    ):
        return None, None
    fields = package["receipts"]["abstract_z2"]["preimage"]
    ontology = protocol["state_ontology"]
    if (
        fields["state_ontology_id"] != ontology["identity"]
        or fields["model_domain"] != ontology["model_domain"]
        or fields["checkpoint_sha256"] != ontology["checkpoint_sha256"]
        or fields["state_type"] != "Z2"
    ):
        return None, None
    projection = {
        "protocol_version": "RNG2_INT_V1",
        "RNG2_authority_domain": "RNG2_D0_GENERIC_STATE_AUTHORITY_V1",
        "state_ontology_id": fields["state_ontology_id"],
        "model_domain": fields["model_domain"],
        "checkpoint_sha256": fields["checkpoint_sha256"],
        "state_type": fields["state_type"],
        "category": fields["category"],
    }
    return RNG.authority_key(fields["category"]), projection


def baseline_key(kind: str, row: dict[str, Any]) -> bytes:
    if kind == "PROMPT":
        payload = row["prompt_text"].encode()
    elif kind == "ANSWER":
        payload = str(row["visible_answer"]).encode()
    elif kind == "GENERIC":
        payload = b"FIXED_GENERIC_AUTHORITY"
    else:
        raise ValueError("BASELINE_KIND")
    return hmac.new(
        sha(b"RNG2_INT_D0_BASELINE_ROOT").encode(),
        b"RNG2_INT_D0_" + kind.encode() + b"\x00" + payload,
        hashlib.sha256,
    ).digest()


def make_jobs(
    rows: list[dict[str, Any]], primitives: list[dict[str, Any]],
    packages: list[dict[str, Any]], receipt_key: bytes, protocol: dict[str, Any],
    rec_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    package_by_id = {row["package_id"]: row for row in packages}
    primitive_by_id = {row["package_id"]: row for row in primitives}
    by_example_role = {
        (value["context_id"], value["role"]): value for value in primitives
    }
    jobs, authorizations = [], []
    for row in rows:
        natural = by_example_role[(row["example_id"], "NATURAL")]
        alias = by_example_role[(row["example_id"], "ALIAS_Z2")]
        pair = [natural, alias]
        payload = RNG.CAR.surface_payload(
            row["prompt_text"].encode(), row["visible_answer"],
            {"surface_semantic_payload": {"schema": "SCB_M1_SURFACE_SEMANTIC_V1"}},
        )
        for condition in ("STATE", "CARRIER_OFF", "PROMPT", "ANSWER", "GENERIC"):
            pairing = sha(
                b"RNG2_INT_D0_PAIRING\x00" + row["example_id"].encode()
                + b"\x00" + condition.encode()
            )
            for side, primitive in enumerate(pair):
                package = package_by_id[primitive["package_id"]]
                decisions = REC.evaluate_bundle(
                    receipt_key, package["receipts"],
                    expected_binding(primitive, rec_config), protocol,
                )
                state_key, projection = state_authority(package, decisions, protocol)
                if condition == "STATE":
                    authority = state_key
                elif condition == "CARRIER_OFF":
                    authority = None
                else:
                    authority = baseline_key(condition, row)
                generation_authorized = (
                    decisions["abstract_z2_state"] == "ACCEPT"
                    and decisions["chain_consistency"] == "ACCEPT"
                )
                if not generation_authorized or state_key is None:
                    raise ValueError("GENERATION_NOT_AUTHORIZED")
                authorizations.append({
                    "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.generation_authorization.v1",
                    "package_id": primitive["package_id"],
                    "condition": condition,
                    "receipt_decisions": decisions,
                    "authenticated_projection": projection,
                    "authority_sha256": sha(authority) if authority is not None else None,
                    "generation_authorized": True,
                    "verification_preceded_authority": True,
                })
                for output_index in range(8):
                    jobs.append({
                        "closure_id": row["closure_id"],
                        "context_id": row["example_id"],
                        "package_id": primitive["package_id"],
                        "side": side,
                        "role": primitive["role"],
                        "condition": condition,
                        "output_index": output_index,
                        "pairing_context_id": pairing,
                        "semantic_payload": payload,
                        "authority_key": authority,
                        "authenticated_category": projection["category"],
                        "public_tokens": [],
                        "order": None,
                        "draws": [],
                    })
    return jobs, authorizations


def generate_jobs(jobs: list[dict[str, Any]], sampler_key: bytes, config: dict[str, Any]) -> list[dict[str, Any]]:
    model = RNG.A0.load_checkpoint(ROOT / "artifacts/dev_only/csw-a0/training/seed1/CHECKPOINT.pt")[0]
    model.eval()
    prompts = torch.tensor(
        [RNG.A0.encode(RNG.A0.prompt_tokens(job["semantic_payload"])) for job in jobs],
        dtype=torch.long,
    )
    with torch.no_grad():
        states, hidden = model.gru(model.embedding(prompts))
        logits = model.output(states[:, -1, :])
        for step in range(49):
            candidates, eligible = RNG._candidate_rows(jobs, step)
            candidate_ids = torch.tensor(
                [[RNG.A0.TO_ID[token] for token in row] for row in candidates],
                dtype=torch.long,
            )
            selected_logits = logits.gather(1, candidate_ids).to(torch.float64)
            if eligible:
                masks = []
                for job, options in zip(jobs, candidates):
                    green = (
                        RNG.A0.green_set(
                            job["authority_key"],
                            sum(token in RNG.A0.ELIGIBLE_TOKEN_SET for token in job["public_tokens"]),
                            job["public_tokens"], options,
                        )
                        if job["authority_key"] is not None else set()
                    )
                    masks.append([token in green for token in options])
                selected_logits += torch.tensor(masks, dtype=torch.float64) * 2.0
            probabilities = torch.softmax(selected_logits, dim=1)
            uniforms = []
            for job in jobs:
                uniform = RNG.D0.position_uniform(
                    sampler_key,
                    RNG.D0.index_for(
                        job["pairing_context_id"], job["closure_id"],
                        job["output_index"], step,
                    ),
                )["uniform"]
                uniforms.append(uniform)
                job["draws"].append(uniform)
            cumulative = probabilities.cumsum(dim=1)
            selections = (
                torch.tensor(uniforms, dtype=torch.float64)[:, None] <= cumulative
            ).to(torch.int64).argmax(dim=1)
            selected_ids = candidate_ids.gather(1, selections[:, None]).squeeze(1)
            tokens = [
                candidates[index][int(selections[index])] for index in range(len(jobs))
            ]
            for job, token in zip(jobs, tokens):
                if step == 0:
                    job["order"] = RNG.A0.ORDERS[0 if token == "<ORDER_A>" else 1]
                elif token != "<NONE>":
                    job["public_tokens"].append(token)
            states, hidden = model.gru(model.embedding(selected_ids[:, None]), hidden)
            logits = model.output(states[:, -1, :])
    outputs = []
    for job in jobs:
        literal = " ".join(job["public_tokens"])
        outputs.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.output.v1",
            **{key: job[key] for key in (
                "closure_id", "context_id", "package_id", "side", "role",
                "condition", "output_index", "pairing_context_id",
                "semantic_payload", "authenticated_category",
            )},
            "authority_sha256": (
                sha(job["authority_key"]) if job["authority_key"] is not None else None
            ),
            "literal_output": literal,
            "literal_output_sha256": sha(literal.encode()),
            "draws": job["draws"],
            "draw_trace_sha256": sha(canonical(job["draws"])),
            "model_steps": 49,
        })
    return outputs


def attacks(primitives: list[dict[str, Any]], packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    package_by_id = {row["package_id"]: row for row in packages}
    natural = [row for row in primitives if row["role"] == "NATURAL"]
    alias_by_context = {
        row["context_id"]: row for row in primitives if row["role"] == "ALIAS_Z2"
    }
    base = natural[0]
    alias = alias_by_context[base["context_id"]]
    same = next(
        row for row in natural[1:] if row["consumed_z2"] == base["consumed_z2"]
    )
    other = next(row for row in natural[1:] if row["consumed_z2"] != base["consumed_z2"])
    malformed = copy.deepcopy(package_by_id[base["package_id"]]["receipts"]["abstract_z2"])
    malformed["tag"] = "0" * len(malformed["tag"])
    bad_chain = copy.deepcopy(package_by_id[base["package_id"]]["receipts"]["chain"])
    bad_chain["tag"] = "0" * len(bad_chain["tag"])
    rows = [
        ("SAME_STATE_DIFFERENT_EXACT", base, same, same, base, None, None),
        ("NATURAL_ALIAS_CROSS", base, alias, alias, alias, None, None),
        ("CORRECT_ABSTRACT_WRONG_EXACT", base, alias, base, base, None, None),
        ("WRONG_STATE_RECEIPT", base, base, alias, base, None, None),
        ("REPLAY_WRONG_CONTEXT", base, other, other, other, None, None),
        ("MISSING_RECEIPT", base, None, None, None, None, None),
        ("MALFORMED_ABSTRACT_RECEIPT", base, base, None, base, malformed, None),
        ("INVALID_CHAIN", base, base, base, None, None, bad_chain),
    ]
    result = []
    for family, output, exact, abstract, chain, inline_abstract, inline_chain in rows:
        result.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.attack.v1",
            "attack_id": sha(canonical([family, output["package_id"]])),
            "family": family,
            "output_package_id": output["package_id"],
            "candidate_exact_package_id": exact["package_id"] if exact else None,
            "abstract_package_id": abstract["package_id"] if abstract else None,
            "chain_package_id": chain["package_id"] if chain else None,
            "inline_abstract_z2": inline_abstract,
            "inline_chain": inline_chain,
        })
    return result


def execute(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    commitments = read_json(stage / "commitments/KEY_COMMITMENTS.json")
    rows = read_rows(stage / "commitments/SELECTED_PUBLIC_ROWS.jsonl")
    receipt_key = Path(config["keys"]["receipt_key_path"]).read_bytes()
    sampler_key = Path(config["keys"]["sampler_key_path"]).read_bytes()
    if (
        key_commitment(receipt_key, config["keys"]["receipt_commitment_domain"])
        != commitments["receipt_key_commitment"]
        or key_commitment(sampler_key, config["keys"]["sampler_commitment_domain"])
        != commitments["sampler_key_commitment"]
    ):
        raise ValueError("KEY_BINDING")
    rec_config = REC.read_json(REC.CONFIG_PATH)
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    model = REC.M1.load_model(REC.M1.read_json(REC.M1.CONFIG_PATH))
    values = REC.M1.tensors(rows)
    with torch.no_grad():
        natural = model.run_natural(values)
        targets = torch.tensor([(row["natural"]["z2"] + 8) % 16 for row in rows])
        alias = model.run_with_z2(values, targets)
        natural_answers = natural["answer_logits"].argmax(-1)
        alias_answers = alias["answer_logits"].argmax(-1)
    primitives, packages = [], []
    for index, row in enumerate(rows):
        predicted = int(natural["z2_selected"][index])
        observations = (
            ("NATURAL", {
                "z2_consumed": int(natural["z2_consumed"][index]),
                "z3_selected": int(natural["z3_selected"][index]),
                "z3_consumed": int(natural["z3_consumed"][index]),
                "answer": int(natural_answers[index]),
            }, predicted),
            ("ALIAS_Z2", {
                "z2_consumed": int(alias["z2_consumed"][index]),
                "z3_selected": int(alias["z3_selected"][index]),
                "z3_consumed": int(alias["z3_consumed"][index]),
                "answer": int(alias_answers[index]),
            }, int(targets[index])),
        )
        for role, observed, target in observations:
            primitive, package = REC.receipt_package(
                receipt_key, protocol, rec_config, row, role, observed, predicted, target
            )
            primitives.append(primitive)
            packages.append(package)
    jobs, authorizations = make_jobs(
        rows, primitives, packages, receipt_key, protocol, rec_config
    )
    outputs = generate_jobs(jobs, sampler_key, config)
    attack_rows = attacks(primitives, packages)
    raw = stage / "raw"
    raw.mkdir(parents=True)
    write_rows(raw / "EXECUTION_ROWS.jsonl", primitives)
    write_rows(raw / "RECEIPT_PACKAGES.jsonl", packages)
    write_rows(raw / "GENERATION_AUTHORIZATIONS.jsonl", authorizations)
    write_rows(raw / "OUTPUTS.jsonl", outputs)
    write_rows(raw / "ATTACK_INPUTS.jsonl", attack_rows)
    names = (
        "EXECUTION_ROWS.jsonl", "RECEIPT_PACKAGES.jsonl",
        "GENERATION_AUTHORIZATIONS.jsonl", "OUTPUTS.jsonl", "ATTACK_INPUTS.jsonl",
    )
    manifest = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.raw_manifest.v1",
        "identity": config["identity"],
        "source_commit": commitments["source_commit"],
        "receipt_key_commitment": commitments["receipt_key_commitment"],
        "sampler_key_commitment": commitments["sampler_key_commitment"],
        "input_rows": len(rows),
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "generation_authorizations": len(authorizations),
        "outputs": len(outputs),
        "model_steps": len(outputs) * 49,
        "attack_inputs": len(attack_rows),
        "files": {name: file_sha(raw / name) for name in names},
        "keys_revealed": False,
        "scores_computed": False,
        "INT_V1_access": 0,
        "protected_access": 0,
    }
    write_json(raw / "RAW_EVIDENCE_COMMITMENT.json", manifest)
    return manifest


def reveal(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    raw = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    receipt_key = Path(config["keys"]["receipt_key_path"]).read_bytes()
    sampler_key = Path(config["keys"]["sampler_key_path"]).read_bytes()
    if raw["keys_revealed"] is not False or raw["scores_computed"] is not False:
        raise ValueError("CHRONOLOGY")
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.key_reveal.v1",
        "identity": config["identity"],
        "receipt_key_hex": receipt_key.hex(),
        "sampler_key_hex": sampler_key.hex(),
        "receipt_key_commitment": raw["receipt_key_commitment"],
        "sampler_key_commitment": raw["sampler_key_commitment"],
        "raw_commitment_sha256": file_sha(stage / "raw/RAW_EVIDENCE_COMMITMENT.json"),
        "reveal_after_raw_commitment": True,
    }
    write_json(stage / "reveal/KEY_REVEAL.json", result)
    return {key: value for key, value in result.items() if not key.endswith("_hex")}


def accepted(scores: list[float], category: int, config: dict[str, Any]) -> bool:
    wrong = max(scores[index] for index in range(16) if index != category)
    return (
        scores[category] > config["carrier"]["T"]
        and scores[category] == max(scores)
        and scores.count(max(scores)) == 1
        and scores[category] - wrong > config["carrier"]["M"]
    )


def score_outputs(outputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    primitive, grouped = [], defaultdict(list)
    for row in outputs:
        parsed, counts, normalized = RNG._score_output(row["literal_output"])
        expected = {key: row["semantic_payload"][key] for key in ("z1", "z2", "z3", "answer")}
        if parsed["values"] != expected or parsed["eligible_count"] != 24:
            raise ValueError("SEMANTIC")
        primitive.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.primitive_score.v1",
            "package_id": row["package_id"],
            "condition": row["condition"],
            "output_index": row["output_index"],
            "green_counts": counts,
            "normalized_scores": normalized,
        })
        grouped[(row["package_id"], row["condition"])].append(normalized)
    aggregates = {}
    for key, vectors in grouped.items():
        aggregates[key] = [
            math.fsum(vector[category] for vector in vectors) / math.sqrt(8)
            for category in range(16)
        ]
    return primitive, aggregates


def receipt_layers(
    receipt_key: bytes, attack: dict[str, Any],
    primitives: dict[str, dict[str, Any]], packages: dict[str, dict[str, Any]],
    protocol: dict[str, Any], rec_config: dict[str, Any],
) -> tuple[str, str, str, int | None]:
    output = primitives[attack["output_package_id"]]
    binding = expected_binding(output, rec_config)
    exact_package = packages.get(attack["candidate_exact_package_id"])
    abstract_package = packages.get(attack["abstract_package_id"])
    chain_package = packages.get(attack["chain_package_id"])
    exact_receipt = (
        exact_package["receipts"]["exact_z2"] if exact_package else None
    )
    exact = REC.S0.verify_receipt(
        receipt_key, exact_receipt, "EXACT_Z2", protocol, binding
    )
    if attack["inline_abstract_z2"] is not None:
        abstract_receipt = attack["inline_abstract_z2"]
        abstract_exact = packages[output["package_id"]]["receipts"]["exact_z2"]
    elif abstract_package:
        abstract_receipt = abstract_package["receipts"]["abstract_z2"]
        abstract_exact = abstract_package["receipts"]["exact_z2"]
    else:
        abstract_receipt = abstract_exact = None
    abstract = REC.ontology_decision(
        receipt_key, abstract_receipt, abstract_exact,
        "ABSTRACT_Z2", "EXACT_Z2", protocol,
    )
    if attack["inline_chain"] is not None:
        chain_receipt = attack["inline_chain"]
        chain_bundle = copy.deepcopy(packages[output["package_id"]]["receipts"])
        chain_bundle["chain"] = chain_receipt
    elif chain_package:
        chain_bundle = chain_package["receipts"]
    else:
        chain_bundle = {key: None for key in ("exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain")}
    chain = REC.evaluate_bundle(
        receipt_key, chain_bundle, binding, protocol
    )["chain_consistency"]
    category = (
        abstract_receipt["preimage"].get("category")
        if abstract == "ACCEPT" and abstract_receipt else None
    )
    return exact, abstract, chain, category


def evaluate(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    raw_manifest = read_json(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    reveal_value = read_json(stage / "reveal/KEY_REVEAL.json")
    if (
        reveal_value["raw_commitment_sha256"]
        != file_sha(stage / "raw/RAW_EVIDENCE_COMMITMENT.json")
    ):
        raise ValueError("RAW_REVEAL")
    for name, expected in raw_manifest["files"].items():
        if file_sha(stage / "raw" / name) != expected:
            raise ValueError("RAW_FILE:" + name)
    receipt_key = bytes.fromhex(reveal_value["receipt_key_hex"])
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    rec_config = REC.read_json(REC.CONFIG_PATH)
    primitive_rows = read_rows(stage / "raw/EXECUTION_ROWS.jsonl")
    package_rows = read_rows(stage / "raw/RECEIPT_PACKAGES.jsonl")
    outputs = read_rows(stage / "raw/OUTPUTS.jsonl")
    attack_rows = read_rows(stage / "raw/ATTACK_INPUTS.jsonl")
    primitives = {row["package_id"]: row for row in primitive_rows}
    packages = {row["package_id"]: row for row in package_rows}
    score_rows, aggregates = score_outputs(outputs)
    genuine = []
    for package in package_rows:
        primitive = primitives[package["package_id"]]
        receipt = REC.evaluate_bundle(
            receipt_key, package["receipts"],
            expected_binding(primitive, rec_config), protocol,
        )
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        carrier = (
            "ACCEPT" if accepted(aggregates[(package["package_id"], "STATE")], category, config)
            else "INDETERMINATE"
        )
        genuine.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.genuine_verdict.v1",
            "package_id": package["package_id"],
            "receipt": receipt,
            "carrier": carrier,
            "integrated": integrated(
                receipt["exact_z2_event"], receipt["abstract_z2_state"],
                receipt["chain_consistency"], carrier,
            ),
        })
    attack_verdicts = []
    for attack in attack_rows:
        exact, abstract, chain, category = receipt_layers(
            receipt_key, attack, primitives, packages, protocol, rec_config
        )
        if abstract == "ABSTAIN":
            carrier = "INDETERMINATE"
        elif abstract == "REJECT" or category is None:
            carrier = "REJECT"
        else:
            carrier = (
                "ACCEPT"
                if accepted(
                    aggregates[(attack["output_package_id"], "STATE")], category, config
                )
                else "REJECT"
            )
        attack_verdicts.append({
            "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.attack_verdict.v1",
            "attack_id": attack["attack_id"],
            "family": attack["family"],
            "exact": exact,
            "abstract": abstract,
            "chain": chain,
            "carrier": carrier,
            "integrated": integrated(exact, abstract, chain, carrier),
        })
    by_context_condition = defaultdict(dict)
    for row in outputs:
        by_context_condition[(row["context_id"], row["condition"], row["output_index"])][row["role"]] = row
    paired_draws = all(
        values["NATURAL"]["draw_trace_sha256"] == values["ALIAS_Z2"]["draw_trace_sha256"]
        for values in by_context_condition.values()
    )
    baseline_equal = all(
        values["NATURAL"]["literal_output_sha256"] == values["ALIAS_Z2"]["literal_output_sha256"]
        for (context, condition, index), values in by_context_condition.items()
        if condition in ("CARRIER_OFF", "PROMPT", "ANSWER", "GENERIC")
    )
    same_state = next(row for row in attack_verdicts if row["family"] == "SAME_STATE_DIFFERENT_EXACT")
    missing = next(row for row in attack_verdicts if row["family"] == "MISSING_RECEIPT")
    invalid_chain = next(row for row in attack_verdicts if row["family"] == "INVALID_CHAIN")
    gates = {
        "genuine_receipts": all(
            all(value == "ACCEPT" for value in row["receipt"].values())
            for row in genuine
        ),
        "genuine_carrier": all(row["carrier"] == "ACCEPT" for row in genuine),
        "exact_abstract_distinction": (
            same_state["integrated"]["exact_execution_provenance"] == "REJECT"
            and same_state["integrated"]["abstract_state_provenance"] == "ACCEPT"
        ),
        "missing_abstains": missing["integrated"]["abstract_state_provenance"] == "ABSTAIN",
        "invalid_chain_rejects_causal": (
            invalid_chain["integrated"]["causal_output_provenance"] == "REJECT"
        ),
        "all_attacks_structured": len(attack_verdicts) == len(config["attacks"]),
        "paired_draws": paired_draws,
        "baseline_outputs_equal": baseline_equal,
        "thresholds_frozen": config["carrier"]["T"] == 4.041451884327381 and config["carrier"]["M"] == 0.0,
        "complete_denominator": (
            len(outputs) == config["carrier"]["outputs"]
            and sum(len(row["draws"]) for row in outputs) == config["carrier"]["model_steps"]
            and len(score_rows) * 16 == config["carrier"]["primitive_authority_scores"]
            and len(aggregates) * 16 == config["carrier"]["aggregate_authority_scores"]
        ),
        "chronology": reveal_value["reveal_after_raw_commitment"] is True,
        "future_roles_sealed": raw_manifest["INT_V1_access"] == 0 and raw_manifest["protected_access"] == 0,
    }
    production = stage / "production"
    production.mkdir(parents=True)
    write_rows(production / "PRIMITIVE_SCORES.jsonl", score_rows)
    write_rows(production / "GENUINE_VERDICTS.jsonl", genuine)
    write_rows(production / "ATTACK_VERDICTS.jsonl", attack_verdicts)
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.official.v1",
        "identity": config["identity"],
        "gates": gates,
        "passed": all(gates.values()),
        "terminal": config["terminals"]["ready"] if all(gates.values()) else config["terminals"]["invalid"],
        "observed": {
            "fixture_rows": raw_manifest["input_rows"],
            "execution_packages": len(package_rows),
            "receipt_envelopes": len(package_rows) * 5,
            "outputs": len(outputs),
            "model_steps": sum(len(row["draws"]) for row in outputs),
            "eligible_positions": sum(24 for _ in outputs),
            "primitive_authority_scores": len(score_rows) * 16,
            "aggregate_authority_scores": len(aggregates) * 16,
            "attack_packages": len(attack_verdicts),
        },
        "evidence": {
            "primitive_sha256": file_sha(production / "PRIMITIVE_SCORES.jsonl"),
            "genuine_sha256": file_sha(production / "GENUINE_VERDICTS.jsonl"),
            "attacks_sha256": file_sha(production / "ATTACK_VERDICTS.jsonl"),
        },
        "claim_boundary": {
            "implementation_ready": all(gates.values()),
            "prospective_integration_validated": False,
            "protected_output_provenance": False,
        },
    }
    write_json(production / "OFFICIAL_RESULT.json", result)
    return result


def finalize(stage: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    official = read_json(stage / "production/OFFICIAL_RESULT.json")
    reference = read_json(stage / "reference/REFERENCE_RESULT.json")
    valid = official["passed"] and reference["passed"]
    audit = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.main_audit.v1",
        "identity": config["identity"],
        "valid": valid,
        "main_verdict": config["terminals"]["ready"] if valid else config["terminals"]["invalid"],
        "gates": official["gates"],
        "observed": official["observed"],
        "independent_agreement": reference["gates"],
        "claim_boundary": official["claim_boundary"],
        "authorizations": {
            "separate_INT_V1_request": valid,
            "INT_V1_release": False,
            "protected_release": False,
            "automatic_successor": False,
        },
        "evidence": {
            **official["evidence"],
            "official_sha256": file_sha(stage / "production/OFFICIAL_RESULT.json"),
            "reference_sha256": file_sha(stage / "reference/REFERENCE_RESULT.json"),
            "raw_commitment_sha256": file_sha(stage / "raw/RAW_EVIDENCE_COMMITMENT.json"),
        },
    }
    write_json(stage / "MAIN_AUDIT.json", audit)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_d0.terminal.v1",
        "terminal": audit["main_verdict"],
        "valid": valid,
        "main_audit_sha256": file_sha(stage / "MAIN_AUDIT.json"),
        "INT_V1_release_authorized": False,
        "protected_release": 0,
    }
    write_json(stage / "TERMINAL.json", terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-commit", required=True)
    prep.add_argument("--output", type=Path, required=True)
    exe = sub.add_parser("execute")
    exe.add_argument("--stage", type=Path, required=True)
    rev = sub.add_parser("reveal")
    rev.add_argument("--stage", type=Path, required=True)
    eva = sub.add_parser("evaluate")
    eva.add_argument("--stage", type=Path, required=True)
    fin = sub.add_parser("finalize")
    fin.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.source_commit, args.output)
    elif args.command == "execute":
        value = execute(args.stage)
    elif args.command == "reveal":
        value = reveal(args.stage)
    elif args.command == "evaluate":
        value = evaluate(args.stage)
    else:
        value = finalize(args.stage)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
