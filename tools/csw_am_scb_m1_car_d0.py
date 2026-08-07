#!/usr/bin/env python3
"""Public receipt-bound 16-state carrier integration for SCB Model 1."""
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-car-d0/CONFIG.json"


def module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


CAR = module("car_s0_for_car_d0", "tools/csw_am_scb_m1_car_s0.py")
REC = module("rec_d0_for_car_d0", "tools/csw_am_scb_m1_rec_d0.py")
M1 = module("scb_m1_for_car_d0", "tools/csw_am_scb_m1.py")
A0 = module("a0_channel_for_car_d0", "tools/csw_a0_stochastic_channel.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("DUPLICATE_KEY:" + key)
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line, object_pairs_hook=pairs)
        if type(value) is not dict:
            raise ValueError("ROW:" + path.name)
        result.append(value)
    return result


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


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def committed(path: Path, commit: str) -> None:
    relative = str(path.resolve().relative_to(ROOT))
    if git_show(commit, relative) != path.read_bytes():
        raise ValueError("COMMIT_BINDING:" + relative)


def selected_manifest_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    population = config["public_population"]
    source = ROOT / population["source_path"]
    if file_sha(source) != population["source_sha256"]:
        raise ValueError("PUBLIC_SOURCE_HASH")
    rows = read_rows(source)
    closures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        closures[row["closure_id"]].append(row)
    if len(closures) != population["source_closures"]:
        raise ValueError("SOURCE_CLOSURES")
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    domain = population["selection_domain"].encode()
    for closure_id, members in closures.items():
        lower = [row for row in members if row["orientation"] == "LOWER_C_ORIENTATION"]
        if len(members) != 2 or len(lower) != 1:
            raise ValueError("CLOSURE_ORIENTATION")
        row = copy.deepcopy(lower[0])
        cell = (
            row["natural"]["z2"] % 8,
            row["natural"]["answer"],
            row["template_id"],
        )
        row["selection_cell"] = {
            "z2_mod8": cell[0], "answer": cell[1], "template_id": cell[2]
        }
        row["selection_priority"] = sha(domain + b"\x00" + closure_id.encode())
        row["dataset_role"] = "CAR_D0_PUBLIC_INTEGRATION"
        cells[cell].append(row)
    if len(cells) != population["reachable_cells"]:
        raise ValueError("REACHABLE_CELLS")
    selected = [
        min(candidates, key=lambda row: (row["selection_priority"], row["closure_id"]))
        for _, candidates in sorted(cells.items())
    ]
    return sorted(
        selected,
        key=lambda row: (
            row["selection_cell"]["z2_mod8"],
            row["selection_cell"]["answer"],
            row["selection_cell"]["template_id"],
        ),
    )


def select(output: Path) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    rows = selected_manifest_rows(config)
    if len(rows) != config["public_population"]["selected_rows"]:
        raise ValueError("SELECTED_ROWS")
    write_rows(output, rows)
    return {
        "closures": len(rows),
        "reachable_cells": len({canonical(row["selection_cell"]) for row in rows}),
        "manifest_sha256": file_sha(output),
    }


def validate_source(source_commit: str) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if len(source_commit) != 40 or config["identity"] != "CSW-AM-SCB-M1-CAR-D0":
        raise ValueError("SOURCE_IDENTITY")
    for relative in config["source_file_paths"]:
        if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    lineage = config["lineage"]
    s0_audit = ROOT / lineage["CAR_S0_main_audit_path"]
    if (
        file_sha(s0_audit) != lineage["CAR_S0_main_audit_sha256"]
        or read_json(s0_audit)["main_verdict"] != lineage["CAR_S0_main_terminal"]
    ):
        raise ValueError("CAR_S0_BINDING")
    for commit, path_key, sha_key in (
        (lineage["REC_D0_source_commit"], "REC_D0_production_path", "REC_D0_production_sha256"),
        (lineage["REC_D0_source_commit"], "REC_D0_reference_path", "REC_D0_reference_sha256"),
        (lineage["M1_source_commit"], "M1_source_path", "M1_source_sha256"),
        (lineage["A0_Q1_immutable_commit"], "A0_production_path", "A0_production_sha256"),
        (lineage["A0_Q1_immutable_commit"], "A0_reference_path", "A0_reference_sha256"),
    ):
        relative = lineage[path_key]
        path = ROOT / relative
        if file_sha(path) != lineage[sha_key] or git_show(commit, relative) != path.read_bytes():
            raise ValueError("FROZEN_DEPENDENCY:" + relative)
    manifest = ROOT / config["public_population"]["manifest_path"]
    if read_rows(manifest) != selected_manifest_rows(config):
        raise ValueError("PUBLIC_MANIFEST")
    return config


def receipt_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(REC.read_json(REC.CONFIG_PATH))
    value["identity"] = config["identity"]
    value["key_lifecycle"]["commitment_domain"] = config["keys"]["receipt_commitment_domain"]
    value["key_lifecycle"]["run_id_domain"] = "EXP33A|SCB-M1|CAR-D0|RECEIPT-RUN|V1"
    value["key_lifecycle"]["nonce_domain"] = "EXP33A|SCB-M1|CAR-D0|RECEIPT-NONCE|V1"
    value["public_population"]["execution_packages"] = config["execution"]["SCB_execution_packages"]
    value["lineage"]["M1_checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
    return value


def key_commitment(key: bytes, domain: str) -> str:
    return sha(domain.encode() + b"\x00" + key)


def create_keys(
    source_commit: str, receipt_out: Path, carrier_out: Path, commitment_out: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    if any(path.exists() for path in (receipt_out, carrier_out, commitment_out)):
        raise FileExistsError("KEY_EXISTS")
    receipt, carrier = secrets.token_bytes(32), secrets.token_bytes(32)
    if receipt == carrier:
        raise RuntimeError("KEY_COLLISION")
    for path, value in ((receipt_out, receipt), (carrier_out, carrier)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = ROOT / config["public_population"]["manifest_path"]
    value = {
        "schema": "exp33a.csw_am_scb_m1_car_d0.key_commitment.v1",
        "source_commit": source_commit,
        "public_manifest_sha256": file_sha(manifest),
        "receipt_key_commitment": key_commitment(
            receipt, config["keys"]["receipt_commitment_domain"]
        ),
        "carrier_key_commitment": key_commitment(
            carrier, config["keys"]["carrier_commitment_domain"]
        ),
        "keys_distinct": True,
        "keys_revealed": False,
    }
    write_json(commitment_out, value)
    return value


def validate_keys(
    config: dict[str, Any], source_commit: str, receipt: bytes, carrier: bytes,
    commitment_path: Path, commitment_commit: str,
) -> dict[str, Any]:
    committed(commitment_path, commitment_commit)
    value = read_json(commitment_path)
    manifest = ROOT / config["public_population"]["manifest_path"]
    if (
        value["source_commit"] != source_commit
        or value["public_manifest_sha256"] != file_sha(manifest)
        or value["receipt_key_commitment"]
        != key_commitment(receipt, config["keys"]["receipt_commitment_domain"])
        or value["carrier_key_commitment"]
        != key_commitment(carrier, config["keys"]["carrier_commitment_domain"])
        or receipt == carrier
        or value["keys_revealed"] is not False
    ):
        raise ValueError("KEY_BINDING")
    return value


def baseline_authority(
    carrier_root: bytes, family: str, prompt_text: str, answer: int,
) -> bytes:
    domains = {
        "PROMPT": "EXP33A|SCB-M1|PROMPT-CARRIER-AUTHORITY|V1",
        "ANSWER": "EXP33A|SCB-M1|ANSWER-CARRIER-AUTHORITY|V1",
        "GENERIC": "EXP33A|SCB-M1|GENERIC-CARRIER-AUTHORITY|V1",
    }
    projections = {
        "PROMPT": {"visible_prompt_sha256": sha(prompt_text.encode())},
        "ANSWER": {"answer": answer},
        "GENERIC": {"constant_origin_label": "SCB_M1_GENERIC_ORIGIN_V1"},
    }
    return hmac.new(
        carrier_root,
        domains[family].encode() + b"\x00" + canonical(projections[family]),
        hashlib.sha256,
    ).digest()


def expected_binding(primitive: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = {
        field: primitive[field]
        for field in ("run_id", "context_id", "execution_id", "nonce")
    }
    result["checkpoint_sha256"] = config["lineage"]["M1_checkpoint_sha256"]
    return result


def load_surface(config: dict[str, Any]):
    path = ROOT / config["lineage"]["A0_checkpoint_path"]
    if file_sha(path) != config["lineage"]["A0_checkpoint_sha256"]:
        raise ValueError("A0_CHECKPOINT")
    return A0.load_checkpoint(path)[0]


def make_output(
    surface: Any, payload: dict[str, Any], seed: bytes, target_key: bytes | None,
    condition: str, side: str, closure_id: str, draw: int, category: int,
    package_id: str, config: dict[str, Any],
) -> dict[str, Any]:
    generated = A0.generate(
        surface, payload, stream_seed=seed, target_key=target_key,
        carrier_enabled=condition != "OFF", bias=config["channel"]["bias"],
        retain_operations=False,
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_car_d0.output.v1",
        "output_id": digest([
            "SCB_M1_CAR_D0_OUTPUT_V1", closure_id, side, condition, draw
        ]),
        "closure_id": closure_id,
        "side": side,
        "condition": condition,
        "draw": draw,
        "source_z2_category": category,
        "receipt_package_id": package_id,
        "semantic_payload": payload,
        "semantic_payload_sha256": digest(payload),
        "stream_seed_sha256": sha(seed),
        "literal_output": generated["literal_output"],
        "output_sha256": generated["output_sha256"],
        "eligible_positions": generated["eligible_positions"],
        "stochastic_draws": generated["stochastic_draws"],
        "parsed": generated["parsed"],
    }


def execute(
    source_commit: str, receipt_key_path: Path, carrier_key_path: Path,
    commitment_path: Path, commitment_commit: str, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    receipt_key, carrier_root = receipt_key_path.read_bytes(), carrier_key_path.read_bytes()
    key_binding = validate_keys(
        config, source_commit, receipt_key, carrier_root,
        commitment_path, commitment_commit,
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    rows = read_rows(ROOT / config["public_population"]["manifest_path"])
    model, surface = M1.load_model(M1.read_json(M1.CONFIG_PATH)), load_surface(config)
    rconfig = receipt_config(config)
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    authority_spec = CAR.read_json(CAR.AUTHORITY_PATH)
    values = M1.tensors(rows)
    with torch.no_grad():
        natural = model.run_natural(values)
        targets = torch.tensor(
            [(row["natural"]["z2"] + 8) % 16 for row in rows], dtype=torch.long
        )
        alias = model.run_with_z2(values, targets)
    natural_answers = natural["answer_logits"].argmax(-1)
    alias_answers = alias["answer_logits"].argmax(-1)
    primitives, packages, output_rows = [], [], []
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
        side_packages = []
        for side, observed, target in observations:
            primitive, package = REC.receipt_package(
                receipt_key, protocol, rconfig, row, side, observed, predicted, target
            )
            decisions = REC.evaluate_bundle(
                receipt_key, package["receipts"],
                expected_binding(primitive, config), protocol,
            )
            primitive["pre_generation_receipt_decisions"] = decisions
            primitive["abstract_authority_derived_before_generation"] = (
                decisions["abstract_z2_state"] == "ACCEPT"
            )
            primitives.append(primitive)
            packages.append(package)
            side_packages.append((side, primitive, package))
        payload = CAR.surface_payload(
            row["prompt_text"].encode(), row["natural"]["answer"],
            CAR.read_json(CAR.BOUNDARY_PATH),
        )
        for side, primitive, package in side_packages:
            category = package["receipts"]["abstract_z2"]["preimage"]["category"]
            state_key = CAR.derive_authority(
                carrier_root, category,
                primitive["pre_generation_receipt_decisions"]["abstract_z2_state"],
                authority_spec,
            )
            if state_key is None:
                raise ValueError("NO_AUTHENTICATED_STATE_AUTHORITY")
            for condition, draws in (
                ("STATE", range(4)), ("OFF", range(4)),
                ("PROMPT", range(1)), ("ANSWER", range(1)), ("GENERIC", range(1)),
            ):
                for draw in draws:
                    seed = CAR.paired_nonce(row["closure_id"], draw)
                    if condition == "STATE":
                        target_key = state_key
                    elif condition == "OFF":
                        target_key = None
                    else:
                        target_key = baseline_authority(
                            carrier_root, condition, row["prompt_text"],
                            row["natural"]["answer"],
                        )
                    output_rows.append(make_output(
                        surface, payload, seed, target_key, condition, side,
                        row["closure_id"], draw, category, package["package_id"], config,
                    ))
    expected = config["execution"]
    if (
        len(primitives) != expected["SCB_execution_packages"]
        or len(packages) * 5 != expected["receipt_envelopes"]
        or len(output_rows) != expected["generated_outputs"]
    ):
        raise ValueError("RAW_DENOMINATOR")
    output.mkdir(parents=True)
    write_rows(output / "PUBLIC_ROWS.jsonl", rows)
    write_rows(output / "EXECUTION_ROWS.jsonl", primitives)
    write_rows(output / "RECEIPT_PACKAGES.jsonl", packages)
    write_rows(output / "OUTPUT_ROWS.jsonl", output_rows)
    names = (
        "PUBLIC_ROWS.jsonl", "EXECUTION_ROWS.jsonl",
        "RECEIPT_PACKAGES.jsonl", "OUTPUT_ROWS.jsonl",
    )
    write_json(output / "RAW_MANIFEST.json", {
        "schema": "exp33a.csw_am_scb_m1_car_d0.raw_manifest.v1",
        "source_commit": source_commit,
        "key_commitment_commit": commitment_commit,
        "receipt_key_commitment": key_binding["receipt_key_commitment"],
        "carrier_key_commitment": key_binding["carrier_key_commitment"],
        "closures": len(rows),
        "execution_packages": len(primitives),
        "receipt_envelopes": len(packages) * 5,
        "generated_outputs": len(output_rows),
        "files": {name: file_sha(output / name) for name in names},
        "detector_scores_computed": 0,
        "output_filtering": 0,
        "retries": 0,
        "protected_releases": 0,
    })
    return {
        "terminal": "CSW_AM_SCB_M1_CAR_D0_RAW_COMMITTED_PENDING_REVEAL",
        "closures": len(rows),
        "outputs": len(output_rows),
        "detector_scores_computed": 0,
    }


def reveal(
    source_commit: str, receipt_key_path: Path, carrier_key_path: Path,
    commitment_path: Path, commitment_commit: str,
    raw_manifest: Path, raw_commit: str, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    receipt, carrier = receipt_key_path.read_bytes(), carrier_key_path.read_bytes()
    validate_keys(
        config, source_commit, receipt, carrier, commitment_path, commitment_commit
    )
    committed(raw_manifest, raw_commit)
    raw = read_json(raw_manifest)
    value = {
        "schema": "exp33a.csw_am_scb_m1_car_d0.key_reveal.v1",
        "source_commit": source_commit,
        "raw_evidence_commit": raw_commit,
        "raw_manifest_sha256": file_sha(raw_manifest),
        "receipt_key_hex": receipt.hex(),
        "carrier_root_hex": carrier.hex(),
        "receipt_key_commitment": raw["receipt_key_commitment"],
        "carrier_key_commitment": raw["carrier_key_commitment"],
    }
    write_json(output, value)
    return {key: value[key] for key in value if not key.endswith("_hex")}


def output_authority(
    row: dict[str, Any], carrier_root: bytes, authority_spec: dict[str, Any],
    public_rows: dict[str, dict[str, Any]],
) -> bytes | None:
    public = public_rows[row["closure_id"]]
    if row["condition"] == "STATE":
        return CAR.derive_authority(
            carrier_root, row["source_z2_category"], "ACCEPT", authority_spec
        )
    if row["condition"] == "OFF":
        return None
    return baseline_authority(
        carrier_root, row["condition"], public["prompt_text"],
        public["natural"]["answer"],
    )


def analysis(
    config: dict[str, Any], receipt_key: bytes, carrier_root: bytes,
    raw: Path, *, detector_fn: Any | None = None, parser_fn: Any | None = None,
    reconstruct_generation: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(raw / "PUBLIC_ROWS.jsonl")
    primitives = read_rows(raw / "EXECUTION_ROWS.jsonl")
    packages = read_rows(raw / "RECEIPT_PACKAGES.jsonl")
    outputs = read_rows(raw / "OUTPUT_ROWS.jsonl")
    public_index = {row["closure_id"]: row for row in rows}
    primitive_index = {row["package_id"]: row for row in primitives}
    package_index = {row["package_id"]: row for row in packages}
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    authority_spec = CAR.read_json(CAR.AUTHORITY_PATH)
    receipt_results = []
    for package in packages:
        primitive = primitive_index[package["package_id"]]
        decisions = REC.evaluate_bundle(
            receipt_key, package["receipts"],
            expected_binding(primitive, config), protocol,
        )
        receipt_results.append({
            "package_id": package["package_id"],
            "closure_id": primitive["closure_id"],
            "side": primitive["role"],
            "decisions": decisions,
        })
    state_keys = [
        CAR.derive_authority(carrier_root, category, "ACCEPT", authority_spec)
        for category in range(16)
    ]
    if any(key is None for key in state_keys):
        raise ValueError("STATE_KEY")
    detect = detector_fn or (
        lambda text, key: A0.detector(
            text, key, config["channel"]["threshold"],
            config["channel"]["minimum_eligible_length"],
        )
    )
    parse = parser_fn or A0.parse_output
    detector_rows, summaries = [], []
    surface = load_surface(config) if reconstruct_generation else None
    generation_matches = 0
    for row in outputs:
        state_decisions = []
        for category, key in enumerate(state_keys):
            assert key is not None
            detected = detect(row["literal_output"], key)
            state_decisions.append({
                "category": category, **detected,
            })
            detector_rows.append({
                "output_id": row["output_id"],
                "authority_family": "STATE",
                "authority_category": category,
                **detected,
            })
        own = None
        if row["condition"] in ("PROMPT", "ANSWER", "GENERIC"):
            key = output_authority(row, carrier_root, authority_spec, public_index)
            assert key is not None
            own = detect(row["literal_output"], key)
            detector_rows.append({
                "output_id": row["output_id"],
                "authority_family": row["condition"],
                "authority_category": None,
                **own,
            })
        matching = state_decisions[row["source_z2_category"]]
        wrong = [
            item for item in state_decisions
            if item["category"] != row["source_z2_category"]
        ]
        scores = [item["score"] for item in state_decisions]
        parsed = parse(row["literal_output"])
        semantic_ok = (
            parsed["valid"]
            and parsed["values"]
            == {field: row["semantic_payload"][field] for field in A0.FIELDS}
        )
        generation_match = True
        if reconstruct_generation:
            key = output_authority(row, carrier_root, authority_spec, public_index)
            regenerated = A0.generate(
                surface, row["semantic_payload"],
                stream_seed=CAR.paired_nonce(row["closure_id"], row["draw"]),
                target_key=key, carrier_enabled=row["condition"] != "OFF",
                bias=config["channel"]["bias"], retain_operations=False,
            )
            generation_match = regenerated["literal_output"] == row["literal_output"]
        generation_matches += int(generation_match)
        summaries.append({
            "output_id": row["output_id"],
            "closure_id": row["closure_id"],
            "side": row["side"],
            "condition": row["condition"],
            "draw": row["draw"],
            "source_z2_category": row["source_z2_category"],
            "matching_decision": matching["decision"],
            "matching_score": matching["score"],
            "max_wrong_score": max(item["score"] for item in wrong),
            "wrong_accept_count": sum(item["decision"] == "ACCEPT" for item in wrong),
            "all_state_accept_count": sum(
                item["decision"] == "ACCEPT" for item in state_decisions
            ),
            "matching_rank": 1 + sum(score > matching["score"] for score in scores),
            "own_family_decision": None if own is None else own["decision"],
            "semantic_valid": semantic_ok,
            "generation_reconstruction_match": generation_match,
        })
    by_key = {
        (row["closure_id"], row["side"], row["condition"], row["draw"]): row
        for row in outputs
    }
    baseline_equal = 0
    baseline_total = 0
    payload_pair_equal = 0
    paired_nonce_equal = 0
    paired_total = 0
    for closure_id in public_index:
        for condition, draws in (
            ("STATE", range(4)), ("OFF", range(4)), ("PROMPT", range(1)),
            ("ANSWER", range(1)), ("GENERIC", range(1)),
        ):
            for draw in draws:
                paired_total += 1
                natural_row = by_key[(closure_id, "NATURAL", condition, draw)]
                alias_row = by_key[(closure_id, "ALIAS_Z2", condition, draw)]
                payload_pair_equal += int(
                    natural_row["semantic_payload_sha256"]
                    == alias_row["semantic_payload_sha256"]
                )
                paired_nonce_equal += int(
                    natural_row["stream_seed_sha256"] == alias_row["stream_seed_sha256"]
                )
        for condition, draws in (
            ("OFF", range(4)), ("PROMPT", range(1)),
            ("ANSWER", range(1)), ("GENERIC", range(1)),
        ):
            for draw in draws:
                baseline_total += 1
                baseline_equal += int(
                    by_key[(closure_id, "NATURAL", condition, draw)]["literal_output"]
                    == by_key[(closure_id, "ALIAS_Z2", condition, draw)]["literal_output"]
                )
    state_summaries = [row for row in summaries if row["condition"] == "STATE"]
    off_summaries = [row for row in summaries if row["condition"] == "OFF"]
    controls = [
        row for row in summaries if row["condition"] in ("PROMPT", "ANSWER", "GENERIC")
    ]
    side_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_summaries:
        side_groups[(row["closure_id"], row["side"])].append(row)
    side_success = {
        key: (
            sum(row["matching_decision"] == "ACCEPT" for row in values) >= 3
            and all(row["wrong_accept_count"] == 0 for row in values)
        )
        for key, values in side_groups.items()
    }
    closure_success = [
        side_success[(closure, "NATURAL")]
        and side_success[(closure, "ALIAS_Z2")]
        for closure in public_index
    ]
    receipt_all = sum(
        all(value == "ACCEPT" for value in row["decisions"].values())
        for row in receipt_results
    )
    categories_to_exact: dict[int, set[str]] = defaultdict(set)
    category_authorities: dict[int, set[bytes]] = defaultdict(set)
    for package in packages:
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        categories_to_exact[category].add(
            package["receipts"]["exact_z2"]["receipt_id"]
        )
        derived = CAR.derive_authority(carrier_root, category, "ACCEPT", authority_spec)
        assert derived is not None
        category_authorities[category].add(derived)
    primitive_by_side = {
        (row["closure_id"], row["role"]): row for row in primitives
    }
    baseline_authority_equal = 0
    baseline_authority_total = 0
    for public in rows:
        for family in ("PROMPT", "ANSWER", "GENERIC"):
            baseline_authority_total += 1
            natural_key = baseline_authority(
                carrier_root, family, public["prompt_text"], public["natural"]["answer"]
            )
            alias_key = baseline_authority(
                carrier_root, family, public["prompt_text"], public["natural"]["answer"]
            )
            baseline_authority_equal += int(natural_key == alias_key)
    first = copy.deepcopy(packages[0])
    first["receipts"]["abstract_z2"]["tag"] = "0" * 64
    invalid_decisions = REC.evaluate_bundle(
        receipt_key, first["receipts"],
        expected_binding(primitive_index[first["package_id"]], config), protocol,
    )
    missing = copy.deepcopy(packages[0]["receipts"])
    missing["abstract_z2"] = None
    missing_decisions = REC.evaluate_bundle(
        receipt_key, missing,
        expected_binding(primitive_index[packages[0]["package_id"]], config), protocol,
    )
    metric_core = {
        "closures": len(rows),
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "generated_outputs": len(outputs),
        "state_authority_scores": len(outputs) * 16,
        "own_family_control_scores": len(controls),
        "detector_decisions": len(detector_rows),
        "eligible_positions": sum(row["eligible_positions"] for row in outputs),
        "receipt_and_chain_accepts": receipt_all,
        "all16_authorities_distinct": len(set(state_keys)) == 16,
        "same_state_categories_with_multiple_exact_executions": sum(
            len(values) >= 2 for values in categories_to_exact.values()
        ),
        "same_state_authority_equal_categories": sum(
            len(values) == 1 for values in category_authorities.values()
        ),
        "payload_pair_equal": payload_pair_equal,
        "payload_pair_total": paired_total,
        "paired_nonce_equal": paired_nonce_equal,
        "paired_nonce_total": paired_total,
        "baseline_authority_equal": baseline_authority_equal,
        "baseline_authority_total": baseline_authority_total,
        "baseline_output_equal": baseline_equal,
        "baseline_output_total": baseline_total,
        "state_authority_switches": sum(
            primitive_by_side[(closure, "NATURAL")]["consumed_z2"]
            != primitive_by_side[(closure, "ALIAS_Z2")]["consumed_z2"]
            for closure in public_index
        ),
        "semantic_successes": sum(row["semantic_valid"] for row in summaries),
        "generation_reconstruction_matches": generation_matches,
        "matching_state_accepts": sum(
            row["matching_decision"] == "ACCEPT" for row in state_summaries
        ),
        "state_outputs": len(state_summaries),
        "state_outputs_with_any_wrong_accept": sum(
            row["wrong_accept_count"] > 0 for row in state_summaries
        ),
        "carrier_off_outputs_with_any_state_accept": sum(
            row["all_state_accept_count"] > 0 for row in off_summaries
        ),
        "carrier_off_outputs": len(off_summaries),
        "side_successes": sum(side_success.values()),
        "sides": len(side_success),
        "closure_pair_switch_successes": sum(closure_success),
        "paired_margin_mean": sum(
            row["matching_score"] - row["max_wrong_score"] for row in state_summaries
        ) / len(state_summaries),
        "control_matching_accepts": sum(
            row["own_family_decision"] == "ACCEPT" for row in controls
        ),
        "control_outputs": len(controls),
        "invalid_receipt_decision": invalid_decisions["abstract_z2_state"],
        "missing_receipt_decision": missing_decisions["abstract_z2_state"],
    }
    return metric_core, receipt_results, summaries, {
        "detector_rows": detector_rows,
        "invalid_decisions": invalid_decisions,
        "missing_decisions": missing_decisions,
    }


def gate_metrics(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, bool]:
    expected, gates = config["execution"], config["gates"]
    return {
        "complete_denominator": (
            metrics["closures"] == config["public_population"]["selected_closures"]
            and metrics["execution_packages"] == expected["SCB_execution_packages"]
            and metrics["receipt_envelopes"] == expected["receipt_envelopes"]
            and metrics["generated_outputs"] == expected["generated_outputs"]
            and metrics["state_authority_scores"] == expected["state_authority_scores"]
            and metrics["own_family_control_scores"] == expected["own_family_control_scores"]
            and metrics["detector_decisions"] == expected["detector_decisions"]
            and metrics["eligible_positions"] == expected["eligible_positions"]
        ),
        "receipt_and_chain": metrics["receipt_and_chain_accepts"]
        == metrics["execution_packages"],
        "authority_equivalence": (
            metrics["all16_authorities_distinct"]
            and metrics["same_state_categories_with_multiple_exact_executions"] == 16
            and metrics["same_state_authority_equal_categories"] == 16
            and metrics["baseline_authority_equal"]
            == metrics["baseline_authority_total"]
            and metrics["state_authority_switches"] == metrics["closures"]
        ),
        "payload_equality": metrics["payload_pair_equal"] == metrics["payload_pair_total"],
        "paired_nonce_equality": metrics["paired_nonce_equal"]
        == metrics["paired_nonce_total"],
        "baseline_output_equality": metrics["baseline_output_equal"]
        == metrics["baseline_output_total"],
        "semantic_validity": metrics["semantic_successes"] == metrics["generated_outputs"],
        "generation_reconstruction": metrics["generation_reconstruction_matches"]
        == metrics["generated_outputs"],
        "matching_power": metrics["matching_state_accepts"] / metrics["state_outputs"]
        >= gates["matching_state_output_power_min"],
        "wrong_state_false_accept": (
            metrics["state_outputs_with_any_wrong_accept"] / metrics["state_outputs"]
            <= gates["state_output_any_wrong_state_false_accept_max"]
        ),
        "carrier_off_false_accept": (
            metrics["carrier_off_outputs_with_any_state_accept"]
            / metrics["carrier_off_outputs"]
            <= gates["carrier_off_any_state_false_accept_max"]
        ),
        "side_success": metrics["side_successes"] / metrics["sides"]
        >= gates["side_success_fraction_min"],
        "closure_pair_switch": (
            metrics["closure_pair_switch_successes"] / metrics["closures"]
            >= gates["closure_pair_switch_success_fraction_min"]
        ),
        "paired_margin": metrics["paired_margin_mean"]
        >= gates["paired_matching_minus_best_wrong_margin_mean_min"],
        "control_power": metrics["control_matching_accepts"] / metrics["control_outputs"]
        >= gates["control_matching_power_min"],
        "receipt_controls": (
            metrics["invalid_receipt_decision"] == gates["invalid_receipt_control"]
            and metrics["missing_receipt_decision"] == gates["missing_receipt_control"]
        ),
    }


def validate_reveal(
    config: dict[str, Any], raw: Path, raw_commit: str,
    reveal_path: Path, reveal_commit: str,
) -> tuple[bytes, bytes]:
    committed(raw / "RAW_MANIFEST.json", raw_commit)
    committed(reveal_path, reveal_commit)
    raw_manifest, reveal = read_json(raw / "RAW_MANIFEST.json"), read_json(reveal_path)
    if (
        reveal["raw_evidence_commit"] != raw_commit
        or reveal["raw_manifest_sha256"] != file_sha(raw / "RAW_MANIFEST.json")
    ):
        raise ValueError("REVEAL_BINDING")
    receipt, carrier = bytes.fromhex(reveal["receipt_key_hex"]), bytes.fromhex(
        reveal["carrier_root_hex"]
    )
    if (
        key_commitment(receipt, config["keys"]["receipt_commitment_domain"])
        != raw_manifest["receipt_key_commitment"]
        or key_commitment(carrier, config["keys"]["carrier_commitment_domain"])
        != raw_manifest["carrier_key_commitment"]
    ):
        raise ValueError("REVEAL_KEYS")
    for name, expected in raw_manifest["files"].items():
        if file_sha(raw / name) != expected:
            raise ValueError("RAW_HASH:" + name)
    return receipt, carrier


def evaluate(
    source_commit: str, raw: Path, raw_commit: str,
    reveal_path: Path, reveal_commit: str, output: Path,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    receipt, carrier = validate_reveal(
        config, raw, raw_commit, reveal_path, reveal_commit
    )
    metrics, receipt_rows, summaries, detail = analysis(
        config, receipt, carrier, raw
    )
    gates = gate_metrics(config, metrics)
    passed = all(gates.values())
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    output.mkdir(parents=True)
    write_rows(output / "RECEIPT_VERDICTS.jsonl", receipt_rows)
    write_rows(output / "OUTPUT_SUMMARIES.jsonl", summaries)
    write_rows(output / "DETECTOR_ROWS.jsonl", detail["detector_rows"])
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_d0.production_result.v1",
        "metric_core": metrics,
        "metric_core_digest": digest(metrics),
        "gates": gates,
        "gate_digest": digest(gates),
        "passed": passed,
    }
    write_json(output / "PRODUCTION_RESULT.json", result)
    write_json(output / "CONTROL_RESULTS.json", {
        "schema": "exp33a.csw_am_scb_m1_car_d0.controls.v1",
        "invalid_receipt": detail["invalid_decisions"],
        "missing_receipt": detail["missing_decisions"],
    })
    return result


def reconcile(
    production_path: Path, reference_path: Path, output: Path,
) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    production, reference = read_json(production_path), read_json(reference_path)
    agreement = (
        production["metric_core_digest"] == reference["metric_core_digest"]
        and production["gate_digest"] == reference["gate_digest"]
        and production["metric_core"] == reference["metric_core"]
        and production["gates"] == reference["gates"]
    )
    passed = production["passed"] and reference["passed"] and agreement
    if not agreement:
        terminal_name = config["terminals"]["evaluator_mismatch"]
    elif passed:
        terminal_name = config["terminals"]["ready"]
    else:
        terminal_name = config["terminals"]["operating_point_failed"]
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_car_d0.terminal.v1",
        "terminal": terminal_name,
        "passed": passed,
        "official_reference_exact_agreement": agreement,
        "CAR_Q1_protocol_request_authorized": passed,
        "protected_role_release_authorized": False,
        "claim_bearing_output_provenance": False,
    }
    write_json(output, terminal)
    return terminal


def preflight() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    rows = selected_manifest_rows(config)
    authority = CAR.read_json(CAR.AUTHORITY_PATH)
    fixture_key = bytes(range(32))
    state = [
        CAR.derive_authority(fixture_key, category, "ACCEPT", authority)
        for category in range(16)
    ]
    row = rows[0]
    model, surface = M1.load_model(M1.read_json(M1.CONFIG_PATH)), load_surface(config)
    values = M1.tensors([row])
    with torch.no_grad():
        natural = model.run_natural(values)
        target = torch.tensor(
            [(row["natural"]["z2"] + 8) % 16], dtype=torch.long
        )
        alias = model.run_with_z2(values, target)
    rconfig = receipt_config(config)
    protocol = REC.read_json(REC.S0_PROTOCOL_PATH)
    receipt_key = bytes(reversed(range(32)))
    predicted = int(natural["z2_selected"][0])
    observed = {
        "z2_consumed": int(natural["z2_consumed"][0]),
        "z3_selected": int(natural["z3_selected"][0]),
        "z3_consumed": int(natural["z3_consumed"][0]),
        "answer": int(natural["answer_logits"].argmax(-1)[0]),
    }
    primitive, package = REC.receipt_package(
        receipt_key, protocol, rconfig, row, "NATURAL",
        observed, predicted, predicted,
    )
    decisions = REC.evaluate_bundle(
        receipt_key, package["receipts"], expected_binding(primitive, config), protocol,
    )
    payload = CAR.surface_payload(
        row["prompt_text"].encode(), row["natural"]["answer"],
        CAR.read_json(CAR.BOUNDARY_PATH),
    )
    state_key = CAR.derive_authority(
        fixture_key,
        package["receipts"]["abstract_z2"]["preimage"]["category"],
        decisions["abstract_z2_state"],
        authority,
    )
    if state_key is None:
        raise ValueError("PREFLIGHT_STATE_KEY")
    seed = CAR.paired_nonce(row["closure_id"], 0)
    generated_a = A0.generate(
        surface, payload, stream_seed=seed, target_key=state_key,
        carrier_enabled=True, bias=config["channel"]["bias"],
        retain_operations=False,
    )
    generated_b = A0.generate(
        surface, payload, stream_seed=seed, target_key=state_key,
        carrier_enabled=True, bias=config["channel"]["bias"],
        retain_operations=False,
    )
    detector_fixture = [
        A0.detector(
            generated_a["literal_output"], key, config["channel"]["threshold"],
            config["channel"]["minimum_eligible_length"],
        )
        for key in state
    ]
    checks = {
        "selection_cells": len(rows) == 128
        and len({canonical(row["selection_cell"]) for row in rows}) == 128,
        "one_row_per_closure": len({row["closure_id"] for row in rows}) == 128,
        "all16_authorities": len(set(state)) == 16,
        "invalid_no_authority": CAR.derive_authority(
            fixture_key, 0, "REJECT", authority
        ) is None,
        "frozen_channel": config["channel"]["retuned_parameters"] == []
        and config["channel"]["sampler"]
        == "FROZEN_A0_Q1_HISTORY_CONDITIONED_COUNTER_UNIFORM",
        "denominators": (
            config["execution"]["generated_outputs"] == 128 * 22
            and config["execution"]["state_authority_scores"] == 128 * 22 * 16
            and config["execution"]["detector_decisions"]
            == config["execution"]["state_authority_scores"] + 128 * 6
        ),
        "literal_receipt_path": all(value == "ACCEPT" for value in decisions.values()),
        "literal_state_intervention": int(alias["z2_consumed"][0]) == int(target[0]),
        "literal_realiser_determinism": (
            generated_a["literal_output"] == generated_b["literal_output"]
            and generated_a["eligible_positions"]
            == config["channel"]["eligible_positions_per_output"]
        ),
        "literal_all16_detector_path": (
            len(detector_fixture) == 16
            and all(item["decision"] in ("ACCEPT", "REJECT") for item in detector_fixture)
        ),
        "zero_protected": config["lifecycle"]["protected_releases"] == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--output", type=Path, required=True)
    sub.add_parser("preflight")
    keys = sub.add_parser("create-keys")
    keys.add_argument("--source-commit", required=True)
    keys.add_argument("--receipt-out", type=Path, required=True)
    keys.add_argument("--carrier-out", type=Path, required=True)
    keys.add_argument("--commitment-out", type=Path, required=True)
    run = sub.add_parser("execute")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--receipt-key", type=Path, required=True)
    run.add_argument("--carrier-key", type=Path, required=True)
    run.add_argument("--commitment", type=Path, required=True)
    run.add_argument("--commitment-commit", required=True)
    run.add_argument("--output", type=Path, required=True)
    reveal_parser = sub.add_parser("reveal")
    reveal_parser.add_argument("--source-commit", required=True)
    reveal_parser.add_argument("--receipt-key", type=Path, required=True)
    reveal_parser.add_argument("--carrier-key", type=Path, required=True)
    reveal_parser.add_argument("--commitment", type=Path, required=True)
    reveal_parser.add_argument("--commitment-commit", required=True)
    reveal_parser.add_argument("--raw-manifest", type=Path, required=True)
    reveal_parser.add_argument("--raw-commit", required=True)
    reveal_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--source-commit", required=True)
    evaluate_parser.add_argument("--raw", type=Path, required=True)
    evaluate_parser.add_argument("--raw-commit", required=True)
    evaluate_parser.add_argument("--reveal", type=Path, required=True)
    evaluate_parser.add_argument("--reveal-commit", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--production", type=Path, required=True)
    reconcile_parser.add_argument("--reference", type=Path, required=True)
    reconcile_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "select":
            result = select(args.output)
        elif args.command == "preflight":
            result = preflight()
        elif args.command == "create-keys":
            result = create_keys(
                args.source_commit, args.receipt_out, args.carrier_out,
                args.commitment_out,
            )
        elif args.command == "execute":
            result = execute(
                args.source_commit, args.receipt_key, args.carrier_key,
                args.commitment, args.commitment_commit, args.output,
            )
        elif args.command == "reveal":
            result = reveal(
                args.source_commit, args.receipt_key, args.carrier_key,
                args.commitment, args.commitment_commit, args.raw_manifest,
                args.raw_commit, args.output,
            )
        elif args.command == "evaluate":
            result = evaluate(
                args.source_commit, args.raw, args.raw_commit,
                args.reveal, args.reveal_commit, args.output,
            )
        else:
            result = reconcile(args.production, args.reference, args.output)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
