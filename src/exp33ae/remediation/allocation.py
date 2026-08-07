"""Prospective DEV allocation, independent-oracle agreement, and persistence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..preconfirmation.allocator import ProtocolExample, allocate, causal_signature, development_exclusions, shortcut_baselines
from ..preconfirmation.pairing import build_pairs
from ..task import Programme, interpret
from .entropy import DevMasterEntropy
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .strict import object_sha256, sha256_file, strict_load_json, strict_load_jsonl, write_json, write_jsonl


def _production_values(row: ProtocolExample) -> dict:
    operations = row.operation_tables
    programme = Programme(*operations, partition=row.partition if row.partition != "clean_target_axis" and row.partition != "clean_parallel_axis" else "causal_evaluation", axis=row.axis)
    value = interpret(programme, row.a, row.b, row.c, row.d)
    return {name: getattr(value, name) for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": value.joint_answer}


def allocation_counts() -> dict[str, int]:
    return dict(load_protocol()["allocation"]["programme_counts"])


def primary_pair_counts() -> dict[str, int]:
    values = load_protocol()["pairs"]
    return {name: values[name] for name in ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")}


def build_allocation(root: Path, output: Path, entropy: DevMasterEntropy) -> tuple[dict, list[dict], dict]:
    value = allocate(root, entropy, allocation_counts(), DEV_NAMESPACE)
    examples = value["examples"]
    mismatches = 0
    for row in examples:
        production = _production_values(row)
        observed = {name: getattr(row, name) for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": row.answer}
        mismatches += production != observed
    if mismatches:
        raise RuntimeError(f"production/independent oracle mismatches: {mismatches}")
    pairs = build_pairs(value, entropy, primary_pair_counts(), DEV_NAMESPACE)
    baselines = shortcut_baselines(value, entropy)
    allocation_dir = output / "allocation"
    write_jsonl(allocation_dir / "programmes.jsonl", value["programmes"])
    write_jsonl(allocation_dir / "examples.jsonl", (row.as_dict() for row in examples))
    write_jsonl(allocation_dir / "pairs.jsonl", pairs)
    exclusions = {key: sorted(items) for key, items in development_exclusions(root).items()}
    write_json(allocation_dir / "development-exclusions.json", {
        "schema": "exp33ae.dev_only_development_exclusions.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        **exclusions,
    })
    files = {
        name: sha256_file(allocation_dir / name)
        for name in ("programmes.jsonl", "examples.jsonl", "pairs.jsonl", "development-exclusions.json")
    }
    manifest = {
        "schema": "exp33ae.dev_only_allocation_manifest.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "stage_id": entropy.stage_id,
        "protocol_sha256": protocol_hash(),
        "programme_count": len(value["programmes"]),
        "example_count": len(examples),
        "pair_count": len(pairs),
        "partition_counts": dict(sorted(Counter(row.partition for row in examples).items())),
        "pair_strata": dict(sorted(Counter(row["stratum"] for row in pairs).items())),
        "production_independent_oracle_mismatches": mismatches,
        "shortcut_baselines": baselines,
        "files": files,
        "development_exclusion_counts": {key: len(items) for key, items in exclusions.items()},
        "confirmation_objects_created": 0,
        "confirmation_objects_accessed": 0,
    }
    manifest["allocation_sha256"] = object_sha256(manifest)
    write_json(allocation_dir / "manifest.json", manifest)
    return value, pairs, manifest


def load_allocation(output: Path) -> tuple[dict, list[dict], dict]:
    directory = output / "allocation"
    manifest = strict_load_json(directory / "manifest.json")
    programmes = strict_load_jsonl(directory / "programmes.jsonl")
    rows = []
    for value in strict_load_jsonl(directory / "examples.jsonl"):
        value["operation_ids"] = tuple(value["operation_ids"])
        rows.append(ProtocolExample(**value))
    pairs = strict_load_jsonl(directory / "pairs.jsonl")
    allocation = {
        "schema": "exp33ae.dev_only_allocation.v2",
        "namespace": DEV_NAMESPACE,
        "stage_id": manifest["stage_id"],
        "claim_bearing": False,
        "programmes": programmes,
        "examples": rows,
    }
    return allocation, pairs, manifest
