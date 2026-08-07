"""Fresh shared allocation excluding every recorded prior development identity."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from ..independent_oracle import oracle
from ..preconfirmation.allocator import ProtocolExample, allocate, shortcut_baselines
from ..preconfirmation.pairing import build_pairs
from ..spec import ALL_OPERATIONS
from ..task import Programme, interpret
from ..remediation.strict import object_sha256, sha256_file, strict_load_json, strict_load_jsonl, write_json, write_jsonl
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash


def _read_jsonl(path: Path) -> list[dict]:
    return strict_load_jsonl(path)


def development_exclusions(root: Path) -> tuple[dict[str, set[str]], list[dict]]:
    values = {key: set() for key in ("programme_ids", "example_ids", "pair_ids", "call_ids", "nonces")}
    sources = []
    for relative in load_protocol()["development_exclusion_sources"]:
        path = root / relative
        rows = _read_jsonl(path)
        for row in rows:
            for field, category in (("programme_id", "programme_ids"), ("example_id", "example_ids"),
                                    ("base_id", "example_ids"), ("source_id", "example_ids"),
                                    ("pair_id", "pair_ids"), ("call_id", "call_ids"), ("nonce", "nonces")):
                if type(row.get(field)) is str:
                    values[category].add(row[field])
        sources.append({"path": relative, "sha256": sha256_file(path), "rows": len(rows)})
    return values, sources


def _production(row: ProtocolExample) -> dict:
    operations = row.operation_tables
    p = Programme(*operations, partition="causal_evaluation", axis=row.axis)
    result = interpret(p, row.a, row.b, row.c, row.d)
    return {name: getattr(result, name) for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": result.joint_answer}


def build(root: Path, output: Path, entropy) -> tuple[dict, list[dict], dict]:
    protocol = load_protocol()
    exclusions, sources = development_exclusions(root)
    allocation = allocate(root, entropy, protocol["allocation"]["programme_counts"], DEV_NAMESPACE, exclusions)
    mismatches = 0
    for row in allocation["examples"]:
        expected = oracle(tuple(ALL_OPERATIONS[i] for i in row.operation_ids), (row.a, row.b, row.c, row.d))
        independent = {name: getattr(row, name) for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": row.answer}
        expected_normalized = {name: expected[name] for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": expected["joint_answer"]}
        mismatches += expected_normalized != independent
        # A genuinely separate production interpreter comparison.
        production = _production(row)
        observed = {name: getattr(row, name) for name in ("z1", "z2", "z3", "n1", "n2", "n3")} | {"answer": row.answer}
        mismatches += production != observed
    if mismatches:
        raise RuntimeError(f"oracle disagreement: {mismatches}")
    pair_counts = {key: protocol["pairs"][key] for key in ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")}
    pairs = build_pairs(allocation, entropy, pair_counts, DEV_NAMESPACE)
    directory = output / "allocation"
    write_jsonl(directory / "programmes.jsonl", allocation["programmes"])
    write_jsonl(directory / "examples.jsonl", (row.as_dict() for row in allocation["examples"]))
    write_jsonl(directory / "pairs.jsonl", pairs)
    exclusion_record = {"schema": "exp33ae.dev_only_reliability_exclusions.v1", "namespace": DEV_NAMESPACE,
                        "claim_bearing": False, "sources": sources,
                        "counts": {key: len(value) for key, value in exclusions.items()},
                        "programme_ids": sorted(exclusions["programme_ids"])}
    write_json(directory / "prior-development-exclusions.json", exclusion_record)
    files = {name: sha256_file(directory / name) for name in
             ("programmes.jsonl", "examples.jsonl", "pairs.jsonl", "prior-development-exclusions.json")}
    manifest = {
        "schema": "exp33ae.dev_only_reliability_allocation_manifest.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "stage_id": entropy.stage_id, "protocol_sha256": protocol_hash(),
        "programme_count": len(allocation["programmes"]), "example_count": len(allocation["examples"]),
        "pair_count": len(pairs), "partition_counts": dict(sorted(Counter(r.partition for r in allocation["examples"]).items())),
        "pair_strata": dict(sorted(Counter(r["stratum"] for r in pairs).items())),
        "oracle_disagreements": mismatches, "shortcut_baselines": None,
        "heldout_baseline_results_generated": False,
        "prior_exclusion_counts": exclusion_record["counts"], "files": files,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    manifest["allocation_sha256"] = object_sha256(manifest)
    write_json(directory / "manifest.json", manifest)
    return allocation, pairs, manifest


def finalize_shortcut_baselines(output: Path, entropy) -> dict:
    """Run held-out shortcut outcomes only after all 20 checkpoints exist."""
    allocation, _, manifest = load(output)
    if manifest["shortcut_baselines"] is not None or manifest["heldout_baseline_results_generated"] is not False:
        raise RuntimeError("shortcut baselines already finalized")
    manifest["shortcut_baselines"] = shortcut_baselines(allocation, entropy)
    manifest["heldout_baseline_results_generated"] = True
    manifest.pop("allocation_sha256")
    manifest["allocation_sha256"] = object_sha256(manifest)
    write_json(output / "allocation" / "manifest.json", manifest)
    return manifest


def load(output: Path) -> tuple[dict, list[dict], dict]:
    directory = output / "allocation"
    manifest = strict_load_json(directory / "manifest.json")
    programmes = strict_load_jsonl(directory / "programmes.jsonl")
    examples = []
    for row in strict_load_jsonl(directory / "examples.jsonl"):
        row["operation_ids"] = tuple(row["operation_ids"])
        examples.append(ProtocolExample(**row))
    return {"schema": "exp33ae.dev_only_reliability_allocation.v1", "namespace": DEV_NAMESPACE,
            "stage_id": manifest["stage_id"], "claim_bearing": False,
            "programmes": programmes, "examples": examples}, strict_load_jsonl(directory / "pairs.jsonl"), manifest
