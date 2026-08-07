"""DEV_ONLY reproducibility cleanup for the immutable reliability-study v1 evidence.

This module never trains, allocates, evaluates checkpoints, or creates confirmation
objects.  It validates and projects the already immutable 78,400-row v1 ledger.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from importlib import metadata
from itertools import zip_longest
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable

from ..remediation.evaluator import _seed_status
from ..remediation.normative import POSITIVE_ANALOGUE
from ..remediation.strict import (
    StrictValidationError,
    canonical_json,
    expect_sha256,
    object_sha256,
    sha256_file,
    strict_load_json,
    strict_loads,
    write_json,
)
from .evaluator import gate_results, select_recipe


SOURCE_TAG = "dev-only-33a-e-training-reliability-candidate-v1"
SOURCE_TAG_OBJECT = "e735d5e91930629f903aed13dbdf063c2e3066d7"
SOURCE_COMMIT = "e5ee6a99807c02687a9824aa05ad300a9e1e08a8"
SOURCE_TREE = "21e08b48148ec1e7a98b3f1f5e7c723752e29199"
CLEANUP_TAG_V1 = "dev-only-33a-e-reliability-reproducibility-cleanup-candidate-v1"
CLEANUP_TAG = "dev-only-33a-e-reliability-reproducibility-cleanup-candidate-v2"
CLEANUP_TAGS = frozenset({CLEANUP_TAG_V1, CLEANUP_TAG})
V1_RELATIVE = Path("artifacts/dev_only/33a-e-training-reliability-v1")
CLEANUP_RELATIVE = Path("artifacts/dev_only/33a-e-reliability-reproducibility-cleanup-v1")
PROTOCOL_RELATIVE = Path("protocol/dev_only/33a-e-reliability-reproducibility-cleanup-v1.json")
DEV_NAMESPACE = "DEV_ONLY_33A_E_RELIABILITY_REPRODUCIBILITY_CLEANUP"
EXPECTED_ROWS = 78_400

SCIENTIFIC_SCHEMA = "exp33ae.dev_only_reliability_scientific_row_projection.v1"
DIAGNOSTIC_SCHEMA = "exp33ae.dev_only_reliability_diagnostic_row_projection.v1"
SCIENTIFIC_DOMAIN = b"exp33ae.dev_only_reliability_scientific_projection.v1\x00"
DIAGNOSTIC_DOMAIN = b"exp33ae.dev_only_reliability_diagnostic_projection.v1\x00"

RAW_KEYS = frozenset({
    "schema", "namespace", "stage_id", "claim_bearing", "call_id", "nonce", "order",
    "registry_row_sha256", "registry_payload_commitment", "model_order", "recipe_id",
    "recipe_order", "seed_order", "model_seed_identity", "checkpoint_file_sha256",
    "checkpoint_tensor_sha256", "source_manifest_sha256", "dependency_manifest_sha256",
    "allocation_sha256", "representation_sha256", "control_manifest_sha256",
    "training_manifest_sha256", "programme_id", "causal_signature", "base_id", "source_id",
    "pair_id", "pair_stratum", "method", "control_id", "basis_sha256", "decoder_sha256",
    "actual_layer", "actual_position", "actual_rank", "actual_strength", "attempt_number",
    "attempted", "completed", "dropped", "error", "base_prediction", "source_prediction",
    "intervened_prediction", "decoded_z2", "intervened_z3", "intervened_n3", "applied_norm",
    "expected_applied_norm", "norm_absolute_error", "base_hidden_norm",
    "activation_delta_sha256", "hook_consumed_by_downstream", "raw_row_sha256",
})
DIAGNOSTIC_SOURCE_FIELDS = frozenset({"base_hidden_norm", "raw_row_sha256"})
SCIENTIFIC_SOURCE_FIELDS = RAW_KEYS - DIAGNOSTIC_SOURCE_FIELDS - {"schema"}

INTEGER_FIELDS = frozenset({
    "order", "model_order", "recipe_order", "seed_order", "model_seed_identity", "actual_layer",
    "actual_position", "actual_rank", "attempt_number", "base_prediction", "source_prediction",
    "intervened_prediction", "decoded_z2", "intervened_z3", "intervened_n3",
})
FLOAT_FIELDS = frozenset({
    "actual_strength", "applied_norm", "expected_applied_norm", "norm_absolute_error",
    "base_hidden_norm",
})
BOOLEAN_FIELDS = frozenset({"claim_bearing", "attempted", "completed", "dropped", "hook_consumed_by_downstream"})
NULLABLE_STRING_FIELDS = frozenset({"control_id", "error"})
SHA256_FIELDS = frozenset({
    "registry_row_sha256", "registry_payload_commitment", "checkpoint_file_sha256",
    "checkpoint_tensor_sha256", "source_manifest_sha256", "dependency_manifest_sha256",
    "allocation_sha256", "representation_sha256", "control_manifest_sha256",
    "training_manifest_sha256", "causal_signature", "basis_sha256", "decoder_sha256",
    "activation_delta_sha256", "raw_row_sha256",
})

RAW_TO_REGISTRY = {
    "call_id": "call_id", "nonce": "nonce", "order": "order", "model_order": "model_order",
    "recipe_id": "recipe_id", "recipe_order": "recipe_order", "seed_order": "seed_order",
    "model_seed_identity": "model_seed_identity", "checkpoint_file_sha256": "checkpoint_file_sha256",
    "checkpoint_tensor_sha256": "checkpoint_tensor_sha256", "source_manifest_sha256": "source_manifest_sha256",
    "dependency_manifest_sha256": "dependency_manifest_sha256", "allocation_sha256": "allocation_sha256",
    "representation_sha256": "representation_sha256", "control_manifest_sha256": "control_manifest_sha256",
    "training_manifest_sha256": "training_manifest_sha256", "programme_id": "programme_id",
    "causal_signature": "causal_signature", "base_id": "base_id", "source_id": "source_id",
    "pair_id": "pair_id", "pair_stratum": "pair_stratum", "method": "method",
    "control_id": "control_id", "basis_sha256": "basis_sha256", "decoder_sha256": "decoder_sha256",
    "actual_layer": "layer", "actual_position": "position", "actual_rank": "rank",
    "actual_strength": "strength", "expected_applied_norm": "expected_applied_norm",
    "registry_row_sha256": "registry_row_sha256", "registry_payload_commitment": "registry_payload_commitment",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def selected_python() -> Path:
    """Return the explicit override or this running interpreter; never infer clone/.venv."""
    configured = os.environ.get("EXP33AE_PYTHON")
    candidate = configured if configured else sys.executable
    located = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
    if not located:
        raise RuntimeError("EXP33AE_PYTHON does not resolve to an executable")
    # Preserve the invoked interpreter path: resolving a venv launcher symlink
    # would bypass its pyvenv.cfg and silently change the dependency context.
    path = Path(located).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("selected Python is not an executable file")
    return path


def clone_subprocess_environment(clone_source: Path) -> dict[str, str]:
    """Preserve platform loader state but replace Python import-path state exactly."""
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(clone_source.resolve())
    return environment


def dependency_preflight() -> dict:
    """Observe and check dependencies without installing, resolving, or downloading."""
    contract = {
        "python_major_minor": [3, 12],
        "torch": "2.7.1",
        "numpy": "2.2.6",
        "pytest": "8.4.1",
    }
    observed = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(selected_python()),
        "torch": metadata.version("torch"),
        "numpy": metadata.version("numpy"),
        "pytest": metadata.version("pytest"),
        "platform": platform.platform(),
    }
    checks = {
        "python": list(sys.version_info[:2]) == contract["python_major_minor"],
        "torch": observed["torch"] == contract["torch"],
        "numpy": observed["numpy"] == contract["numpy"],
        "pytest": observed["pytest"] == contract["pytest"],
    }
    return {
        "schema": "exp33ae.dev_only_offline_dependency_preflight.v1",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "contract": contract,
        "observed": observed,
        "checks": checks,
        "contract_satisfied": all(checks.values()),
        "installation_attempted": False,
        "downloads_attempted": False,
        "dependency_reconstruction_claimed": False,
        "boundary": (
            "A clean clone can run with an externally supplied locked interpreter. "
            "Reuse of that interpreter is same-locked-environment execution, not independent dependency reconstruction."
        ),
    }


def _validate_raw_types(row: dict, ordinal: int) -> None:
    if set(row) != RAW_KEYS:
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} exact keys")
    for key, value in row.items():
        if key in INTEGER_FIELDS:
            valid = type(value) is int
        elif key in FLOAT_FIELDS:
            valid = type(value) is float and value == value and abs(value) != float("inf")
        elif key in BOOLEAN_FIELDS:
            valid = type(value) is bool
        elif key in NULLABLE_STRING_FIELDS:
            valid = value is None or type(value) is str
        else:
            valid = type(value) is str
        if not valid:
            raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} field {key} type")
    for field in SHA256_FIELDS:
        expect_sha256(row[field], f"raw row {ordinal} {field}")
    if row["schema"] != "exp33ae.dev_only_reliability_raw.v1":
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} source schema")
    if row["namespace"] != "DEV_ONLY_33A_E_TRAINING_RELIABILITY" or row["claim_bearing"] is not False:
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} namespace")
    if not (0 <= row["model_order"] < 20 and 0 <= row["recipe_order"] < 4 and 0 <= row["seed_order"] < 5):
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} model identity range")
    if not (0 <= row["actual_layer"] <= 3 and 0 <= row["actual_position"] <= 16 and row["actual_rank"] == 2):
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} intervention range")
    if any(not 0 <= row[field] <= maximum for field, maximum in (
        ("base_prediction", 3), ("source_prediction", 3), ("intervened_prediction", 3),
        ("decoded_z2", 1), ("intervened_z3", 1), ("intervened_n3", 1),
    )):
        raise StrictValidationError("PROJECTION_SCHEMA", f"raw row {ordinal} prediction range")


def validate_projection_source_row(row: dict, plan: dict, ordinal: int, seen: set[str]) -> None:
    _validate_raw_types(row, ordinal)
    if row["order"] != ordinal:
        raise StrictValidationError("PROJECTION_ORDER", f"row {ordinal} records order {row['order']}")
    if row["call_id"] in seen:
        raise StrictValidationError("PROJECTION_DUPLICATE", f"call {row['call_id']}")
    if any(row[raw_key] != plan[registry_key] for raw_key, registry_key in RAW_TO_REGISTRY.items()):
        raise StrictValidationError("PROJECTION_LINKAGE", f"raw/registry row {ordinal}")
    if row["attempted"] is not True or row["completed"] is not True or row["dropped"] is not False:
        raise StrictValidationError("PROJECTION_COMPLETION", f"row {ordinal}")
    if row["attempt_number"] != 1 or row["error"] is not None:
        raise StrictValidationError("PROJECTION_COMPLETION", f"row {ordinal} attempt")
    unhashed = dict(row)
    observed_hash = unhashed.pop("raw_row_sha256")
    if object_sha256(unhashed) != observed_hash:
        raise StrictValidationError("PROJECTION_LINKAGE", f"raw hash row {ordinal}")
    tolerance = plan["norm_tolerance_absolute"] + plan["norm_tolerance_relative"] * abs(plan["expected_applied_norm"])
    if abs(row["applied_norm"] - plan["expected_applied_norm"]) > tolerance:
        raise StrictValidationError("PROJECTION_LINKAGE", f"norm row {ordinal}")
    seen.add(row["call_id"])


def scientific_row_projection(row: dict) -> dict:
    projected = {
        "schema": SCIENTIFIC_SCHEMA,
        "source_schema": row["schema"],
    }
    projected.update({key: row[key] for key in sorted(SCIENTIFIC_SOURCE_FIELDS)})
    return projected


def diagnostic_row_projection(row: dict) -> dict:
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "source_schema": row["schema"],
        "namespace": row["namespace"],
        "stage_id": row["stage_id"],
        "order": row["order"],
        "call_id": row["call_id"],
        "base_hidden_norm": row["base_hidden_norm"],
        "source_raw_row_sha256": row["raw_row_sha256"],
    }


def canonical_projection_line(value: dict) -> bytes:
    return canonical_json(value).encode("utf-8") + b"\n"


def _empty_accumulator() -> dict:
    return {
        "identity": None,
        "task_target": 0, "task_parallel": 0, "decode": 0, "iia_A": 0, "same_state": 0, "iia_D": 0,
        "programme_correct": Counter(), "programme_total": Counter(),
        "transport": Counter(), "descendant": Counter(), "non_descendant": Counter(),
        "control_iia": Counter(), "base_groups": Counter(), "source_groups": Counter(),
        "group_denominators": Counter(),
    }


def _accumulate(accumulator: dict, plan: dict, row: dict) -> None:
    if accumulator["identity"] is None:
        accumulator["identity"] = {
            "model_order": plan["model_order"], "recipe_id": plan["recipe_id"],
            "recipe_order": plan["recipe_order"], "seed_order": plan["seed_order"],
            "model_seed_identity": plan["model_seed_identity"],
        }
    group = repr((plan["method"], plan["control_id"], plan["pair_stratum"]))
    accumulator["group_denominators"][group] += 1
    accumulator["base_groups"][group] += row["base_prediction"] == plan["base_answer"]
    accumulator["source_groups"][group] += row["source_prediction"] == plan["source_answer"]
    if plan["method"] == "clean_task_decode":
        if plan["axis"] == "blinded_within_run_target_composition":
            accumulator["task_target"] += row["intervened_prediction"] == plan["expected_answer"]
        elif plan["axis"] == "blinded_within_run_parallel_composition":
            accumulator["task_parallel"] += row["intervened_prediction"] == plan["expected_answer"]
        else:
            raise StrictValidationError("PROJECTION_LINKAGE", "unknown clean axis")
        accumulator["programme_total"][plan["programme_id"]] += 1
        accumulator["programme_correct"][plan["programme_id"]] += row["intervened_prediction"] == plan["expected_answer"]
        accumulator["decode"] += row["decoded_z2"] == plan["expected_z2"]
    elif plan["method"] == "declared_iit_alignment":
        stratum = plan["pair_stratum"]
        accumulator["transport"][stratum] += row["decoded_z2"] == plan["expected_z2"]
        accumulator["descendant"][stratum] += row["intervened_z3"] == plan["expected_z3"]
        accumulator["non_descendant"][stratum] += row["intervened_n3"] == plan["expected_n3"]
        if stratum == "A_answer_changing":
            accumulator["iia_A"] += row["intervened_prediction"] == plan["expected_answer"]
        elif stratum == "C_same_state":
            accumulator["same_state"] += row["intervened_prediction"] == plan["base_answer"]
        elif stratum == "D_same_observable_answer":
            accumulator["iia_D"] += row["intervened_prediction"] == plan["expected_answer"]
    elif plan["control_id"] is not None:
        accumulator["control_iia"][plan["control_id"]] += row["intervened_prediction"] == plan["expected_answer"]
    else:
        raise StrictValidationError("PROJECTION_LINKAGE", "unknown method")


def _finalize_metrics(accumulator: dict, shortcuts: dict) -> dict:
    strata = ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")
    return {
        "task_target": accumulator["task_target"],
        "task_parallel": accumulator["task_parallel"],
        "worst_programme": min(accumulator["programme_correct"].values()),
        "decode": accumulator["decode"],
        "iia_A": accumulator["iia_A"],
        "transport": {name: accumulator["transport"][name] for name in strata},
        "descendant": {name: accumulator["descendant"][name] for name in strata},
        "non_descendant": {name: accumulator["non_descendant"][name] for name in strata},
        "same_state": accumulator["same_state"],
        "iia_D": accumulator["iia_D"],
        "control_iia": {key: accumulator["control_iia"][key] for key in sorted(accumulator["control_iia"])},
        "base_groups": dict(accumulator["base_groups"]),
        "source_groups": dict(accumulator["source_groups"]),
        "group_denominators": dict(accumulator["group_denominators"]),
        "shortcut_target": max(value["correct"] for value in shortcuts["clean_target_axis"].values()),
        "shortcut_parallel": max(value["correct"] for value in shortcuts["clean_parallel_axis"].values()),
    }


def analyze_v1_ledger(root: Path | None = None) -> dict:
    root = root or repository_root()
    output = root / V1_RELATIVE
    raw_path = output / "access/buffer.jsonl"
    registry_path = output / "registry/registry.jsonl"
    scientific_hash = sha256()
    scientific_commitment = sha256(SCIENTIFIC_DOMAIN)
    diagnostic_hash = sha256()
    diagnostic_commitment = sha256(DIAGNOSTIC_DOMAIN)
    seen: set[str] = set()
    accumulators = [_empty_accumulator() for _ in range(20)]
    count = 0
    with raw_path.open() as raw_handle, registry_path.open() as registry_handle:
        for ordinal, pair in enumerate(zip_longest(raw_handle, registry_handle)):
            raw_line, registry_line = pair
            if raw_line is None or registry_line is None:
                raise StrictValidationError("PROJECTION_OMISSION", f"stream length differs at {ordinal}")
            if not raw_line.strip() or not registry_line.strip():
                raise StrictValidationError("PROJECTION_SCHEMA", f"blank line {ordinal}")
            row = strict_loads(raw_line)
            plan = strict_loads(registry_line)
            if type(row) is not dict or type(plan) is not dict:
                raise StrictValidationError("PROJECTION_SCHEMA", f"row {ordinal} object")
            validate_projection_source_row(row, plan, ordinal, seen)
            scientific_line = canonical_projection_line(scientific_row_projection(row))
            diagnostic_line = canonical_projection_line(diagnostic_row_projection(row))
            scientific_hash.update(scientific_line); scientific_commitment.update(scientific_line)
            diagnostic_hash.update(diagnostic_line); diagnostic_commitment.update(diagnostic_line)
            _accumulate(accumulators[row["model_order"]], plan, row)
            count += 1
    if count != EXPECTED_ROWS or len(seen) != EXPECTED_ROWS:
        raise StrictValidationError("PROJECTION_OMISSION", f"rows={count} unique={len(seen)}")
    allocation = strict_load_json(output / "allocation/manifest.json")
    models = []
    for accumulator in accumulators:
        if accumulator["identity"] is None:
            raise StrictValidationError("PROJECTION_OMISSION", "empty model")
        metrics = _finalize_metrics(accumulator, allocation["shortcut_baselines"])
        gates = gate_results(metrics)
        model_status = _seed_status(metrics)
        identity = accumulator["identity"]
        models.append({
            **identity,
            "model_status": model_status,
            "all_gates_pass": model_status == POSITIVE_ANALOGUE and all(gates.values()),
            "failed_gates": [key for key, value in gates.items() if not value],
            "gates": gates,
            "metrics": metrics,
        })
    recipes, selected, status = select_recipe(models)
    reconstructed = {
        "status": status, "selected_recipe": selected, "recipes": recipes, "models": models,
        "registry_calls": count, "attempted_calls": count, "completed_calls": count,
    }
    stored = strict_load_json(output / "derived-status.json")
    for field in reconstructed:
        if reconstructed[field] != stored[field]:
            raise StrictValidationError("PROJECTION_RECONSTRUCTION", field)
    return {
        "row_count": count,
        "unique_call_ids": len(seen),
        "scientific_payload_sha256": scientific_hash.hexdigest(),
        "scientific_payload_commitment_sha256": scientific_commitment.hexdigest(),
        "diagnostic_payload_sha256": diagnostic_hash.hexdigest(),
        "diagnostic_payload_commitment_sha256": diagnostic_commitment.hexdigest(),
        "reconstructed_status": status,
        "reconstructed_selected_recipe": selected,
        "reconstructed_status_sha256": object_sha256(reconstructed),
    }


def _tag_snapshot(root: Path, exclude_cleanup: bool) -> str:
    lines = _git(root, "for-each-ref", "--sort=refname", "--format=%(objectname) %(refname)", "refs/tags").splitlines()
    if exclude_cleanup:
        stable = []
        for line in lines:
            _, reference = line.split(" ", 1)
            name = reference.rsplit("refs/tags/", 1)[-1]
            if name in CLEANUP_TAGS:
                continue
            commit = _git(root, "rev-parse", f"{reference}^{{commit}}")
            # The cleanup snapshot commits the predecessor tag set at the
            # immutable reliability-science source.  Later lineage tags must
            # not retroactively alter that historical manifest.
            strict_descendant = commit != SOURCE_COMMIT and subprocess.run(
                ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, commit], cwd=root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).returncode == 0
            if not strict_descendant:
                stable.append(line)
        lines = stable
    return "\n".join(lines) + "\n"


def build_supplemental_manifest(root: Path | None = None, destination: Path | None = None) -> dict:
    root = root or repository_root()
    destination = destination or root / CLEANUP_RELATIVE / "supplemental-manifest.json"
    analysis = analyze_v1_ledger(root)
    output = root / V1_RELATIVE
    subtree_listing = _git(root, "ls-tree", "-r", SOURCE_TAG, str(V1_RELATIVE)) + "\n"
    value = {
        "schema": "exp33ae.dev_only_reliability_reproducibility_cleanup_manifest.v1",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "source": {
            "tag": SOURCE_TAG, "tag_object": _git(root, "rev-parse", f"{SOURCE_TAG}^{{tag}}"),
            "commit": _git(root, "rev-parse", f"{SOURCE_TAG}^{{commit}}"),
            "tree": _git(root, "rev-parse", f"{SOURCE_TAG}^{{tree}}"),
            "v1_scientific_subtree_listing_sha256": sha256(subtree_listing.encode("utf-8")).hexdigest(),
            "v1_scientific_subtree_entries": len(subtree_listing.rstrip("\n").splitlines()),
        },
        "immutable_v1_identities": {
            "registry_jsonl_sha256": sha256_file(output / "registry/registry.jsonl"),
            "registry_manifest_sha256": sha256_file(output / "registry/manifest.json"),
            "raw_ledger_jsonl_sha256": sha256_file(output / "access/buffer.jsonl"),
            "derived_status_file_sha256": sha256_file(output / "derived-status.json"),
            "release_envelope_file_sha256": sha256_file(output / "release-envelope.json"),
        },
        "cleanup_protocol_file_sha256": sha256_file(root / PROTOCOL_RELATIVE),
        "projection": {
            "scientific_schema": SCIENTIFIC_SCHEMA,
            "diagnostic_schema": DIAGNOSTIC_SCHEMA,
            "canonical_encoding": "UTF-8 JSON Lines; keys lexicographically sorted; no insignificant whitespace; separators comma/colon; ensure_ascii=false; finite JSON numbers only; LF after every row",
            "ordering": "strict source order with order integers exactly 0..78399",
            "scientific_fields": ["schema", "source_schema", *sorted(SCIENTIFIC_SOURCE_FIELDS)],
            "diagnostic_fields": ["schema", "source_schema", "namespace", "stage_id", "order", "call_id", "base_hidden_norm", "source_raw_row_sha256"],
            "scientific_hash_construction": "SHA-256 over the exact canonical scientific JSONL payload bytes",
            "scientific_commitment_construction": "SHA-256(domain UTF-8 bytes plus NUL plus exact canonical scientific JSONL payload bytes)",
            "diagnostic_hash_construction": "SHA-256 over the exact canonical diagnostic JSONL payload bytes",
            "diagnostic_commitment_construction": "SHA-256(domain UTF-8 bytes plus NUL plus exact canonical diagnostic JSONL payload bytes)",
            **analysis,
        },
        "predecessor_tag_snapshot": {
            "count": len(_tag_snapshot(root, True).splitlines()),
            "canonical_sha256": sha256(_tag_snapshot(root, True).encode("utf-8")).hexdigest(),
            "excluded_cleanup_candidate_tags": sorted(CLEANUP_TAGS),
        },
        "confirmation": {
            "master_entropy": 0, "derived_seeds": 0, "programmes": 0, "datasets": 0,
            "pair_allocations": 0, "registries": 0, "checkpoints": 0, "calls": 0,
            "results": 0, "access_records": 0, "accesses": 0,
        },
        "assertions": {
            "modifies_v1_raw_evidence": False,
            "replaces_v1_raw_evidence": False,
            "modifies_v1_release_identity": False,
            "replaces_v1_release_identity": False,
            "modifies_v1_status_or_conclusion": False,
            "replaces_v1_status_or_conclusion": False,
            "projection_is_trained_model_evidence": False,
            "synthetic_mutations_are_trained_model_evidence": False,
            "protocol_sealed": False,
            "confirmation_authorized": False,
            "independent_review_completed_by_repository": False,
        },
        "status": "READY_FOR_INDEPENDENT_CLEANUP_REVIEW",
    }
    if value["source"] != {
        **value["source"], "tag_object": SOURCE_TAG_OBJECT, "commit": SOURCE_COMMIT, "tree": SOURCE_TREE,
    }:
        raise RuntimeError("immutable source tag identity changed")
    value["manifest_sha256"] = object_sha256(value)
    write_json(destination, value)
    return value


def verify_supplemental_manifest(root: Path | None = None, manifest_path: Path | None = None) -> dict:
    root = root or repository_root()
    manifest_path = manifest_path or root / CLEANUP_RELATIVE / "supplemental-manifest.json"
    observed = strict_load_json(manifest_path)
    temporary = manifest_path.with_name(".supplemental-manifest.reconstructed.json")
    try:
        expected = build_supplemental_manifest(root, temporary)
    finally:
        temporary.unlink(missing_ok=True)
    if observed != expected:
        raise StrictValidationError("PROJECTION_MANIFEST", "supplemental manifest differs from reconstruction")
    return observed


def run_cleanup_audit(root: Path | None = None, require_candidate: bool = False) -> dict:
    root = root or repository_root()
    checks: list[dict] = []
    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
    check("source_tag_object", _git(root, "rev-parse", f"{SOURCE_TAG}^{{tag}}") == SOURCE_TAG_OBJECT, SOURCE_TAG_OBJECT)
    check("source_commit_tree", _git(root, "rev-parse", f"{SOURCE_TAG}^{{commit}}") == SOURCE_COMMIT and _git(root, "rev-parse", f"{SOURCE_TAG}^{{tree}}") == SOURCE_TREE, {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE})
    v1_diff = subprocess.run(["git", "diff", "--exit-code", SOURCE_TAG, "--", str(V1_RELATIVE)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check("v1_evidence_byte_preservation", v1_diff.returncode == 0, v1_diff.stdout + v1_diff.stderr)
    manifest = verify_supplemental_manifest(root)
    check("projection_reconstruction", manifest["projection"]["row_count"] == EXPECTED_ROWS and manifest["projection"]["reconstructed_status"] == "RELIABLE_TRAINING_RECIPE_IDENTIFIED_DEV_ONLY" and manifest["projection"]["reconstructed_selected_recipe"] == "R3_MORE_STEPS_AND_WIDTH", manifest["projection"])
    preflight = dependency_preflight()
    check("offline_dependency_preflight", preflight["contract_satisfied"] and not preflight["installation_attempted"] and not preflight["downloads_attempted"], preflight)
    assumptions = subprocess.run(["rg", "-n", r"\.venv/bin/" + "python", ".", "--glob", "!.git/**"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check("no_clone_local_python_assumption", assumptions.returncode == 1, assumptions.stdout)
    chronology = (root / "docs/DEV_ONLY_RELIABILITY_REPRODUCIBILITY_CLEANUP_HANDOFF.md").read_text()
    check("chronology_and_future_freeze", all(token in chronology for token in ("533f89f", "cc23259", "dbe3134", "e5ee6a9", "literal backslash-n", "no repair exception")), "handoff tokens")
    check("predecessor_tags_preserved", sha256(_tag_snapshot(root, True).encode("utf-8")).hexdigest() == manifest["predecessor_tag_snapshot"]["canonical_sha256"], manifest["predecessor_tag_snapshot"])
    check("zero_confirmation", all(value == 0 for value in manifest["confirmation"].values()), manifest["confirmation"])
    check("nonclaiming_supplement", manifest["claim_bearing"] is False and not manifest["assertions"]["modifies_v1_status_or_conclusion"], manifest["assertions"])
    if require_candidate:
        tag = subprocess.run(["git", "rev-parse", f"{CLEANUP_TAG}^{{commit}}"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check("cleanup_candidate_tag", tag.returncode == 0 and tag.stdout.strip() == _git(root, "rev-parse", "HEAD^{commit}"), tag.stdout.strip())
    return {
        "schema": "exp33ae.dev_only_reliability_reproducibility_cleanup_non_independent_audit.v1",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "independent": False,
        "passed": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
        "confirmation_objects_created": 0,
        "confirmation_objects_accessed": 0,
        "status": "READY_FOR_INDEPENDENT_CLEANUP_REVIEW" if all(item["passed"] for item in checks) else "CLEANUP_REMEDIATION_INCOMPLETE",
    }
