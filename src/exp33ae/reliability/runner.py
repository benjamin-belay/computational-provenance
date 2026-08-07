"""Two-phase prepare/freeze then actual-evaluation reliability runner."""

from __future__ import annotations

import time
from pathlib import Path

from ..remediation.access import DevOnlyAccessController
from ..remediation.strict import object_sha256, strict_load_json, strict_load_jsonl, write_json
from .allocation import build as build_allocation, finalize_shortcut_baselines, load as load_allocation
from .entropy import StudyEntropy, canonical_master_text
from .evaluator import evaluate, load_training
from .indexing import build_indexes
from .inference import execute_segment
from .normative import DEV_NAMESPACE, DEV_STAGE_ID, FREEZE_TAG, build_source_manifest, git, protocol_hash, require_dev_stage
from .planning import build_registry, representation_manifest
from .runtime import dependency_manifest, runtime_observation
from .training import train_all


def output_default(root: Path) -> Path:
    return root / "artifacts/dev_only/33a-e-training-reliability-v1"


def _entropy_record(entropy: StudyEntropy) -> dict:
    value = {"schema": "exp33ae.dev_only_reliability_entropy_record.v1", "namespace": DEV_NAMESPACE,
             "claim_bearing": False, "stage_id": entropy.stage_id, "protocol_sha256": protocol_hash(),
             "freeze_commit": entropy.freeze_commit, "canonical_master_text": canonical_master_text(entropy.freeze_commit),
             "dev_only_master_hex": entropy.master.hex(), "commitment": entropy.commitment,
             "model_seed_identities": list(entropy.model_seeds()), "confirmation_master_entropy_created": False,
             "confirmation_derived_objects_created": 0}
    value["record_sha256"] = object_sha256(value); return value


def prepare(root: Path, output: Path, freeze_tag: str = FREEZE_TAG) -> dict:
    require_dev_stage(DEV_STAGE_ID)
    if freeze_tag != FREEZE_TAG: raise RuntimeError("unexpected freeze tag")
    if output.exists() and any(output.iterdir()): raise RuntimeError("prepare output must be empty")
    freeze_commit = git(root, "rev-parse", f"{freeze_tag}^{{commit}}")
    entropy = StudyEntropy(freeze_commit); output.mkdir(parents=True, exist_ok=True)
    source = build_source_manifest(root, freeze_commit, git(root, "rev-parse", "HEAD^{commit}")); dependency = dependency_manifest()
    write_json(output / "source-manifest.json", source); write_json(output / "dependency-manifest.json", dependency)
    write_json(output / "entropy.json", _entropy_record(entropy)); write_json(output / "runtime" / "host.json", runtime_observation())
    allocation, pairs, allocation_manifest = build_allocation(root, output, entropy)
    representation = representation_manifest(); write_json(output / "representation.json", representation)
    bindings = {"source_manifest_sha256": source["manifest_sha256"], "dependency_manifest_sha256": dependency["manifest_sha256"]}
    training_rows = [r for r in allocation["examples"] if r.partition == "training"]
    started = time.perf_counter(); training = train_all(training_rows, entropy, output, bindings)
    write_json(output / "runtime" / "training-total.json", {"schema": "exp33ae.dev_only_reliability_training_total_runtime.v1",
        "namespace": DEV_NAMESPACE, "claim_bearing": False, "models": 20, "runtime_seconds": time.perf_counter()-started,
        "scientific_identity": False})
    # Only now, after all twenty final checkpoints exist, aggregate held-out
    # shortcut outcomes. No model held-out metric is computed in prepare.
    allocation_manifest = finalize_shortcut_baselines(output, entropy)
    registry = build_registry(output, entropy, allocation, pairs, allocation_manifest, training, representation, source, dependency)
    value = {"schema": "exp33ae.dev_only_reliability_prepare_status.v1", "namespace": DEV_NAMESPACE,
             "claim_bearing": False, "status": "TRAINING_RECIPE_STUDY_RUNTIME_STOPPED",
             "phase": "PRE_EVALUATION_REGISTRY_READY_REQUIRES_GIT_FREEZE", "models_trained": 20,
             "registry_rows": len(registry["rows"]), "heldout_metrics_inspected": False,
             "confirmation_objects_created": 0, "confirmation_objects_accessed": 0}
    write_json(output / "prepare-status.json", value); return value | {"master_hex": entropy.master.hex(), "seeds": list(entropy.model_seeds())}


def _require_committed_registry(root: Path, output: Path) -> str:
    if git(root, "status", "--porcelain"):
        raise RuntimeError("evaluation requires a clean committed registry/checkpoint freeze")
    relative = str((output / "registry" / "registry.jsonl").relative_to(root))
    git(root, "ls-files", "--error-unmatch", relative)
    for forbidden in (output / "access", output / "derived-status.json", output / "release-envelope.json"):
        if forbidden.exists(): raise RuntimeError("evaluation output already exists")
    return git(root, "rev-parse", "HEAD^{commit}")


def evaluate_all(root: Path, output: Path) -> dict:
    registry_freeze_commit = _require_committed_registry(root, output)
    entropy_record = strict_load_json(output / "entropy.json"); entropy = StudyEntropy(entropy_record["freeze_commit"])
    allocation, _, _ = load_allocation(output); training = load_training(output)
    registry_manifest = strict_load_json(output / "registry" / "manifest.json")
    registry = strict_load_jsonl(output / "registry" / "registry.jsonl")
    controller = DevOnlyAccessController(output / "access", DEV_STAGE_ID, entropy.derive("access-token", 0, 32),
                                         registry_manifest["payload_commitment"], [r["call_id"] for r in registry])
    with controller.acquire(initialize=True):
        start = controller.begin(resume=False)
        if start != 0: raise RuntimeError("fresh evaluation did not begin at call zero")
        examples = {r.example_id: r for r in allocation["examples"]}; by_model = {r["model_order"]: r for r in training}
        started = time.perf_counter()
        for cursor in range(0, len(registry), 256):
            controller.append_rows(execute_segment(output, registry[cursor:cursor+256], examples, by_model))
        controller.mark_buffer_complete()
        write_json(output / "runtime" / "evaluation.json", {"schema": "exp33ae.dev_only_reliability_evaluation_runtime.v1",
            "namespace": DEV_NAMESPACE, "claim_bearing": False, "calls": 78400,
            "runtime_seconds": time.perf_counter()-started, "scientific_identity": False})
        status = evaluate(root, output, controller.validate_chain())
        write_json(output / "derived-status.json", status)
        status_hash = object_sha256(status); controller.record_release(status_hash, status["status"])
    write_json(output / "registry-freeze-identity.json", {"schema": "exp33ae.dev_only_registry_freeze_identity.v1",
        "namespace": DEV_NAMESPACE, "claim_bearing": False, "registry_freeze_commit": registry_freeze_commit,
        "registry_file_sha256": registry_manifest["registry_file_sha256"], "registry_payload_commitment": registry_manifest["payload_commitment"]})
    scientific, runtime = build_indexes(output)
    release = {"schema": "exp33ae.dev_only_reliability_release.v1", "namespace": DEV_NAMESPACE, "claim_bearing": False,
               "status": status["status"], "selected_recipe": status.get("selected_recipe"),
               "derived_status_sha256": status_hash, "canonical_scientific_identity": scientific["canonical_scientific_identity"],
               "runtime_observation_identity": runtime["runtime_observation_identity"], "registry_calls": 78400,
               "confirmation_status_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0}
    release["release_sha256"] = object_sha256(release); write_json(output / "release-envelope.json", release)
    return release
