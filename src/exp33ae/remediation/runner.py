"""Generic future-shaped runner with only a DEV_ONLY enabled stage."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

from .access import DevOnlyAccessController
from .allocation import build_allocation, load_allocation
from .binding_polarity import run_binding_polarity
from .entropy import DevMasterEntropy
from .indexing import build_prerelease_index
from .inference import execute_registry_segment
from .normative import DEV_NAMESPACE, build_operative_manifest, protocol_hash, require_dev_stage
from .planning import build_registry, representation_manifest
from .polarity import run_polarity
from .release import evaluator_derived_release
from .runtime import environment_manifest
from .schemas import generated_schemas
from .strict import object_sha256, strict_load_json, strict_load_jsonl, write_json
from .training import train_population


def _entropy_record(entropy: DevMasterEntropy) -> dict:
    value = {
        "schema": "exp33ae.dev_only_entropy_record.v2", "namespace": DEV_NAMESPACE,
        "stage_id": entropy.stage_id, "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "dev_only_master_hex": entropy.master.hex(), "commitment": entropy.commitment,
        "model_seed_identities": list(entropy.model_seeds()),
        "confirmation_master_entropy_created": False, "confirmation_derived_objects_created": 0,
    }
    value["record_sha256"] = object_sha256(value)
    return value


def initialize_run(root: Path, output: Path, entropy: DevMasterEntropy, source_commit: str) -> dict:
    require_dev_stage(entropy.stage_id)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("initial DEV_ONLY runner output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    operative = build_operative_manifest(root, source_commit)
    environment = environment_manifest()
    write_json(output / "operative-source-manifest.json", operative)
    write_json(output / "environment.json", environment)
    write_json(output / "entropy.json", _entropy_record(entropy))
    for name, schema in generated_schemas().items():
        write_json(output / "schemas" / name, schema)
    allocation, pairs, allocation_manifest = build_allocation(root, output, entropy)
    representation = representation_manifest()
    write_json(output / "representation.json", representation)
    bindings = {"operative_manifest_sha256": operative["manifest_sha256"], "environment_sha256": environment["manifest_sha256"]}
    training_rows = [row for row in allocation["examples"] if row.partition == "training"]
    training_started = time.perf_counter()
    training = train_population(training_rows, entropy, output, bindings)
    write_json(output / "runtime" / "training-total.json", {
        "schema": "exp33ae.dev_only_runtime_observation.v2", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "models": 3, "steps_each": 1400,
        "runtime_seconds": time.perf_counter() - training_started,
    })
    registry = build_registry(
        output, entropy, allocation, pairs, allocation_manifest, training,
        representation, operative, environment,
    )
    return {
        "operative": operative, "environment": environment, "allocation": allocation,
        "pairs": pairs, "allocation_manifest": allocation_manifest, "representation": representation,
        "training": training, "registry": registry,
    }


def load_run(output: Path) -> dict:
    entropy_record = strict_load_json(output / "entropy.json")
    entropy = DevMasterEntropy(entropy_record["stage_id"], bytes.fromhex(entropy_record["dev_only_master_hex"]))
    allocation, pairs, allocation_manifest = load_allocation(output)
    training = [strict_load_json(output / "training" / f"model-{order}.json") for order in range(3)]
    registry_manifest = strict_load_json(output / "registry" / "manifest.json")
    registry = strict_load_jsonl(output / "registry" / "registry.jsonl")
    return {
        "entropy": entropy, "allocation": allocation, "pairs": pairs,
        "allocation_manifest": allocation_manifest, "training": training,
        "registry": {"rows": registry, "manifest": registry_manifest},
    }


def run(
    root: Path,
    output: Path,
    master_hex: str,
    stage_id: str,
    source_commit: str,
    resume: bool = False,
    stop_after_calls: int | None = None,
) -> dict:
    # This is intentionally the first operation. Invalid stages create no path,
    # commitment, seed, or derived object.
    require_dev_stage(stage_id)
    if type(master_hex) is not str or len(master_hex) != 64:
        raise ValueError("DEV_ONLY master hex must encode exactly 32 bytes")
    entropy = DevMasterEntropy(stage_id, bytes.fromhex(master_hex))
    if resume:
        package = load_run(output)
        if package["entropy"] != entropy:
            raise RuntimeError("resume entropy differs from frozen run")
    else:
        package = initialize_run(root, output, entropy, source_commit)
        package["entropy"] = entropy
    registry = package["registry"]
    rows = registry["rows"]
    access_token = entropy.derive("access-token", 0, 32)
    controller = DevOnlyAccessController(
        output / "access", stage_id, access_token,
        registry["manifest"]["payload_commitment"], [row["call_id"] for row in rows],
    )
    with controller.acquire(initialize=not resume):
        start = controller.begin(resume=resume)
        limit = len(rows) if stop_after_calls is None else min(len(rows), start + stop_after_calls)
        examples = {row.example_id: row for row in package["allocation"]["examples"]}
        inference_started = time.perf_counter()
        cursor = start
        while cursor < limit:
            end = min(limit, cursor + 256)
            observed = execute_registry_segment(output, rows[cursor:end], examples, package["training"])
            controller.append_rows(observed)
            cursor = end
        write_json(output / "runtime" / "inference-latest.json", {
            "schema": "exp33ae.dev_only_runtime_observation.v2", "namespace": DEV_NAMESPACE,
            "claim_bearing": False, "start_order": start, "end_order": cursor,
            "runtime_seconds": time.perf_counter() - inference_started,
        })
        if cursor < len(rows):
            return {
                "terminal_status": "33A_E_REMEDIATION_RUNTIME_STOPPED", "next_call_order": cursor,
                "expected_calls": len(rows), "claim_bearing": False,
            }
        controller.mark_buffer_complete()
        # Replay the strict row-level polarity suite before the evidence index
        # and release are frozen.  Synthetic threshold cases are labelled as
        # such in the report and never substitute for the trained-model rows.
        run_polarity(output, output / "polarity" / "report.json")
        prerelease = build_prerelease_index(output)
        run_binding_polarity(root, output, output / "polarity" / "binding-report.json")
        # The binding report is itself part of the release package, so refresh
        # the prospectively defined non-circular index once before release.
        prerelease = build_prerelease_index(output)
        released = evaluator_derived_release(root, output, controller)
        return {
            "terminal_status": released["status"]["status"],
            "model_seed_identities": list(entropy.model_seeds()),
            "calls": len(rows), "prerelease_index_sha256": prerelease["payload_sha256"],
            "postrelease_index_sha256": released["postrelease_index"]["payload_sha256"],
            "claim_bearing": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
        }


def fixed_master_hex() -> str:
    return sha256(b"DEV_ONLY 33A-E FINAL REMEDIATION V2 MASTER; CONFIRMATION-INELIGIBLE").hexdigest()
