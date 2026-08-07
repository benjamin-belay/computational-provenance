"""Materialize the complete deterministic DEV_ONLY protocol fixture package."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch

from ..artifacts import canonical_json, sha256_file, write_json, write_jsonl
from .access import DevOnlyAccessSession
from .allocator import allocate, development_exclusions, shortcut_baselines
from .controls import build_controls
from .entropy import dev_golden_entropy
from .evaluator import evaluate, healthy_raw
from .normative import DEV_NAMESPACE, PROTOCOL_RELATIVE_PATH, load_protocol, protocol_hash, repository_root
from .pairing import build_pairs
from .registry import build_registry
from .representation import golden_vectors
from .schemas import schemas
from ..model import ModelConfig, TinyProgrammeTransformer, tensor_state_hash
from .training_recipe import dev_only_schedule_golden, normative_training_table, validate_training_recipe_against_implementation


def _source_manifest(root: Path, schema_directory: Path) -> dict:
    paths = [root / PROTOCOL_RELATIVE_PATH]
    paths.extend(sorted((root / "src/exp33ae/preconfirmation").glob("*.py")))
    paths.extend(sorted(schema_directory.glob("*.json")))
    rows = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    return {"files": rows, "manifest_sha256": sha256(canonical_json(rows).encode()).hexdigest()}


def build_dev_only_fixture(root: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite a nonempty protocol fixture directory")
    output.mkdir(parents=True, exist_ok=True)
    schema_directory = root / "schemas/dev_only/preconfirmation-v1"
    schema_directory.mkdir(parents=True, exist_ok=True)
    schema_values = schemas()
    for name, value in schema_values.items():
        write_json(schema_directory / name, value)
    source_manifest = _source_manifest(root, schema_directory)

    entropy = dev_golden_entropy()
    derived = {
        domain: [entropy.derive(domain, index, 32).hex() for index in range(3)]
        for domain in load_protocol()["entropy"]["derived_domains"]
    }
    entropy_report = {
        "schema": "exp33ae.dev_only_entropy_golden.v1", "namespace": DEV_NAMESPACE,
        "stage_id": entropy.stage_id, "claim_bearing": False, "master_hex": entropy.master.hex(),
        "commitment": entropy.commitment, "protocol_sha256": protocol_hash(),
        "ordered_model_seed_identities": list(entropy.ordered_model_seeds()), "derived_golden_vectors": derived,
        "confirmation_master_entropy_generated": False, "confirmation_commitment_generated": False,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    allocation = allocate(root, entropy)
    baselines = shortcut_baselines(allocation, entropy)
    pairs = build_pairs(allocation, entropy)
    controls = build_controls(entropy)
    registry = build_registry(allocation, pairs, controls, entropy)
    raw = healthy_raw(registry["rows"])
    exclusions = development_exclusions(root)
    exclusion_json = {key: sorted(value) for key, value in exclusions.items()}
    readout_manifest = [
        {
            "namespace": DEV_NAMESPACE, "stage_id": entropy.stage_id, "model_order": index,
            "model_seed_identity": seed, "eligible": True,
            "basis_sha256": registry["rows"][index * 3920 + 1600]["basis_sha256"],
            "readout_sha256": registry["rows"][index * 3920]["readout_sha256"],
            "fit_partition": "localization_fit", "selection_partition": "localization_selection",
            "frozen_before_access": True,
        }
        for index, seed in enumerate(entropy.ordered_model_seeds())
    ]

    write_json(output / "entropy-golden.json", entropy_report)
    write_json(output / "normative-training-table.json", {
        "schema": "exp33ae.normative_training_table.v1", "namespace": "DEV_ONLY_PROTOCOL_SPECIFICATION",
        "protocol_sha256": protocol_hash(), "training": normative_training_table(),
    })
    write_json(output / "training-recipe-consistency.json", validate_training_recipe_against_implementation())
    write_json(output / "training-schedule-golden.json", dev_only_schedule_golden(entropy))
    write_json(output / "source-schema-manifest.json", source_manifest)
    write_json(output / "development-exclusions.json", {"schema": "exp33ae.development_exclusions.v1", "namespace": DEV_NAMESPACE, **exclusion_json})
    write_jsonl(output / "programme-allocation.jsonl", allocation["programmes"])
    write_jsonl(output / "example-allocation.jsonl", (row.as_dict() for row in allocation["examples"]))
    allocation_report = {
        key: value for key, value in allocation.items() if key not in {"programmes", "examples"}
    }
    allocation_report.update({
        "programme_count": len(allocation["programmes"]), "example_count": len(allocation["examples"]),
        "programme_allocation_sha256": sha256_file(output / "programme-allocation.jsonl"),
        "example_allocation_sha256": sha256_file(output / "example-allocation.jsonl"),
        "shortcut_baselines": baselines,
        "globally_unseen_compositions_claimed": False,
        "globally_unseen_primitive_functions_claimed": False,
        "within_run_heldout_only": True,
    })
    write_json(output / "allocation-audit.json", allocation_report)
    write_json(output / "representation-golden.json", golden_vectors())
    write_jsonl(output / "primary-pairs.jsonl", pairs)
    write_json(output / "controls.json", {"schema": "exp33ae.dev_only_controls.v1", "namespace": DEV_NAMESPACE, "controls": controls})
    write_json(output / "readout-manifest.json", {"schema": "exp33ae.dev_only_readouts.v1", "namespace": DEV_NAMESPACE, "readouts": readout_manifest})
    torch.manual_seed(entropy.ordered_model_seeds()[0])
    incompetent = TinyProgrammeTransformer(ModelConfig())
    with torch.no_grad():
        for parameter in incompetent.parameters():
            parameter.zero_()
    incompetent_path = output / "dev_only-incompetent-checkpoint-fixture.pt"
    torch.save({
        "schema": "exp33ae.dev_only_incompetent_checkpoint_fixture.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "stage_id": entropy.stage_id, "model_seed_identity": entropy.ordered_model_seeds()[0],
        "model_config": ModelConfig().as_dict(), "state_dict": incompetent.state_dict(),
        "tensor_state_sha256": tensor_state_hash(incompetent),
        "confirmation_checkpoint_materialized": False,
    }, incompetent_path)
    write_json(output / "incompetent-checkpoint-manifest.json", {
        "schema": "exp33ae.dev_only_incompetent_checkpoint_manifest.v1", "namespace": DEV_NAMESPACE,
        "path": incompetent_path.name, "file_sha256": sha256_file(incompetent_path),
        "tensor_state_sha256": tensor_state_hash(incompetent), "model_seed_identity": entropy.ordered_model_seeds()[0],
        "claim_bearing": False, "confirmation_checkpoint_materialized": False,
    })
    write_jsonl(output / "registry.jsonl", registry["rows"])
    write_json(output / "registry-manifest.json", {
        "schema": registry["schema"], "namespace": DEV_NAMESPACE, "stage_id": entropy.stage_id,
        "claim_bearing": False, "model_seed_identities": registry["model_seed_identities"],
        "payload_commitment": registry["payload_commitment"], "row_count": len(registry["rows"]),
        "registry_file_sha256": sha256_file(output / "registry.jsonl"),
        "confirmation_registry_materialized": False,
    })
    write_jsonl(output / "raw-results.jsonl", raw)

    access_directory = output / "access-happy"
    session = DevOnlyAccessSession(access_directory, entropy.stage_id, [row["call_id"] for row in registry["rows"]])
    session.authorize_once()
    session.buffer_outputs(raw)
    package = {
        "schema": "exp33ae.dev_only_evaluation_package.v1", "namespace": DEV_NAMESPACE,
        "stage_id": entropy.stage_id, "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "source_manifest_expected": source_manifest["manifest_sha256"], "source_manifest_observed": source_manifest["manifest_sha256"],
        "payload_index_expected": registry["payload_commitment"], "payload_index_observed": registry["payload_commitment"],
        "requested_status": None, "access_report": session.report(), "allocation": allocation,
        "registry": registry, "raw_rows": raw, "expected_model_seed_identities": list(entropy.ordered_model_seeds()),
        "development_exclusions": exclusion_json, "readout_manifest": readout_manifest, "shortcut_baselines": baselines,
    }
    decision = evaluate(package, root)
    release = session.atomic_release(decision)
    write_json(output / "simulated-evaluator-result.json", decision)
    write_json(output / "access-happy-report.json", session.report())
    write_json(output / "release-happy-summary.json", release)
    build_report = {
        "schema": "exp33ae.protocol_revision_fixture_build.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(), "source_manifest_sha256": source_manifest["manifest_sha256"],
        "schema_hashes": {name: sha256_file(schema_directory / name) for name in schema_values},
        "counts": {"programmes": len(allocation["programmes"]), "examples": len(allocation["examples"]), "pairs": len(pairs), "controls": len(controls), "registry": len(registry["rows"]), "raw": len(raw)},
        "dev_only_simulated_status": decision["simulated_status"], "terminal_status_materialized": False,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    write_json(output / "build-report.json", build_report)
    return {"report": build_report, "package": package, "decision": decision, "output": output}
