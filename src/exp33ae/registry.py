"""Pre-execution DEV_ONLY call registry and evaluation freeze."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .artifacts import sha256_file, tensor_sha256, write_json, write_jsonl
from .catalog import records
from .model import orthogonal_basis, random_basis, tensor_state_hash
from .pairing import eligible_pairs
from .spec import CALL_COUNTS_PER_SEED, MODEL_SEEDS, N2_POSITION, NAMESPACE, Z2_POSITION, stable_digest
from .training import checkpoint_path, load_checkpoint


def _alignment_path(root: Path, kind: str, seed: int) -> Path:
    return root / "artifacts/dev_only/alignments" / f"{kind}-seed-{seed}.pt"


def _candidate_id(candidate: dict) -> str:
    return f"{candidate['method']}-L{candidate['layer']}-P{candidate['position']}-R{candidate['rank']}-S{candidate['strength']:g}"


def create_registry(root: Path) -> dict:
    causal_rows = records("causal_evaluation")
    registry_rows = []
    control_index = []
    for kind in ("ordinary", "iit_primary"):
        for seed in MODEL_SEEDS[kind]:
            checkpoint = checkpoint_path(root, kind, seed)
            model, _ = load_checkpoint(checkpoint)
            checkpoint_file_hash = sha256_file(checkpoint)
            model_state_hash = tensor_state_hash(model)
            alignment_path = _alignment_path(root, kind, seed)
            alignment = torch.load(alignment_path, map_location="cpu", weights_only=False)
            if alignment.get("namespace") != NAMESPACE:
                raise RuntimeError("alignment is not DEV_ONLY")
            candidate = alignment["selected_candidate"]
            candidate_id = _candidate_id(candidate)
            primary_basis = alignment["basis"]
            random_direction = random_basis(model.config.d_model, candidate["rank"], seed + 70001)
            orthogonal_direction = orthogonal_basis(primary_basis, candidate["rank"], seed + 70002)
            random_layer = (candidate["layer"] + 2) % model.config.n_layers
            random_location = N2_POSITION if candidate["position"] != N2_POSITION else Z2_POSITION
            control_path = root / "artifacts/dev_only/control-bases" / f"{kind}-seed-{seed}.pt"
            control_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "exp33ae.control_bases.v1",
                    "namespace": NAMESPACE,
                    "model_kind": kind,
                    "model_seed": seed,
                    "primary_basis_sha256": tensor_sha256(primary_basis),
                    "random_direction": random_direction,
                    "random_direction_sha256": tensor_sha256(random_direction),
                    "orthogonal_direction": orthogonal_direction,
                    "orthogonal_direction_sha256": tensor_sha256(orthogonal_direction),
                    "random_layer": random_layer,
                    "random_location": random_location,
                },
                control_path,
            )
            controls = torch.load(control_path, map_location="cpu", weights_only=False)
            control_file_hash = sha256_file(control_path)
            control_index.append(
                {
                    "model_kind": kind,
                    "model_seed": seed,
                    "control_artifact": str(control_path.relative_to(root)),
                    "control_artifact_sha256": control_file_hash,
                }
            )
            pairs_by_kind = {
                "different_state": eligible_pairs(
                    causal_rows, "different_state", CALL_COUNTS_PER_SEED["causal_different_state"], seed + 50000
                ),
                "same_state": eligible_pairs(
                    causal_rows, "same_state", CALL_COUNTS_PER_SEED["causal_same_state"], seed + 50000
                ),
                "same_answer_different_state": eligible_pairs(
                    causal_rows,
                    "same_answer_different_state",
                    CALL_COUNTS_PER_SEED["causal_same_answer_different_state"],
                    seed + 50000,
                ),
            }
            plans = []
            for pair in pairs_by_kind["different_state"]:
                plans.extend(
                    (
                        (pair, "causal_candidate", candidate["layer"], candidate["position"], "primary", tensor_sha256(primary_basis)),
                        (pair, "matched_random_direction", candidate["layer"], candidate["position"], "random_direction", controls["random_direction_sha256"]),
                        (pair, "matched_orthogonal_direction", candidate["layer"], candidate["position"], "orthogonal_direction", controls["orthogonal_direction_sha256"]),
                        (pair, "matched_random_location", candidate["layer"], random_location, "primary", tensor_sha256(primary_basis)),
                        (pair, "matched_random_layer", random_layer, candidate["position"], "primary", tensor_sha256(primary_basis)),
                    )
                )
            for pair in pairs_by_kind["same_state"] + pairs_by_kind["same_answer_different_state"]:
                plans.append((pair, "causal_candidate", candidate["layer"], candidate["position"], "primary", tensor_sha256(primary_basis)))
            for pair, family, layer, position, basis_kind, basis_hash in plans:
                raw_id = f"{kind}:{seed}:{candidate_id}:{family}:{pair.pair_id}"
                registry_rows.append(
                    {
                        "schema": "exp33ae.call_plan.v1",
                        "namespace": NAMESPACE,
                        "call_id": "devcall-" + stable_digest(raw_id)[:24],
                        "model_kind": kind,
                        "model_seed": seed,
                        "checkpoint": str(checkpoint.relative_to(root)),
                        "checkpoint_file_sha256": checkpoint_file_hash,
                        "model_tensor_state_sha256": model_state_hash,
                        "alignment": str(alignment_path.relative_to(root)),
                        "alignment_file_sha256": sha256_file(alignment_path),
                        "control_artifact": str(control_path.relative_to(root)),
                        "control_artifact_sha256": control_file_hash,
                        "candidate_id": candidate_id,
                        "dataset_partition": "causal_evaluation",
                        "base_id": pair.base.example_id,
                        "source_id": pair.source.example_id,
                        "base_axis": pair.base.axis,
                        "source_axis": pair.source.axis,
                        "base_programme_id": pair.base.programme.programme_id,
                        "source_programme_id": pair.source.programme.programme_id,
                        "pair_id": pair.pair_id,
                        "pair_kind": pair.kind,
                        "control_family": family,
                        "intervention_target": "z2",
                        "layer": layer,
                        "position": position,
                        "rank": candidate["rank"],
                        "strength": candidate["strength"],
                        "basis_kind": basis_kind,
                        "basis_sha256": basis_hash,
                        "expected_z3": pair.expected_z3,
                        "expected_n3": pair.base.values.n3,
                        "expected_joint": pair.expected_joint,
                        "base_z2": pair.base.values.z2,
                        "source_z2": pair.source.values.z2,
                        "eligible": True,
                        "no_op": pair.no_op,
                        "planned": True,
                    }
                )
    registry_path = root / "registries/dev_only/evaluation-call-registry.jsonl"
    write_jsonl(registry_path, registry_rows)
    expected_group_counts = {}
    expected_axis_counts = {}
    for row in registry_rows:
        key = f"{row['model_kind']}:{row['model_seed']}:{row['control_family']}:{row['pair_kind']}"
        expected_group_counts[key] = expected_group_counts.get(key, 0) + 1
        if row["control_family"] == "causal_candidate":
            for role in ("base", "source"):
                axis_key = (
                    f"{row['model_kind']}:{row['model_seed']}:{row['pair_kind']}:"
                    f"{role}:{row[f'{role}_axis']}"
                )
                expected_axis_counts[axis_key] = expected_axis_counts.get(axis_key, 0) + 1
    freeze = {
        "schema": "exp33ae.evaluation_freeze.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "registry_created_before_calls": True,
        "call_registry": str(registry_path.relative_to(root)),
        "call_registry_sha256": sha256_file(registry_path),
        "expected_call_count": len(registry_rows),
        "expected_unique_call_ids": len({row["call_id"] for row in registry_rows}),
        "expected_group_counts": dict(sorted(expected_group_counts.items())),
        "expected_axis_counts": dict(sorted(expected_axis_counts.items())),
        "expected_unique_pair_ids_per_seed": sum(CALL_COUNTS_PER_SEED[name] for name in (
            "causal_different_state", "causal_same_state", "causal_same_answer_different_state"
        )),
        "expected_seed_identities": {kind: list(MODEL_SEEDS[kind]) for kind in ("ordinary", "iit_primary")},
        "control_artifacts": control_index,
        "split_manifest": "registries/dev_only/split-manifest.json",
        "split_manifest_sha256": sha256_file(root / "registries/dev_only/split-manifest.json"),
        "method_search": "results/dev_only/method-search.json",
        "method_search_sha256": sha256_file(root / "results/dev_only/method-search.json"),
        "future_generation_seed": None,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    write_json(root / "registries/dev_only/evaluation-freeze.json", freeze)
    return freeze
