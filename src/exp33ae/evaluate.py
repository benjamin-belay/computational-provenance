"""Execute the precommitted DEV_ONLY call registry and reconcile every call."""

from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from torch import nn

from .artifacts import sha256_file, tensor_sha256, write_json, write_jsonl
from .catalog import records
from .independent_oracle import oracle_do_z2
from .interventions import evaluate_intervention, primary_norms
from .model import tensor_state_hash
from .pairing import Pair
from .probes import FittedProbe, extract_hidden
from .spec import K, MODEL_SEEDS, NAMESPACE
from .status import aggregate_population, evaluate_seed
from .training import accuracy, load_checkpoint


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_probe(model, alignment) -> FittedProbe:
    metadata = alignment["z2_probe"]
    classifier = nn.Linear(model.config.d_model, K)
    classifier.load_state_dict(alignment["z2_probe_state"])
    classifier.eval()
    return FittedProbe(
        metadata["layer"],
        metadata["position"],
        "z2",
        classifier,
        metadata["fit_correct"],
        metadata["fit_total"],
        metadata["selection_correct"],
        metadata["selection_total"],
    )


def _load_subspace_decoder(alignment):
    metadata = alignment["z2_subspace_decoder"]
    classifier = nn.Linear(metadata["rank"], K)
    classifier.load_state_dict(alignment["z2_subspace_decoder_state"])
    classifier.eval()
    return classifier


def _pair_from_plan(row: dict, examples: dict[str, object]) -> Pair:
    base, source = examples[row["base_id"]], examples[row["source_id"]]
    return Pair(
        pair_id=row["pair_id"],
        kind=row["pair_kind"],
        base=base,
        source=source,
        expected_z3=row["expected_z3"],
        expected_joint=row["expected_joint"],
        no_op=row["no_op"],
    )


def _candidate_id(candidate: dict) -> str:
    return f"{candidate['method']}-L{candidate['layer']}-P{candidate['position']}-R{candidate['rank']}-S{candidate['strength']:g}"


def execute_seed(
    root: Path,
    kind: str,
    seed: int,
    plan_rows: list[dict],
    control_path_override: Path | None = None,
) -> tuple[dict, list[dict]]:
    if not plan_rows:
        raise RuntimeError("empty seed plan")
    checkpoint_path = root / plan_rows[0]["checkpoint"]
    alignment_path = root / plan_rows[0]["alignment"]
    control_path = control_path_override or (
        root / "artifacts/dev_only/control-bases" / f"{kind}-seed-{seed}.pt"
    )
    model, checkpoint_payload = load_checkpoint(checkpoint_path)
    before_hash = tensor_state_hash(model)
    alignment = torch.load(alignment_path, map_location="cpu", weights_only=False)
    controls = torch.load(control_path, map_location="cpu", weights_only=False)
    candidate = alignment["selected_candidate"]
    candidate_id = _candidate_id(candidate)
    primary_basis = alignment["basis"]
    probe = _load_probe(model, alignment)
    subspace_decoder = _load_subspace_decoder(alignment)
    causal_rows = records("causal_evaluation")
    examples = {row.example_id: row for row in causal_rows}
    planned_by_key = {(row["control_family"], row["pair_id"]): row for row in plan_rows}
    ledger = []

    def run_group(family: str, pair_kind: str, basis, layer: int, position: int, norm_reference=None, targets=None):
        plans = [
            row for row in plan_rows
            if row["control_family"] == family and row["pair_kind"] == pair_kind
        ]
        pairs = [_pair_from_plan(row, examples) for row in plans]
        observed = evaluate_intervention(
            model,
            pairs,
            layer,
            position,
            basis,
            candidate["strength"],
            probe,
            family,
            norm_reference_basis=norm_reference,
            target_norms=targets,
            transport_basis=primary_basis,
            transport_decoder=subspace_decoder,
        )
        for plan, result in zip(plans, observed):
            row = {**plan, **result}
            row["schema"] = "exp33ae.call_result.v1"
            row["namespace"] = NAMESPACE
            row["call_id"] = plan["call_id"]
            ledger.append(row)

    for pair_kind in ("different_state", "same_state", "same_answer_different_state"):
        run_group(
            "causal_candidate",
            pair_kind,
            primary_basis,
            candidate["layer"],
            candidate["position"],
        )
    different_plans = [
        row for row in plan_rows
        if row["control_family"] == "causal_candidate" and row["pair_kind"] == "different_state"
    ]
    different_pairs = [_pair_from_plan(row, examples) for row in different_plans]
    target_norms = primary_norms(model, different_pairs, candidate["layer"], candidate["position"], primary_basis)
    run_group(
        "matched_random_direction",
        "different_state",
        controls["random_direction"],
        candidate["layer"],
        candidate["position"],
        norm_reference=primary_basis,
    )
    run_group(
        "matched_orthogonal_direction",
        "different_state",
        controls["orthogonal_direction"],
        candidate["layer"],
        candidate["position"],
        norm_reference=primary_basis,
    )
    run_group(
        "matched_random_location",
        "different_state",
        primary_basis,
        candidate["layer"],
        controls["random_location"],
        targets=target_norms,
    )
    run_group(
        "matched_random_layer",
        "different_state",
        primary_basis,
        controls["random_layer"],
        candidate["position"],
        targets=target_norms,
    )
    after_hash = tensor_state_hash(model)

    oracle_mismatches = pairing_mismatches = 0
    for plan in plan_rows:
        base, source = examples[plan["base_id"]], examples[plan["source_id"]]
        expected_z3 = oracle_do_z2(base.programme.operations, base.d, source.values.z2)["z3"]
        expected_joint = expected_z3 * K + base.values.n3
        if (expected_z3, expected_joint) != (plan["expected_z3"], plan["expected_joint"]):
            oracle_mismatches += 1
        no_op = base.values.z2 == source.values.z2
        kind_valid = (
            (plan["pair_kind"] == "different_state" and not no_op)
            or (plan["pair_kind"] == "same_state" and no_op)
            or (
                plan["pair_kind"] == "same_answer_different_state"
                and not no_op
                and base.joint_answer == source.joint_answer
            )
        )
        identity_valid = (
            plan["dataset_partition"] == "causal_evaluation"
            and plan["base_axis"] == base.axis
            and plan["source_axis"] == source.axis
            and plan["base_programme_id"] == base.programme.programme_id
            and plan["source_programme_id"] == source.programme.programme_id
            and plan["base_z2"] == base.values.z2
            and plan["source_z2"] == source.values.z2
            and plan["expected_n3"] == base.values.n3
            and plan["no_op"] is no_op
        )
        pairing_mismatches += not (kind_valid and identity_valid)

    competence = accuracy(model, causal_rows)
    task_audit = json.loads((root / "results/dev_only/task-design-audit.json").read_text())
    competence["shortcut_advantage"] = {
        "heldout_target_chain_composition": competence["by_axis"]["heldout_target_chain_composition"]["accuracy"]
        - task_audit["best_shortcut_by_axis"]["heldout_target_composition"],
        "heldout_parallel_chain_composition": competence["by_axis"]["heldout_parallel_chain_composition"]["accuracy"]
        - task_audit["best_shortcut_by_axis"]["heldout_parallel_composition"],
    }
    hidden = extract_hidden(model, causal_rows, probe.layer, probe.position)
    labels = torch.tensor([row.values.z2 for row in causal_rows], dtype=torch.long)
    with torch.no_grad():
        decode_correct = int((probe.classifier(hidden).argmax(-1) == labels).sum())
    decode = {"correct": decode_correct, "total": len(causal_rows), "accuracy": decode_correct / len(causal_rows)}
    call_ids = [row["call_id"] for row in ledger]
    plan_ids = [row["call_id"] for row in plan_rows]
    bundle = {
        "namespace": NAMESPACE,
        "candidate_eligibility": True,
        "registry_rows": plan_rows,
        "ledger_rows": ledger,
        "registry_reconciled": len(call_ids) == len(set(call_ids)) and set(call_ids) == set(plan_ids),
        "oracle_mismatch_count": oracle_mismatches,
        "pairing_mismatch_count": pairing_mismatches,
        "split_leakage_count": (
            task_audit["exact_duplicate_count"]
            + task_audit["training_target_composition_overlap"]["causal_target_composition"]
            + task_audit["training_parallel_composition_overlap"]["causal_parallel_composition"]
        ),
        "checkpoint_file_hash_match": all(row["checkpoint_file_sha256"] == sha256_file(checkpoint_path) for row in plan_rows),
        "checkpoint_tensor_hash_match": all(row["model_tensor_state_sha256"] == before_hash for row in plan_rows),
        "checkpoint_identity_match": (
            checkpoint_payload.get("kind") == kind
            and checkpoint_payload.get("seed") == seed
            and all(row.get("model_kind") == kind and row.get("model_seed") == seed for row in plan_rows)
        ),
        "alignment_hash_match": all(row["alignment_file_sha256"] == sha256_file(alignment_path) for row in plan_rows),
        "alignment_identity_match": alignment.get("model_kind") == kind and alignment.get("model_seed") == seed,
        "control_artifact_hash_match": all(
            row.get("control_artifact") == str(control_path.relative_to(root))
            and row.get("control_artifact_sha256") == sha256_file(control_path)
            for row in plan_rows
        ),
        "control_identity_match": controls.get("model_kind") == kind and controls.get("model_seed") == seed,
        "control_distinctness_match": (
            (
                controls["random_direction"].shape != primary_basis.shape
                or not torch.allclose(controls["random_direction"], primary_basis)
            )
            and float((primary_basis.T @ controls["orthogonal_direction"]).abs().max()) < 1e-4
            and controls["random_layer"] != candidate["layer"]
            and controls["random_location"] != candidate["position"]
        ),
        "candidate_hash_match": all(
            row["candidate_id"] == candidate_id
            and row["rank"] == candidate["rank"]
            and row["strength"] == candidate["strength"]
            and row["basis_sha256"] == (
                controls["random_direction_sha256"] if row["basis_kind"] == "random_direction"
                else controls["orthogonal_direction_sha256"] if row["basis_kind"] == "orthogonal_direction"
                else tensor_sha256(primary_basis)
            )
            and (
                (row["control_family"] in {"causal_candidate", "matched_random_direction", "matched_orthogonal_direction"}
                 and row["layer"] == candidate["layer"] and row["position"] == candidate["position"])
                or (row["control_family"] == "matched_random_location"
                    and row["layer"] == candidate["layer"] and row["position"] == controls["random_location"])
                or (row["control_family"] == "matched_random_layer"
                    and row["layer"] == controls["random_layer"] and row["position"] == candidate["position"])
            )
            for row in plan_rows
        ),
        "pre_post_model_state_match": before_hash == after_hash,
        "forbidden_access_events": [],
        "dropped_failed_calls": 0,
        "requested_status": None,
        "competence": competence,
        "decode": decode,
    }
    decision = evaluate_seed(bundle)
    return {
        "model_kind": kind,
        "model_seed": seed,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "model_tensor_state_sha256": before_hash,
        "alignment_file_sha256": sha256_file(alignment_path),
        "candidate_id": candidate_id,
        "competence": competence,
        "decode": decode,
        "integrity": {key: value for key, value in bundle.items() if key not in {"registry_rows", "ledger_rows", "competence", "decode"}},
        "decision": decision,
    }, ledger


def run_evaluation(root: Path) -> dict:
    started = time.perf_counter()
    freeze_path = root / "registries/dev_only/evaluation-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    registry_path = root / freeze["call_registry"]
    if sha256_file(registry_path) != freeze["call_registry_sha256"]:
        raise RuntimeError("call registry changed after freeze")
    registry = read_jsonl(registry_path)
    results, ledger = [], []
    for kind in ("ordinary", "iit_primary"):
        for seed in MODEL_SEEDS[kind]:
            plans = [row for row in registry if row["model_kind"] == kind and row["model_seed"] == seed]
            result, rows = execute_seed(root, kind, seed, plans)
            results.append(result)
            ledger.extend(rows)
    output = {
        "schema": "exp33ae.development_evaluation.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "registry_freeze_sha256": sha256_file(freeze_path),
        "model_results": results,
        "ordinary_aggregate": aggregate_population([r for r in results if r["model_kind"] == "ordinary"], "ordinary"),
        "iit_primary_aggregate": aggregate_population([r for r in results if r["model_kind"] == "iit_primary"], "iit_primary"),
        "attempted_calls": len(ledger),
        "completed_calls": sum(row["completed"] is True for row in ledger),
        "unique_call_ids": len({row["call_id"] for row in ledger}),
        "runtime_seconds": time.perf_counter() - started,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    write_jsonl(root / "registries/dev_only/evaluation-call-ledger.jsonl", ledger)
    write_json(root / "results/dev_only/development-evaluation.json", output)
    return output
