"""Real-upstream DEV_ONLY polarity exercises through the production evaluator."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from .artifacts import sha256_file, tensor_sha256, write_json, write_jsonl
from .catalog import records
from .evaluate import execute_seed, read_jsonl
from .independent_oracle import oracle_do_z2
from .interventions import summarize
from .model import orthogonal_basis, random_basis, tensor_state_hash
from .probes import extract_hidden, fit_probe, fit_subspace_decoder
from .spec import K, MODEL_SEEDS, N2_POSITION, NAMESPACE, Z2_POSITION
from .status import DevelopmentStatus, aggregate_population, evaluate_readiness, evaluate_seed
from .task import ForbiddenNamespace, require_dev_partition
from .training import TrainConfig, checkpoint_path, load_checkpoint, train_model


def _candidate_id(candidate: dict) -> str:
    return f"{candidate['method']}-L{candidate['layer']}-P{candidate['position']}-R{candidate['rank']}-S{candidate['strength']:g}"


def _load_state(root: Path):
    evaluation = json.loads((root / "results/dev_only/development-evaluation.json").read_text())
    registry = read_jsonl(root / "registries/dev_only/evaluation-call-registry.jsonl")
    ledger = read_jsonl(root / "registries/dev_only/evaluation-call-ledger.jsonl")
    return evaluation, registry, ledger


def _seed_context(root: Path, seed: int = 1112):
    evaluation, registry, ledger = _load_state(root)
    result = next(r for r in evaluation["model_results"] if r["model_kind"] == "iit_primary" and r["model_seed"] == seed)
    plans = [r for r in registry if r["model_kind"] == "iit_primary" and r["model_seed"] == seed]
    calls = [r for r in ledger if r["model_kind"] == "iit_primary" and r["model_seed"] == seed]
    bundle = {
        **deepcopy(result["integrity"]),
        "registry_rows": deepcopy(plans),
        "ledger_rows": deepcopy(calls),
        "competence": deepcopy(result["competence"]),
        "decode": deepcopy(result["decode"]),
    }
    return evaluation, result, plans, calls, bundle


def _aggregate(evaluation: dict, decision: dict, seed: int = 1112):
    rows = []
    for expected in MODEL_SEEDS["iit_primary"]:
        if expected == seed:
            rows.append({"model_seed": seed, "decision": decision})
        else:
            healthy = next(
                r for r in evaluation["model_results"]
                if r["model_kind"] == "iit_primary" and r["model_seed"] == expected
            )
            rows.append({"model_seed": expected, "decision": healthy["decision"]})
    return aggregate_population(rows, "iit_primary")


def _readiness(aggregate: dict):
    return evaluate_readiness(
        aggregate,
        {
            "all_required_controls": aggregate.get("status") != DevelopmentStatus.INVALID.value,
            "pipeline_completed": True,
            "seed_identity_check_executed": True,
            "zero_confirmation_access": True,
        },
    )


def _record(name, upstream, decision, aggregate, readiness, expected, details=None, substantive_check=True):
    statuses = {
        "seed": decision.get("status"),
        "population": aggregate.get("status"),
        "readiness": readiness.get("status"),
    }
    allowed = expected if isinstance(expected, (list, tuple, set)) else [expected]
    return {
        "namespace": NAMESPACE,
        "scenario": name,
        "upstream_object": upstream,
        "direct_summary_or_terminal_mutation": False,
        "statuses": statuses,
        "expected_seed_status": list(allowed),
        "passed": statuses["seed"] in allowed and substantive_check is True,
        "substantive_check": substantive_check,
        "faults": decision.get("faults", []),
        "details": details or {},
    }


def _refresh_reconciliation(bundle):
    plan_ids = [r.get("call_id") for r in bundle["registry_rows"]]
    call_ids = [r.get("call_id") for r in bundle["ledger_rows"]]
    bundle["registry_reconciled"] = (
        len(plan_ids) == len(set(plan_ids))
        and len(call_ids) == len(set(call_ids))
        and set(plan_ids) == set(call_ids)
    )


def _oracle_and_pairing_counts(plans):
    examples = {row.example_id: row for row in records("causal_evaluation")}
    oracle_mismatches = pairing_mismatches = 0
    for plan in plans:
        base = examples.get(plan.get("base_id"))
        source = examples.get(plan.get("source_id"))
        if base is None or source is None:
            pairing_mismatches += 1
            continue
        expected_z3 = oracle_do_z2(base.programme.operations, base.d, source.values.z2)["z3"]
        expected_joint = expected_z3 * K + base.values.n3
        oracle_mismatches += (plan.get("expected_z3"), plan.get("expected_joint")) != (expected_z3, expected_joint)
        no_op = base.values.z2 == source.values.z2
        kind = plan.get("pair_kind")
        kind_valid = (
            (kind == "different_state" and not no_op)
            or (kind == "same_state" and no_op)
            or (kind == "same_answer_different_state" and not no_op and base.joint_answer == source.joint_answer)
        )
        fields_valid = (
            plan.get("no_op") is no_op
            and plan.get("base_axis") == base.axis
            and plan.get("source_axis") == source.axis
            and plan.get("base_z2") == base.values.z2
            and plan.get("source_z2") == source.values.z2
            and plan.get("expected_n3") == base.values.n3
        )
        pairing_mismatches += not (kind_valid and fields_valid)
    return int(oracle_mismatches), int(pairing_mismatches)


def _alignment_variant(root, slug: str, seed: int, basis, candidate: dict):
    model, _ = load_checkpoint(checkpoint_path(root, "iit_primary", seed))
    fit_rows, selection_rows = records("localization_fit"), records("localization_selection")
    probe = fit_probe(model, fit_rows, selection_rows, candidate["layer"], candidate["position"], "z2", seed + 210000)
    subspace = fit_subspace_decoder(
        model, fit_rows, selection_rows, candidate["layer"], candidate["position"], basis, "z2", seed + 220000
    )
    path = root / "artifacts/dev_only/polarity" / f"{slug}-alignment.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "exp33ae.polarity_alignment.v1",
            "namespace": NAMESPACE,
            "model_kind": "iit_primary",
            "model_seed": seed,
            "scenario": slug,
            "selected_candidate": candidate,
            "basis": basis,
            "z2_probe_state": probe.classifier.state_dict(),
            "z2_probe": {
                "layer": probe.layer, "position": probe.position,
                "fit_correct": probe.fit_correct, "fit_total": probe.fit_total,
                "selection_correct": probe.selection_correct, "selection_total": probe.selection_total,
            },
            "z2_subspace_decoder_state": subspace.classifier.state_dict(),
            "z2_subspace_decoder": {
                "layer": subspace.layer, "position": subspace.position, "rank": basis.shape[1],
                "fit_correct": subspace.fit_correct, "fit_total": subspace.fit_total,
                "selection_correct": subspace.selection_correct, "selection_total": subspace.selection_total,
                "definition": "DEV_ONLY polarity selected-basis decoder",
            },
        },
        path,
    )
    return path


def _control_variant(root, slug: str, seed: int, basis, candidate: dict, contaminate_random=False):
    random_direction = basis.clone() if contaminate_random else random_basis(basis.shape[0], basis.shape[1], seed + 230001)
    orthogonal = orthogonal_basis(basis, basis.shape[1], seed + 230002)
    path = root / "artifacts/dev_only/polarity" / f"{slug}-controls.pt"
    payload = {
        "schema": "exp33ae.polarity_controls.v1", "namespace": NAMESPACE,
        "model_kind": "iit_primary", "model_seed": seed, "scenario": slug,
        "primary_basis_sha256": tensor_sha256(basis),
        "random_direction": random_direction, "random_direction_sha256": tensor_sha256(random_direction),
        "orthogonal_direction": orthogonal, "orthogonal_direction_sha256": tensor_sha256(orthogonal),
        "random_layer": (candidate["layer"] + 2) % 4,
        "random_location": N2_POSITION if candidate["position"] != N2_POSITION else Z2_POSITION,
    }
    torch.save(payload, path)
    return path, payload


def _variant_plan(root, plans, alignment_path, control_path, controls, candidate, basis):
    result = deepcopy(plans)
    candidate_id = _candidate_id(candidate)
    for row in result:
        row["alignment"] = str(alignment_path.relative_to(root))
        row["alignment_file_sha256"] = sha256_file(alignment_path)
        row["control_artifact"] = str(control_path.relative_to(root))
        row["control_artifact_sha256"] = sha256_file(control_path)
        row["candidate_id"] = candidate_id
        row["rank"] = candidate["rank"]
        row["strength"] = candidate["strength"]
        family = row["control_family"]
        if family in {"causal_candidate", "matched_random_direction", "matched_orthogonal_direction"}:
            row["layer"], row["position"] = candidate["layer"], candidate["position"]
        elif family == "matched_random_location":
            row["layer"], row["position"] = candidate["layer"], controls["random_location"]
        elif family == "matched_random_layer":
            row["layer"], row["position"] = controls["random_layer"], candidate["position"]
        row["basis_sha256"] = (
            controls["random_direction_sha256"] if row["basis_kind"] == "random_direction"
            else controls["orthogonal_direction_sha256"] if row["basis_kind"] == "orthogonal_direction"
            else tensor_sha256(basis)
        )
    return result


def _execute_variant(root, slug, plans, basis, candidate, contaminate_random=False):
    alignment = _alignment_variant(root, slug, 1112, basis, candidate)
    control_path, controls = _control_variant(root, slug, 1112, basis, candidate, contaminate_random)
    variant = _variant_plan(root, plans, alignment, control_path, controls, candidate, basis)
    return execute_seed(root, "iit_primary", 1112, variant, control_path), variant, alignment, control_path


def _label_shuffled_basis(model, rows, layer: int, position: int, seed: int):
    torch.manual_seed(seed)
    hidden = extract_hidden(model, rows, layer, position)
    natural = torch.tensor([row.values.z2 for row in rows], dtype=torch.long)
    shuffled = natural[torch.randperm(len(natural), generator=torch.Generator().manual_seed(seed + 1))]
    classifier = nn.Linear(hidden.shape[1], K)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(300):
        loss = nn.functional.cross_entropy(classifier(hidden), shuffled)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    _, _, vh = torch.linalg.svd(classifier.weight.detach(), full_matrices=False)
    return torch.linalg.qr(vh[:2].T, mode="reduced").Q[:, :2]


def _run_bundle_scenario(evaluation, bundle, expected):
    decision = evaluate_seed(bundle)
    aggregate = _aggregate(evaluation, decision)
    return decision, aggregate, _readiness(aggregate)


def run_polarity(root: Path) -> dict:
    evaluation, healthy_result, plans, calls, healthy = _seed_context(root)
    scenarios = []

    decision, aggregate, readiness = _run_bundle_scenario(evaluation, deepcopy(healthy), DevelopmentStatus.OBSERVED.value)
    scenarios.append(_record("healthy_observation", "frozen model/task/registry/calls", decision, aggregate, readiness, DevelopmentStatus.OBSERVED.value))

    # Actual 20-step IIT checkpoint with an honestly updated scenario registry.
    incompetent_path = root / "artifacts/dev_only/polarity/incompetent-iit-seed-1112.pt"
    if not incompetent_path.exists():
        train_model("iit_primary", 1112, incompetent_path, TrainConfig(steps=20, log_every=5))
    incompetent_model, _ = load_checkpoint(incompetent_path)
    incompetent_plans = deepcopy(plans)
    for row in incompetent_plans:
        row["checkpoint"] = str(incompetent_path.relative_to(root))
        row["checkpoint_file_sha256"] = sha256_file(incompetent_path)
        row["model_tensor_state_sha256"] = tensor_state_hash(incompetent_model)
    result, _ = execute_seed(root, "iit_primary", 1112, incompetent_plans)
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record("incompetent_checkpoint", "actual 20-step IIT checkpoint", result["decision"], aggregate, _readiness(aggregate), DevelopmentStatus.COMPETENCE_FAILED.value))

    # Decoder-positive but causal-negative random alignment with matched fresh controls.
    random_primary = random_basis(64, 2, 331112)
    random_candidate = {"method": "random_alignment", "layer": 2, "position": 14, "rank": 2, "strength": 1.0}
    (result, _), _, alignment_path, _ = _execute_variant(
        root, "decoder-positive-causal-negative", plans, random_primary, random_candidate
    )
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record(
        "decoder_positive_causal_negative", "actual random alignment + 100%-decode frozen probe",
        result["decision"], aggregate, _readiness(aggregate),
        [DevelopmentStatus.DECODE_ONLY.value, DevelopmentStatus.NEGATIVE.value],
        {
            "alignment_sha256": sha256_file(alignment_path),
            "decode_accuracy": result["decode"]["accuracy"],
            "different_state_iia": result["decision"].get("metrics", {}).get("different_state_iia"),
        },
    ))

    # Label-shuffled alignment is separately materialized and evaluated.
    shuffled_basis = _label_shuffled_basis(model=load_checkpoint(checkpoint_path(root, "iit_primary", 1112))[0], rows=records("localization_fit"), layer=2, position=14, seed=441112)
    shuffled_candidate = {"method": "label_shuffled_alignment", "layer": 2, "position": 14, "rank": 2, "strength": 1.0}
    (result, _), _, shuffled_alignment, _ = _execute_variant(
        root, "label-shuffled", plans, shuffled_basis, shuffled_candidate
    )
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record(
        "label_shuffled_alignment", "fixed-seed shuffled-label alignment artifact",
        result["decision"], aggregate, _readiness(aggregate),
        [DevelopmentStatus.DECODE_ONLY.value, DevelopmentStatus.NEGATIVE.value],
        {"alignment_sha256": sha256_file(shuffled_alignment)},
    ))

    # A contaminated random control really uses the primary basis; identity checks must invalidate it.
    original_alignment = torch.load(root / plans[0]["alignment"], map_location="cpu", weights_only=False)
    primary = original_alignment["basis"]
    primary_candidate = original_alignment["selected_candidate"]
    (result, contaminated_ledger), _, _, contaminated_control = _execute_variant(
        root, "random-direction-success", plans, primary, primary_candidate, contaminate_random=True
    )
    aggregate = _aggregate(evaluation, result["decision"])
    random_success = summarize([
        row for row in contaminated_ledger
        if row["control_family"] == "matched_random_direction" and row["pair_kind"] == "different_state"
    ])["iia"]
    scenarios.append(_record(
        "random_direction_success", "actual control artifact with random direction equal to causal basis",
        result["decision"], aggregate, _readiness(aggregate), DevelopmentStatus.INVALID.value,
        {"control_sha256": sha256_file(contaminated_control), "actual_random_direction_iia": random_success},
        substantive_check=random_success >= 0.85,
    ))

    # Actual alternative activation locations/bases target behaviour and non-descendant code.
    model, _ = load_checkpoint(checkpoint_path(root, "iit_primary", 1112))
    fit_rows, selection_rows = records("localization_fit"), records("localization_selection")
    same_probe = fit_probe(model, fit_rows, selection_rows, 2, 16, "n3", 551112)
    same_basis = same_probe.basis(2)
    same_candidate = {"method": "same_state_instability", "layer": 2, "position": 16, "rank": 2, "strength": 5.0}
    (result, _), _, _, _ = _execute_variant(root, "same-state-instability", plans, same_basis, same_candidate)
    aggregate = _aggregate(evaluation, result["decision"])
    same_preservation = result["decision"].get("metrics", {}).get("same_state_preservation")
    scenarios.append(_record(
        "same_state_instability", "actual n3-oriented query activation intervention",
        result["decision"], aggregate, _readiness(aggregate),
        [DevelopmentStatus.DECODE_ONLY.value, DevelopmentStatus.NEGATIVE.value],
        {"same_state_preservation": same_preservation},
        substantive_check=type(same_preservation) is float and same_preservation < 0.95,
    ))

    output_weights = model.output_head.weight.detach()
    n3_direction = (output_weights[[1, 3]].mean(0) - output_weights[[0, 2]].mean(0)).unsqueeze(1)
    extra = torch.randn(64, 1, generator=torch.Generator().manual_seed(881112))
    extra = extra - n3_direction @ (n3_direction.T @ extra) / (n3_direction.T @ n3_direction)
    nondesc_basis = torch.linalg.qr(torch.cat((n3_direction, extra), dim=1), mode="reduced").Q
    nondesc_candidate = {"method": "non_descendant_corruption", "layer": 3, "position": 16, "rank": 2, "strength": 1.0}
    (result, _), _, _, _ = _execute_variant(root, "non-descendant-corruption", plans, nondesc_basis, nondesc_candidate)
    aggregate = _aggregate(evaluation, result["decision"])
    nondesc_invariance = result["decision"].get("metrics", {}).get("non_descendant_invariance")
    scenarios.append(_record(
        "non_descendant_corruption", "actual output-head n3-direction activation intervention",
        result["decision"], aggregate, _readiness(aggregate),
        [DevelopmentStatus.DECODE_ONLY.value, DevelopmentStatus.NEGATIVE.value],
        {"non_descendant_invariance": nondesc_invariance},
        substantive_check=type(nondesc_invariance) is float and nondesc_invariance < 0.95,
    ))

    # Registry/oracle/call-ledger faults operate on full real seed objects.
    def mutated(name, mutate, expected=DevelopmentStatus.INVALID.value, upstream="call registry or ledger"):
        bundle = deepcopy(healthy)
        details = mutate(bundle) or {}
        _refresh_reconciliation(bundle)
        oracle_count, pairing_count = _oracle_and_pairing_counts(bundle["registry_rows"])
        bundle["oracle_mismatch_count"] = oracle_count
        bundle["pairing_mismatch_count"] = pairing_count
        decision, aggregate, readiness = _run_bundle_scenario(evaluation, bundle, expected)
        scenarios.append(_record(name, upstream, decision, aggregate, readiness, expected, details))

    def wrong_oracle(bundle):
        row = bundle["registry_rows"][0]
        row["expected_z3"] = 1 - row["expected_z3"]
        row["expected_joint"] = row["expected_z3"] * K + row["expected_n3"]
        return {"call_id": row["call_id"]}
    mutated("wrong_oracle_result", wrong_oracle, upstream="mutated independent-oracle result in call plan")

    def pairing_fault(bundle):
        row = next(r for r in bundle["registry_rows"] if r["pair_kind"] == "same_state")
        replacement = next(r for r in records("causal_evaluation") if r.values.z2 != row["base_z2"])
        row["source_id"] = replacement.example_id
        return {"call_id": row["call_id"], "replacement_source": replacement.example_id}
    mutated("source_base_pairing_mismatch", pairing_fault, upstream="actual source/base call plan")

    def missing_call(bundle):
        row = bundle["ledger_rows"].pop()
        return {"call_id": row["call_id"]}
    mutated("missing_call", missing_call)

    def duplicate_call(bundle):
        bundle["ledger_rows"].append(deepcopy(bundle["ledger_rows"][0]))
        return {"call_id": bundle["ledger_rows"][0]["call_id"]}
    mutated("duplicate_call", duplicate_call)

    def unplanned_call(bundle):
        row = deepcopy(bundle["ledger_rows"][0]); row["call_id"] = "devcall-unplanned-polarity"
        bundle["ledger_rows"].append(row); return {"call_id": row["call_id"]}
    mutated("unplanned_call", unplanned_call)

    def dropped_failed(bundle):
        failed = bundle["ledger_rows"].pop(0)
        failed["completed"] = False; failed["error"] = "DEV_ONLY injected evaluator exception"
        bundle["dropped_failed_calls"] = 1
        return {"dropped_call_id": failed["call_id"]}
    mutated("dropped_failed_call", dropped_failed)

    def duplicate_plan(bundle):
        bundle["registry_rows"].append(deepcopy(bundle["registry_rows"][0])); return {"call_id": bundle["registry_rows"][0]["call_id"]}
    mutated("duplicate_planned_call", duplicate_plan, upstream="pre-call registry")

    def incomplete_control(bundle):
        target = next(r["call_id"] for r in bundle["ledger_rows"] if r["control_family"] == "matched_orthogonal_direction")
        bundle["ledger_rows"] = [r for r in bundle["ledger_rows"] if r["call_id"] != target]
        bundle["registry_rows"] = [r for r in bundle["registry_rows"] if r["call_id"] != target]
        return {"removed_call_id": target}
    mutated("incomplete_control_group", incomplete_control)

    def empty_random(bundle):
        ids = {r["call_id"] for r in bundle["ledger_rows"] if r["control_family"] == "matched_random_direction"}
        bundle["ledger_rows"] = [r for r in bundle["ledger_rows"] if r["call_id"] not in ids]
        bundle["registry_rows"] = [r for r in bundle["registry_rows"] if r["call_id"] not in ids]
        return {"removed_calls": len(ids)}
    mutated("empty_random_controls", empty_random)

    def unknown_status(bundle):
        manifest = {"namespace": NAMESPACE, "requested_status": "DEVELOPMENT_ALIEN_STATUS"}
        path = root / "artifacts/dev_only/polarity/unknown-status-manifest.json"
        write_json(path, manifest); bundle["requested_status"] = manifest["requested_status"]
        return {"manifest": str(path.relative_to(root)), "sha256": sha256_file(path)}
    mutated("unknown_status", unknown_status, upstream="frozen status-source manifest")

    # Real split-manifest leakage and guarded forbidden namespace event.
    leaked_rows = read_jsonl(root / "registries/dev_only/example-index.jsonl")
    duplicate = deepcopy(next(r for r in leaked_rows if r["partition"] == "causal_evaluation"))
    duplicate["partition"] = "training"
    leaked_rows.append(duplicate)
    leak_path = root / "artifacts/dev_only/polarity/leaked-example-index.jsonl"
    write_jsonl(leak_path, leaked_rows)
    partitions_by_id = {}
    for row in leaked_rows:
        partitions_by_id.setdefault(row["example_id"], set()).add(row["partition"])
    leak_bundle = deepcopy(healthy)
    leak_bundle["split_leakage_count"] = sum(len(partitions) > 1 for partitions in partitions_by_id.values())
    decision, aggregate, readiness = _run_bundle_scenario(evaluation, leak_bundle, DevelopmentStatus.INVALID.value)
    scenarios.append(_record("split_leakage", "actual duplicated example-index row", decision, aggregate, readiness, DevelopmentStatus.INVALID.value, {"artifact": str(leak_path.relative_to(root)), "sha256": sha256_file(leak_path)}))

    access_bundle = deepcopy(healthy)
    try:
        require_dev_partition("confirmation")
    except ForbiddenNamespace as error:
        access_bundle["forbidden_access_events"] = [str(error)]
    decision, aggregate, readiness = _run_bundle_scenario(evaluation, access_bundle, DevelopmentStatus.INVALID.value)
    scenarios.append(_record("forbidden_confirmation_namespace_event", "real namespace guard event; no data created or read", decision, aggregate, readiness, DevelopmentStatus.INVALID.value, {"accessed_objects": 0}))

    # A real registry basis-hash mutation is passed back through execute_seed.
    wrong_candidate_plans = deepcopy(plans)
    wrong_candidate_plans[0]["basis_sha256"] = "0" * 64
    result, _ = execute_seed(root, "iit_primary", 1112, wrong_candidate_plans)
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record(
        "wrong_candidate_hash", "candidate hash in frozen registry",
        result["decision"], aggregate, _readiness(aggregate), DevelopmentStatus.INVALID.value,
        {"call_id": wrong_candidate_plans[0]["call_id"], "mutated_basis_sha256": "0" * 64},
    ))

    # Actual checkpoint drift.
    payload = torch.load(checkpoint_path(root, "iit_primary", 1112), map_location="cpu", weights_only=False)
    first = next(iter(payload["state_dict"])); payload["state_dict"][first] = payload["state_dict"][first].clone(); payload["state_dict"][first].view(-1)[0] += 1e-3
    drift_path = root / "artifacts/dev_only/polarity/drifted-iit-seed-1112.pt"; torch.save(payload, drift_path)
    drift_plans = deepcopy(plans)
    for row in drift_plans: row["checkpoint"] = str(drift_path.relative_to(root))
    result, _ = execute_seed(root, "iit_primary", 1112, drift_plans)
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record("checkpoint_drift", "actual tensor-mutated checkpoint", result["decision"], aggregate, _readiness(aggregate), DevelopmentStatus.INVALID.value, {"sha256": sha256_file(drift_path)}))

    # Actual selected-method drift artifact with an honestly updated file hash but stale frozen candidate rows.
    drift_alignment_payload = deepcopy(original_alignment); drift_alignment_payload["selected_candidate"] = {**primary_candidate, "layer": 1}; drift_alignment_payload["schema"] = "exp33ae.polarity_alignment.v1"; drift_alignment_payload["scenario"] = "selected_method_drift"
    method_path = root / "artifacts/dev_only/polarity/selected-method-drift-alignment.pt"; torch.save(drift_alignment_payload, method_path)
    method_plans = deepcopy(plans)
    for row in method_plans:
        row["alignment"] = str(method_path.relative_to(root)); row["alignment_file_sha256"] = sha256_file(method_path)
    result, _ = execute_seed(root, "iit_primary", 1112, method_plans)
    aggregate = _aggregate(evaluation, result["decision"])
    scenarios.append(_record("selected_method_drift", "actual alignment artifact changed after selection freeze", result["decision"], aggregate, _readiness(aggregate), DevelopmentStatus.INVALID.value, {"sha256": sha256_file(method_path)}))

    # No-eligible is generated from an empty candidate registry, not a terminal flag.
    no_eligible = {
        "namespace": NAMESPACE, "candidate_eligibility": False, "registry_rows": [], "ledger_rows": [],
        "selection_manifest_frozen": True, "forbidden_access_events": [],
    }
    decision = evaluate_seed(no_eligible); aggregate = _aggregate(evaluation, decision)
    scenarios.append(_record("no_eligible_candidate_registry", "frozen empty candidate registry", decision, aggregate, _readiness(aggregate), DevelopmentStatus.NO_ELIGIBLE.value))

    # Aggregation faults use real healthy seed results but faulty selection manifests.
    healthy_seed_rows = [
        {"model_seed": r["model_seed"], "decision": r["decision"]}
        for r in evaluation["model_results"] if r["model_kind"] == "iit_primary"
    ]
    for name, rows in (
        ("missing_seed", healthy_seed_rows[:2]),
        ("duplicate_seed", [healthy_seed_rows[0], healthy_seed_rows[0], healthy_seed_rows[2]]),
        ("best_seed_only_selection", [healthy_seed_rows[0]]),
    ):
        aggregate = aggregate_population(rows, "iit_primary")
        decision = {"status": aggregate["status"], "faults": [aggregate.get("reason", "aggregation rejected")]}
        scenarios.append(_record(name, "actual model-selection seed list", decision, aggregate, _readiness(aggregate), DevelopmentStatus.INVALID.value, {"observed_seed_rows": [r["model_seed"] for r in rows]}))

    report = {
        "schema": "exp33ae.real_upstream_polarity.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "scenario_count": len(scenarios),
        "passed_count": sum(s["passed"] for s in scenarios),
        "all_faults_traverse_production_status": all(s["direct_summary_or_terminal_mutation"] is False for s in scenarios),
        "scenarios": scenarios,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    write_json(root / "results/dev_only/real-upstream-polarity.json", report)
    return report
