"""Fail-closed evidence validator, metric reconstruction, and 5/5 recipe selection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..artifacts import tensor_sha256
from ..model import declared_iit_basis
from ..remediation.evaluator import _metrics_for_seed, _seed_status
from ..remediation.normative import POSITIVE_ANALOGUE
from ..remediation.strict import object_sha256, sha256_file, strict_load_json, strict_load_jsonl, StrictValidationError
from .allocation import load as load_allocation
from .entropy import StudyEntropy
from .inference import execute_segment
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash, verify_source_manifest
from .runtime import dependency_manifest, load_checkpoint


STUDY_INVALID = "TRAINING_RECIPE_STUDY_INVALID"
STUDY_NONE = "NO_RELIABLE_TRAINING_RECIPE_FOUND"
STUDY_POSITIVE = "RELIABLE_TRAINING_RECIPE_IDENTIFIED_DEV_ONLY"


def _fail(reason: str, detail: str) -> None:
    raise StrictValidationError(reason, detail)


def _verify_hash(value: dict, field: str, reason: str) -> None:
    copy = dict(value); observed = copy.pop(field, None)
    if observed != object_sha256(copy): _fail(reason, field)


def load_training(output: Path) -> list[dict]:
    values = []
    for recipe in load_protocol()["recipes"]:
        for seed_order in range(5):
            values.append(strict_load_json(output / "training" / recipe["id"] / f"seed-{seed_order}.json"))
    return values


def validate_registry(rows: list[dict], manifest: dict, training: list[dict], entropy: StudyEntropy) -> None:
    if len(rows) != 78400 or manifest.get("row_count") != 78400 or manifest.get("rows_per_model") != 3920 or manifest.get("rows_per_recipe") != 19600 or manifest.get("models") != 20:
        _fail("REGISTRY_STRUCTURE", "exact counts")
    _verify_hash(manifest, "registry_manifest_sha256", "REGISTRY_BINDING")
    if manifest.get("registry_file_sha256") is None or manifest.get("confirmation_registry_materialized") is not False:
        _fail("REGISTRY_STRUCTURE", "manifest")
    if manifest.get("model_seed_identities") != list(entropy.model_seeds()): _fail("SEED_IDENTITY", "manifest seeds")
    payload = [{k: v for k, v in row.items() if k != "registry_payload_commitment"} for row in rows]
    if object_sha256(payload) != manifest["payload_commitment"] or any(r.get("registry_payload_commitment") != manifest["payload_commitment"] for r in rows):
        _fail("REGISTRY_BINDING", "payload")
    keys = set(rows[0]); calls = []; nonces = []
    expected_methods = Counter({"clean_task_decode": 1600, "declared_iit_alignment": 400,
                                "random_direction": 640, "orthogonal_direction": 640,
                                "alternate_location": 480, "alternate_early_layer": 160})
    for ordinal, row in enumerate(rows):
        if set(row) != keys or row.get("schema") != "exp33ae.dev_only_reliability_registry_row.v1" or row.get("namespace") != DEV_NAMESPACE or row.get("claim_bearing") is not False:
            _fail("SCHEMA", f"registry row {ordinal}")
        model_order = ordinal // 3920; train = training[model_order]
        if row["order"] != ordinal or row["model_order"] != model_order or row["recipe_order"] != model_order // 5 or row["seed_order"] != model_order % 5:
            _fail("REGISTRY_STRUCTURE", f"order {ordinal}")
        for field in ("recipe_id", "model_seed_identity", "checkpoint_file_sha256", "checkpoint_tensor_sha256", "training_manifest_sha256"):
            expected = train["final_tensor_state_sha256"] if field == "checkpoint_tensor_sha256" else train["scientific_manifest_sha256"] if field == "training_manifest_sha256" else train[field]
            if row[field] != expected: _fail("REGISTRY_BINDING", f"{field} {ordinal}")
        copy = dict(row); observed = copy.pop("registry_row_sha256"); copy.pop("registry_payload_commitment")
        if observed != object_sha256(copy): _fail("REGISTRY_BINDING", f"row hash {ordinal}")
        if row["planned_attempts"] != 1 or row["retry_policy"] != "resume-first-missing-no-reexecution" or row["eligible"] is not True:
            _fail("REGISTRY_STRUCTURE", f"policy {ordinal}")
        calls.append(row["call_id"]); nonces.append(row["nonce"])
    if len(set(calls)) != 78400 or len(set(nonces)) != 78400: _fail("REGISTRY_STRUCTURE", "duplicate call or nonce")
    for model_order in range(20):
        block = rows[model_order*3920:(model_order+1)*3920]
        if Counter(r["method"] for r in block) != expected_methods: _fail("REGISTRY_STRUCTURE", f"methods model {model_order}")
        if Counter(r["pair_stratum"] for r in block if r["method"] == "declared_iit_alignment") != Counter({"A_answer_changing":160,"B_outcome_collision":80,"C_same_state":80,"D_same_observable_answer":80}):
            _fail("REGISTRY_STRUCTURE", f"strata model {model_order}")
        for family, n in (("random_direction",4),("orthogonal_direction",4),("alternate_location",3),("alternate_early_layer",1)):
            counts = Counter(r["control_id"] for r in block if r["method"] == family)
            if len(counts) != n or set(counts.values()) != {160}: _fail("REGISTRY_STRUCTURE", f"controls {model_order} {family}")


def validate_raw(raw: list[dict], registry: list[dict]) -> None:
    if len(raw) != 78400: _fail("CALL_RECONCILIATION", "exact raw count")
    keys = set(raw[0]); seen = set()
    identities = {"call_id":"call_id","nonce":"nonce","order":"order","model_order":"model_order","recipe_id":"recipe_id",
                  "recipe_order":"recipe_order","seed_order":"seed_order","model_seed_identity":"model_seed_identity",
                  "checkpoint_file_sha256":"checkpoint_file_sha256","checkpoint_tensor_sha256":"checkpoint_tensor_sha256",
                  "source_manifest_sha256":"source_manifest_sha256","dependency_manifest_sha256":"dependency_manifest_sha256",
                  "allocation_sha256":"allocation_sha256","representation_sha256":"representation_sha256",
                  "control_manifest_sha256":"control_manifest_sha256","training_manifest_sha256":"training_manifest_sha256",
                  "programme_id":"programme_id","causal_signature":"causal_signature","base_id":"base_id","source_id":"source_id",
                  "pair_id":"pair_id","pair_stratum":"pair_stratum","method":"method","control_id":"control_id",
                  "basis_sha256":"basis_sha256","decoder_sha256":"decoder_sha256","actual_layer":"layer","actual_position":"position",
                  "actual_rank":"rank","actual_strength":"strength","expected_applied_norm":"expected_applied_norm",
                  "registry_row_sha256":"registry_row_sha256","registry_payload_commitment":"registry_payload_commitment"}
    for ordinal, (row, plan) in enumerate(zip(raw, registry)):
        if set(row) != keys or row.get("schema") != "exp33ae.dev_only_reliability_raw.v1" or row.get("namespace") != DEV_NAMESPACE or row.get("claim_bearing") is not False:
            _fail("SCHEMA", f"raw {ordinal}")
        if any(row[a] != plan[b] for a,b in identities.items()): _fail("RAW_BINDING", f"identity {ordinal}")
        if row["attempted"] is not True or row["completed"] is not True or row["dropped"] is not False or row["attempt_number"] != 1 or row["error"] is not None:
            _fail("CALL_RECONCILIATION", f"completion {ordinal}")
        copy = dict(row); observed = copy.pop("raw_row_sha256")
        if observed != object_sha256(copy): _fail("RAW_BINDING", f"row hash {ordinal}")
        tolerance = plan["norm_tolerance_absolute"] + plan["norm_tolerance_relative"] * abs(plan["expected_applied_norm"])
        if abs(row["applied_norm"] - plan["expected_applied_norm"]) > tolerance: _fail("NORM_MISMATCH", str(ordinal))
        if row["call_id"] in seen: _fail("CALL_RECONCILIATION", "duplicate raw call")
        seen.add(row["call_id"])


def gate_results(metrics: dict) -> dict[str, bool]:
    g = {
        "target_task": metrics["task_target"] >= 680, "parallel_task": metrics["task_parallel"] >= 680,
        "worst_programme": metrics["worst_programme"] >= 12,
        "shortcut_target": metrics["task_target"] - metrics["shortcut_target"] >= 160,
        "shortcut_parallel": metrics["task_parallel"] - metrics["shortcut_parallel"] >= 160,
        "registered_rank2_decode": metrics["decode"] >= 1440, "answer_changing_iia": metrics["iia_A"] >= 136,
        "transport_pooled": sum(metrics["transport"][n] for n in ("A_answer_changing","B_outcome_collision","D_same_observable_answer")) >= 288,
        "transport_A": metrics["transport"]["A_answer_changing"] >= 136, "transport_B": metrics["transport"]["B_outcome_collision"] >= 68,
        "transport_D": metrics["transport"]["D_same_observable_answer"] >= 68,
        "descendant_A": metrics["descendant"]["A_answer_changing"] >= 136, "descendant_B": metrics["descendant"]["B_outcome_collision"] >= 68,
        "descendant_D": metrics["descendant"]["D_same_observable_answer"] >= 68,
        "non_descendant_A": metrics["non_descendant"]["A_answer_changing"] >= 152,
        "non_descendant_B": metrics["non_descendant"]["B_outcome_collision"] >= 76,
        "non_descendant_C": metrics["non_descendant"]["C_same_state"] >= 76,
        "non_descendant_D": metrics["non_descendant"]["D_same_observable_answer"] >= 76,
        "same_state": metrics["same_state"] >= 76, "same_observable_answer_iia": metrics["iia_D"] >= 64,
        "all_individual_control_gaps": len(metrics["control_iia"]) == 12 and all(metrics["iia_A"] - v >= 48 for v in metrics["control_iia"].values()),
        "base_accuracy_all_groups": all(20*v >= 17*metrics["group_denominators"][k] for k,v in metrics["base_groups"].items()),
        "source_accuracy_all_groups": all(20*v >= 17*metrics["group_denominators"][k] for k,v in metrics["source_groups"].items()),
    }
    return g


def select_recipe(per_model: list[dict]) -> tuple[list[dict], str | None, str]:
    recipes = []
    for recipe in load_protocol()["recipes"]:
        rows = [r for r in per_model if r["recipe_id"] == recipe["id"]]
        eligible = len(rows) == 5 and all(r["all_gates_pass"] for r in rows)
        recipes.append({"recipe_id": recipe["id"], "recipe_order": recipe["order"], "passed_models": sum(r["all_gates_pass"] for r in rows),
                        "required_models": 5, "eligible": eligible, "model_statuses": [r["model_status"] for r in rows]})
    selected = next((r["recipe_id"] for r in recipes if r["eligible"]), None)
    return recipes, selected, STUDY_POSITIVE if selected else STUDY_NONE


def evaluate(root: Path, output: Path, access_snapshot: dict) -> dict:
    try:
        if access_snapshot["state"]["state"] != "BUFFER_COMPLETE" or access_snapshot["state"]["buffered_calls"] != 78400:
            _fail("ACCESS_STATE", "complete 78400-call buffer required")
        entropy_record = strict_load_json(output / "entropy.json"); _verify_hash(entropy_record, "record_sha256", "ENTROPY_BINDING")
        if entropy_record.get("namespace") != DEV_NAMESPACE or entropy_record.get("claim_bearing") is not False or entropy_record.get("confirmation_master_entropy_created") is not False:
            _fail("STAGE_GUARD", "entropy namespace")
        entropy = StudyEntropy(entropy_record["freeze_commit"])
        if entropy.master.hex() != entropy_record["dev_only_master_hex"] or list(entropy.model_seeds()) != entropy_record["model_seed_identities"]:
            _fail("ENTROPY_BINDING", "derivation")
        source = strict_load_json(output / "source-manifest.json"); verify_source_manifest(root, source)
        dependency = strict_load_json(output / "dependency-manifest.json"); _verify_hash(dependency, "manifest_sha256", "DEPENDENCY_BINDING")
        if dependency != dependency_manifest(): _fail("DEPENDENCY_BINDING", "live environment")
        allocation, pairs, allocation_manifest = load_allocation(output); _verify_hash(allocation_manifest, "allocation_sha256", "ALLOCATION_BINDING")
        for name, digest in allocation_manifest["files"].items():
            if sha256_file(output / "allocation" / name) != digest: _fail("ALLOCATION_BINDING", name)
        if len(allocation["programmes"]) != 702 or len(allocation["examples"]) != 11232 or len(pairs) != 400 or allocation_manifest["oracle_disagreements"] != 0:
            _fail("ALLOCATION_BINDING", "counts/oracles")
        representation = strict_load_json(output / "representation.json"); _verify_hash(representation, "representation_sha256", "REPRESENTATION_BINDING")
        training = load_training(output)
        if len(training) != 20: _fail("TRAINING_BINDING", "model count")
        seeds = entropy.model_seeds()
        for model_order, train in enumerate(training):
            _verify_hash(train, "scientific_manifest_sha256", "TRAINING_BINDING")
            recipe = load_protocol()["recipes"][model_order // 5]
            if (train["model_order"], train["recipe_id"], train["seed_order"], train["model_seed_identity"], train["steps"], train["model_config"]["d_model"], train["model_config"]["d_mlp"]) != (model_order, recipe["id"], model_order%5, seeds[model_order%5], recipe["steps"], recipe["d_model"], recipe["d_mlp"]):
                _fail("TRAINING_BINDING", str(model_order))
            load_checkpoint(output / train["checkpoint_relative_path"], {"recipe_id": train["recipe_id"], "recipe_order": train["recipe_order"],
                "seed_order": train["seed_order"], "model_seed_identity": train["model_seed_identity"], "source_manifest_sha256": source["manifest_sha256"],
                "dependency_manifest_sha256": dependency["manifest_sha256"], "tensor_state_sha256": train["final_tensor_state_sha256"],
                "checkpoint_file_sha256": train["checkpoint_file_sha256"]})
            control = strict_load_json(output / "controls" / train["recipe_id"] / f"seed-{train['seed_order']}.json"); _verify_hash(control, "control_manifest_sha256", "CONTROL_BINDING")
            if Counter(c["family"] for c in control["controls"]) != Counter({"random_direction":4,"orthogonal_direction":4,"alternate_location":3,"alternate_early_layer":1}): _fail("CONTROL_BINDING", str(model_order))
        manifest = strict_load_json(output / "registry" / "manifest.json"); rows = strict_load_jsonl(output / "registry" / "registry.jsonl")
        if manifest["registry_file_sha256"] != sha256_file(output / "registry" / "registry.jsonl"): _fail("REGISTRY_BINDING", "file")
        validate_registry(rows, manifest, training, entropy); raw = access_snapshot["buffer_rows"]; validate_raw(raw, rows)
        # Independent actual-inference reconstruction prevents coherent raw-row substitution.
        examples = {r.example_id: r for r in allocation["examples"]}; by_model = {r["model_order"]: r for r in training}; rebuilt = []
        for start in range(0, len(rows), 256): rebuilt.extend(execute_segment(output, rows[start:start+256], examples, by_model))
        if rebuilt != raw: _fail("RAW_BINDING", "independent actual-inference reconstruction")
        per_model = []
        for model_order, train in enumerate(training):
            plans = rows[model_order*3920:(model_order+1)*3920]; observations = raw[model_order*3920:(model_order+1)*3920]
            metrics = _metrics_for_seed(plans, observations, allocation_manifest["shortcut_baselines"]); gates = gate_results(metrics)
            status = _seed_status(metrics); all_pass = status == POSITIVE_ANALOGUE and all(gates.values())
            per_model.append({"model_order": model_order, "recipe_id": train["recipe_id"], "recipe_order": train["recipe_order"],
                              "seed_order": train["seed_order"], "model_seed_identity": train["model_seed_identity"],
                              "model_status": status, "all_gates_pass": all_pass,
                              "failed_gates": [k for k,v in gates.items() if not v], "gates": gates, "metrics": metrics})
        recipes, selected, status = select_recipe(per_model)
        return {"schema": "exp33ae.dev_only_reliability_derived_status.v1", "namespace": DEV_NAMESPACE,
                "claim_bearing": False, "status": status, "selected_recipe": selected,
                "selection_rule": "first 5/5 eligible recipe in R0,R1,R2,R3 order", "recipes": recipes,
                "models": per_model, "registry_calls": 78400, "attempted_calls": 78400, "completed_calls": 78400,
                "confirmation_status_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
                "invalidity_reasons": []}
    except Exception as error:
        reason = error.reason if isinstance(error, StrictValidationError) else "INTERNAL_VALIDATION_EXCEPTION"
        return {"schema": "exp33ae.dev_only_reliability_derived_status.v1", "namespace": DEV_NAMESPACE,
                "claim_bearing": False, "status": STUDY_INVALID, "selected_recipe": None, "recipes": [], "models": [],
                "invalidity_reasons": [reason], "invalidity_detail": str(error),
                "confirmation_status_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0}
