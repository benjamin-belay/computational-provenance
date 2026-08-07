"""Strict typed development status logic with invalidity precedence."""

from __future__ import annotations

from enum import Enum
import math

from .interventions import summarize
from .spec import GATES, MODEL_SEEDS


class DevelopmentStatus(str, Enum):
    OBSERVED = "DEVELOPMENT_ENGINEERED_CAUSAL_REPRESENTATION_OBSERVED"
    COMPETENCE_FAILED = "DEVELOPMENT_MODEL_COMPETENCE_FAILED"
    DECODE_ONLY = "DEVELOPMENT_DECODE_ONLY_CAUSALITY_FAILED"
    NO_ELIGIBLE = "DEVELOPMENT_NO_ELIGIBLE_REPRESENTATION"
    NEGATIVE = "DEVELOPMENT_EXPERIMENT_NEGATIVE"
    INVALID = "DEVELOPMENT_EXPERIMENT_INVALID"
    NOT_READY = "DEVELOPMENT_NOT_READY"
    READY = "DEVELOPMENT_READY_FOR_33A_E_PROTOCOL_FREEZE"


REQUIRED_GROUPS = {
    ("causal_candidate", "different_state"): 160,
    ("causal_candidate", "same_state"): 80,
    ("causal_candidate", "same_answer_different_state"): 80,
    ("matched_random_direction", "different_state"): 160,
    ("matched_orthogonal_direction", "different_state"): 160,
    ("matched_random_location", "different_state"): 160,
    ("matched_random_layer", "different_state"): 160,
}


def _exact_bool(value) -> bool:
    return type(value) is bool


def _finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return True


def _number(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def validate_bundle(bundle: dict) -> list[str]:
    faults = []
    if type(bundle) is not dict or bundle.get("namespace") != "DEV_ONLY":
        return ["namespace_or_schema: bundle is not a DEV_ONLY object"]
    if not _exact_bool(bundle.get("candidate_eligibility")) or bundle.get("candidate_eligibility") is not True:
        faults.append("selection: candidate_eligibility must be exact true for an evaluation bundle")
    plans = bundle.get("registry_rows")
    calls = bundle.get("ledger_rows")
    if type(plans) is not list or not plans:
        faults.append("registry: absent or empty call plan")
        plans = []
    if type(calls) is not list or not calls:
        faults.append("ledger: absent or empty call ledger")
        calls = []
    if not _finite_tree(bundle):
        faults.append("schema: nonfinite numeric value")
    plan_ids = [row.get("call_id") for row in plans if type(row) is dict]
    call_ids = [row.get("call_id") for row in calls if type(row) is dict]
    if len(plan_ids) != len(set(plan_ids)):
        faults.append("registry: duplicate call id")
    if len(call_ids) != len(set(call_ids)):
        faults.append("ledger: duplicate call id")
    missing = set(plan_ids) - set(call_ids)
    unplanned = set(call_ids) - set(plan_ids)
    if missing:
        faults.append(f"ledger: {len(missing)} missing planned calls")
    if unplanned:
        faults.append(f"ledger: {len(unplanned)} unplanned calls")
    for row in plans:
        if type(row) is not dict or row.get("namespace") != "DEV_ONLY":
            faults.append("registry: malformed or wrong-namespace row")
            break
        if not _exact_bool(row.get("planned")) or row["planned"] is not True:
            faults.append("registry: planned must be exact true")
            break
        if not _exact_bool(row.get("eligible")) or row["eligible"] is not True:
            faults.append("registry: eligible must be exact true")
            break
        if not _exact_bool(row.get("no_op")):
            faults.append("registry: no_op must be an exact boolean")
            break
    for row in calls:
        if type(row) is not dict or row.get("namespace") != "DEV_ONLY":
            faults.append("ledger: malformed or wrong-namespace row")
            break
        if not _exact_bool(row.get("attempted")) or row["attempted"] is not True:
            faults.append("ledger: attempted must be exact true")
            break
        if not _exact_bool(row.get("completed")):
            faults.append("ledger: completed must be exact boolean")
            break
        required_call_numbers = (
            "base_prediction", "base_joint", "source_prediction", "source_joint",
            "expected_joint", "intervened_prediction", "decoded_intervened_z2",
            "source_z2", "intervened_z3", "expected_z3", "intervened_n3",
            "expected_n3", "intervention_norm", "base_hidden_norm", "norm_ratio",
        )
        if any(not _number(row.get(name)) for name in required_call_numbers):
            faults.append("ledger: missing, wrong-type, or nonfinite metric field")
            break
    if any(row.get("completed") is not True for row in calls):
        faults.append("denominator: one or more planned calls did not complete")
    observed_groups = {}
    for row in calls:
        key = (row.get("control_family"), row.get("pair_kind"))
        observed_groups[key] = observed_groups.get(key, 0) + 1
    if observed_groups != REQUIRED_GROUPS:
        faults.append(f"controls: exact group counts mismatch {observed_groups}")
    if bundle.get("registry_reconciled") is not True or not _exact_bool(bundle.get("registry_reconciled")):
        faults.append("registry: reconciliation is not exact true")
    if bundle.get("oracle_mismatch_count") != 0 or type(bundle.get("oracle_mismatch_count")) is not int:
        faults.append("oracle: counterfactual mismatch")
    if bundle.get("pairing_mismatch_count") != 0 or type(bundle.get("pairing_mismatch_count")) is not int:
        faults.append("pairing: source/base semantics mismatch")
    if bundle.get("split_leakage_count") != 0 or type(bundle.get("split_leakage_count")) is not int:
        faults.append("leakage: split overlap detected")
    for name in (
        "checkpoint_file_hash_match",
        "checkpoint_tensor_hash_match",
        "checkpoint_identity_match",
        "alignment_hash_match",
        "alignment_identity_match",
        "control_artifact_hash_match",
        "control_identity_match",
        "control_distinctness_match",
        "candidate_hash_match",
        "pre_post_model_state_match",
    ):
        value = bundle.get(name)
        if not _exact_bool(value) or value is not True:
            faults.append(f"freeze: {name} is not exact true")
    if bundle.get("forbidden_access_events") != [] or type(bundle.get("forbidden_access_events")) is not list:
        faults.append("namespace_or_access: forbidden access event")
    if bundle.get("requested_status") not in {None, *(status.value for status in DevelopmentStatus)}:
        faults.append("status: unknown requested status")
    if bundle.get("dropped_failed_calls") != 0 or type(bundle.get("dropped_failed_calls")) is not int:
        faults.append("denominator: failed calls were dropped")
    competence = bundle.get("competence")
    decode = bundle.get("decode")
    axes = ("heldout_target_chain_composition", "heldout_parallel_chain_composition")
    if type(competence) is not dict or type(competence.get("by_axis")) is not dict:
        faults.append("competence: absent or malformed")
    else:
        if any(
            type(competence["by_axis"].get(axis)) is not dict
            or not _number(competence["by_axis"][axis].get("accuracy"))
            for axis in axes
        ):
            faults.append("competence: missing or malformed held-out axis")
        if not _number(competence.get("minimum_programme_accuracy")):
            faults.append("competence: malformed minimum programme accuracy")
        advantages = competence.get("shortcut_advantage")
        if type(advantages) is not dict or any(not _number(advantages.get(axis)) for axis in axes):
            faults.append("competence: malformed shortcut advantage")
    if type(decode) is not dict or not _number(decode.get("accuracy")):
        faults.append("decode: absent or malformed")
    return faults


def evaluate_seed(bundle: dict) -> dict:
    if type(bundle) is dict and bundle.get("candidate_eligibility") is False:
        allowed = (
            bundle.get("namespace") == "DEV_ONLY"
            and bundle.get("registry_rows") == []
            and bundle.get("ledger_rows") == []
            and bundle.get("selection_manifest_frozen") is True
            and _exact_bool(bundle.get("selection_manifest_frozen"))
            and bundle.get("forbidden_access_events") == []
        )
        if allowed:
            return {"status": DevelopmentStatus.NO_ELIGIBLE.value, "faults": [], "metrics": {}}
        return {"status": DevelopmentStatus.INVALID.value, "faults": ["selection: malformed no-eligible-candidate bundle"], "metrics": {}}
    faults = validate_bundle(bundle)
    if faults:
        return {"status": DevelopmentStatus.INVALID.value, "faults": faults, "metrics": {}}
    calls = bundle["ledger_rows"]
    groups = {
        key: [row for row in calls if (row["control_family"], row["pair_kind"]) == key]
        for key in REQUIRED_GROUPS
    }
    summaries = {f"{key[0]}:{key[1]}": summarize(rows) for key, rows in groups.items()}
    causal = summaries["causal_candidate:different_state"]
    same = summaries["causal_candidate:same_state"]
    same_answer = summaries["causal_candidate:same_answer_different_state"]
    random_direction = summaries["matched_random_direction:different_state"]
    orthogonal = summaries["matched_orthogonal_direction:different_state"]
    random_location = summaries["matched_random_location:different_state"]
    random_layer = summaries["matched_random_layer:different_state"]
    competence = bundle["competence"]
    decode = bundle["decode"]
    metrics = {
        "competence": competence,
        "decode": decode,
        "different_state_iia": causal["iia"],
        "state_transport": causal["state_transport"],
        "descendant_correctness": causal["descendant_correctness"],
        "non_descendant_invariance": causal["non_descendant_invariance"],
        "same_state_preservation": same["same_state_preservation"],
        "same_answer_different_state_iia": same_answer["iia"],
        "random_direction_iia": random_direction["iia"],
        "orthogonal_direction_iia": orthogonal["iia"],
        "random_location_iia": random_location["iia"],
        "random_layer_iia": random_layer["iia"],
        "random_direction_gap": causal["iia"] - random_direction["iia"],
        "orthogonal_direction_gap": causal["iia"] - orthogonal["iia"],
        "random_location_gap": causal["iia"] - random_location["iia"],
        "random_layer_gap": causal["iia"] - random_layer["iia"],
        "base_competence_all_groups": min(summary["base_competence"] for summary in summaries.values()),
        "source_competence_all_groups": min(summary["source_competence"] for summary in summaries.values()),
        "call_completion": sum(summary["completed"] for summary in summaries.values()) / sum(summary["attempted"] for summary in summaries.values()),
        "max_intervention_norm_ratio": max(summary["max_norm_ratio"] for summary in summaries.values()),
        "group_summaries": summaries,
    }
    competence_pass = (
        competence["by_axis"]["heldout_target_chain_composition"]["accuracy"] >= GATES["task_accuracy_each_axis"]
        and competence["by_axis"]["heldout_parallel_chain_composition"]["accuracy"] >= GATES["task_accuracy_each_axis"]
        and competence["minimum_programme_accuracy"] >= GATES["minimum_programme_accuracy"]
        and competence["shortcut_advantage"]["heldout_target_chain_composition"] >= GATES["shortcut_advantage"]
        and competence["shortcut_advantage"]["heldout_parallel_chain_composition"] >= GATES["shortcut_advantage"]
        and metrics["base_competence_all_groups"] >= GATES["call_base_source_accuracy"]
        and metrics["source_competence_all_groups"] >= GATES["call_base_source_accuracy"]
    )
    causal_pass = (
        decode["accuracy"] >= GATES["decode_accuracy"]
        and metrics["different_state_iia"] >= GATES["different_state_iia"]
        and metrics["state_transport"] >= GATES["state_transport"]
        and metrics["descendant_correctness"] >= GATES["descendant_correctness"]
        and metrics["non_descendant_invariance"] >= GATES["non_descendant_invariance"]
        and metrics["same_state_preservation"] >= GATES["same_state_preservation"]
        and metrics["same_answer_different_state_iia"] >= GATES["same_answer_different_state_iia"]
        and metrics["random_direction_gap"] >= GATES["random_direction_gap"]
        and metrics["orthogonal_direction_gap"] >= GATES["orthogonal_direction_gap"]
        and metrics["random_location_gap"] >= GATES["random_location_gap"]
        and metrics["random_layer_gap"] >= GATES["random_layer_gap"]
        and metrics["call_completion"] == GATES["call_completion"]
    )
    if not competence_pass:
        status = DevelopmentStatus.COMPETENCE_FAILED
    elif causal_pass:
        status = DevelopmentStatus.OBSERVED
    elif decode["accuracy"] >= GATES["decode_accuracy"] and metrics["different_state_iia"] < GATES["different_state_iia"]:
        status = DevelopmentStatus.DECODE_ONLY
    else:
        status = DevelopmentStatus.NEGATIVE
    return {"status": status.value, "faults": [], "metrics": metrics, "gates": GATES}


def aggregate_population(seed_results: list[dict], kind: str) -> dict:
    expected = list(MODEL_SEEDS[kind])
    observed = [result.get("model_seed") for result in seed_results]
    if observed != expected or len(set(observed)) != len(expected):
        return {"status": DevelopmentStatus.INVALID.value, "reason": "missing, extra, reordered, or duplicate seed"}
    statuses = [result["decision"]["status"] for result in seed_results]
    if DevelopmentStatus.INVALID.value in statuses:
        status = DevelopmentStatus.INVALID
    elif DevelopmentStatus.COMPETENCE_FAILED.value in statuses:
        status = DevelopmentStatus.COMPETENCE_FAILED
    elif kind == "iit_primary" and all(value == DevelopmentStatus.OBSERVED.value for value in statuses):
        status = DevelopmentStatus.OBSERVED
    elif DevelopmentStatus.DECODE_ONLY.value in statuses:
        status = DevelopmentStatus.DECODE_ONLY
    else:
        status = DevelopmentStatus.NEGATIVE
    return {"status": status.value, "expected_seeds": expected, "observed_seeds": observed, "seed_statuses": statuses}


def evaluate_readiness(primary_aggregate: dict, required_checks: dict) -> dict:
    """Map a scientific development result plus real audit checks to readiness."""
    if type(primary_aggregate) is not dict or type(required_checks) is not dict or not required_checks:
        return {"status": DevelopmentStatus.INVALID.value, "reason": "malformed readiness inputs"}
    known = {status.value for status in DevelopmentStatus}
    primary_status = primary_aggregate.get("status")
    if primary_status not in known:
        return {"status": DevelopmentStatus.INVALID.value, "reason": "unknown scientific status"}
    if any(not _exact_bool(value) for value in required_checks.values()):
        return {"status": DevelopmentStatus.INVALID.value, "reason": "readiness checks must be exact booleans"}
    if primary_status == DevelopmentStatus.INVALID.value:
        return {"status": DevelopmentStatus.INVALID.value, "reason": "scientific pipeline invalid"}
    if not all(required_checks.values()):
        failed = sorted(name for name, value in required_checks.items() if not value)
        return {"status": DevelopmentStatus.NOT_READY.value, "failed_checks": failed}
    if primary_status != DevelopmentStatus.OBSERVED.value:
        return {"status": DevelopmentStatus.NOT_READY.value, "reason": "primary development observation absent"}
    return {"status": DevelopmentStatus.READY.value, "checks": dict(sorted(required_checks.items()))}
