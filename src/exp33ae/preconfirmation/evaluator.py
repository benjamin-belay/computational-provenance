"""Strict future-status evaluator over real upstream DEV_ONLY fixture objects."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from enum import Enum
from hashlib import sha256
import math

from ..artifacts import canonical_json
from ..independent_oracle import oracle
from ..spec import ALL_OPERATIONS
from .allocator import causal_signature, development_exclusions, programme_id
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .registry import reconcile


class FutureStatus(str, Enum):
    POSITIVE = "CONFIRMED_ENGINEERED_CAUSAL_ABSTRACTION_POSITIVE"
    COMPETENCE_FAILED = "MODEL_COMPETENCE_FAILED"
    DECODE_ONLY = "DECODE_ONLY_CAUSALITY_FAILED"
    NO_ELIGIBLE = "NO_ELIGIBLE_CAUSAL_REPRESENTATION"
    NEGATIVE = "EXPERIMENT_NEGATIVE"
    INVALID = "EXPERIMENT_INVALID"


def healthy_raw(registry_rows: list[dict]) -> list[dict]:
    rows = []
    for plan in registry_rows:
        primary = plan["method"] == "declared_iit_alignment"
        clean = plan["method"] == "clean_task_decode"
        if clean or primary:
            prediction = plan["expected_answer"]
            decoded = plan["expected_z2"]
            z3, n3 = plan["expected_z3"], plan["expected_n3"]
        else:
            prediction = plan["base_answer"]
            decoded = plan["base_z2"]
            z3, n3 = plan["base_z3"], plan["base_n3"]
        rows.append({
            "schema": "exp33ae.dev_only_raw_call.v1", "namespace": DEV_NAMESPACE,
            "stage_id": plan["stage_id"], "claim_bearing": False,
            "call_id": plan["call_id"], "nonce": plan["nonce"], "model_order": plan["model_order"],
            "model_seed_identity": plan["model_seed_identity"], "checkpoint_sha256": plan["checkpoint_sha256"],
            "programme_id": plan["programme_id"], "pair_id": plan["pair_id"], "method": plan["method"],
            "basis_sha256": plan["basis_sha256"], "layer": plan["layer"], "position": plan["position"],
            "rank": plan["rank"], "strength": plan["strength"], "attempt_number": 1,
            "attempted": True, "completed": True, "dropped": False, "error": None,
            "base_prediction": plan["base_answer"], "source_prediction": plan["source_answer"],
            "intervened_prediction": prediction, "decoded_z2": decoded,
            "intervened_z3": z3, "intervened_n3": n3,
            "applied_norm": 0.0 if clean else 1.0,
        })
    return rows


def _exact_bool(value) -> bool:
    return type(value) is bool


def _finite_number(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _validate_allocation(root, allocation: dict) -> list[str]:
    faults = []
    if type(allocation) is not dict or allocation.get("namespace") != DEV_NAMESPACE:
        return ["allocation:schema_or_namespace"]
    programmes = allocation.get("programmes")
    examples = allocation.get("examples")
    if type(programmes) is not list or type(examples) is not list:
        return ["allocation:missing_rows"]
    exclusions = development_exclusions(root)
    ids, signatures = [], []
    for row in programmes:
        try:
            opids = tuple(row["operation_ids"])
            pid, signature = programme_id(opids), causal_signature(opids)
            ids.append(row["programme_id"]); signatures.append(row["causal_signature"])
            if row["programme_id"] != pid or row["causal_signature"] != signature:
                faults.append("allocation:wrong_programme_or_signature")
                break
        except Exception:
            faults.append("allocation:malformed_programme")
            break
    if len(ids) != len(set(ids)) or len(signatures) != len(set(signatures)):
        faults.append("allocation:semantic_or_exact_duplicate")
    if set(ids) & exclusions["programme_ids"]:
        faults.append("allocation:development_programme_overlap")
    example_ids = []
    for row in examples:
        try:
            example_ids.append(row.example_id)
            values = oracle(tuple(ALL_OPERATIONS[index] for index in row.operation_ids), (row.a, row.b, row.c, row.d))
            if any(getattr(row, key) != values[key] for key in ("z1", "z2", "z3", "n1", "n2", "n3")) or row.answer != values["joint_answer"]:
                faults.append("allocation:example_oracle_mismatch")
                break
        except Exception:
            faults.append("allocation:malformed_example")
            break
    if len(example_ids) != len(set(example_ids)) or set(example_ids) & exclusions["example_ids"]:
        faults.append("allocation:example_duplicate_or_development_overlap")
    expected = {name: count * 16 for name, count in load_protocol()["allocation"]["counts_programmes"].items()}
    observed = Counter(row.partition for row in examples)
    if dict(observed) != expected:
        faults.append("allocation:exact_count_mismatch")
    for partition, count in expected.items():
        rows = [row for row in examples if row.partition == partition]
        wanted = {
            "z2": Counter({0: count//2, 1: count//2}), "z3": Counter({0: count//2, 1: count//2}),
            "n3": Counter({0: count//2, 1: count//2}), "answer": Counter({0: count//4, 1: count//4, 2: count//4, 3: count//4}),
        }
        if any(Counter(getattr(row, name) for row in rows) != target for name, target in wanted.items()):
            faults.append(f"allocation:imbalance:{partition}")
    train_t = {row["target_composition_id"] for row in programmes if row["partition"] == "training"}
    train_n = {row["parallel_composition_id"] for row in programmes if row["partition"] == "training"}
    target_eval = {row["target_composition_id"] for row in programmes if row["partition"] == "clean_target_axis"}
    parallel_eval = {row["parallel_composition_id"] for row in programmes if row["partition"] == "clean_parallel_axis"}
    if train_t & target_eval or train_n & parallel_eval:
        faults.append("allocation:heldout_axis_overlap")
    return faults


def _validate_registry(package: dict) -> list[str]:
    faults = []
    registry = package.get("registry")
    if type(registry) is not dict or registry.get("namespace") != DEV_NAMESPACE:
        return ["registry:schema_or_namespace"]
    rows = registry.get("rows")
    if type(rows) is not list or len(rows) != 11760:
        return ["registry:exact_count_mismatch"]
    seeds = registry.get("model_seed_identities")
    if type(seeds) is not list or len(seeds) != 3 or len(set(seeds)) != 3:
        faults.append("registry:seed_identity_or_uniqueness")
    if seeds != package.get("expected_model_seed_identities"):
        faults.append("registry:seed_order_or_identity")
    payload = []
    for row in rows:
        if type(row) is not dict:
            faults.append("registry:malformed_row"); break
        copy = dict(row); copy.pop("registry_payload_commitment", None); payload.append(copy)
        if (
            row.get("schema") != "exp33ae.preconfirmation_registry_row.v1"
            or row.get("domain") != DEV_NAMESPACE or row.get("stage_id") != package.get("stage_id")
            or row.get("protocol_sha256") != protocol_hash() or row.get("claim_bearing") is not False
            or not _exact_bool(row.get("eligible")) or not _exact_bool(row.get("collision"))
            or not _exact_bool(row.get("same_observable_answer")) or not _exact_bool(row.get("no_op"))
        ):
            faults.append("registry:row_schema_type_or_domain"); break
    commitment = sha256(canonical_json(payload).encode()).hexdigest()
    if registry.get("payload_commitment") != commitment or any(row.get("registry_payload_commitment") != commitment for row in rows):
        faults.append("registry:commitment_mismatch")
    calls, nonces = [row.get("call_id") for row in rows], [row.get("nonce") for row in rows]
    if len(set(calls)) != len(rows) or len(set(nonces)) != len(rows):
        faults.append("registry:duplicate_call_or_nonce")
    exclusions = package["development_exclusions"]
    pair_ids = {row.get("pair_id") for row in rows}
    if set(calls) & set(exclusions["call_ids"]) or set(nonces) & set(exclusions["nonces"]) or pair_ids & set(exclusions["pair_ids"]):
        faults.append("registry:development_identity_overlap")
    examples = {row.example_id: row for row in package["allocation"]["examples"]}
    for row in rows:
        base, source = examples.get(row.get("base_id")), examples.get(row.get("source_id"))
        if base is None or source is None:
            faults.append("registry:unknown_base_or_source"); break
        if row["method"] == "clean_task_decode":
            valid = (
                base.example_id == source.example_id and row["pair_stratum"] == "clean_task_decode"
                and row["programme_id"] == base.programme_id and row["causal_signature"] == base.causal_signature
                and row["expected_z2"] == base.z2 and row["expected_z3"] == base.z3
                and row["expected_n3"] == base.n3 and row["expected_answer"] == base.answer
            )
        else:
            values_match = all(row[f"base_{name}"] == getattr(base, "answer" if name == "answer" else name) for name in ("z2", "z3", "n3", "answer")) and all(
                row[f"source_{name}"] == getattr(source, "answer" if name == "answer" else name) for name in ("z2", "z3", "n3", "answer")
            )
            expected_values = oracle(tuple(ALL_OPERATIONS[index] for index in base.operation_ids), (base.a, base.b, base.c, base.d))
            # The independent do-operation changes z2 only and preserves n3.
            z3 = ALL_OPERATIONS[base.operation_ids[2]][2 * source.z2 + base.d]
            expected_answer = 2 * z3 + base.n3
            stratum = row["pair_stratum"]
            predicate = {
                "A_answer_changing": base.z2 != source.z2 and expected_answer != base.answer and base.answer != source.answer,
                "B_outcome_collision": base.z2 != source.z2 and expected_answer == base.answer,
                "C_same_state": base.z2 == source.z2 and base.example_id != source.example_id,
                "D_same_observable_answer": base.z2 != source.z2 and base.answer == source.answer and expected_answer != base.answer,
            }.get(stratum, False)
            valid = (
                values_match and predicate and row["programme_id"] == base.programme_id
                and row["causal_signature"] == base.causal_signature and row["axis"] == base.axis
                and row["expected_z2"] == source.z2 and row["expected_z3"] == z3
                and row["expected_n3"] == base.n3 and row["expected_answer"] == expected_answer
                and row["no_op"] is (base.z2 == source.z2)
                and row["collision"] is (expected_answer == base.answer)
                and row["same_observable_answer"] is (base.answer == source.answer)
            )
        if not valid:
            faults.append("registry:pair_eligibility_or_oracle_mismatch"); break
    expected_seeds = package.get("expected_model_seed_identities", [])
    for order, seed in enumerate(expected_seeds):
        model = [row for row in rows if row.get("model_order") == order and row.get("model_seed_identity") == seed]
        groups = Counter(row["method"] for row in model)
        if len(model) != 3920 or groups != Counter({
            "clean_task_decode": 1600, "declared_iit_alignment": 400, "random_direction": 640,
            "orthogonal_direction": 640, "alternate_location": 480, "alternate_early_layer": 160,
        }):
            faults.append(f"registry:group_counts:{order}")
        if Counter(row["pair_stratum"] for row in model if row["method"] == "declared_iit_alignment") != Counter({
            "A_answer_changing": 160, "B_outcome_collision": 80, "C_same_state": 80, "D_same_observable_answer": 80,
        }):
            faults.append(f"registry:primary_strata:{order}")
        for family, expected_controls in (("random_direction", 4), ("orthogonal_direction", 4), ("alternate_location", 3), ("alternate_early_layer", 1)):
            control_ids = Counter(row["control_id"] for row in model if row["method"] == family)
            if len(control_ids) != expected_controls or set(control_ids.values()) != {160}:
                faults.append(f"registry:control_strata:{order}:{family}")
    return faults


def _validate_raw(package: dict) -> list[str]:
    plans = package["registry"]["rows"]
    raw = package.get("raw_rows")
    if type(raw) is not list or not raw:
        return ["raw:missing_or_empty"]
    reconciliation = reconcile(plans, raw)
    faults = [] if reconciliation["exact"] else ["raw:reconciliation:" + canonical_json(reconciliation)]
    plan_by_id = {row["call_id"]: row for row in plans}
    allowed_numbers = ("base_prediction", "source_prediction", "intervened_prediction", "decoded_z2", "intervened_z3", "intervened_n3", "applied_norm")
    for row in raw:
        plan = plan_by_id.get(row.get("call_id"))
        if plan is None:
            continue
        if (
            row.get("schema") != "exp33ae.dev_only_raw_call.v1" or row.get("namespace") != DEV_NAMESPACE
            or row.get("claim_bearing") is not False or not _exact_bool(row.get("attempted"))
            or not _exact_bool(row.get("completed")) or not _exact_bool(row.get("dropped"))
            or any(not _finite_number(row.get(field)) for field in allowed_numbers)
        ):
            faults.append("raw:schema_type_or_nonfinite"); break
        for field in ("nonce", "model_order", "model_seed_identity", "checkpoint_sha256", "programme_id", "pair_id", "method", "basis_sha256", "layer", "position", "rank", "strength"):
            if row.get(field) != plan.get(field):
                faults.append(f"raw:identity_mismatch:{field}"); break
    # Norm match every individual control to the corresponding primary A pair.
    primary_norm = {(row["model_seed_identity"], row["pair_id"]): row["applied_norm"] for row in raw if row["method"] == "declared_iit_alignment" and plan_by_id[row["call_id"]]["pair_stratum"] == "A_answer_changing"}
    for row in raw:
        if row["method"] in {"random_direction", "orthogonal_direction", "alternate_location", "alternate_early_layer"}:
            reference = primary_norm.get((row["model_seed_identity"], row["pair_id"]))
            if reference is None or abs(row["applied_norm"] - reference) > 1e-6 + 1e-5 * abs(reference):
                faults.append("raw:norm_mismatch"); break
    return faults


def _metrics_for_seed(plans: list[dict], raw: list[dict], shortcut: dict) -> dict:
    observed = {row["call_id"]: row for row in raw}
    clean = [plan for plan in plans if plan["method"] == "clean_task_decode"]
    primary = [plan for plan in plans if plan["method"] == "declared_iit_alignment"]
    controls = [plan for plan in plans if plan["method"] not in {"clean_task_decode", "declared_iit_alignment"}]

    def correct(rows, predicate):
        return sum(predicate(plan, observed[plan["call_id"]]) for plan in rows)

    axes = {
        axis: [plan for plan in clean if plan["axis"] == axis]
        for axis in ("blinded_within_run_target_composition", "blinded_within_run_parallel_composition")
    }
    programme_scores = Counter(); programme_totals = Counter()
    for plan in clean:
        programme_totals[plan["programme_id"]] += 1
        programme_scores[plan["programme_id"]] += observed[plan["call_id"]]["intervened_prediction"] == plan["expected_answer"]
    strata = {name: [plan for plan in primary if plan["pair_stratum"] == name] for name in ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")}
    metrics = {
        "task_target": correct(axes["blinded_within_run_target_composition"], lambda p, r: r["intervened_prediction"] == p["expected_answer"]),
        "task_parallel": correct(axes["blinded_within_run_parallel_composition"], lambda p, r: r["intervened_prediction"] == p["expected_answer"]),
        "worst_programme": min(programme_scores[pid] for pid in programme_totals),
        "decode": correct(clean, lambda p, r: r["decoded_z2"] == p["expected_z2"]),
        "transport": {}, "descendant": {}, "non_descendant": {},
        "iia_A": correct(strata["A_answer_changing"], lambda p, r: r["intervened_prediction"] == p["expected_answer"]),
        "same_state": correct(strata["C_same_state"], lambda p, r: r["intervened_prediction"] == p["base_answer"]),
        "iia_D": correct(strata["D_same_observable_answer"], lambda p, r: r["intervened_prediction"] == p["expected_answer"]),
        "control_iia": {},
    }
    for name, rows in strata.items():
        metrics["transport"][name] = correct(rows, lambda p, r: r["decoded_z2"] == p["expected_z2"])
        metrics["descendant"][name] = correct(rows, lambda p, r: r["intervened_z3"] == p["expected_z3"])
        metrics["non_descendant"][name] = correct(rows, lambda p, r: r["intervened_n3"] == p["expected_n3"])
    for control_id in sorted({plan["control_id"] for plan in controls}):
        rows = [plan for plan in controls if plan["control_id"] == control_id]
        metrics["control_iia"][control_id] = correct(rows, lambda p, r: r["intervened_prediction"] == p["expected_answer"])
    groups = defaultdict(list)
    for plan in plans:
        groups[(plan["method"], plan["control_id"], plan["pair_stratum"])].append(plan)
    metrics["base_groups"] = {str(key): correct(rows, lambda p, r: r["base_prediction"] == p["base_answer"]) for key, rows in groups.items()}
    metrics["source_groups"] = {str(key): correct(rows, lambda p, r: r["source_prediction"] == p["source_answer"]) for key, rows in groups.items()}
    metrics["group_denominators"] = {str(key): len(rows) for key, rows in groups.items()}
    metrics["shortcut_target"] = max(value["correct"] for value in shortcut["clean_target_axis"].values())
    metrics["shortcut_parallel"] = max(value["correct"] for value in shortcut["clean_parallel_axis"].values())
    return metrics


def evaluate(package: dict, root) -> dict:
    faults = []
    if type(package) is not dict or package.get("namespace") != DEV_NAMESPACE or package.get("claim_bearing") is not False:
        faults.append("package:schema_or_namespace")
    if package.get("protocol_sha256") != protocol_hash(): faults.append("package:protocol_hash")
    if package.get("source_manifest_expected") != package.get("source_manifest_observed"): faults.append("package:frozen_source_drift")
    if package.get("payload_index_expected") != package.get("payload_index_observed"): faults.append("package:stale_result_or_index")
    if package.get("requested_status") is not None: faults.append("package:unknown_or_user_supplied_status")
    access = package.get("access_report")
    if type(access) is not dict or access.get("namespace") != DEV_NAMESPACE or access.get("invalid") is not False or access.get("access_consumed") is not True:
        faults.append("access:invalid_missing_second_or_partial")
    try: faults.extend(_validate_allocation(root, package.get("allocation")))
    except Exception as error: faults.append("allocation:exception:" + type(error).__name__)
    try: faults.extend(_validate_registry(package))
    except Exception as error: faults.append("registry:exception:" + type(error).__name__)
    try: faults.extend(_validate_raw(package))
    except Exception as error: faults.append("raw:exception:" + type(error).__name__)
    readouts = package.get("readout_manifest")
    seeds = package.get("expected_model_seed_identities")
    if type(readouts) is not list or type(seeds) is not list or [row.get("model_seed_identity") for row in readouts if type(row) is dict] != seeds:
        faults.append("readout:identity_order_or_missing")
    elif any(not _exact_bool(row.get("eligible")) or type(row.get("readout_sha256")) is not str for row in readouts):
        faults.append("readout:type_or_hash")
    if faults:
        return {"schema": "exp33ae.dev_only_simulated_decision.v1", "namespace": DEV_NAMESPACE, "claim_bearing": False, "simulated_status": FutureStatus.INVALID.value, "faults": sorted(set(faults)), "terminal_status_materialized": False}

    plans, raw = package["registry"]["rows"], package["raw_rows"]
    decisions, metrics_by_seed = [], {}
    for seed in seeds:
        seed_plans = [row for row in plans if row["model_seed_identity"] == seed]
        seed_raw = [row for row in raw if row["model_seed_identity"] == seed]
        metrics = _metrics_for_seed(seed_plans, seed_raw, package["shortcut_baselines"])
        metrics_by_seed[str(seed)] = metrics
        competence = (
            metrics["task_target"] >= 680 and metrics["task_parallel"] >= 680 and metrics["worst_programme"] >= 12
            and metrics["task_target"] - metrics["shortcut_target"] >= 160
            and metrics["task_parallel"] - metrics["shortcut_parallel"] >= 160
            and all(20 * correct >= 17 * metrics["group_denominators"][key] for key, correct in metrics["base_groups"].items())
            and all(20 * correct >= 17 * metrics["group_denominators"][key] for key, correct in metrics["source_groups"].items())
        )
        eligible = next(row["eligible"] for row in readouts if row["model_seed_identity"] == seed) and metrics["decode"] >= 1440
        causal = metrics["iia_A"] >= 136
        other = (
            metrics["transport"]["A_answer_changing"] >= 136
            and metrics["transport"]["B_outcome_collision"] >= 68
            and metrics["transport"]["D_same_observable_answer"] >= 68
            and sum(metrics["transport"][name] for name in ("A_answer_changing", "B_outcome_collision", "D_same_observable_answer")) >= 288
            and metrics["descendant"]["A_answer_changing"] >= 136
            and metrics["descendant"]["B_outcome_collision"] >= 68
            and metrics["descendant"]["D_same_observable_answer"] >= 68
            and metrics["non_descendant"]["A_answer_changing"] >= 152
            and all(metrics["non_descendant"][name] >= 76 for name in ("B_outcome_collision", "C_same_state", "D_same_observable_answer"))
            and metrics["same_state"] >= 76 and metrics["iia_D"] >= 64
            and all(metrics["iia_A"] - value >= 48 for value in metrics["control_iia"].values())
        )
        if not competence: status = FutureStatus.COMPETENCE_FAILED
        elif not eligible: status = FutureStatus.NO_ELIGIBLE
        elif not causal: status = FutureStatus.DECODE_ONLY
        elif not other: status = FutureStatus.NEGATIVE
        else: status = FutureStatus.POSITIVE
        decisions.append(status.value)
    if FutureStatus.COMPETENCE_FAILED.value in decisions: aggregate = FutureStatus.COMPETENCE_FAILED
    elif FutureStatus.NO_ELIGIBLE.value in decisions: aggregate = FutureStatus.NO_ELIGIBLE
    elif FutureStatus.DECODE_ONLY.value in decisions: aggregate = FutureStatus.DECODE_ONLY
    elif FutureStatus.NEGATIVE.value in decisions: aggregate = FutureStatus.NEGATIVE
    elif len(decisions) == 3 and all(value == FutureStatus.POSITIVE.value for value in decisions): aggregate = FutureStatus.POSITIVE
    else: aggregate = FutureStatus.INVALID
    return {
        "schema": "exp33ae.dev_only_simulated_decision.v1", "namespace": DEV_NAMESPACE, "claim_bearing": False,
        "simulated_status": aggregate.value, "simulated_seed_statuses": decisions, "metrics_by_seed": metrics_by_seed,
        "faults": [], "terminal_status_materialized": False,
    }
