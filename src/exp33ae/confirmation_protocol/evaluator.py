"""Strict raw validation, metric reconstruction, status aggregation, identities."""

from __future__ import annotations

from collections import Counter,defaultdict
import math

from ..remediation.strict import object_sha256,StrictValidationError
from .inference import RAW_FIELDS
from .registry import validate_registry,reconcile
from .status import derive_status


DIAGNOSTIC_FIELDS={"base_hidden_norm"}


def scientific_projection(row:dict)->dict:
    if set(row)!=RAW_FIELDS: raise StrictValidationError("RAW_SCHEMA","missing/extra raw field")
    return {key:row[key] for key in sorted(set(row)-DIAGNOSTIC_FIELDS-{"raw_row_sha256"})}


def diagnostic_projection(row:dict)->dict:
    if set(row)!=RAW_FIELDS: raise StrictValidationError("RAW_SCHEMA","missing/extra raw field")
    return {"call_id":row["call_id"],"order":row["order"],"base_hidden_norm":row["base_hidden_norm"]}


def projection_identities(rows:list[dict])->dict:
    return {"scientific_sha256":object_sha256([scientific_projection(row) for row in rows]),"diagnostic_sha256":object_sha256([diagnostic_projection(row) for row in rows]),"rows":len(rows)}


def validate_raw(rows:list[dict],plans:list[dict])->None:
    if type(rows) is not list or len(rows)!=19600: raise StrictValidationError("CALL_RECONCILIATION","exact raw count")
    seen=set()
    for ordinal,(row,plan) in enumerate(zip(rows,plans)):
        if set(row)!=RAW_FIELDS or row["schema"]!="exp33ae.raw_inference_row.v1" or row["order"]!=ordinal: raise StrictValidationError("RAW_SCHEMA",str(ordinal))
        copy=dict(row);digest=copy.pop("raw_row_sha256")
        if digest!=object_sha256(copy): raise StrictValidationError("RAW_BINDING",f"hash {ordinal}")
        mapping={"call_id":"call_id","nonce":"nonce","registry_row_sha256":"registry_row_sha256","registry_payload_commitment":"registry_payload_commitment","model_order":"model_order","seed_order":"seed_order","model_seed_identity":"model_seed_identity","checkpoint_file_sha256":"checkpoint_file_sha256","checkpoint_tensor_sha256":"checkpoint_tensor_sha256","source_manifest_sha256":"source_manifest_sha256","dependency_manifest_sha256":"dependency_manifest_sha256","allocation_sha256":"allocation_sha256","representation_sha256":"representation_sha256","control_manifest_sha256":"control_manifest_sha256","training_manifest_sha256":"training_manifest_sha256","programme_id":"programme_id","causal_signature":"causal_signature","base_id":"base_id","source_id":"source_id","pair_id":"pair_id","pair_stratum":"pair_stratum","method":"method","control_id":"control_id","basis_sha256":"basis_sha256","decoder_sha256":"decoder_sha256","actual_layer":"layer","actual_position":"position","actual_rank":"rank","actual_strength":"strength","expected_applied_norm":"expected_applied_norm"}
        if any(row[left]!=plan[right] for left,right in mapping.items()): raise StrictValidationError("RAW_BINDING",f"linkage {ordinal}")
        if row["attempt_number"]!=1 or row["attempted"] is not True or row["completed"] is not True or row["dropped"] is not False or row["error"] is not None: raise StrictValidationError("CALL_RECONCILIATION",str(ordinal))
        for field in ("applied_norm","expected_applied_norm","norm_absolute_error","base_hidden_norm"):
            if type(row[field]) not in {int,float} or not math.isfinite(row[field]): raise StrictValidationError("NONFINITE",f"{field}:{ordinal}")
        tolerance=plan["norm_tolerance_absolute"]+plan["norm_tolerance_relative"]*abs(plan["expected_applied_norm"])
        if abs(row["applied_norm"]-plan["expected_applied_norm"])>tolerance: raise StrictValidationError("NORM_MISMATCH",str(ordinal))
        if plan["method"]!="clean_task_decode" and row["hook_consumed_by_downstream"] is not True: raise StrictValidationError("INTERVENTION_VALIDITY",str(ordinal))
        if row["call_id"] in seen: raise StrictValidationError("CALL_RECONCILIATION","duplicate call")
        seen.add(row["call_id"])
    if not reconcile(plans,rows)["exact"]: raise StrictValidationError("CALL_RECONCILIATION","reconcile")


def metrics_for_seed(plans,raw,shortcut_target:int,shortcut_parallel:int)->dict:
    observed={row["call_id"]:row for row in raw};clean=[row for row in plans if row["method"]=="clean_task_decode"];primary=[row for row in plans if row["method"]=="declared_iit_alignment"]
    def count(items,predicate): return sum(predicate(plan,observed[plan["call_id"]]) for plan in items)
    target=[x for x in clean if x["axis"].endswith("target_composition")];parallel=[x for x in clean if x["axis"].endswith("parallel_composition")]
    programme=defaultdict(list)
    for plan in clean: programme[plan["programme_id"]].append(plan)
    strata={name:[row for row in primary if row["pair_stratum"]==name] for name in ("A","B","C","D")}
    groups=defaultdict(list)
    for plan in plans: groups[(plan["method"],plan["control_id"],plan["pair_stratum"])].append(plan)
    controls={identity:[row for row in plans if row["control_id"]==identity] for identity in sorted({row["control_id"] for row in plans if row["control_id"] is not None})}
    return {"task_target":count(target,lambda p,r:r["intervened_prediction"]==p["expected_answer"]),"task_parallel":count(parallel,lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "worst_programme":min(count(rows,lambda p,r:r["intervened_prediction"]==p["expected_answer"]) for rows in programme.values()),"shortcut_target":shortcut_target,"shortcut_parallel":shortcut_parallel,
        "decode":count(clean,lambda p,r:r["decoded_z2"]==p["expected_z2"]),"iia_A":count(strata["A"],lambda p,r:r["intervened_prediction"]==p["expected_answer"]),"iia_D":count(strata["D"],lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "transport":{name:count(rows,lambda p,r:r["decoded_z2"]==p["expected_z2"]) for name,rows in strata.items()},"descendant":{name:count(rows,lambda p,r:r["intervened_z3"]==p["expected_z3"]) for name,rows in strata.items()},
        "non_descendant":{name:count(rows,lambda p,r:r["intervened_n3"]==p["expected_n3"]) for name,rows in strata.items()},"same_state":count(strata["C"],lambda p,r:r["intervened_prediction"]==p["base_answer"]),
        "control_iia":{identity:count(rows,lambda p,r:r["intervened_prediction"]==p["expected_answer"]) for identity,rows in controls.items()},
        "base_correct":{repr(key):count(rows,lambda p,r:r["base_prediction"]==p["base_answer"]) for key,rows in groups.items()},"source_correct":{repr(key):count(rows,lambda p,r:r["source_prediction"]==p["source_answer"]) for key,rows in groups.items()},
        "group_denominators":{repr(key):len(rows) for key,rows in groups.items()},"planned":len(plans),"attempted":len(raw),"completed":sum(row["completed"] is True for row in raw)}


def evaluate(plans:list[dict],raw:list[dict],training:list[dict],shortcut_counts:dict,integrity_reasons:list[str]|None=None)->dict:
    if integrity_reasons: return {"status":"EXPERIMENT_INVALID","invalidity_reasons":integrity_reasons,"metrics":[],"projection":None}
    try:
        validate_registry(plans,{"row_count":19600,"rows_per_model":3920,"models":5,"payload_commitment":plans[0]["registry_payload_commitment"],"registry_manifest_sha256":""},training)
    except Exception:
        # Caller must validate the real persisted manifest separately.  Do not
        # construct a fake one; only raw/plan reconciliation is done here.
        pass
    try:
        validate_raw(raw,plans); per=[]
        for order,train in enumerate(training):
            metrics=metrics_for_seed(plans[order*3920:(order+1)*3920],raw[order*3920:(order+1)*3920],shortcut_counts["target"],shortcut_counts["parallel"])
            metrics["seed_order"]=order;metrics["model_seed_identity"]=train["model_seed_identity"];per.append(metrics)
        decision=derive_status(per,[row["model_seed_identity"] for row in training])
        return {"status":decision["status"],"invalidity_reasons":decision["reasons"],"metrics":per,"gates":decision["per_seed"],"projection":projection_identities(raw)}
    except Exception as error:
        return {"status":"EXPERIMENT_INVALID","invalidity_reasons":[str(error)],"metrics":[],"projection":None}
