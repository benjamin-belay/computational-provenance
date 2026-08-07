"""Evaluator-derived release; caller-supplied metrics/status are prohibited."""

from __future__ import annotations

from pathlib import Path

from ..remediation.strict import object_sha256,sha256_file,strict_load_json,strict_load_jsonl,StrictValidationError
from .access import AccessController,AccessViolation,atomic_json
from .allocation import allocate,build_pairs
from .baselines import frozen_shortcuts
from .evaluator import evaluate,validate_raw
from .inference import execute_segment
from .normative import offline_dependency_preflight,verify_live_source
from .package import load_allocation,verify_scientific_index
from .registry import validate_registry,build_registry
from .representation import representation_manifest,control_manifest
from .runtime import dependency_manifest


def release(root:Path,output:Path,controller:AccessController,context,caller_supplied=None)->dict:
    if caller_supplied is not None:
        controller._invalidate("RELEASE_BINDING");raise AccessViolation("caller-supplied metrics/gates/status prohibited")
    snapshot=controller.validate()
    if snapshot["state"]["state"]!="BUFFER_COMPLETE":raise AccessViolation("release requires complete buffer")
    source=strict_load_json(output/"source-manifest.json");verify_live_source(root,source)
    observed=offline_dependency_preflight();stored_dependency=strict_load_json(output/"dependency-manifest.json");current=dependency_manifest(observed,stored_dependency["namespace"],stored_dependency["stage"],stored_dependency["claim_bearing"])
    if current!=stored_dependency:raise AccessViolation("live dependency drift")
    entropy_record=strict_load_json(output/"entropy-commitment.json");copy=dict(entropy_record);digest=copy.pop("record_sha256",None)
    if digest!=object_sha256(copy) or entropy_record.get("commitment")!=context.commitment or entropy_record.get("model_seed_identities")!=list(context.model_seeds()) or entropy_record.get("master_persisted") is not False:raise AccessViolation("entropy commitment binding")
    allocation,pairs,allocation_manifest=load_allocation(output);expected_allocation=allocate(root,context);expected_pairs=build_pairs(expected_allocation,context)
    if allocation["programmes"]!=expected_allocation["programmes"] or [row.as_dict() for row in allocation["examples"]]!=[row.as_dict() for row in expected_allocation["examples"]] or pairs!=expected_pairs:raise AccessViolation("independent allocation/pair reconstruction")
    representation=strict_load_json(output/"representation.json");copy=dict(representation);digest=copy.pop("representation_sha256")
    if digest!=object_sha256(copy):raise AccessViolation("representation binding")
    if representation!=representation_manifest(context.namespace,context.stage,context.claim_bearing):raise AccessViolation("representation reconstruction")
    training=[strict_load_json(output/"training"/f"model-{index}.json") for index in range(5)];controls=[strict_load_json(output/"controls"/f"model-{index}.json") for index in range(5)]
    registry_manifest=strict_load_json(output/"registry"/"manifest.json");plans=strict_load_jsonl(output/"registry"/"registry.jsonl")
    if registry_manifest["registry_file_sha256"]!=sha256_file(output/"registry"/"registry.jsonl"):raise AccessViolation("registry bytes")
    validate_registry(plans,registry_manifest,training)
    rebuilt_registry=build_registry(expected_allocation,expected_pairs,context,representation,training,output=output)
    persisted_core={key:value for key,value in registry_manifest.items() if key not in {"registry_file_sha256","registry_manifest_sha256"}}
    rebuilt_core={key:value for key,value in rebuilt_registry["manifest"].items() if key!="registry_manifest_sha256"}
    if plans!=rebuilt_registry["rows"] or persisted_core!=rebuilt_core or controls!=rebuilt_registry["controls"]:raise AccessViolation("independent registry/control reconstruction")
    raw=snapshot["buffer_rows"];validate_raw(raw,plans)
    examples={row.example_id:row for row in allocation["examples"]};rebuilt=[]
    for start in range(0,19600,256):rebuilt.extend(execute_segment(output,plans[start:start+256],examples,training,controls))
    if rebuilt!=raw:raise AccessViolation("independent actual inference reconstruction failed")
    index=verify_scientific_index(output)
    if not index["matches"]:raise AccessViolation("scientific payload index drift")
    shortcuts=strict_load_json(output/"shortcut-baselines.json");expected_shortcuts=frozen_shortcuts(expected_allocation,context)
    if shortcuts!=expected_shortcuts:raise AccessViolation("shortcut reconstruction")
    shortcut_counts={"target":max(value["correct"] for value in shortcuts["axes"]["clean_target_axis"].values()),"parallel":max(value["correct"] for value in shortcuts["axes"]["clean_parallel_axis"].values())}
    result=evaluate(plans,raw,training,shortcut_counts)
    atomic_json(output/"derived-status.json",result)
    envelope={"schema":"exp33ae.evaluator_derived_release.v1","namespace":controller.namespace,"stage":controller.stage,"claim_bearing":controller.claim_bearing,"status":result["status"],"derived_status_sha256":sha256_file(output/"derived-status.json"),"raw_buffer_sha256":sha256_file(controller.buffer_path),"registry_commitment":controller.registry_commitment,"scientific_index_payload_sha256":index["stored"]["payload_sha256"],"caller_supplied_result_used":False}
    envelope["release_payload_sha256"]=object_sha256(envelope);atomic_json(output/"release-envelope.json",envelope);controller.record_release(sha256_file(output/"release-envelope.json"),result["status"]);return {"status":result,"release":envelope}
