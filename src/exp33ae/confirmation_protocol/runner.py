"""One generic future-shaped runner; claim launch is unavailable without external authority."""

from __future__ import annotations

from pathlib import Path

from ..remediation.strict import object_sha256,write_json,strict_load_json,strict_load_jsonl
from .access import AccessController
from .allocation import allocate,build_pairs
from .baselines import frozen_shortcuts
from .finalizer import release
from .guard import preflight_claim_launch,preflight_claim_resume
from .inference import execute_segment
from .normative import offline_dependency_preflight,source_manifest
from .package import write_allocation,write_registry,freeze_scientific_index,load_allocation
from .registry import build_registry
from .representation import representation_manifest
from .runtime import dependency_manifest,runtime_observation
from .training import train_population


def run_confirmation(root:Path,output:Path,seal:dict|None,authorization:dict|None,master_hex:str|None,resume:bool=False)->dict:
    """Run only after a later independently accepted seal and explicit authority.

    No caller in this protocol-freeze goal supplies those objects, so this path
    is tested solely for fail-closed rejection and creates nothing now.
    """
    if resume:
        plan=preflight_claim_resume(root,output,seal,authorization,master_hex);context=plan.context
        allocation,pairs,_=load_allocation(output);training=[strict_load_json(output/"training"/f"model-{index}.json") for index in range(5)]
        controls=[strict_load_json(output/"controls"/f"model-{index}.json") for index in range(5)]
        registry_manifest=strict_load_json(output/"registry/manifest.json");rows=strict_load_jsonl(output/"registry/registry.jsonl");controller=AccessController(output/"access",context.namespace,context.stage,True,context.derive("access-token",0,32),registry_manifest["payload_commitment"],[row["call_id"] for row in rows])
        with controller.acquire():
            start=controller.begin(resume=True);examples={row.example_id:row for row in allocation["examples"]}
            for offset in range(start,19600,256):controller.append(execute_segment(output,rows[offset:offset+256],examples,training,controls))
            controller.complete();return release(root,output,controller,context)
    plan=preflight_claim_launch(root,output,seal,authorization,master_hex)
    output.mkdir(parents=True,exist_ok=False)
    context=plan.context;entropy_record={"schema":"exp33ae.entropy_record.v1","namespace":context.namespace,"stage":context.stage,"claim_bearing":True,"commitment":context.commitment,"model_seed_identities":list(context.model_seeds()),"master_persisted":False};entropy_record["record_sha256"]=object_sha256(entropy_record);write_json(output/"entropy-commitment.json",entropy_record)
    source=source_manifest(root,seal["operative_source_commit"]);write_json(output/"source-manifest.json",source);preflight=offline_dependency_preflight();dependency=dependency_manifest(preflight,context.namespace,context.stage,True);write_json(output/"dependency-manifest.json",dependency);write_json(output/"runtime-observations.json",runtime_observation(preflight))
    allocation=allocate(root,context);pairs=build_pairs(allocation,context);allocation_manifest=write_allocation(output,allocation,pairs);shortcuts=frozen_shortcuts(allocation,context);write_json(output/"shortcut-baselines.json",shortcuts)
    representation=representation_manifest(context.namespace,context.stage,True);write_json(output/"representation.json",representation)
    bindings={"source_manifest_sha256":source["manifest_sha256"],"dependency_manifest_sha256":dependency["manifest_sha256"],"allocation_sha256":allocation_manifest["allocation_sha256"]};training=train_population(allocation,context,output,bindings)
    registry=build_registry(allocation,pairs,context,representation,training,output=output)
    for item in registry["controls"]:write_json(output/"controls"/f"model-{item['seed_order']}.json",item)
    write_registry(output,registry);freeze_scientific_index(output)
    ids=[row["call_id"] for row in registry["rows"]];controller=AccessController(output/"access",context.namespace,context.stage,True,context.derive("access-token",0,32),registry["manifest"]["payload_commitment"],ids)
    with controller.acquire(initialize=True):
        start=controller.begin();examples={row.example_id:row for row in allocation["examples"]}
        for offset in range(start,19600,256):controller.append(execute_segment(output,registry["rows"][offset:offset+256],examples,training,registry["controls"]))
        controller.complete();return release(root,output,controller,context)
