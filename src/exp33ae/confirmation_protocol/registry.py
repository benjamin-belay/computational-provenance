"""Exact 19,600-row pre-evaluation registry and fail-closed reconciliation."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

import torch

from ..model import declared_iit_basis
from ..remediation.strict import object_sha256, StrictValidationError
from .normative import protocol_hash
from .representation import control_manifest
from .runtime import load_checkpoint


METHOD_COUNTS = Counter({"clean_task_decode":1600,"declared_iit_alignment":400,"random_direction":640,"orthogonal_direction":640,"alternate_location":480,"alternate_early_layer":160})
REGISTRY_FIELDS=frozenset({"schema","namespace","stage","claim_bearing","protocol_sha256","order","model_order","seed_order","model_seed_identity","source_manifest_sha256","dependency_manifest_sha256","allocation_sha256","representation_sha256","control_manifest_sha256","training_manifest_sha256","checkpoint_relative_path","checkpoint_file_sha256","checkpoint_tensor_sha256","programme_id","causal_signature","axis","base_id","source_id","pair_id","pair_stratum","eligible","collision","same_observable_answer","no_op","base_z2","base_z3","base_n3","base_answer","source_z2","source_z3","source_n3","source_answer","expected_z2","expected_z3","expected_n3","expected_answer","method","control_id","basis_sha256","decoder_sha256","layer","position","rank","strength","expected_applied_norm","norm_tolerance_absolute","norm_tolerance_relative","call_id","nonce","planned_attempts","retry_policy","registry_row_sha256","registry_payload_commitment"})


def _primary_norms(model,pairs,examples)->dict[str,float]:
    basis=declared_iit_basis(model.config); values={}; model.eval()
    with torch.no_grad():
        for start in range(0,len(pairs),256):
            batch=pairs[start:start+256]; bases=[examples[x["base_id"]] for x in batch]; sources=[examples[x["source_id"]] for x in batch]
            bt=torch.tensor([x.tokens for x in bases]); btab=torch.tensor([x.operation_tables for x in bases],dtype=torch.float32)
            st=torch.tensor([x.tokens for x in sources]); stab=torch.tensor([x.operation_tables for x in sources],dtype=torch.float32)
            _,acts,_=model(bt,btab,return_activations=True); source=model.source_activation(st,stab,2,14)
            norms=(((source-acts[2][:,14,:])@basis)@basis.T).norm(dim=-1)
            values.update({plan["pair_id"]:float(norm) for plan,norm in zip(batch,norms.tolist())})
    return values


def _fixture_hash(label:str,seed:int)->str:
    return sha256(f"DEV_ONLY|{label}|{seed}".encode()).hexdigest()


def fixture_model_bindings(entropy)->list[dict]:
    return [{"seed_order":order,"model_seed_identity":seed,"checkpoint_relative_path":f"DEV_ONLY/checkpoints/model-{order}.pt",
             "checkpoint_file_sha256":_fixture_hash("checkpoint-file",seed),"final_tensor_state_sha256":_fixture_hash("tensor-state",seed),
             "scientific_manifest_sha256":_fixture_hash("training",seed),"source_manifest_sha256":_fixture_hash("source",0),
             "dependency_manifest_sha256":_fixture_hash("dependency",0),"allocation_sha256":_fixture_hash("allocation",0)}
            for order,seed in enumerate(entropy.model_seeds())]


def build_registry(allocation:dict,pairs:list[dict],entropy,representation:dict,training:list[dict],output:Path|None=None,fixture_norm:float|None=None)->dict:
    if len(training)!=5 or [x["seed_order"] for x in training]!=list(range(5)) or [x["model_seed_identity"] for x in training]!=list(entropy.model_seeds()):
        raise StrictValidationError("SEED_IDENTITY","exact ordered five-model bindings")
    examples={row.example_id:row for row in allocation["examples"]}; clean=[row for row in allocation["examples"] if row.partition in {"clean_target_axis","clean_parallel_axis"}]
    if len(clean)!=1600 or Counter(row["stratum"] for row in pairs)!=Counter({"A":160,"B":80,"C":80,"D":80}): raise StrictValidationError("REGISTRY_STRUCTURE","clean/pair counts")
    A=[row for row in pairs if row["stratum"]=="A"]; rows=[]; controls_by_model=[]; ordinal=0
    for train in training:
        control=control_manifest(entropy,train["seed_order"],train["model_seed_identity"],representation); controls_by_model.append(control)
        if fixture_norm is None:
            if output is None: raise StrictValidationError("REGISTRY_BINDING","checkpoint output required")
            model,_=load_checkpoint(output/train["checkpoint_relative_path"],{"seed_order":train["seed_order"],"model_seed_identity":train["model_seed_identity"],"checkpoint_file_sha256":train["checkpoint_file_sha256"],"tensor_state_sha256":train["final_tensor_state_sha256"]})
            norms=_primary_norms(model,pairs,examples)
        else:
            if entropy.claim_bearing: raise StrictValidationError("STAGE_GUARD","fixture norms forbidden for claim registry")
            norms={pair["pair_id"]:float(fixture_norm) for pair in pairs}

        def add(base,source,pair,method,control_row):
            nonlocal ordinal
            clean_call=pair is None
            if clean_call:
                stratum="clean"; pair_id="clean-"+sha256(f"{train['model_seed_identity']}|{base.example_id}".encode()).hexdigest()[:24]
                expected={"z2":base.z2,"z3":base.z3,"n3":base.n3,"answer":base.answer}; layer=2;position=14;rank=2;strength=0.0;control_id=None;basis=representation["basis_sha256"];expected_norm=0.0;no_op=True;collision=False;same_answer=True
            else:
                stratum=pair["stratum"];pair_id=pair["pair_id"];expected={name:pair[f"expected_{name}"] for name in ("z2","z3","n3","answer")};no_op=pair["no_op"];collision=pair["collision"];same_answer=pair["same_observable_answer"];expected_norm=norms[pair_id]
                if control_row is None: layer=2;position=14;rank=2;strength=1.0;control_id=None;basis=representation["basis_sha256"]
                else: layer=control_row["layer"];position=control_row["position"];rank=control_row["rank"];strength=control_row["strength"];control_id=control_row["control_id"];basis=control_row["basis_sha256"]
            prefix="devonly-cpcall-" if not entropy.claim_bearing else "exp33ae-cnfcall-"; nprefix="devonly-cpnonce-" if not entropy.claim_bearing else "exp33ae-cnfnonce-"
            row={"schema":"exp33ae.registry_row.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,"protocol_sha256":protocol_hash(),
                "order":ordinal,"model_order":train["seed_order"],"seed_order":train["seed_order"],"model_seed_identity":train["model_seed_identity"],
                "source_manifest_sha256":train["source_manifest_sha256"],"dependency_manifest_sha256":train["dependency_manifest_sha256"],"allocation_sha256":train["allocation_sha256"],
                "representation_sha256":representation["representation_sha256"],"control_manifest_sha256":control["control_manifest_sha256"],"training_manifest_sha256":train["scientific_manifest_sha256"],
                "checkpoint_relative_path":train["checkpoint_relative_path"],"checkpoint_file_sha256":train["checkpoint_file_sha256"],"checkpoint_tensor_sha256":train["final_tensor_state_sha256"],
                "programme_id":base.programme_id,"causal_signature":base.causal_signature,"axis":base.axis,"base_id":base.example_id,"source_id":source.example_id,"pair_id":pair_id,
                "pair_stratum":stratum,"eligible":True,"collision":collision,"same_observable_answer":same_answer,"no_op":no_op,
                "base_z2":base.z2,"base_z3":base.z3,"base_n3":base.n3,"base_answer":base.answer,"source_z2":source.z2,"source_z3":source.z3,"source_n3":source.n3,"source_answer":source.answer,
                "expected_z2":expected["z2"],"expected_z3":expected["z3"],"expected_n3":expected["n3"],"expected_answer":expected["answer"],
                "method":method,"control_id":control_id,"basis_sha256":basis,"decoder_sha256":representation["representation_sha256"],"layer":layer,"position":position,"rank":rank,"strength":strength,
                "expected_applied_norm":expected_norm,"norm_tolerance_absolute":1e-6,"norm_tolerance_relative":1e-5,
                "call_id":prefix+entropy.derive("call-id",ordinal,16).hex(),"nonce":nprefix+entropy.derive("nonce",ordinal,16).hex(),"planned_attempts":1,"retry_policy":"same-consumed-session-first-missing-no-reexecution"}
            row["registry_row_sha256"]=object_sha256(row); rows.append(row); ordinal+=1
        for base in clean: add(base,base,None,"clean_task_decode",None)
        for pair in pairs: add(examples[pair["base_id"]],examples[pair["source_id"]],pair,"declared_iit_alignment",None)
        for control_row in control["controls"]:
            for pair in A: add(examples[pair["base_id"]],examples[pair["source_id"]],pair,control_row["family"],control_row)
    if len(rows)!=19600: raise StrictValidationError("REGISTRY_STRUCTURE",f"row count {len(rows)}")
    payload=object_sha256(rows)
    for row in rows: row["registry_payload_commitment"]=payload
    manifest={"schema":"exp33ae.registry_manifest.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
        "protocol_sha256":protocol_hash(),"model_seed_identities":list(entropy.model_seeds()),"row_count":19600,"rows_per_model":3920,"models":5,
        "payload_commitment":payload,"generated_before_evaluation":True,"method_counts":dict(sorted(Counter(row["method"] for row in rows).items()))}
    manifest["registry_manifest_sha256"]=object_sha256(manifest); validate_registry(rows,manifest,training); return {"rows":rows,"manifest":manifest,"controls":controls_by_model}


def validate_registry(rows:list[dict],manifest:dict,training:list[dict])->None:
    if type(rows) is not list or len(rows)!=19600 or manifest.get("row_count")!=19600 or manifest.get("rows_per_model")!=3920 or manifest.get("models")!=5: raise StrictValidationError("REGISTRY_STRUCTURE","exact totals")
    copy=dict(manifest); observed=copy.pop("registry_manifest_sha256",None)
    if observed!=object_sha256(copy): raise StrictValidationError("REGISTRY_BINDING","manifest hash")
    payload=[{key:value for key,value in row.items() if key!="registry_payload_commitment"} for row in rows]
    if object_sha256(payload)!=manifest["payload_commitment"]: raise StrictValidationError("REGISTRY_BINDING","payload")
    calls=[]; nonces=[]
    for ordinal,row in enumerate(rows):
        if set(row)!=REGISTRY_FIELDS or row.get("schema")!="exp33ae.registry_row.v1" or type(row.get("claim_bearing")) is not bool or row.get("protocol_sha256")!=protocol_hash() or row["order"]!=ordinal or row["model_order"]!=ordinal//3920 or row["seed_order"]!=ordinal//3920: raise StrictValidationError("REGISTRY_STRUCTURE",str(ordinal))
        train=training[ordinal//3920]
        for field,source in (("model_seed_identity","model_seed_identity"),("checkpoint_file_sha256","checkpoint_file_sha256"),("checkpoint_tensor_sha256","final_tensor_state_sha256"),("training_manifest_sha256","scientific_manifest_sha256")):
            if row[field]!=train[source]: raise StrictValidationError("REGISTRY_BINDING",f"{field}:{ordinal}")
        copy=dict(row); digest=copy.pop("registry_row_sha256");copy.pop("registry_payload_commitment")
        if digest!=object_sha256(copy) or row["registry_payload_commitment"]!=manifest["payload_commitment"]: raise StrictValidationError("REGISTRY_BINDING",str(ordinal))
        if row["planned_attempts"]!=1 or row["eligible"] is not True or row["retry_policy"]!="same-consumed-session-first-missing-no-reexecution" or row["rank"]!=2 or row["strength"] not in {0.0,1.0}: raise StrictValidationError("REGISTRY_STRUCTURE","policy")
        calls.append(row["call_id"]);nonces.append(row["nonce"])
    if len(set(calls))!=19600 or len(set(nonces))!=19600: raise StrictValidationError("REGISTRY_STRUCTURE","duplicate call/nonce")
    for model in range(5):
        block=rows[model*3920:(model+1)*3920]
        if Counter(row["method"] for row in block)!=METHOD_COUNTS: raise StrictValidationError("REGISTRY_STRUCTURE",f"methods {model}")
        if Counter(row["pair_stratum"] for row in block if row["method"]=="declared_iit_alignment")!=Counter({"A":160,"B":80,"C":80,"D":80}): raise StrictValidationError("REGISTRY_STRUCTURE",f"strata {model}")


def reconcile(plans:list[dict],raw:list[dict])->dict:
    planned=[row.get("call_id") for row in plans];observed=[row.get("call_id") for row in raw];missing=set(planned)-set(observed);unplanned=set(observed)-set(planned)
    result={"planned":len(plans),"attempted":len(raw),"completed":sum(row.get("completed") is True for row in raw),"missing":len(missing),"unplanned":len(unplanned),
        "duplicates":len(observed)-len(set(observed)),"failed":sum(row.get("completed") is False for row in raw),"retried":sum(row.get("attempt_number")!=1 for row in raw),"dropped":sum(row.get("dropped") is True for row in raw),"reordered":observed!=planned}
    result["exact"]=result=={"planned":19600,"attempted":19600,"completed":19600,"missing":0,"unplanned":0,"duplicates":0,"failed":0,"retried":0,"dropped":0,"reordered":False};return result
