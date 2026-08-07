"""Actual frozen-checkpoint inference for clean, primary, and matched controls."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path
import math

import torch

from ..artifacts import tensor_sha256
from ..model import declared_iit_basis,tensor_state_hash
from ..remediation.strict import object_sha256,StrictValidationError
from .representation import decode_rank2
from .runtime import load_checkpoint


RAW_FIELDS={"schema","namespace","stage","claim_bearing","call_id","nonce","order","registry_row_sha256","registry_payload_commitment","model_order","seed_order","model_seed_identity","checkpoint_file_sha256","checkpoint_tensor_sha256","source_manifest_sha256","dependency_manifest_sha256","allocation_sha256","representation_sha256","control_manifest_sha256","training_manifest_sha256","programme_id","causal_signature","base_id","source_id","pair_id","pair_stratum","method","control_id","basis_sha256","decoder_sha256","actual_layer","actual_position","actual_rank","actual_strength","attempt_number","attempted","completed","dropped","error","base_prediction","source_prediction","intervened_prediction","decoded_z2","intervened_z3","intervened_n3","applied_norm","expected_applied_norm","norm_absolute_error","base_hidden_norm","activation_delta_sha256","hook_consumed_by_downstream","raw_row_sha256"}


def _raw(plan,values):
    row={"schema":"exp33ae.raw_inference_row.v1","namespace":plan["namespace"],"stage":plan["stage"],"claim_bearing":plan["claim_bearing"],"call_id":plan["call_id"],"nonce":plan["nonce"],"order":plan["order"],
        "registry_row_sha256":plan["registry_row_sha256"],"registry_payload_commitment":plan["registry_payload_commitment"],"model_order":plan["model_order"],"seed_order":plan["seed_order"],"model_seed_identity":plan["model_seed_identity"],
        "checkpoint_file_sha256":plan["checkpoint_file_sha256"],"checkpoint_tensor_sha256":plan["checkpoint_tensor_sha256"],"source_manifest_sha256":plan["source_manifest_sha256"],"dependency_manifest_sha256":plan["dependency_manifest_sha256"],
        "allocation_sha256":plan["allocation_sha256"],"representation_sha256":plan["representation_sha256"],"control_manifest_sha256":plan["control_manifest_sha256"],"training_manifest_sha256":plan["training_manifest_sha256"],
        "programme_id":plan["programme_id"],"causal_signature":plan["causal_signature"],"base_id":plan["base_id"],"source_id":plan["source_id"],"pair_id":plan["pair_id"],"pair_stratum":plan["pair_stratum"],
        "method":plan["method"],"control_id":plan["control_id"],"basis_sha256":plan["basis_sha256"],"decoder_sha256":plan["decoder_sha256"],"actual_layer":plan["layer"],"actual_position":plan["position"],"actual_rank":plan["rank"],"actual_strength":plan["strength"],
        "attempt_number":1,"attempted":True,"completed":True,"dropped":False,"error":None,**values}
    row["raw_row_sha256"]=object_sha256(row);return row


def _clean(model,plans,examples):
    result=[]
    for start in range(0,len(plans),256):
        batch=plans[start:start+256]; base=[examples[x["base_id"]] for x in batch]; tokens=torch.tensor([x.tokens for x in base]);tables=torch.tensor([x.operation_tables for x in base],dtype=torch.float32)
        with torch.no_grad(): logits,acts,_=model(tokens,tables,return_activations=True)
        predictions=logits.argmax(-1);decoded=decode_rank2(acts[2][:,14,:]);zero=tensor_sha256(torch.zeros(model.config.d_model,dtype=acts[2].dtype))
        for index,plan in enumerate(batch):
            prediction=int(predictions[index]);result.append(_raw(plan,{"base_prediction":prediction,"source_prediction":prediction,"intervened_prediction":prediction,"decoded_z2":int(decoded[index]),"intervened_z3":prediction//2,"intervened_n3":prediction%2,
                "applied_norm":0.0,"expected_applied_norm":0.0,"norm_absolute_error":0.0,"base_hidden_norm":float(acts[2][index,14,:].norm()),"activation_delta_sha256":zero,"hook_consumed_by_downstream":False}))
    return result


def _intervene(model,plans,examples,basis):
    result=[]
    for start in range(0,len(plans),256):
        batch=plans[start:start+256];base=[examples[x["base_id"]] for x in batch];source=[examples[x["source_id"]] for x in batch]
        bt=torch.tensor([x.tokens for x in base]);btab=torch.tensor([x.operation_tables for x in base],dtype=torch.float32);st=torch.tensor([x.tokens for x in source]);stab=torch.tensor([x.operation_tables for x in source],dtype=torch.float32)
        layer,position=batch[0]["layer"],batch[0]["position"]
        with torch.no_grad():
            base_logits,base_acts,_=model(bt,btab,return_activations=True);source_logits,source_acts,_=model(st,stab,return_activations=True);base_hidden=base_acts[layer][:,position,:];source_hidden=source_acts[layer][:,position,:]
            projected=((source_hidden-base_hidden)@basis)@basis.T
            if batch[0]["method"]!="declared_iit_alignment":
                target=torch.tensor([x["expected_applied_norm"] for x in batch],dtype=projected.dtype);observed=projected.norm(dim=-1)
                if bool(((observed<1e-12)&(target>1e-6)).any()): raise StrictValidationError("NORM_MISMATCH","unmatchable zero norm")
                projected=projected*(target/observed.clamp_min(1e-12)).unsqueeze(1);source_for_hook=base_hidden+projected
            else: source_for_hook=source_hidden
            after,after_acts,applied=model(bt,btab,intervention={"layer":layer,"position":position,"source":source_for_hook,"basis":basis,"strength":1.0},return_activations=True)
        bp=base_logits.argmax(-1);sp=source_logits.argmax(-1);pred=after.argmax(-1);decoded=decode_rank2(after_acts[2][:,14,:]);norms=applied.norm(dim=-1)
        for index,plan in enumerate(batch):
            value=int(pred[index]);actual=float(norms[index]);expected=float(plan["expected_applied_norm"])
            if not all(math.isfinite(x) for x in (actual,expected)): raise StrictValidationError("NONFINITE","norm")
            result.append(_raw(plan,{"base_prediction":int(bp[index]),"source_prediction":int(sp[index]),"intervened_prediction":value,"decoded_z2":int(decoded[index]),"intervened_z3":value//2,"intervened_n3":value%2,
                "applied_norm":actual,"expected_applied_norm":expected,"norm_absolute_error":abs(actual-expected),"base_hidden_norm":float(base_hidden[index].norm()),"activation_delta_sha256":tensor_sha256(applied[index]),"hook_consumed_by_downstream":True}))
    return result


def execute_segment(output:Path,plans:list[dict],examples:dict,training:list[dict],controls:list[dict])->list[dict]:
    result=[];training_by={x["seed_order"]:x for x in training};control_by={x["seed_order"]:{c["control_id"]:c for c in x["controls"]} for x in controls}
    for model_order,iterator in groupby(plans,key=lambda row:row["model_order"]):
        block=list(iterator);train=training_by[model_order];model,_=load_checkpoint(output/train["checkpoint_relative_path"],{"seed_order":train["seed_order"],"model_seed_identity":train["model_seed_identity"],"checkpoint_file_sha256":train["checkpoint_file_sha256"],"tensor_state_sha256":train["final_tensor_state_sha256"]});before=tensor_state_hash(model)
        for (_,control_id),sub in groupby(block,key=lambda row:(row["method"],row["control_id"])):
            group=list(sub)
            if group[0]["method"]=="clean_task_decode": result.extend(_clean(model,group,examples));continue
            basis=declared_iit_basis(model.config) if control_id is None else torch.tensor(control_by[model_order][control_id]["basis"],dtype=torch.float32)
            if tensor_sha256(basis)!=group[0]["basis_sha256"]: raise StrictValidationError("CONTROL_BINDING","basis bytes")
            result.extend(_intervene(model,group,examples,basis))
        if tensor_state_hash(model)!=before: raise StrictValidationError("CHECKPOINT_IMMUTABILITY","model changed")
    return result
