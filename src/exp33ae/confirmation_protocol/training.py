"""Exact five-model final-step-only R3 IIT training implementation."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from ..model import ModelConfig, TinyProgrammeTransformer, declared_iit_basis, tensor_state_hash
from ..spec import K,N1_POSITION,N2_POSITION,Z1_POSITION,Z2_POSITION
from ..remediation.strict import object_sha256, write_json
from ..remediation.training import examples_to_tensors
from .normative import load_protocol, protocol_hash
from .runtime import configure,save_checkpoint


def train_one(rows,entropy,seed_order:int,output:Path,bindings:dict)->dict:
    p=load_protocol(); recipe=p["model_and_training"]; seed=entropy.model_seeds()[seed_order]
    batch_seed=entropy.stream_seed("training-batch-stream",seed_order); source_seed=entropy.stream_seed("training-interchange-stream",seed_order)
    if batch_seed==source_seed: raise RuntimeError("training stream collision")
    configure(seed); config=ModelConfig(d_model=96,d_mlp=192,n_heads=4,n_layers=4); model=TinyProgrammeTransformer(config); initial=tensor_state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"],betas=tuple(recipe["betas"]),eps=recipe["eps"])
    bg=torch.Generator().manual_seed(batch_seed); sg=torch.Generator().manual_seed(source_seed); data=examples_to_tensors(rows); basis=declared_iit_basis(config); curve=[]; model.train()
    for step in range(1,recipe["steps"]+1):
        bi=torch.randint(0,len(rows),(recipe["batch_size"],),generator=bg); si=torch.randint(0,len(rows),(recipe["batch_size"],),generator=sg)
        tokens,tables,answer=data["tokens"][bi],data["tables"][bi],data["answer"][bi]
        logits,acts,_=model(tokens,tables,return_activations=True)
        with torch.no_grad(): _,sacts,_=model(data["tokens"][si],data["tables"][si],return_activations=True)
        source_z2=data["z2"][si]; expected_z3=data["t3"][bi].gather(1,(2*source_z2+data["d"][bi]).unsqueeze(1)).squeeze(1); expected_answer=2*expected_z3+data["n3"][bi]
        cf=model(tokens,tables,intervention={"layer":2,"position":14,"source":sacts[2][:,14,:],"basis":basis,"strength":1.0})
        answer_loss=nn.functional.cross_entropy(logits,answer); iit_loss=nn.functional.cross_entropy(cf,expected_answer)
        target=(nn.functional.cross_entropy(acts[0][:,Z1_POSITION,:K],data["z1"][bi])+nn.functional.cross_entropy(acts[1][:,Z2_POSITION,:K],data["z2"][bi])+nn.functional.cross_entropy(acts[2][:,Z2_POSITION,:K],data["z2"][bi]))/3
        parallel=(nn.functional.cross_entropy(acts[0][:,N1_POSITION,:K],data["n1"][bi])+nn.functional.cross_entropy(acts[1][:,N2_POSITION,:K],data["n2"][bi])+nn.functional.cross_entropy(acts[2][:,N2_POSITION,K:2*K],data["n3"][bi]))/3
        w=recipe["loss_weights"]; loss=w["answer"]*answer_loss+w["iit_interchange"]*iit_loss+w["direct_intermediate_code"]*target+w["parallel_code"]*parallel
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),recipe["gradient_clip_norm"]); optimizer.step()
        if step==1 or step%100==0 or step==recipe["steps"]: curve.append({"step":step,"loss":float(loss.detach()),"answer_loss":float(answer_loss.detach()),"iit_loss":float(iit_loss.detach()),"target_code_loss":float(target.detach()),"parallel_code_loss":float(parallel.detach())})
    model.eval(); relative=Path("checkpoints")/f"model-{seed_order}.pt"; checkpoint=save_checkpoint(output/relative,model,seed_order,seed,bindings,entropy.namespace,entropy.stage,entropy.claim_bearing)
    manifest={"schema":"exp33ae.training_manifest.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
        "protocol_sha256":protocol_hash(),"recipe_id":recipe["recipe_id"],"seed_order":seed_order,"model_seed_identity":seed,"batch_stream_seed":batch_seed,"interchange_stream_seed":source_seed,
        "model_config":config.as_dict(),"parameter_count":sum(x.numel() for x in model.parameters()),"initial_tensor_state_sha256":initial,"final_tensor_state_sha256":checkpoint["tensor_state_sha256"],
        "checkpoint_relative_path":str(relative),"checkpoint_file_sha256":checkpoint["checkpoint_file_sha256"],"checkpoint_bytes":checkpoint["checkpoint_bytes"],
        "steps":recipe["steps"],"batch_size":recipe["batch_size"],"interchange_pairs":recipe["steps"]*recipe["batch_size"],"final_step_only":True,"curve":curve,**bindings}
    manifest["scientific_manifest_sha256"]=object_sha256(manifest); write_json(output/"training"/f"model-{seed_order}.json",manifest); return manifest


def train_population(allocation:dict,entropy,output:Path,bindings:dict)->list[dict]:
    training=[row for row in allocation["examples"] if row.partition=="training"]
    values=[train_one(training,entropy,index,output,bindings) for index in range(5)]
    if len(values)!=5: raise RuntimeError("exact five models required")
    return values
