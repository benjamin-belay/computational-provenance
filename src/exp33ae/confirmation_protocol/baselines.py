"""Frozen shortcut baselines used only for competence advantage gates."""

from __future__ import annotations

from collections import Counter,defaultdict
import random
import torch
from torch import nn

from ..independent_oracle import oracle
from ..spec import ALL_OPERATIONS,JOINT_CLASSES,SEQ_LEN,VOCAB_SIZE
from ..remediation.strict import object_sha256


def _table_accuracy(train,evaluation,key):
    tables=defaultdict(Counter)
    for row in train: tables[key(row)][row.answer]+=1
    correct=0
    for row in evaluation:
        counts=tables.get(key(row),Counter({0:1}));prediction=sorted(counts.items(),key=lambda item:(-item[1],item[0]))[0][0];correct+=prediction==row.answer
    return correct


class ShallowMLP(nn.Module):
    def __init__(self): super().__init__();self.net=nn.Sequential(nn.Linear(SEQ_LEN*VOCAB_SIZE+24,64),nn.GELU(),nn.Linear(64,JOINT_CLASSES))
    def forward(self,tokens,tables): return self.net(torch.cat((nn.functional.one_hot(tokens,VOCAB_SIZE).float().flatten(1),tables.float().flatten(1)),dim=1))


def frozen_shortcuts(allocation:dict,entropy)->dict:
    train=[row for row in allocation["examples"] if row.partition=="training"];sets={axis:[row for row in allocation["examples"] if row.partition==axis] for axis in ("clean_target_axis","clean_parallel_axis")}
    keys={"exact_programme_input_lookup":lambda r:(r.programme_id,r.a,r.b,r.c,r.d),"programme_only_lookup":lambda r:r.programme_id,
        "target_composition_input_lookup":lambda r:(r.target_composition_id,r.a,r.b,r.c,r.d),"parallel_composition_input_lookup":lambda r:(r.parallel_composition_id,r.a,r.b,r.c,r.d),
        "input_only_lookup":lambda r:(r.a,r.b,r.c,r.d),"last_two_inputs_lookup":lambda r:(r.c,r.d),"operation_unigram_bag_lookup":lambda r:tuple(sorted(r.operation_ids))}
    result={axis:{name:{"correct":_table_accuracy(train,rows,key),"denominator":800} for name,key in keys.items()} for axis,rows in sets.items()}
    train_programmes={row.programme_id:row.operation_ids for row in train}
    for axis,rows in sets.items():
        correct=0
        for row in rows:
            nearest=max(train_programmes.items(),key=lambda item:(sum(left==right for left,right in zip(row.operation_ids,item[1])),item[0]))[1]
            correct+=oracle(tuple(ALL_OPERATIONS[index] for index in nearest),(row.a,row.b,row.c,row.d))["joint_answer"]==row.answer
        result[axis]["nearest_programme_transition"]={"correct":correct,"denominator":800};result[axis]["direct_output_marginal"]={"correct":200,"denominator":800}
    seed=int.from_bytes(entropy.derive("programme-allocation",999,8),"big")&((1<<63)-1);random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);model=ShallowMLP();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-3)
    tokens=torch.tensor([r.tokens for r in train]);tables=torch.tensor([r.operation_tables for r in train],dtype=torch.float32);labels=torch.tensor([r.answer for r in train]);generator=torch.Generator().manual_seed((seed+1)&((1<<63)-1));curve=[];model.train()
    for step in range(1,601):
        index=torch.randint(0,len(train),(512,),generator=generator);loss=nn.functional.cross_entropy(model(tokens[index],tables[index]),labels[index]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step in {1,100,200,400,600}:curve.append({"step":step,"loss":float(loss.detach())})
    model.eval()
    with torch.no_grad():
        for axis,rows in sets.items():
            x=torch.tensor([r.tokens for r in rows]);t=torch.tensor([r.operation_tables for r in rows],dtype=torch.float32);correct=int((model(x,t).argmax(-1)==torch.tensor([r.answer for r in rows])).sum());result[axis]["shallow_position_mlp"]={"correct":correct,"denominator":800}
    value={"schema":"exp33ae.frozen_shortcut_results.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,"seed":seed,"shallow_steps":600,"shallow_curve":curve,"axes":result}
    value["results_sha256"]=object_sha256(value);return value
