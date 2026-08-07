"""Exact mathematical rank-2 representation and matched controls."""

from __future__ import annotations

import torch

from ..artifacts import tensor_sha256
from ..model import ModelConfig, declared_iit_basis
from ..remediation.strict import object_sha256, StrictValidationError
from .normative import protocol_hash


def representation_manifest(namespace: str, stage: str, claim_bearing: bool) -> dict:
    basis=declared_iit_basis(ModelConfig(d_model=96,d_mlp=192,n_heads=4,n_layers=4)).float()
    value={"schema":"exp33ae.rank2_representation.v1","namespace":namespace,"stage":stage,"claim_bearing":claim_bearing,
        "protocol_sha256":protocol_hash(),"target":"z2","layer":2,"position":14,"rank":2,"strength":1.0,
        "basis":basis.tolist(),"basis_sha256":tensor_sha256(basis),"basis_convention":"B=[e0,e1] in R^96",
        "projection":"B^T h","replacement":"h + B(B^T h_source - B^T h_base)",
        "decoder":{"kind":"fixed_registered_coordinate_argmax","coordinates":[0,1],"classes":2},
        "state_codes":{"0":[1.0,0.0],"1":[0.0,1.0]},"primary_method":"declared_iit_alignment",
        "secondary_methods":[],"das_and_probes_adjudicate":False}
    value["representation_sha256"]=object_sha256(value); return value


def _random_basis(seed: int) -> torch.Tensor:
    generator=torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(96,2,generator=generator,dtype=torch.float64),mode="reduced").Q.float()


def control_manifest(entropy, seed_order: int, seed_identity: int, representation: dict) -> dict:
    primary=torch.tensor(representation["basis"],dtype=torch.float32); controls=[]
    for index in range(4):
        seed=int.from_bytes(entropy.derive("random-direction",seed_order*4+index,8),"big")&((1<<63)-1)
        basis=_random_basis(seed)
        if tensor_sha256(basis)==representation["basis_sha256"]: raise StrictValidationError("CONTROL_BINDING","random equals primary")
        controls.append({"control_id":f"random_direction_{index}","family":"random_direction","index":index,"layer":2,"position":14,"rank":2,"strength":1.0,"basis":basis.tolist(),"basis_sha256":tensor_sha256(basis)})
    primary64=primary.double()
    for index in range(4):
        seed=int.from_bytes(entropy.derive("orthogonal-direction",seed_order*4+index,8),"big")&((1<<63)-1)
        raw=_random_basis(seed).double(); residual=raw-primary64@(primary64.T@raw); basis=torch.linalg.qr(residual,mode="reduced").Q.float()
        if float((primary.T@basis).abs().max())>1e-5: raise StrictValidationError("CONTROL_BINDING","not orthogonal")
        controls.append({"control_id":f"orthogonal_direction_{index}","family":"orthogonal_direction","index":index,"layer":2,"position":14,"rank":2,"strength":1.0,"basis":basis.tolist(),"basis_sha256":tensor_sha256(basis)})
    for index,position in enumerate((12,13,15)):
        controls.append({"control_id":f"alternate_location_{index}","family":"alternate_location","index":index,"layer":2,"position":position,"rank":2,"strength":1.0,"basis":primary.tolist(),"basis_sha256":tensor_sha256(primary)})
    controls.append({"control_id":"alternate_early_layer_0","family":"alternate_early_layer","index":0,"layer":0,"position":14,"rank":2,"strength":1.0,"basis":primary.tolist(),"basis_sha256":tensor_sha256(primary)})
    value={"schema":"exp33ae.control_manifest.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
        "protocol_sha256":protocol_hash(),"seed_order":seed_order,"model_seed_identity":seed_identity,"controls":controls,
        "norm_tolerance_absolute":1e-6,"norm_tolerance_relative":1e-5}
    value["control_manifest_sha256"]=object_sha256(value); return value


def decode_rank2(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.shape[-1]<2: raise StrictValidationError("READOUT_BINDING","residual too narrow")
    return hidden[...,0:2].argmax(dim=-1)
