"""Frozen runtime configuration and exact checkpoint byte/tensor bindings."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import torch

from ..model import ModelConfig, TinyProgrammeTransformer, tensor_state_hash
from ..remediation.strict import object_sha256, sha256_file, StrictValidationError
from .normative import load_protocol, protocol_hash


def configure(seed: int) -> None:
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4)


def dependency_manifest(preflight: dict, namespace: str, stage: str, claim_bearing: bool) -> dict:
    if not preflight.get("passed"):
        raise StrictValidationError("DEPENDENCY_BINDING", "offline preflight failed")
    module_files = {}
    for name, module in (("torch", torch), ("numpy", np)):
        path = Path(module.__file__).resolve()
        module_files[name] = {"entry_file_sha256": sha256_file(path), "entry_file_bytes": path.stat().st_size}
    result = {
        "schema":"exp33ae.dependency_manifest.v1","namespace":namespace,"stage":stage,"claim_bearing":claim_bearing,
        "protocol_sha256":protocol_hash(),"required_versions":preflight["required"],
        "observed_versions":{name:preflight["observed"][name] for name in ("python","torch","numpy","pytest")},
        "python_implementation":preflight["observed"]["implementation"],
        "python_executable_sha256":sha256_file(Path(sys.executable).resolve()),"module_entry_files":module_files,
        "requirements_lock_sha256":sha256_file(Path(__file__).resolve().parents[3]/"requirements-lock.txt"),
        "boundary":"locked externally supplied interpreter; not independent dependency reconstruction",
        "downloads_used":False,"external_services_used":False,
    }
    result["manifest_sha256"]=object_sha256(result); return result


def runtime_observation(preflight: dict) -> dict:
    return {"schema":"exp33ae.runtime_observation.v1","scientific_identity":False,
            "platform":preflight["observed"]["platform"],"executable":preflight["observed"]["executable"]}


def save_checkpoint(path: Path, model: TinyProgrammeTransformer, seed_order: int, seed: int, bindings: dict, namespace: str, stage: str, claim_bearing: bool) -> dict:
    protocol=load_protocol(); recipe=protocol["model_and_training"]
    payload={"schema":"exp33ae.checkpoint.v1","namespace":namespace,"stage":stage,"claim_bearing":claim_bearing,
        "protocol_sha256":protocol_hash(),"recipe_id":recipe["recipe_id"],"seed_order":seed_order,"model_seed_identity":seed,
        "final_step":recipe["steps"],"model_config":model.config.as_dict(),"source_manifest_sha256":bindings["source_manifest_sha256"],
        "dependency_manifest_sha256":bindings["dependency_manifest_sha256"],"allocation_sha256":bindings["allocation_sha256"],
        "tensor_state_sha256":tensor_state_hash(model),"state_dict":model.state_dict()}
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise StrictValidationError("CHECKPOINT_BINDING","checkpoint path already exists")
    torch.save(payload,path)
    return {"checkpoint_file_sha256":sha256_file(path),"tensor_state_sha256":payload["tensor_state_sha256"],"checkpoint_bytes":path.stat().st_size}


def load_checkpoint(path: Path, expected: dict) -> tuple[TinyProgrammeTransformer,dict]:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    keys={"schema","namespace","stage","claim_bearing","protocol_sha256","recipe_id","seed_order","model_seed_identity","final_step","model_config","source_manifest_sha256","dependency_manifest_sha256","allocation_sha256","tensor_state_sha256","state_dict"}
    if type(payload) is not dict or set(payload)!=keys or payload["schema"]!="exp33ae.checkpoint.v1" or payload["protocol_sha256"]!=protocol_hash():
        raise StrictValidationError("CHECKPOINT_BINDING","schema")
    for field,value in expected.items():
        if field=="checkpoint_file_sha256":
            if sha256_file(path)!=value: raise StrictValidationError("CHECKPOINT_BINDING","file bytes")
        elif payload.get(field)!=value: raise StrictValidationError("CHECKPOINT_BINDING",field)
    model=TinyProgrammeTransformer(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["state_dict"]); model.eval()
    if tensor_state_hash(model)!=payload["tensor_state_sha256"]: raise StrictValidationError("CHECKPOINT_BINDING","canonical tensor state")
    return model,payload
