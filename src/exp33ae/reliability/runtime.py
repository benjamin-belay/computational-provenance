"""Locked dependency boundary and variable-width checkpoint bindings."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

from ..model import ModelConfig, TinyProgrammeTransformer, tensor_state_hash
from ..remediation.strict import object_sha256, sha256_file, StrictValidationError
from .normative import DEV_NAMESPACE, protocol_hash


def dependency_manifest() -> dict:
    module_files = {}
    for name, module in (("torch", torch), ("numpy", np)):
        path = Path(module.__file__).resolve()
        module_files[name] = {"path_basename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    value = {
        "schema": "exp33ae.dev_only_reliability_dependency_manifest.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "module_entry_files": module_files, "requirements_lock_sha256": sha256_file(Path(__file__).resolve().parents[3] / "requirements-lock.txt"),
        "boundary": "same locked local numerical environment; not independent dependency reconstruction",
        "downloads_used": False, "external_services_used": False,
    }
    value["manifest_sha256"] = object_sha256(value)
    return value


def runtime_observation() -> dict:
    return {
        "schema": "exp33ae.dev_only_reliability_runtime_observation.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "pid": os.getpid(), "torch_threads": 4,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "scientific_identity": False,
    }


def configure(seed: int) -> None:
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4)


def save_checkpoint(path: Path, model: TinyProgrammeTransformer, recipe: dict, recipe_order: int,
                    seed_order: int, seed: int, bindings: dict) -> dict:
    payload = {
        "schema": "exp33ae.dev_only_reliability_checkpoint.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "recipe_id": recipe["id"], "recipe_order": recipe_order, "seed_order": seed_order,
        "model_seed_identity": seed, "final_step": recipe["steps"],
        "model_config": model.config.as_dict(), "source_manifest_sha256": bindings["source_manifest_sha256"],
        "dependency_manifest_sha256": bindings["dependency_manifest_sha256"],
        "tensor_state_sha256": tensor_state_hash(model), "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    return {"checkpoint_file_sha256": sha256_file(path), "tensor_state_sha256": payload["tensor_state_sha256"],
            "checkpoint_bytes": path.stat().st_size}


def load_checkpoint(path: Path, expected: dict | None = None) -> tuple[TinyProgrammeTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = {"schema", "namespace", "claim_bearing", "protocol_sha256", "recipe_id", "recipe_order",
            "seed_order", "model_seed_identity", "final_step", "model_config", "source_manifest_sha256",
            "dependency_manifest_sha256", "tensor_state_sha256", "state_dict"}
    if type(payload) is not dict or set(payload) != keys or payload["schema"] != "exp33ae.dev_only_reliability_checkpoint.v1" or payload["namespace"] != DEV_NAMESPACE or payload["claim_bearing"] is not False or payload["protocol_sha256"] != protocol_hash():
        raise StrictValidationError("CHECKPOINT_BINDING", "payload schema")
    model = TinyProgrammeTransformer(ModelConfig(**payload["model_config"])); model.load_state_dict(payload["state_dict"]); model.eval()
    if tensor_state_hash(model) != payload["tensor_state_sha256"]:
        raise StrictValidationError("CHECKPOINT_BINDING", "tensor state")
    if expected:
        for field, value in expected.items():
            if field == "checkpoint_file_sha256":
                if sha256_file(path) != value: raise StrictValidationError("CHECKPOINT_BINDING", "file")
            elif payload.get(field) != value:
                raise StrictValidationError("CHECKPOINT_BINDING", field)
    return model, payload

