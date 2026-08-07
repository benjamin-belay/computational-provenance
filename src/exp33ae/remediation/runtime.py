"""Runtime, environment, checkpoint, and deterministic I/O bindings."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from ..model import ModelConfig, TinyProgrammeTransformer, tensor_state_hash
from .normative import DEV_NAMESPACE, protocol_hash
from .strict import canonical_json, object_sha256, sha256_file, StrictValidationError


def environment_manifest() -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    value = {
        "schema": "exp33ae.environment_manifest.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "protocol_sha256": protocol_hash(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_basename": Path(sys.executable).name,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": 4,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "numpy_configuration": buffer.getvalue(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    value["manifest_sha256"] = object_sha256(value)
    return value


def configure_determinism(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def checkpoint_payload(model: TinyProgrammeTransformer, model_order: int, seed: int, step: int, bindings: dict) -> dict:
    return {
        "schema": "exp33ae.dev_only_remediation_checkpoint.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "protocol_sha256": protocol_hash(),
        "model_order": model_order,
        "model_seed_identity": seed,
        "final_step": step,
        "model_config": model.config.as_dict(),
        "operative_manifest_sha256": bindings["operative_manifest_sha256"],
        "environment_sha256": bindings["environment_sha256"],
        "tensor_state_sha256": tensor_state_hash(model),
        "state_dict": model.state_dict(),
    }


def save_checkpoint(path: Path, model: TinyProgrammeTransformer, model_order: int, seed: int, step: int, bindings: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, model_order, seed, step, bindings), path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "tensor_state_sha256": tensor_state_hash(model),
        "bytes": path.stat().st_size,
    }


def load_checkpoint(path: Path, expected: dict | None = None) -> tuple[TinyProgrammeTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema", "namespace", "claim_bearing", "protocol_sha256", "model_order", "model_seed_identity",
        "final_step", "model_config", "operative_manifest_sha256", "environment_sha256", "tensor_state_sha256",
        "state_dict",
    }
    if type(payload) is not dict or set(payload) != required:
        raise StrictValidationError("CHECKPOINT_FILE_BINDING", "checkpoint payload keys")
    if payload["schema"] != "exp33ae.dev_only_remediation_checkpoint.v2" or payload["namespace"] != DEV_NAMESPACE or payload["claim_bearing"] is not False:
        raise StrictValidationError("CHECKPOINT_FILE_BINDING", "checkpoint namespace/schema")
    if payload["protocol_sha256"] != protocol_hash() or payload["final_step"] != 1400:
        raise StrictValidationError("TRAINING_BINDING", "checkpoint protocol or final step")
    model = TinyProgrammeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if tensor_state_hash(model) != payload["tensor_state_sha256"]:
        raise StrictValidationError("CHECKPOINT_TENSOR_BINDING", "canonical tensor state mismatch")
    if expected:
        for field in ("model_order", "model_seed_identity", "operative_manifest_sha256", "environment_sha256", "tensor_state_sha256"):
            if payload[field] != expected[field]:
                raise StrictValidationError("CHECKPOINT_TENSOR_BINDING", f"checkpoint field {field}")
        if sha256_file(path) != expected["checkpoint_file_sha256"]:
            raise StrictValidationError("CHECKPOINT_FILE_BINDING", "checkpoint file hash")
    return model, payload
