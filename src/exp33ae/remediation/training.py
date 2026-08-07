"""Actual 1,400-step IIT training consuming independently derived streams."""

from __future__ import annotations

from hashlib import sha256
import time
from pathlib import Path

import torch
from torch import nn

from ..model import ModelConfig, TinyProgrammeTransformer, tensor_state_hash
from ..spec import K, N1_POSITION, N2_POSITION, Z1_POSITION, Z2_POSITION
from .entropy import DevMasterEntropy
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .runtime import configure_determinism, save_checkpoint
from .strict import object_sha256, write_json


def examples_to_tensors(rows) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([row.tokens for row in rows], dtype=torch.long),
        "tables": torch.tensor([row.operation_tables for row in rows], dtype=torch.float32),
        "answer": torch.tensor([row.answer for row in rows], dtype=torch.long),
        "z1": torch.tensor([row.z1 for row in rows], dtype=torch.long),
        "z2": torch.tensor([row.z2 for row in rows], dtype=torch.long),
        "n1": torch.tensor([row.n1 for row in rows], dtype=torch.long),
        "n2": torch.tensor([row.n2 for row in rows], dtype=torch.long),
        "n3": torch.tensor([row.n3 for row in rows], dtype=torch.long),
        "d": torch.tensor([row.d for row in rows], dtype=torch.long),
        "t3": torch.tensor([row.operation_tables[2] for row in rows], dtype=torch.long),
    }


def primary_basis(dtype=torch.float32) -> torch.Tensor:
    basis = torch.zeros(64, 2, dtype=dtype)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return basis


def train_one(
    rows,
    entropy: DevMasterEntropy,
    model_order: int,
    output_directory: Path,
    bindings: dict,
) -> dict:
    protocol = load_protocol()["training"]
    seed = entropy.model_seeds()[model_order]
    batch_seed = entropy.stream_seed("training-batch-stream", model_order)
    source_seed = entropy.stream_seed("training-interchange-stream", model_order)
    if batch_seed == source_seed:
        raise RuntimeError("training streams collided")
    configure_determinism(seed)
    model = TinyProgrammeTransformer(ModelConfig())
    initial_hash = tensor_state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
    )
    batch_generator = torch.Generator().manual_seed(batch_seed)
    source_generator = torch.Generator().manual_seed(source_seed)
    data = examples_to_tensors(rows)
    basis = primary_basis()
    curve = []
    started = time.perf_counter()
    model.train()
    for step in range(1, protocol["steps"] + 1):
        base_index = torch.randint(0, len(rows), (protocol["batch_size"],), generator=batch_generator)
        source_index = torch.randint(0, len(rows), (protocol["batch_size"],), generator=source_generator)
        tokens = data["tokens"][base_index]
        tables = data["tables"][base_index]
        answer = data["answer"][base_index]
        logits, activations, _ = model(tokens, tables, return_activations=True)
        with torch.no_grad():
            _, source_activations, _ = model(
                data["tokens"][source_index], data["tables"][source_index], return_activations=True
            )
        source_z2 = data["z2"][source_index]
        t3_index = 2 * source_z2 + data["d"][base_index]
        expected_z3 = data["t3"][base_index].gather(1, t3_index.unsqueeze(1)).squeeze(1)
        expected_answer = 2 * expected_z3 + data["n3"][base_index]
        counterfactual_logits = model(
            tokens,
            tables,
            intervention={
                "layer": 2,
                "position": 14,
                "source": source_activations[2][:, 14, :],
                "basis": basis,
                "strength": 1.0,
            },
        )
        answer_loss = nn.functional.cross_entropy(logits, answer)
        iit_loss = nn.functional.cross_entropy(counterfactual_logits, expected_answer)
        target_code_loss = (
            nn.functional.cross_entropy(activations[0][:, Z1_POSITION, :K], data["z1"][base_index])
            + nn.functional.cross_entropy(activations[1][:, Z2_POSITION, :K], data["z2"][base_index])
            + nn.functional.cross_entropy(activations[2][:, Z2_POSITION, :K], data["z2"][base_index])
        ) / 3
        parallel_code_loss = (
            nn.functional.cross_entropy(activations[0][:, N1_POSITION, :K], data["n1"][base_index])
            + nn.functional.cross_entropy(activations[1][:, N2_POSITION, :K], data["n2"][base_index])
            + nn.functional.cross_entropy(activations[2][:, N2_POSITION, K:2*K], data["n3"][base_index])
        ) / 3
        weights = protocol["loss_weights"]
        loss = (
            weights["answer"] * answer_loss
            + weights["iit_interchange"] * iit_loss
            + weights["direct_intermediate_code"] * target_code_loss
            + weights["parallel_code"] * parallel_code_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == protocol["steps"]:
            curve.append({
                "step": step,
                "loss": float(loss.detach()),
                "answer_loss": float(answer_loss.detach()),
                "iit_loss": float(iit_loss.detach()),
                "target_code_loss": float(target_code_loss.detach()),
                "parallel_code_loss": float(parallel_code_loss.detach()),
                "batch_correct": int((logits.argmax(-1) == answer).sum()),
                "batch_denominator": protocol["batch_size"],
            })
    runtime = time.perf_counter() - started
    model.eval()
    checkpoint_path = output_directory / "checkpoints" / f"model-{model_order}.pt"
    checkpoint = save_checkpoint(checkpoint_path, model, model_order, seed, protocol["steps"], bindings)
    manifest = {
        "schema": "exp33ae.dev_only_training_manifest.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "protocol_sha256": protocol_hash(),
        "model_order": model_order,
        "model_seed_identity": seed,
        "batch_stream_seed": batch_seed,
        "interchange_stream_seed": source_seed,
        "initial_tensor_state_sha256": initial_hash,
        "final_tensor_state_sha256": checkpoint["tensor_state_sha256"],
        "checkpoint_relative_path": str(checkpoint_path.relative_to(output_directory)),
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "steps": protocol["steps"],
        "batch_size": protocol["batch_size"],
        "interchange_pairs": protocol["steps"] * protocol["batch_size"],
        "final_step_only": True,
        "curve": curve,
        "operative_manifest_sha256": bindings["operative_manifest_sha256"],
        "environment_sha256": bindings["environment_sha256"],
    }
    manifest["scientific_manifest_sha256"] = object_sha256(manifest)
    write_json(output_directory / "training" / f"model-{model_order}.json", manifest)
    write_json(output_directory / "runtime" / f"training-model-{model_order}.json", {
        "schema": "exp33ae.dev_only_runtime_observation.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "model_order": model_order,
        "model_seed_identity": seed,
        "runtime_seconds": runtime,
    })
    return manifest


def train_population(rows, entropy: DevMasterEntropy, output_directory: Path, bindings: dict) -> list[dict]:
    return [train_one(rows, entropy, order, output_directory, bindings) for order in range(3)]
