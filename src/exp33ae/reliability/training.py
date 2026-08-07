"""Exact four-recipe/five-seed IIT training population."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from ..model import ModelConfig, TinyProgrammeTransformer, declared_iit_basis, tensor_state_hash
from ..spec import K, N1_POSITION, N2_POSITION, Z1_POSITION, Z2_POSITION
from ..remediation.strict import object_sha256, write_json
from ..remediation.training import examples_to_tensors
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .runtime import configure, save_checkpoint


def parameter_count(recipe: dict) -> int:
    model = TinyProgrammeTransformer(ModelConfig(d_model=recipe["d_model"], d_mlp=recipe["d_mlp"],
                                                   n_heads=recipe["heads"], n_layers=recipe["layers"]))
    return sum(p.numel() for p in model.parameters())


def train_one(rows, entropy, recipe: dict, seed_order: int, output: Path, bindings: dict) -> dict:
    protocol = load_protocol(); train = protocol["training"]
    recipe_order = recipe["order"]; model_order = recipe_order * 5 + seed_order
    seed = entropy.model_seeds()[seed_order]
    batch_seed = entropy.stream_seed("training-batch-stream", recipe_order, seed_order)
    interchange_seed = entropy.stream_seed("training-interchange-stream", recipe_order, seed_order)
    if batch_seed == interchange_seed: raise RuntimeError("training streams collided")
    configure(seed)
    config = ModelConfig(d_model=recipe["d_model"], d_mlp=recipe["d_mlp"], n_heads=recipe["heads"], n_layers=recipe["layers"])
    model = TinyProgrammeTransformer(config); initial_hash = tensor_state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"],
                                  betas=tuple(train["betas"]), eps=train["eps"])
    batch_generator = torch.Generator().manual_seed(batch_seed)
    source_generator = torch.Generator().manual_seed(interchange_seed)
    data = examples_to_tensors(rows); basis = declared_iit_basis(config)
    curve = []; started = time.perf_counter(); model.train()
    for step in range(1, recipe["steps"] + 1):
        base_index = torch.randint(0, len(rows), (train["batch_size"],), generator=batch_generator)
        source_index = torch.randint(0, len(rows), (train["batch_size"],), generator=source_generator)
        tokens, tables, answer = data["tokens"][base_index], data["tables"][base_index], data["answer"][base_index]
        logits, activations, _ = model(tokens, tables, return_activations=True)
        with torch.no_grad():
            _, source_activations, _ = model(data["tokens"][source_index], data["tables"][source_index], return_activations=True)
        source_z2 = data["z2"][source_index]
        expected_z3 = data["t3"][base_index].gather(1, (2 * source_z2 + data["d"][base_index]).unsqueeze(1)).squeeze(1)
        expected_answer = 2 * expected_z3 + data["n3"][base_index]
        counterfactual = model(tokens, tables, intervention={"layer": 2, "position": 14,
            "source": source_activations[2][:, 14, :], "basis": basis, "strength": 1.0})
        answer_loss = nn.functional.cross_entropy(logits, answer)
        iit_loss = nn.functional.cross_entropy(counterfactual, expected_answer)
        target_loss = (nn.functional.cross_entropy(activations[0][:, Z1_POSITION, :K], data["z1"][base_index]) +
                       nn.functional.cross_entropy(activations[1][:, Z2_POSITION, :K], data["z2"][base_index]) +
                       nn.functional.cross_entropy(activations[2][:, Z2_POSITION, :K], data["z2"][base_index])) / 3
        parallel_loss = (nn.functional.cross_entropy(activations[0][:, N1_POSITION, :K], data["n1"][base_index]) +
                         nn.functional.cross_entropy(activations[1][:, N2_POSITION, :K], data["n2"][base_index]) +
                         nn.functional.cross_entropy(activations[2][:, N2_POSITION, K:2*K], data["n3"][base_index])) / 3
        weights = train["loss_weights"]
        loss = weights["answer"]*answer_loss + weights["iit_interchange"]*iit_loss + weights["direct_intermediate_code"]*target_loss + weights["parallel_code"]*parallel_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"]); optimizer.step()
        if step == 1 or step % 100 == 0 or step == recipe["steps"]:
            curve.append({"step": step, "loss": float(loss.detach()), "answer_loss": float(answer_loss.detach()),
                          "iit_loss": float(iit_loss.detach()), "target_code_loss": float(target_loss.detach()),
                          "parallel_code_loss": float(parallel_loss.detach()),
                          "batch_correct": int((logits.argmax(-1) == answer).sum()), "batch_denominator": train["batch_size"]})
    runtime_seconds = time.perf_counter() - started; model.eval()
    relative = Path("checkpoints") / recipe["id"] / f"seed-{seed_order}.pt"
    checkpoint = save_checkpoint(output / relative, model, recipe, recipe_order, seed_order, seed, bindings)
    manifest = {
        "schema": "exp33ae.dev_only_reliability_training_manifest.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(), "recipe_id": recipe["id"],
        "recipe_order": recipe_order, "seed_order": seed_order, "model_order": model_order,
        "model_seed_identity": seed, "batch_stream_seed": batch_seed, "interchange_stream_seed": interchange_seed,
        "model_config": config.as_dict(), "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_tensor_state_sha256": initial_hash, "final_tensor_state_sha256": checkpoint["tensor_state_sha256"],
        "checkpoint_relative_path": str(relative), "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
        "checkpoint_bytes": checkpoint["checkpoint_bytes"], "steps": recipe["steps"], "batch_size": train["batch_size"],
        "interchange_pairs": recipe["steps"] * train["batch_size"], "final_step_only": True,
        "curve": curve, "source_manifest_sha256": bindings["source_manifest_sha256"],
        "dependency_manifest_sha256": bindings["dependency_manifest_sha256"],
    }
    manifest["scientific_manifest_sha256"] = object_sha256(manifest)
    path = output / "training" / recipe["id"] / f"seed-{seed_order}.json"; write_json(path, manifest)
    write_json(output / "runtime" / "training" / recipe["id"] / f"seed-{seed_order}.json", {
        "schema": "exp33ae.dev_only_reliability_training_runtime.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "recipe_id": recipe["id"], "seed_order": seed_order,
        "model_seed_identity": seed, "runtime_seconds": runtime_seconds, "scientific_identity": False})
    return manifest


def train_all(rows, entropy, output: Path, bindings: dict) -> list[dict]:
    results = []
    for recipe in load_protocol()["recipes"]:
        for seed_order in range(5):
            results.append(train_one(rows, entropy, recipe, seed_order, output, bindings))
    if len(results) != 20: raise RuntimeError("exactly 20 training results required")
    return results
