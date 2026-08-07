"""Actual clean and activation-intervention calls for the frozen registry."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import torch

from ..artifacts import tensor_sha256
from ..model import tensor_state_hash
from .normative import DEV_NAMESPACE
from .runtime import load_checkpoint
from .strict import object_sha256, strict_load_json


def _raw(plan: dict, values: dict) -> dict:
    row = {
        "schema": "exp33ae.dev_only_raw_inference.v2",
        "namespace": DEV_NAMESPACE,
        "stage_id": plan["stage_id"],
        "claim_bearing": False,
        "call_id": plan["call_id"],
        "nonce": plan["nonce"],
        "order": plan["order"],
        "registry_row_sha256": plan["registry_row_sha256"],
        "registry_payload_commitment": plan["registry_payload_commitment"],
        "model_order": plan["model_order"],
        "model_seed_identity": plan["model_seed_identity"],
        "checkpoint_file_sha256": plan["checkpoint_file_sha256"],
        "checkpoint_tensor_sha256": plan["checkpoint_tensor_sha256"],
        "operative_manifest_sha256": plan["operative_manifest_sha256"],
        "environment_sha256": plan["environment_sha256"],
        "allocation_sha256": plan["allocation_sha256"],
        "representation_sha256": plan["representation_sha256"],
        "control_manifest_sha256": plan["control_manifest_sha256"],
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "programme_id": plan["programme_id"],
        "causal_signature": plan["causal_signature"],
        "base_id": plan["base_id"],
        "source_id": plan["source_id"],
        "pair_id": plan["pair_id"],
        "pair_stratum": plan["pair_stratum"],
        "method": plan["method"],
        "control_id": plan["control_id"],
        "basis_sha256": plan["basis_sha256"],
        "decoder_sha256": plan["decoder_sha256"],
        "actual_layer": plan["layer"],
        "actual_position": plan["position"],
        "actual_rank": plan["rank"],
        "actual_strength": plan["strength"],
        "attempt_number": 1,
        "attempted": True,
        "completed": True,
        "dropped": False,
        "error": None,
        **values,
    }
    row["raw_row_sha256"] = object_sha256(row)
    return row


def _clean(model, plans, examples) -> list[dict]:
    rows = []
    for start in range(0, len(plans), 256):
        batch = plans[start:start+256]
        base = [examples[plan["base_id"]] for plan in batch]
        tokens = torch.tensor([row.tokens for row in base], dtype=torch.long)
        tables = torch.tensor([row.operation_tables for row in base], dtype=torch.float32)
        with torch.no_grad():
            logits, activations, _ = model(tokens, tables, return_activations=True)
        prediction = logits.argmax(-1)
        decoded = activations[2][:, 14, :2].argmax(-1)
        for index, plan in enumerate(batch):
            pred = int(prediction[index])
            rows.append(_raw(plan, {
                "base_prediction": pred, "source_prediction": pred, "intervened_prediction": pred,
                "decoded_z2": int(decoded[index]), "intervened_z3": pred // 2, "intervened_n3": pred % 2,
                "applied_norm": 0.0, "expected_applied_norm": 0.0, "norm_absolute_error": 0.0,
                "base_hidden_norm": float(activations[2][index, 14, :].norm()),
                "activation_delta_sha256": tensor_sha256(torch.zeros(model.config.d_model, dtype=activations[2].dtype)),
                "hook_consumed_by_downstream": False,
            }))
    return rows


def _intervene(model, plans, examples, basis: torch.Tensor) -> list[dict]:
    rows = []
    for start in range(0, len(plans), 256):
        batch = plans[start:start+256]
        base = [examples[plan["base_id"]] for plan in batch]
        source = [examples[plan["source_id"]] for plan in batch]
        bt = torch.tensor([row.tokens for row in base], dtype=torch.long)
        btab = torch.tensor([row.operation_tables for row in base], dtype=torch.float32)
        st = torch.tensor([row.tokens for row in source], dtype=torch.long)
        stab = torch.tensor([row.operation_tables for row in source], dtype=torch.float32)
        layer, position = batch[0]["layer"], batch[0]["position"]
        with torch.no_grad():
            base_logits, base_activations, _ = model(bt, btab, return_activations=True)
            source_logits, source_activations, _ = model(st, stab, return_activations=True)
            base_hidden = base_activations[layer][:, position, :]
            source_hidden = source_activations[layer][:, position, :]
            projected = (source_hidden - base_hidden) @ basis @ basis.T
            if batch[0]["method"] != "declared_iit_alignment":
                targets = torch.tensor([plan["expected_applied_norm"] for plan in batch], dtype=projected.dtype)
                observed = projected.norm(dim=-1)
                impossible = (observed < 1e-12) & (targets > 1e-6)
                if bool(impossible.any()):
                    raise RuntimeError("control projection has zero norm and cannot be matched")
                projected = projected * (targets / observed.clamp_min(1e-12)).unsqueeze(1)
                source_for_hook = base_hidden + projected
            else:
                source_for_hook = source_hidden
            after_logits, after_activations, applied = model(
                bt, btab,
                intervention={"layer": layer, "position": position, "source": source_for_hook, "basis": basis, "strength": 1.0},
                return_activations=True,
            )
        base_prediction = base_logits.argmax(-1)
        source_prediction = source_logits.argmax(-1)
        prediction = after_logits.argmax(-1)
        decoded = after_activations[2][:, 14, :2].argmax(-1)
        norms = applied.norm(dim=-1)
        for index, plan in enumerate(batch):
            pred = int(prediction[index])
            actual_norm = float(norms[index])
            expected_norm = float(plan["expected_applied_norm"])
            rows.append(_raw(plan, {
                "base_prediction": int(base_prediction[index]),
                "source_prediction": int(source_prediction[index]),
                "intervened_prediction": pred,
                "decoded_z2": int(decoded[index]),
                "intervened_z3": pred // 2,
                "intervened_n3": pred % 2,
                "applied_norm": actual_norm,
                "expected_applied_norm": expected_norm,
                "norm_absolute_error": abs(actual_norm - expected_norm),
                "base_hidden_norm": float(base_hidden[index].norm()),
                "activation_delta_sha256": tensor_sha256(applied[index]),
                "hook_consumed_by_downstream": True,
            }))
    return rows


def execute_registry_segment(output: Path, plans: list[dict], examples: dict, training_manifests: list[dict]) -> list[dict]:
    if not plans:
        return []
    results = []
    for model_order, model_plans_iterator in groupby(plans, key=lambda row: row["model_order"]):
        model_plans = list(model_plans_iterator)
        training = training_manifests[model_order]
        model, _ = load_checkpoint(output / training["checkpoint_relative_path"], {
            "model_order": model_order, "model_seed_identity": training["model_seed_identity"],
            "operative_manifest_sha256": model_plans[0]["operative_manifest_sha256"],
            "environment_sha256": model_plans[0]["environment_sha256"],
            "tensor_state_sha256": training["final_tensor_state_sha256"],
            "checkpoint_file_sha256": training["checkpoint_file_sha256"],
        })
        before = tensor_state_hash(model)
        controls = strict_load_json(output / "controls" / f"model-{model_order}.json")
        control_by_id = {row["control_id"]: row for row in controls["controls"]}
        for (_, control_id), iterator in groupby(model_plans, key=lambda row: (row["method"], row["control_id"])):
            group = list(iterator)
            if group[0]["method"] == "clean_task_decode":
                results.extend(_clean(model, group, examples))
            else:
                basis_values = (
                    [[1.0, 0.0], [0.0, 1.0]] if control_id is None
                    else control_by_id[control_id]["basis"]
                )
                if control_id is None:
                    basis = torch.zeros(64, 2, dtype=torch.float32); basis[0, 0] = basis[1, 1] = 1.0
                else:
                    basis = torch.tensor(basis_values, dtype=torch.float32)
                if tensor_sha256(basis) != group[0]["basis_sha256"]:
                    raise RuntimeError("actual intervention basis differs from registry")
                results.extend(_intervene(model, group, examples, basis))
        if tensor_state_hash(model) != before:
            raise RuntimeError("model state changed during frozen inference")
    return results
