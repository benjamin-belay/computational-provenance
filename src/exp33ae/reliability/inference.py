"""Actual checkpoint inference for every reliability-study registry row."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import torch

from ..artifacts import tensor_sha256
from ..model import declared_iit_basis, tensor_state_hash
from ..remediation.strict import object_sha256, strict_load_json
from .normative import DEV_NAMESPACE
from .runtime import load_checkpoint


def _raw(plan: dict, values: dict) -> dict:
    row = {
        "schema": "exp33ae.dev_only_reliability_raw.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "stage_id": plan["stage_id"], "call_id": plan["call_id"], "nonce": plan["nonce"],
        "order": plan["order"], "registry_row_sha256": plan["registry_row_sha256"],
        "registry_payload_commitment": plan["registry_payload_commitment"], "model_order": plan["model_order"],
        "recipe_id": plan["recipe_id"], "recipe_order": plan["recipe_order"], "seed_order": plan["seed_order"],
        "model_seed_identity": plan["model_seed_identity"], "checkpoint_file_sha256": plan["checkpoint_file_sha256"],
        "checkpoint_tensor_sha256": plan["checkpoint_tensor_sha256"], "source_manifest_sha256": plan["source_manifest_sha256"],
        "dependency_manifest_sha256": plan["dependency_manifest_sha256"], "allocation_sha256": plan["allocation_sha256"],
        "representation_sha256": plan["representation_sha256"], "control_manifest_sha256": plan["control_manifest_sha256"],
        "training_manifest_sha256": plan["training_manifest_sha256"], "programme_id": plan["programme_id"],
        "causal_signature": plan["causal_signature"], "base_id": plan["base_id"], "source_id": plan["source_id"],
        "pair_id": plan["pair_id"], "pair_stratum": plan["pair_stratum"], "method": plan["method"],
        "control_id": plan["control_id"], "basis_sha256": plan["basis_sha256"], "decoder_sha256": plan["decoder_sha256"],
        "actual_layer": plan["layer"], "actual_position": plan["position"], "actual_rank": plan["rank"],
        "actual_strength": plan["strength"], "attempt_number": 1, "attempted": True, "completed": True,
        "dropped": False, "error": None, **values,
    }
    row["raw_row_sha256"] = object_sha256(row); return row


def _clean(model, plans, examples) -> list[dict]:
    output = []
    for start in range(0, len(plans), 256):
        batch = plans[start:start+256]; base = [examples[p["base_id"]] for p in batch]
        tokens = torch.tensor([r.tokens for r in base]); tables = torch.tensor([r.operation_tables for r in base], dtype=torch.float32)
        with torch.no_grad(): logits, activations, _ = model(tokens, tables, return_activations=True)
        predictions = logits.argmax(-1); decoded = activations[2][:, 14, :2].argmax(-1)
        zero = tensor_sha256(torch.zeros(model.config.d_model, dtype=activations[2].dtype))
        for i, plan in enumerate(batch):
            pred = int(predictions[i]); output.append(_raw(plan, {
                "base_prediction": pred, "source_prediction": pred, "intervened_prediction": pred,
                "decoded_z2": int(decoded[i]), "intervened_z3": pred // 2, "intervened_n3": pred % 2,
                "applied_norm": 0.0, "expected_applied_norm": 0.0, "norm_absolute_error": 0.0,
                "base_hidden_norm": float(activations[2][i, 14, :].norm()), "activation_delta_sha256": zero,
                "hook_consumed_by_downstream": False}))
    return output


def _intervene(model, plans, examples, basis: torch.Tensor) -> list[dict]:
    output = []
    for start in range(0, len(plans), 256):
        batch = plans[start:start+256]; base = [examples[p["base_id"]] for p in batch]; source = [examples[p["source_id"]] for p in batch]
        bt = torch.tensor([r.tokens for r in base]); btab = torch.tensor([r.operation_tables for r in base], dtype=torch.float32)
        st = torch.tensor([r.tokens for r in source]); stab = torch.tensor([r.operation_tables for r in source], dtype=torch.float32)
        layer, position = batch[0]["layer"], batch[0]["position"]
        with torch.no_grad():
            base_logits, base_acts, _ = model(bt, btab, return_activations=True)
            source_logits, source_acts, _ = model(st, stab, return_activations=True)
            base_hidden, source_hidden = base_acts[layer][:, position, :], source_acts[layer][:, position, :]
            projected = ((source_hidden - base_hidden) @ basis) @ basis.T
            if batch[0]["method"] != "declared_iit_alignment":
                target = torch.tensor([p["expected_applied_norm"] for p in batch], dtype=projected.dtype)
                observed = projected.norm(dim=-1)
                if bool(((observed < 1e-12) & (target > 1e-6)).any()): raise RuntimeError("unmatchable zero-norm control")
                projected = projected * (target / observed.clamp_min(1e-12)).unsqueeze(1); source_for_hook = base_hidden + projected
            else: source_for_hook = source_hidden
            after, after_acts, applied = model(bt, btab, intervention={"layer": layer, "position": position,
                "source": source_for_hook, "basis": basis, "strength": 1.0}, return_activations=True)
        base_pred, source_pred, pred = base_logits.argmax(-1), source_logits.argmax(-1), after.argmax(-1)
        decoded = after_acts[2][:, 14, :2].argmax(-1); norms = applied.norm(dim=-1)
        for i, plan in enumerate(batch):
            value = int(pred[i]); actual = float(norms[i]); expected = float(plan["expected_applied_norm"])
            output.append(_raw(plan, {"base_prediction": int(base_pred[i]), "source_prediction": int(source_pred[i]),
                "intervened_prediction": value, "decoded_z2": int(decoded[i]), "intervened_z3": value // 2,
                "intervened_n3": value % 2, "applied_norm": actual, "expected_applied_norm": expected,
                "norm_absolute_error": abs(actual - expected), "base_hidden_norm": float(base_hidden[i].norm()),
                "activation_delta_sha256": tensor_sha256(applied[i]), "hook_consumed_by_downstream": True}))
    return output


def execute_segment(output: Path, plans: list[dict], examples: dict, training_by_model: dict[int, dict]) -> list[dict]:
    results = []
    for model_order, iterator in groupby(plans, key=lambda r: r["model_order"]):
        model_plans = list(iterator); train = training_by_model[model_order]
        expected = {"recipe_id": train["recipe_id"], "recipe_order": train["recipe_order"], "seed_order": train["seed_order"],
                    "model_seed_identity": train["model_seed_identity"], "source_manifest_sha256": model_plans[0]["source_manifest_sha256"],
                    "dependency_manifest_sha256": model_plans[0]["dependency_manifest_sha256"],
                    "tensor_state_sha256": train["final_tensor_state_sha256"], "checkpoint_file_sha256": train["checkpoint_file_sha256"]}
        model, _ = load_checkpoint(output / train["checkpoint_relative_path"], expected); before = tensor_state_hash(model)
        controls = strict_load_json(output / "controls" / train["recipe_id"] / f"seed-{train['seed_order']}.json")
        control_by_id = {r["control_id"]: r for r in controls["controls"]}
        for (_, control_id), sub in groupby(model_plans, key=lambda r: (r["method"], r["control_id"])):
            group = list(sub)
            if group[0]["method"] == "clean_task_decode": results.extend(_clean(model, group, examples)); continue
            basis = declared_iit_basis(model.config) if control_id is None else torch.tensor(control_by_id[control_id]["basis"], dtype=torch.float32)
            if tensor_sha256(basis) != group[0]["basis_sha256"]: raise RuntimeError("actual basis differs from registry")
            results.extend(_intervene(model, group, examples, basis))
        if tensor_state_hash(model) != before: raise RuntimeError("checkpoint changed during evaluation")
    return results
