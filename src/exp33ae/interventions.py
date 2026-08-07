"""Actual activation interventions, controls, and call-level metrics."""

from __future__ import annotations

import torch

from .model import TinyProgrammeTransformer
from .spec import K


def decode_transport(hidden, full_probe, basis: torch.Tensor | None = None, subspace_decoder=None):
    """Decode transported state from the registered subspace when one is declared."""
    if basis is not None and subspace_decoder is not None:
        return subspace_decoder(hidden @ basis).argmax(-1)
    return full_probe.predict(hidden)


def pair_tensors(pairs):
    base_tokens = torch.tensor([p.base.tokens for p in pairs], dtype=torch.long)
    base_tables = torch.tensor([p.base.operation_tables for p in pairs], dtype=torch.float32)
    source_tokens = torch.tensor([p.source.tokens for p in pairs], dtype=torch.long)
    source_tables = torch.tensor([p.source.operation_tables for p in pairs], dtype=torch.float32)
    return base_tokens, base_tables, source_tokens, source_tables


def primary_norms(model, pairs, layer: int, position: int, basis: torch.Tensor) -> torch.Tensor:
    base_tokens, base_tables, source_tokens, source_tables = pair_tensors(pairs)
    with torch.no_grad():
        _, base_acts, _ = model(base_tokens, base_tables, return_activations=True)
        source = model.source_activation(source_tokens, source_tables, layer, position)
        delta = source - base_acts[layer][:, position, :]
        return ((delta @ basis) @ basis.T).norm(dim=-1)


def evaluate_intervention(
    model: TinyProgrammeTransformer,
    pairs,
    layer: int,
    position: int,
    basis: torch.Tensor,
    strength: float,
    z2_probe,
    control_family: str,
    norm_reference_basis: torch.Tensor | None = None,
    target_norms: torch.Tensor | None = None,
    transport_basis: torch.Tensor | None = None,
    transport_decoder=None,
):
    base_tokens, base_tables, source_tokens, source_tables = pair_tensors(pairs)
    model.eval()
    with torch.no_grad():
        base_logits, base_acts, _ = model(base_tokens, base_tables, return_activations=True)
        source_logits = model(source_tokens, source_tables)
        source_activation = model.source_activation(source_tokens, source_tables, layer, position)
        intervention = {
            "layer": layer,
            "position": position,
            "source": source_activation,
            "basis": basis,
            "strength": strength,
        }
        if norm_reference_basis is not None:
            intervention["norm_reference_basis"] = norm_reference_basis
        after_logits, after_acts, applied = model(base_tokens, base_tables, intervention=intervention, return_activations=True)
        if target_norms is not None:
            # Re-run with an explicitly per-call norm-matched source activation.
            base_hidden = base_acts[layer][:, position, :]
            delta = source_activation - base_hidden
            projected = (delta @ basis) @ basis.T
            scaled = projected * (target_norms.unsqueeze(1) / projected.norm(dim=-1, keepdim=True).clamp_min(1e-9))
            synthetic_source = base_hidden + scaled
            intervention = {
                "layer": layer,
                "position": position,
                "source": synthetic_source,
                "basis": basis,
                "strength": strength,
            }
            after_logits, after_acts, applied = model(
                base_tokens, base_tables, intervention=intervention, return_activations=True
            )
        base_pred = base_logits.argmax(-1)
        source_pred = source_logits.argmax(-1)
        after_pred = after_logits.argmax(-1)
        transport_hidden = after_acts[z2_probe.layer][:, z2_probe.position, :]
        decoded_z2 = decode_transport(transport_hidden, z2_probe, transport_basis, transport_decoder)
        applied_norm = applied.norm(dim=-1)
        base_hidden_norm = base_acts[layer][:, position, :].norm(dim=-1)

    rows = []
    for i, pair in enumerate(pairs):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "pair_kind": pair.kind,
                "control_family": control_family,
                "base_id": pair.base.example_id,
                "source_id": pair.source.example_id,
                "base_axis": pair.base.axis,
                "source_axis": pair.source.axis,
                "base_programme_id": pair.base.programme.programme_id,
                "source_programme_id": pair.source.programme.programme_id,
                "base_z2": pair.base.values.z2,
                "source_z2": pair.source.values.z2,
                "base_z3": pair.base.values.z3,
                "base_n3": pair.base.values.n3,
                "expected_z3": pair.expected_z3,
                "expected_n3": pair.base.values.n3,
                "base_joint": pair.base.joint_answer,
                "expected_joint": pair.expected_joint,
                "base_prediction": int(base_pred[i]),
                "source_joint": pair.source.joint_answer,
                "source_prediction": int(source_pred[i]),
                "intervened_prediction": int(after_pred[i]),
                "decoded_intervened_z2": int(decoded_z2[i]),
                "intervened_z3": int(after_pred[i]) // K,
                "intervened_n3": int(after_pred[i]) % K,
                "intervention_norm": float(applied_norm[i]),
                "base_hidden_norm": float(base_hidden_norm[i]),
                "norm_ratio": float(applied_norm[i] / base_hidden_norm[i].clamp_min(1e-9)),
                "no_op": pair.no_op,
                "attempted": True,
                "completed": True,
                "error": None,
            }
        )
    return rows


def summarize(rows) -> dict:
    denominator = len(rows)
    completed = sum(type(row.get("completed")) is bool and row["completed"] for row in rows)
    def rate(predicate):
        return sum(bool(row.get("completed")) and predicate(row) for row in rows) / denominator if denominator else float("nan")
    return {
        "attempted": denominator,
        "completed": completed,
        "failed": denominator - completed,
        "iia": rate(lambda r: r["intervened_prediction"] == r["expected_joint"]),
        "state_transport": rate(lambda r: r["decoded_intervened_z2"] == r["source_z2"]),
        "descendant_correctness": rate(lambda r: r["intervened_z3"] == r["expected_z3"]),
        "non_descendant_invariance": rate(lambda r: r["intervened_n3"] == r["expected_n3"]),
        "same_state_preservation": rate(lambda r: r["pair_kind"] != "same_state" or r["intervened_prediction"] == r["base_joint"]),
        "base_competence": rate(lambda r: r["base_prediction"] == r["base_joint"]),
        "source_competence": rate(lambda r: r["source_prediction"] == r["source_joint"]),
        "mean_intervention_norm": sum(float(r.get("intervention_norm", 0)) for r in rows) / denominator if denominator else float("nan"),
        "max_norm_ratio": max((float(r.get("norm_ratio", 0)) for r in rows), default=float("nan")),
    }
