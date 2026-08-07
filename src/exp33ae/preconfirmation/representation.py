"""Canonical mathematical registered representation and golden vectors."""

from __future__ import annotations

import torch

from ..artifacts import tensor_sha256


def primary_basis() -> torch.Tensor:
    basis = torch.zeros(64, 2, dtype=torch.float64)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return basis


def state_code(z2: int) -> torch.Tensor:
    if z2 not in (0, 1):
        raise ValueError("z2 must be Boolean")
    return torch.tensor([1.0, 0.0] if z2 == 0 else [0.0, 1.0], dtype=torch.float64)


def coordinates(hidden: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    basis = primary_basis() if basis is None else basis
    if hidden.shape[-1] != 64 or basis.shape != (64, 2):
        raise ValueError("registered coordinates require hidden[...,64] and basis[64,2]")
    return hidden.to(torch.float64) @ basis


def projected_interchange(base: torch.Tensor, source: torch.Tensor, strength: float = 1.0, basis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    basis = primary_basis() if basis is None else basis
    if type(strength) not in {int, float} or float(strength) != 1.0:
        raise ValueError("registered primary intervention strength is exactly 1.0")
    delta = (source.to(torch.float64) - base.to(torch.float64)) @ basis @ basis.T
    return base.to(torch.float64) + delta, delta


def decode_subspace(hidden: torch.Tensor, basis: torch.Tensor | None = None) -> int:
    return int(coordinates(hidden, basis).argmax().item())


def golden_vectors() -> dict:
    basis = primary_basis()
    base = torch.arange(64, dtype=torch.float64) / 16.0
    source = torch.flip(base, dims=(0,))
    swapped, delta = projected_interchange(base, source)
    return {
        "schema": "exp33ae.representation_golden.v1",
        "namespace": "DEV_ONLY_PRECONFIRMATION_FIXTURE",
        "claim_bearing": False,
        "basis_sha256": tensor_sha256(basis),
        "basis_orthonormal_max_error": float((basis.T @ basis - torch.eye(2, dtype=torch.float64)).abs().max()),
        "base_coordinates": coordinates(base).tolist(),
        "source_coordinates": coordinates(source).tolist(),
        "swapped_coordinates": coordinates(swapped).tolist(),
        "delta_nonzero_indices": [index for index, value in enumerate(delta.tolist()) if value != 0.0],
        "state_codes": {"0": state_code(0).tolist(), "1": state_code(1).tolist()},
        "confirmation_objects_created": 0,
        "confirmation_objects_accessed": 0
    }
