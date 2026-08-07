"""Deterministically frozen confirmation-shaped controls for DEV_ONLY fixtures."""

from __future__ import annotations

import torch

from ..artifacts import tensor_sha256
from .entropy import DevEntropy
from .normative import DEV_NAMESPACE, load_protocol
from .representation import primary_basis


def _seed(entropy: DevEntropy, domain: str, index: int) -> int:
    return int.from_bytes(entropy.derive(domain, index, 8), "big") & ((1 << 63) - 1)


def _random_basis(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(64, 2, generator=generator, dtype=torch.float64), mode="reduced").Q


def build_controls(entropy: DevEntropy) -> list[dict]:
    protocol = load_protocol()
    primary = primary_basis()
    rows = []
    for index in range(4):
        basis = _random_basis(_seed(entropy, "random-direction", index))
        if torch.allclose(basis, primary):
            raise RuntimeError("random direction accidentally equals primary")
        rows.append({"family": "random_direction", "index": index, "basis": basis})
    for index in range(4):
        raw = _random_basis(_seed(entropy, "orthogonal-direction", index))
        residual = raw - primary @ (primary.T @ raw)
        basis = torch.linalg.qr(residual, mode="reduced").Q
        if float((primary.T @ basis).abs().max()) > 1e-10:
            raise RuntimeError("orthogonal direction is not orthogonal")
        rows.append({"family": "orthogonal_direction", "index": index, "basis": basis})
    for index, position in enumerate(protocol["controls"]["alternate_locations"]["positions"]):
        rows.append({"family": "alternate_location", "index": index, "basis": primary.clone(), "position": position})
    rows.append({"family": "alternate_early_layer", "index": 0, "basis": primary.clone(), "layer": 0})
    result = []
    for row in rows:
        family, index, basis = row["family"], row["index"], row["basis"]
        result.append({
            "schema": "exp33ae.dev_only_control.v1", "namespace": DEV_NAMESPACE, "stage_id": entropy.stage_id,
            "control_id": f"devonly-{family}-{index}", "family": family, "index": index,
            "basis_sha256": tensor_sha256(basis), "basis": basis.tolist(),
            "layer": row.get("layer", 2), "position": row.get("position", 14), "rank": 2, "strength": 1.0,
            "paired_stratum": "A_answer_changing", "paired_calls": 160,
        })
    if len(result) != 12 or len({row["basis_sha256"] for row in result if row["family"] in {"random_direction", "orthogonal_direction"}}) != 8:
        raise RuntimeError("control identity count/collision failure")
    return result
