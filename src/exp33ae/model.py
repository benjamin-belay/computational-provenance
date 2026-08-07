"""Small causal transformer with explicit residual-stream interchange hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import torch
from torch import nn

from .spec import IIT_RANK, JOINT_CLASSES, QUERY_POSITION, ROLE_POSITIONS, SEQ_LEN, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_mlp: int = 128
    output_classes: int = JOINT_CLASSES

    def as_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_mlp),
            nn.GELU(),
            nn.Linear(config.d_mlp, config.d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.ln1(x)
        attended, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        x = x + attended
        return x + self.mlp(self.ln2(x))


def projected_swap(
    base: torch.Tensor,
    source: torch.Tensor,
    basis: torch.Tensor,
    strength: float,
    norm_reference_basis: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = source - base
    projected = (delta @ basis) @ basis.T
    if norm_reference_basis is not None:
        reference = (delta @ norm_reference_basis) @ norm_reference_basis.T
        target_norm = reference.norm(dim=-1, keepdim=True)
        projected = projected * (target_norm / projected.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    applied = float(strength) * projected
    return base + applied, applied


class TinyProgrammeTransformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.position_embedding = nn.Parameter(torch.empty(c.seq_len, c.d_model))
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_ln = nn.LayerNorm(c.d_model)
        self.output_head = nn.Linear(c.d_model, c.output_classes)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        tables: torch.Tensor,
        intervention: dict[str, Any] | None = None,
        return_activations: bool = False,
    ):
        if tokens.ndim != 2 or tokens.shape[1] != self.config.seq_len:
            raise ValueError(f"expected [batch,{self.config.seq_len}], got {tuple(tokens.shape)}")
        x = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        if tables.shape != (tokens.shape[0], 6, 4):
            raise ValueError(f"expected operation tables [batch,6,4], got {tuple(tables.shape)}")
        table_features = torch.zeros(tokens.shape[0], 6, self.config.d_model, device=tokens.device)
        table_features[:, :, :4] = tables.float() * 2.0 - 1.0
        x = x.clone()
        x[:, ROLE_POSITIONS, :] = x[:, ROLE_POSITIONS, :] + table_features
        mask = torch.triu(
            torch.ones(self.config.seq_len, self.config.seq_len, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        activations = []
        applied_delta = None
        for layer, block in enumerate(self.blocks):
            x = block(x, mask)
            if intervention is not None and layer == int(intervention["layer"]):
                position = int(intervention["position"])
                swapped, applied_delta = projected_swap(
                    x[:, position, :],
                    intervention["source"],
                    intervention["basis"],
                    float(intervention["strength"]),
                    intervention.get("norm_reference_basis"),
                )
                x = x.clone()
                x[:, position, :] = swapped
            activations.append(x)
        logits = self.output_head(self.final_ln(x[:, QUERY_POSITION, :]))
        if return_activations:
            return logits, activations, applied_delta
        return logits

    def source_activation(self, tokens: torch.Tensor, tables: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        _, activations, _ = self.forward(tokens, tables, return_activations=True)
        return activations[layer][:, position, :]


def declared_iit_basis(config: ModelConfig | None = None) -> torch.Tensor:
    config = config or ModelConfig()
    basis = torch.zeros(config.d_model, IIT_RANK)
    basis[:IIT_RANK, :] = torch.eye(IIT_RANK)
    return basis


def random_basis(d_model: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d_model, rank, generator=generator), mode="reduced").Q


def orthogonal_basis(primary: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    raw = random_basis(primary.shape[0], rank, seed)
    residual = raw - primary @ (primary.T @ raw)
    if int(torch.linalg.matrix_rank(residual)) < rank:
        residual = residual + 1e-4 * random_basis(primary.shape[0], rank, seed + 1)
    return torch.linalg.qr(residual, mode="reduced").Q[:, :rank]


def tensor_state_hash(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()
