"""Strict two-stage transformer with hard, non-bypassable z2 and z3 interfaces."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


STATE_CARDINALITY = 16
ANSWER_CARDINALITY = 8
D_MODEL = 128
N_HEADS = 4
D_MLP = 256
LAYERS_PER_STAGE = 2


def hard_one_hot(categories: torch.Tensor) -> torch.Tensor:
    if categories.dtype != torch.int64 or categories.ndim != 1:
        raise ValueError("CATEGORIES_MUST_BE_INT64_VECTOR")
    if torch.any(categories < 0) or torch.any(categories >= STATE_CARDINALITY):
        raise ValueError("CATEGORY_OUT_OF_RANGE")
    return torch.nn.functional.one_hot(categories, STATE_CARDINALITY).to(torch.float32)


def validate_one_hot(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != STATE_CARDINALITY:
        raise ValueError("STATE_MUST_BE_FLOAT32_BATCH_BY_16")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError("STATE_NOT_BINARY")
    if not torch.all(value.sum(dim=-1) == 1):
        raise ValueError("STATE_NOT_ONE_HOT")
    return value.argmax(dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MLP,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden)


class Z2Transformer(nn.Module):
    """Stage 1 sees only a, b, and c and emits 16-way z2 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.a_embedding = nn.Embedding(STATE_CARDINALITY, D_MODEL)
        self.b_embedding = nn.Embedding(STATE_CARDINALITY, D_MODEL)
        self.c_embedding = nn.Embedding(STATE_CARDINALITY, D_MODEL)
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.position_embedding = nn.Embedding(4, D_MODEL)
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(LAYERS_PER_STAGE)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.state_head = nn.Linear(D_MODEL, STATE_CARDINALITY)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not (a.dtype == b.dtype == c.dtype == torch.int64 and a.ndim == b.ndim == c.ndim == 1):
            raise ValueError("Z2_INPUTS_MUST_BE_INT64_VECTORS")
        if not (a.shape == b.shape == c.shape):
            raise ValueError("Z2_INPUT_SHAPES")
        batch = a.shape[0]
        hidden = torch.stack(
            (
                self.cls.expand(batch, -1, -1)[:, 0, :],
                self.a_embedding(a),
                self.b_embedding(b),
                self.c_embedding(c),
            ),
            dim=1,
        )
        positions = torch.arange(4, device=hidden.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        final_hidden = self.final_norm(hidden)
        return self.state_head(final_hidden[:, 0, :]), final_hidden


class Z3Transformer(nn.Module):
    """Stage 2 sees only a hard z2 category and d and emits 16-way z3 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.z2_embedding = nn.Embedding(STATE_CARDINALITY, D_MODEL)
        self.d_embedding = nn.Embedding(STATE_CARDINALITY, D_MODEL)
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.position_embedding = nn.Embedding(3, D_MODEL)
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(LAYERS_PER_STAGE)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.state_head = nn.Linear(D_MODEL, STATE_CARDINALITY)

    def forward(self, hard_z2: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z2_consumed = validate_one_hot(hard_z2)
        if d.dtype != torch.int64 or d.ndim != 1 or d.shape != z2_consumed.shape:
            raise ValueError("D_MUST_MATCH_Z2_BATCH")
        batch = d.shape[0]
        hidden = torch.stack(
            (
                self.cls.expand(batch, -1, -1)[:, 0, :],
                self.z2_embedding(z2_consumed),
                self.d_embedding(d),
            ),
            dim=1,
        )
        positions = torch.arange(3, device=hidden.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        final_hidden = self.final_norm(hidden)
        return self.state_head(final_hidden[:, 0, :]), z2_consumed, final_hidden


class EngineeredCausalTransformer(nn.Module):
    """Composition with literal hard-state entry points and no residual bypass."""

    def __init__(self) -> None:
        super().__init__()
        self.z2_transformer = Z2Transformer()
        self.z3_transformer = Z3Transformer()
        self.answer_head = nn.Linear(STATE_CARDINALITY, ANSWER_CARDINALITY)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_values(values: torch.Tensor) -> None:
        if values.dtype != torch.int64 or values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("VALUES_MUST_BE_INT64_BATCH_BY_4")
        if torch.any(values < 0) or torch.any(values >= STATE_CARDINALITY):
            raise ValueError("VALUE_OUT_OF_RANGE")

    def predict_z2(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, residual = self.z2_transformer(a, b, c)
        selected = logits.argmax(dim=-1)
        return logits, hard_one_hot(selected), residual

    def transition(self, hard_z2: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, z2_consumed, residual = self.z3_transformer(hard_z2, d)
        selected = logits.argmax(dim=-1)
        return logits, z2_consumed, hard_one_hot(selected), residual

    def answer(self, hard_z3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z3_consumed = validate_one_hot(hard_z3)
        return self.answer_head(hard_z3), z3_consumed

    def run_natural(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_values(values)
        z2_logits, hard_z2, stage_1_residual = self.predict_z2(values[:, 0], values[:, 1], values[:, 2])
        z3_logits, z2_consumed, hard_z3, stage_2_residual = self.transition(hard_z2, values[:, 3])
        answer_logits, z3_consumed = self.answer(hard_z3)
        return {
            "z2_logits": z2_logits,
            "z2_selected": hard_z2.argmax(dim=-1),
            "z2_consumed": z2_consumed,
            "z3_logits": z3_logits,
            "z3_selected": hard_z3.argmax(dim=-1),
            "z3_consumed": z3_consumed,
            "answer_logits": answer_logits,
            "stage_1_residual": stage_1_residual,
            "stage_2_residual": stage_2_residual,
        }

    def run_with_z2(self, values: torch.Tensor, target_z2: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_values(values)
        hard_z2 = hard_one_hot(target_z2)
        z3_logits, z2_consumed, hard_z3, stage_2_residual = self.transition(hard_z2, values[:, 3])
        answer_logits, z3_consumed = self.answer(hard_z3)
        return {
            "z2_consumed": z2_consumed,
            "z3_logits": z3_logits,
            "z3_selected": hard_z3.argmax(dim=-1),
            "z3_consumed": z3_consumed,
            "answer_logits": answer_logits,
            "stage_2_residual": stage_2_residual,
        }

    def run_with_z3(self, target_z3: torch.Tensor) -> dict[str, torch.Tensor]:
        hard_z3 = hard_one_hot(target_z3)
        answer_logits, z3_consumed = self.answer(hard_z3)
        return {"z3_consumed": z3_consumed, "answer_logits": answer_logits}

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.run_natural(values)


def parameter_count(model: nn.Module | None = None) -> int:
    selected = EngineeredCausalTransformer() if model is None else model
    return sum(parameter.numel() for parameter in selected.parameters())


def architecture_record(model: EngineeredCausalTransformer | None = None) -> dict[str, Any]:
    selected = EngineeredCausalTransformer() if model is None else model
    stage_1_parameters = {id(parameter) for parameter in selected.z2_transformer.parameters()}
    stage_2_parameters = {id(parameter) for parameter in selected.z3_transformer.parameters()}
    return {
        "parameter_count": parameter_count(selected),
        "stage_1_parameter_count": sum(parameter.numel() for parameter in selected.z2_transformer.parameters()),
        "stage_2_parameter_count": sum(parameter.numel() for parameter in selected.z3_transformer.parameters()),
        "answer_head_parameter_count": sum(parameter.numel() for parameter in selected.answer_head.parameters()),
        "shared_stage_parameter_objects": len(stage_1_parameters & stage_2_parameters),
        "stage_1_sequence_length": 4,
        "stage_2_sequence_length": 3,
        "layers_per_stage": LAYERS_PER_STAGE,
        "d_model": D_MODEL,
        "attention_heads": N_HEADS,
        "mlp_width": D_MLP,
    }
