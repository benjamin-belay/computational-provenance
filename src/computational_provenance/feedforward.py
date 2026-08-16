"""Strict feed-forward model with hard, non-bypassable z2 and z3 interfaces."""
from __future__ import annotations

import torch
from torch import nn


def hard_one_hot(categories: torch.Tensor) -> torch.Tensor:
    if categories.dtype != torch.int64 or categories.ndim != 1:
        raise ValueError("categories must be an int64 vector")
    return torch.nn.functional.one_hot(categories, 16).to(torch.float32)


def validate_one_hot(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 16 or not value.dtype.is_floating_point:
        raise ValueError("state must be batch-by-16 one-hot")
    if not torch.all((value == 0) | (value == 1)) or not torch.all(value.sum(-1) == 1):
        raise ValueError("invalid hard state")
    return value.argmax(-1)


class CausalStrictBottleneck(nn.Module):
    """The 12,648-parameter architecture used for the final feed-forward result."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Embedding(16, 16)
        self.b = nn.Embedding(16, 16)
        self.c = nn.Embedding(16, 16)
        self.d = nn.Embedding(16, 16)
        self.z2_module = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Linear(128, 16))
        self.z3_module = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16))
        self.answer_module = nn.Linear(16, 8)

    def natural_z2(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.z2_module(torch.cat((self.a(values[:, 0]), self.b(values[:, 1]), self.c(values[:, 2])), -1))
        return logits, hard_one_hot(logits.argmax(-1))

    def transition(self, hard_z2: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validate_one_hot(hard_z2)
        logits = self.z3_module(torch.cat((hard_z2, self.d(d)), -1))
        return logits, hard_one_hot(logits.argmax(-1))

    def answer(self, hard_z3: torch.Tensor) -> torch.Tensor:
        validate_one_hot(hard_z3)
        return self.answer_module(hard_z3)

    def run_natural(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        z2_logits, z2 = self.natural_z2(values)
        z3_logits, z3 = self.transition(z2, values[:, 3])
        return {"z2_logits": z2_logits, "z2_selected": z2.argmax(-1), "z2_consumed": validate_one_hot(z2),
                "z3_logits": z3_logits, "z3_selected": z3.argmax(-1), "z3_consumed": validate_one_hot(z3),
                "answer_logits": self.answer(z3)}

    def run_with_z2(self, values: torch.Tensor, target_z2: torch.Tensor) -> dict[str, torch.Tensor]:
        z2 = hard_one_hot(target_z2)
        z3_logits, z3 = self.transition(z2, values[:, 3])
        return {"z2_consumed": validate_one_hot(z2), "z3_logits": z3_logits,
                "z3_selected": z3.argmax(-1), "z3_consumed": validate_one_hot(z3),
                "answer_logits": self.answer(z3)}

    def run_with_z3(self, target_z3: torch.Tensor) -> dict[str, torch.Tensor]:
        z3 = hard_one_hot(target_z3)
        return {"z3_consumed": validate_one_hot(z3), "answer_logits": self.answer(z3)}

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.run_natural(values)
