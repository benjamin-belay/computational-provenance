"""Deterministic training recipes for the two released architectures."""
from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .feedforward import CausalStrictBottleneck
from .transformer import EngineeredCausalTransformer


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def configure(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def tensors(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    values = torch.tensor([[row["inputs"][key] for key in ("a", "b", "c", "d")] for row in rows], dtype=torch.int64)
    target = "targets" if "targets" in rows[0] else "natural"
    return (values, *(torch.tensor([row[target][name] for row in rows], dtype=torch.int64)
                      for name in ("z2", "z3", "answer")))


def _straight_through(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return hard + probabilities - probabilities.detach()


def _feedforward_training_output(model: CausalStrictBottleneck, values: torch.Tensor) -> dict[str, torch.Tensor]:
    z2_logits = model.z2_module(torch.cat((model.a(values[:, 0]), model.b(values[:, 1]), model.c(values[:, 2])), -1))
    z2 = _straight_through(z2_logits)
    z3_logits = model.z3_module(torch.cat((z2, model.d(values[:, 3])), -1))
    z3 = _straight_through(z3_logits)
    return {"z2": z2_logits, "z3": z3_logits, "answer": model.answer_module(z3)}


def _rate(step: int, *, peak: float, floor: float, warmup: int, horizon: int) -> float:
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (horizon - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


TRAINING_PROFILES = {
    "feedforward": {"designated": [1062101], "robustness": [952101, 952102, 952103, 952104, 952105]},
    "transformer": {"designated": [783201], "robustness": [783101, 783102, 783103]},
}


def train(architecture: str, rows: list[dict[str, Any]], *, seed: int | None = None,
          steps: int | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    if architecture == "feedforward":
        settings = {"seed": 1062101, "steps": 5000, "batch": 256, "peak": 1e-3,
                    "floor": 5e-5, "warmup": 750, "horizon": 10000, "weight_decay": 1e-4}
        model: torch.nn.Module = CausalStrictBottleneck()
    elif architecture == "transformer":
        settings = {"seed": 783201, "steps": 12000, "batch": 512, "peak": 5e-4,
                    "floor": 5e-5, "warmup": 800, "horizon": 12000, "weight_decay": 1e-2}
        model = EngineeredCausalTransformer()
    else:
        raise ValueError("architecture")
    if seed is not None:
        settings["seed"] = seed
    if steps is not None:
        settings["steps"] = steps
    configure(settings["seed"])
    # Recreate after seeding so parameter initialisation is claim-recipe deterministic.
    model = CausalStrictBottleneck() if architecture == "feedforward" else EngineeredCausalTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["peak"], weight_decay=settings["weight_decay"])
    values, z2, z3, answer = tensors(rows)
    generator = torch.Generator().manual_seed(settings["seed"] + 1000003)
    trace = []
    model.train()
    for step in range(1, settings["steps"] + 1):
        index = torch.randint(0, len(rows), (settings["batch"],), generator=generator)
        if architecture == "feedforward":
            output = _feedforward_training_output(model, values[index])
        else:
            raw = model(values[index])
            output = {"z2": raw["z2_logits"], "z3": raw["z3_logits"], "answer": raw["answer_logits"]}
        losses = {"z2": F.cross_entropy(output["z2"], z2[index]),
                  "z3": F.cross_entropy(output["z3"], z3[index]),
                  "answer": F.cross_entropy(output["answer"], answer[index])}
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rate = _rate(step, peak=settings["peak"], floor=settings["floor"],
                     warmup=settings["warmup"], horizon=settings["horizon"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == settings["steps"]:
            trace.append({"step": step, "learning_rate": rate, "loss": float(loss.detach())})
    return model.eval(), {"architecture": architecture, **settings, "training_rows": len(rows), "trace": trace}
