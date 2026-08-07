"""Deterministic ordinary and engineered-IIT DEV_ONLY training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import sha256_file
from .catalog import records
from .model import ModelConfig, TinyProgrammeTransformer, declared_iit_basis, tensor_state_hash
from .spec import IIT_LAYER, IIT_POSITION, K, MODEL_SEEDS, N1_POSITION, N2_POSITION, NAMESPACE, Z1_POSITION, Z2_POSITION


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1400
    batch_size: int = 256
    learning_rate: float = 0.0015
    weight_decay: float = 0.01
    log_every: int = 100
    iit_weight: float = 1.0
    z2_code_weight: float = 1.0
    n3_code_weight: float = 1.0


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def rows_to_tensors(rows) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([row.tokens for row in rows], dtype=torch.long),
        "tables": torch.tensor([row.operation_tables for row in rows], dtype=torch.float32),
        "answer": torch.tensor([row.joint_answer for row in rows], dtype=torch.long),
        "z1": torch.tensor([row.values.z1 for row in rows], dtype=torch.long),
        "z2": torch.tensor([row.values.z2 for row in rows], dtype=torch.long),
        "n1": torch.tensor([row.values.n1 for row in rows], dtype=torch.long),
        "n2": torch.tensor([row.values.n2 for row in rows], dtype=torch.long),
        "n3": torch.tensor([row.values.n3 for row in rows], dtype=torch.long),
        "d": torch.tensor([row.d for row in rows], dtype=torch.long),
        "t3_table": torch.tensor([row.programme.t3 for row in rows], dtype=torch.long),
    }


def accuracy(model: TinyProgrammeTransformer, rows, batch_size: int = 1024) -> dict:
    correct = total = 0
    by_axis: dict[str, list[int]] = {}
    by_programme: dict[str, list[int]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            x = torch.tensor([row.tokens for row in batch], dtype=torch.long)
            y = torch.tensor([row.joint_answer for row in batch], dtype=torch.long)
            tables = torch.tensor([row.operation_tables for row in batch], dtype=torch.float32)
            pred = model(x, tables).argmax(-1)
            for row, ok in zip(batch, (pred == y).tolist()):
                correct += bool(ok)
                total += 1
                by_axis.setdefault(row.axis, [0, 0])
                by_axis[row.axis][0] += bool(ok)
                by_axis[row.axis][1] += 1
                by_programme.setdefault(row.programme.programme_id, [0, 0])
                by_programme[row.programme.programme_id][0] += bool(ok)
                by_programme[row.programme.programme_id][1] += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "by_axis": {axis: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for axis, v in by_axis.items()},
        "minimum_programme_accuracy": min(v[0] / v[1] for v in by_programme.values()),
    }


def train_model(kind: str, seed: int, output_path: Path, config: TrainConfig | None = None) -> dict:
    if kind not in {"ordinary", "iit_primary", "undertrained"}:
        raise ValueError(kind)
    config = config or TrainConfig()
    if kind == "undertrained":
        config = TrainConfig(steps=20, batch_size=256, log_every=5)
    set_determinism(seed)
    model_config = ModelConfig()
    model = TinyProgrammeTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_rows = records("training")
    data = rows_to_tensors(train_rows)
    generator = torch.Generator().manual_seed(seed + 330000)
    basis = declared_iit_basis(model_config)
    curve = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        idx = torch.randint(0, len(train_rows), (config.batch_size,), generator=generator)
        tokens = data["tokens"][idx]
        tables = data["tables"][idx]
        answer = data["answer"][idx]
        logits, activations, _ = model(tokens, tables, return_activations=True)
        task_loss = nn.functional.cross_entropy(logits, answer)
        loss = task_loss
        iit_loss = torch.tensor(0.0)
        z2_code_loss = torch.tensor(0.0)
        n3_code_loss = torch.tensor(0.0)
        if kind == "iit_primary":
            source_idx = torch.randint(0, len(train_rows), (config.batch_size,), generator=generator)
            source_tokens = data["tokens"][source_idx]
            source_tables = data["tables"][source_idx]
            source_z2 = data["z2"][source_idx]
            _, source_acts, _ = model(source_tokens, source_tables, return_activations=True)
            t3_index = 2 * source_z2 + data["d"][idx]
            expected_z3 = data["t3_table"][idx].gather(1, t3_index.unsqueeze(1)).squeeze(1)
            expected_joint = expected_z3 * K + data["n3"][idx]
            cf_logits = model(
                tokens,
                tables,
                intervention={
                    "layer": IIT_LAYER,
                    "position": IIT_POSITION,
                    "source": source_acts[IIT_LAYER][:, IIT_POSITION, :],
                    "basis": basis,
                    "strength": 1.0,
                },
            )
            iit_loss = nn.functional.cross_entropy(cf_logits, expected_joint)
            hidden = activations[IIT_LAYER][:, IIT_POSITION, :]
            z2_target = data["z2"][idx]
            n3_target = data["n3"][idx]
            z1_target = data["z1"][idx]
            n1_target = data["n1"][idx]
            n2_target = data["n2"][idx]
            z2_code_loss = (
                nn.functional.cross_entropy(activations[0][:, Z1_POSITION, :K], z1_target)
                + nn.functional.cross_entropy(activations[1][:, Z2_POSITION, :K], z2_target)
                + nn.functional.cross_entropy(hidden[:, :K], z2_target)
            ) / 3
            n3_code_loss = (
                nn.functional.cross_entropy(activations[0][:, N1_POSITION, :K], n1_target)
                + nn.functional.cross_entropy(activations[1][:, N2_POSITION, :K], n2_target)
                + nn.functional.cross_entropy(activations[2][:, N2_POSITION, K : 2 * K], n3_target)
            ) / 3
            loss = (
                task_loss
                + config.iit_weight * iit_loss
                + config.z2_code_weight * z2_code_loss
                + config.n3_code_weight * n3_code_loss
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "iit_loss": float(iit_loss.detach()),
                    "z2_code_loss": float(z2_code_loss.detach()),
                    "n3_code_loss": float(n3_code_loss.detach()),
                    "batch_accuracy": float((logits.argmax(-1) == answer).float().mean()),
                }
            )
    runtime = time.perf_counter() - started
    model.eval()
    summary = {
        "schema": "exp33ae.training_summary.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "kind": kind,
        "seed": seed,
        "architecture": model_config.as_dict(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_config": asdict(config),
        "optimizer": "AdamW",
        "runtime_seconds": runtime,
        "device": "cpu",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "training_sample_accuracy": accuracy(model, train_rows[::8]),
        "model_selection_accuracy": accuracy(model, records("model_selection")),
        "curve": curve,
        "tensor_state_sha256": tensor_state_hash(model),
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "exp33ae.checkpoint.v1",
            "namespace": NAMESPACE,
            "kind": kind,
            "seed": seed,
            "model_config": model_config.as_dict(),
            "train_config": asdict(config),
            "state_dict": model.state_dict(),
            "summary": summary,
        },
        output_path,
    )
    summary["checkpoint_sha256"] = sha256_file(output_path)
    return summary


def load_checkpoint(path: Path) -> tuple[TinyProgrammeTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("namespace") != NAMESPACE:
        raise RuntimeError("refusing checkpoint outside DEV_ONLY namespace")
    model = TinyProgrammeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def checkpoint_path(root: Path, kind: str, seed: int, profile: str = "full") -> Path:
    directory = root / "checkpoints/dev_only"
    if profile == "smoke":
        directory = directory / "smoke"
    return directory / f"{kind}-seed-{seed}.pt"


def train_population(root: Path, profile: str = "full") -> list[dict]:
    if profile not in {"smoke", "full"}:
        raise ValueError(profile)
    summaries = []
    populations = (
        ("ordinary", MODEL_SEEDS["ordinary"]),
        ("iit_primary", MODEL_SEEDS["iit_primary"]),
        ("undertrained", MODEL_SEEDS["undertrained"]),
    )
    for kind, seeds in populations:
        for seed in seeds:
            config = TrainConfig(steps=80, batch_size=128, log_every=20) if profile == "smoke" and kind != "undertrained" else None
            summaries.append(train_model(kind, seed, checkpoint_path(root, kind, seed, profile), config))
    return summaries
