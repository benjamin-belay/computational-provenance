"""Prospectively declared shortcut baselines for the task-design audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import random

import torch
from torch import nn

from .spec import JOINT_CLASSES, SEQ_LEN, VOCAB_SIZE


def _majority_tables(train_rows, key_fn):
    tables = defaultdict(Counter)
    for row in train_rows:
        tables[key_fn(row)][row.joint_answer] += 1
    global_label = Counter(row.joint_answer for row in train_rows).most_common(1)[0][0]
    return tables, global_label


def table_accuracy(train_rows, eval_rows, key_fn) -> float:
    tables, global_label = _majority_tables(train_rows, key_fn)
    correct = 0
    for row in eval_rows:
        counts = tables.get(key_fn(row))
        pred = counts.most_common(1)[0][0] if counts else global_label
        correct += pred == row.joint_answer
    return correct / len(eval_rows)


def nearest_programme_accuracy(train_rows, eval_rows) -> float:
    train_programs = {row.programme.programme_id: row.programme for row in train_rows}
    eval_programs = {row.programme.programme_id: row.programme for row in eval_rows}
    nearest = {}
    for eval_id, ep in eval_programs.items():
        nearest[eval_id] = max(
            train_programs.values(),
            key=lambda tp: (sum(a == b for a, b in zip(ep.operations, tp.operations)), tp.programme_id),
        )
    from .task import interpret

    correct = 0
    for row in eval_rows:
        surrogate = nearest[row.programme.programme_id]
        pred = interpret(surrogate, row.a, row.b, row.c, row.d).joint_answer
        correct += pred == row.joint_answer
    return correct / len(eval_rows)


class ShallowPositionMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SEQ_LEN * VOCAB_SIZE + 24, hidden), nn.GELU(), nn.Linear(hidden, JOINT_CLASSES))

    def forward(self, tokens: torch.Tensor, tables: torch.Tensor) -> torch.Tensor:
        onehot = nn.functional.one_hot(tokens, VOCAB_SIZE).float().flatten(1)
        return self.net(torch.cat((onehot, tables.float().flatten(1)), dim=1))


def shallow_mlp_baseline(train_rows, evaluation_sets: dict[str, tuple], seed: int = 330501) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = ShallowPositionMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    tokens = torch.tensor([row.tokens for row in train_rows], dtype=torch.long)
    tables = torch.tensor([row.operation_tables for row in train_rows], dtype=torch.float32)
    labels = torch.tensor([row.joint_answer for row in train_rows], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed + 1)
    curve = []
    model.train()
    for step in range(1, 601):
        idx = torch.randint(0, len(train_rows), (512,), generator=generator)
        logits = model(tokens[idx], tables[idx])
        loss = nn.functional.cross_entropy(logits, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {1, 100, 200, 400, 600}:
            curve.append({"step": step, "loss": float(loss.detach())})

    def accuracy(rows) -> float:
        correct = total = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), 1024):
                batch = rows[start : start + 1024]
                x = torch.tensor([row.tokens for row in batch], dtype=torch.long)
                t = torch.tensor([row.operation_tables for row in batch], dtype=torch.float32)
                y = torch.tensor([row.joint_answer for row in batch])
                correct += int((model(x, t).argmax(-1) == y).sum())
                total += len(batch)
        return correct / total

    return {
        "schema": "exp33ae.shallow_baseline.v1",
        "namespace": "DEV_ONLY",
        "architecture": f"position-onehot+24 table bits -> Linear({SEQ_LEN * VOCAB_SIZE + 24},64) -> GELU -> Linear(64,{JOINT_CLASSES})",
        "seed": seed,
        "steps": 600,
        "curve": curve,
        "accuracy": {name: accuracy(rows) for name, rows in evaluation_sets.items()},
    }


def shortcut_suite(train_rows, evaluation_sets: dict[str, tuple]) -> dict:
    key_fns = {
        "exact_programme_input_lookup": lambda r: (r.programme.programme_id, r.a, r.b, r.c, r.d),
        "programme_only_lookup": lambda r: r.programme.programme_id,
        "target_composition_input_lookup": lambda r: (r.programme.target_composition_id, r.a, r.b, r.c, r.d),
        "parallel_composition_input_lookup": lambda r: (r.programme.parallel_composition_id, r.a, r.b, r.c, r.d),
        "input_only_lookup": lambda r: (r.a, r.b, r.c, r.d),
        "last_two_inputs_lookup": lambda r: (r.c, r.d),
        "operation_unigram_bag_lookup": lambda r: tuple(sorted(r.programme.operations)),
    }
    results = {}
    for name, rows in evaluation_sets.items():
        results[name] = {baseline: table_accuracy(train_rows, rows, key_fn) for baseline, key_fn in key_fns.items()}
        results[name]["nearest_programme_transition"] = nearest_programme_accuracy(train_rows, rows)
        results[name]["chance"] = 1 / JOINT_CLASSES
    return results
