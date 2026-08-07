#!/usr/bin/env python3
"""CSW-A0: bounded stochastic structured-prose channel development."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocol/csw-a0/CSW_A0_SPECIFICATION.json"
ARTIFACT = ROOT / "artifacts/dev_only/csw-a0"
CORPUS_DIR = ARTIFACT / "corpus"
LIFECYCLE = ARTIFACT / "lifecycle-v2"
TRAINING = ARTIFACT / "training"
DEVELOPMENT = ARTIFACT / "development"
QUALIFICATION = ARTIFACT / "qualification"
REFERENCE = ROOT / "tools/csw_a0_reference.py"

GROUPS: tuple[tuple[str, ...], ...] = (
    ("Here", "Now", "Presently", "Directly", "Briefly", "Simply", "Accordingly", "Formally"),
    ("record", "calculation", "derivation", "account", "summary", "analysis", "result", "trace"),
    ("plainly", "explicitly", "carefully", "concisely", "firmly", "notably", "precisely", "transparently"),
    ("shows", "states", "gives", "reports", "records", "yields", "lists", "presents"),
    ("that", "how", "namely", "specifically", "directly-as", "in-form", "as-value", "with-value"),
    ("indeed", "therefore", "accordingly", "thus", "thereby", "consistently", "exactly", "formally-so"),
)
OPTIONALS = (
    "<NONE>",
    "Additionally-noted",
    "Separately-recorded",
    "Explicitly-retained",
    "Carefully-preserved",
    "Directly-confirmed",
    "Formally-listed",
    "Briefly-restated",
    "Transparently-given",
)
ORDERS = (
    ("z1", "z2", "z3", "answer"),
    ("z1", "z3", "z2", "answer"),
)
FIELDS = ("z1", "z2", "z3", "answer")
VALUES = tuple(str(index) for index in range(8))
PROMPT_MARKERS = ("<BOS>", "<SEM>", "<Z1>", "<Z2>", "<Z3>", "<ANSWER>")
INTERNAL = ("<ORDER_A>", "<ORDER_B>", "<EOS>", "the", "is", ".")
ALL_TOKENS = tuple(
    dict.fromkeys(
        PROMPT_MARKERS
        + INTERNAL
        + FIELDS
        + VALUES
        + tuple(token for group in GROUPS for token in group)
        + OPTIONALS
    )
)
TO_ID = {token: index for index, token in enumerate(ALL_TOKENS)}
TO_TOKEN = {index: token for token, index in TO_ID.items()}
ELIGIBLE_TOKEN_SET = frozenset(token for group in GROUPS for token in group)
VISIBLE_OPTIONALS = frozenset(OPTIONALS[1:])
REQUIRED_PREDICATES = ("schema_valid", "ordinal_valid", "payload_valid", "mac_valid")


class A0Error(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as stream:
        for row in rows:
            line = canonical(row) + b"\n"
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def append_jsonl(stream: Any, row: dict[str, Any], digest: Any) -> None:
    line = canonical(row) + b"\n"
    stream.write(line)
    stream.flush()
    os.fsync(stream.fileno())
    digest.update(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def git_bytes(commit: str, relative: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=ROOT
    )


def semantic_object(z1: int, z2: int, z3: int) -> dict[str, Any]:
    value = {
        "schema": "exp33a.csw_a0.semantic_object.v1",
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "answer": z3,
    }
    value["semantic_id"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def all_semantics() -> list[dict[str, Any]]:
    rows = [
        semantic_object(z1, z2, z3)
        for z1 in range(8)
        for z2 in range(8)
        for z3 in range(8)
    ]
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                b"CSW_A0_PARTITION_V1\x00" + row["semantic_id"].encode("ascii")
            ).digest(),
            row["semantic_id"],
        ),
    )


def split_semantics() -> dict[str, list[dict[str, Any]]]:
    rows = all_semantics()
    return {
        "training": rows[:320],
        "development": rows[320:416],
        "qualification": rows[416:512],
    }


def deterministic_index(domain: str, semantic_id: str, variant: int, modulus: int) -> int:
    digest = hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + semantic_id.encode("ascii")
        + variant.to_bytes(4, "big")
    ).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def prompt_tokens(row: dict[str, Any]) -> list[str]:
    return [
        "<BOS>",
        "<SEM>",
        "<Z1>",
        str(row["z1"]),
        "<Z2>",
        str(row["z2"]),
        "<Z3>",
        str(row["z3"]),
        "<ANSWER>",
        str(row["answer"]),
    ]


def deterministic_target(row: dict[str, Any], variant: int) -> list[str]:
    order_index = deterministic_index("CSW_A0_ORDER", row["semantic_id"], variant, 2)
    tokens = ["<ORDER_A>" if order_index == 0 else "<ORDER_B>"]
    order = ORDERS[order_index]
    slot = 0
    for field in order:
        choices = []
        for group_index, group in enumerate(GROUPS):
            choices.append(
                group[
                    deterministic_index(
                        f"CSW_A0_GROUP_{group_index}",
                        row["semantic_id"],
                        variant * 32 + slot,
                        len(group),
                    )
                ]
            )
            slot += 1
        optional = OPTIONALS[
            deterministic_index(
                "CSW_A0_OPTIONAL", row["semantic_id"], variant * 8 + len(tokens), len(OPTIONALS)
            )
        ]
        tokens.extend(
            [
                choices[0],
                "the",
                choices[1],
                choices[2],
                choices[3],
                choices[4],
                field,
                "is",
                str(row[field]),
                choices[5],
                ".",
                optional,
            ]
        )
    tokens.append("<EOS>")
    return tokens


def prepare_corpus() -> None:
    spec = read_json(SPEC_PATH)
    splits = split_semantics()
    seen = set()
    partition_rows = []
    for name in ("training", "development", "qualification"):
        for ordinal, row in enumerate(splits[name]):
            if row["semantic_id"] in seen:
                raise A0Error("partition overlap")
            seen.add(row["semantic_id"])
            partition_rows.append({"partition": name, "ordinal": ordinal, **row})
    if len(seen) != spec["corpus"]["semantic_object_count"]:
        raise A0Error("semantic count")
    write_jsonl(CORPUS_DIR / "PARTITIONS.jsonl", partition_rows)
    training_rows = []
    for row in splits["training"]:
        for variant in range(spec["corpus"]["training_variants_per_object"]):
            training_rows.append(
                {
                    "schema": "exp33a.csw_a0.corpus_row.v1",
                    "semantic_id": row["semantic_id"],
                    "semantic": {field: row[field] for field in FIELDS},
                    "variant": variant,
                    "prompt_tokens": prompt_tokens(row),
                    "target_tokens": deterministic_target(row, variant),
                }
            )
    count, digest = write_jsonl(CORPUS_DIR / "TRAINING_CORPUS.jsonl", training_rows)
    write_json(
        CORPUS_DIR / "CORPUS_MANIFEST.json",
        {
            "schema": "exp33a.csw_a0.corpus_manifest.v1",
            "corpus_seed": spec["corpus"]["seed"],
            "semantic_objects": len(seen),
            "partition_counts": {key: len(value) for key, value in splits.items()},
            "training_rows": count,
            "training_corpus_sha256": digest,
            "partition_sha256": sha256_file(CORPUS_DIR / "PARTITIONS.jsonl"),
            "tokenizer_tokens": list(ALL_TOKENS),
            "tokenizer_sha256": hashlib.sha256(canonical(list(ALL_TOKENS))).hexdigest(),
            "authority_inputs": 0,
            "p4_objects": 0,
        },
    )


class SurfaceGRU(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(ALL_TOKENS), embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, len(ALL_TOKENS))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.embedding(token_ids))
        return self.output(states)

    def initialize(self, prompt_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.tensor([list(prompt_ids)], dtype=torch.long)
        states, hidden = self.gru(self.embedding(values))
        return hidden, self.output(states[:, -1, :])[0]

    def advance(self, hidden: torch.Tensor, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.tensor([[token_id]], dtype=torch.long)
        states, hidden = self.gru(self.embedding(values), hidden)
        return hidden, self.output(states[:, -1, :])[0]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def encode(tokens: Sequence[str]) -> list[int]:
    return [TO_ID[token] for token in tokens]


def corpus_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (CORPUS_DIR / "TRAINING_CORPUS.jsonl").read_bytes().splitlines()
    ]


def teacher_tensors(rows: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, int]:
    inputs = []
    labels = []
    for row in rows:
        prompt = encode(row["prompt_tokens"])
        target = encode(row["target_tokens"])
        inputs.append(prompt + target[:-1])
        labels.append(target)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        len(rows[0]["prompt_tokens"]),
    )


def teacher_nll(model: SurfaceGRU, rows: Sequence[dict[str, Any]], batch_size: int = 64) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs, labels, prompt_length = teacher_tensors(rows[start : start + batch_size])
            logits = model(inputs)[:, prompt_length - 1 :, :]
            loss = F.cross_entropy(
                logits.reshape(-1, len(ALL_TOKENS)), labels.reshape(-1), reduction="mean"
            )
            losses.append((float(loss), labels.numel()))
    return sum(value * count for value, count in losses) / sum(count for _, count in losses)


def load_checkpoint(path: Path) -> tuple[SurfaceGRU, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "exp33a.csw_a0.checkpoint.v1":
        raise A0Error("checkpoint schema")
    if payload["tokenizer_sha256"] != hashlib.sha256(canonical(list(ALL_TOKENS))).hexdigest():
        raise A0Error("checkpoint tokenizer")
    model = SurfaceGRU(payload["embedding_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def train(seed: int, output: Path) -> dict[str, Any]:
    spec = read_json(SPEC_PATH)
    set_deterministic(seed)
    rows = corpus_rows()
    training_ids = {row["semantic_id"] for row in rows}
    development_semantics = split_semantics()["development"]
    dev_rows = []
    for semantic in development_semantics:
        for variant in range(2):
            dev_rows.append(
                {
                    "prompt_tokens": prompt_tokens(semantic),
                    "target_tokens": deterministic_target(semantic, variant + 100),
                }
            )
    model = SurfaceGRU(
        spec["generator"]["embedding_dim"], spec["generator"]["hidden_dim"]
    )
    settings = spec["generator"]["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    rng = random.Random(seed ^ 0xA0A0)
    history = []
    best = None
    started = time.monotonic()
    for epoch in range(settings["epochs"]):
        order = list(range(len(rows)))
        rng.shuffle(order)
        epoch_losses = []
        model.train()
        for start in range(0, len(order), settings["batch_size"]):
            batch = [rows[index] for index in order[start : start + settings["batch_size"]]]
            inputs, labels, prompt_length = teacher_tensors(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)[:, prompt_length - 1 :, :]
            loss = F.cross_entropy(logits.reshape(-1, len(ALL_TOKENS)), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise A0Error("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        dev_nll = teacher_nll(model, dev_rows)
        entry = {
            "epoch": epoch + 1,
            "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
            "development_nll": dev_nll,
        }
        history.append(entry)
        if best is None or (dev_nll, epoch + 1) < (best["development_nll"], best["epoch"]):
            best = {
                **entry,
                "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
            }
        if time.monotonic() - started > settings["resource_cap_seconds"]:
            raise A0Error("training resource cap")
    assert best is not None
    model.load_state_dict(best["state_dict"])
    parser_passes = 0
    for semantic in development_semantics:
        result = generate(
            model,
            semantic,
            stream_seed=hashlib.sha256(
                b"CSW_A0_TRAIN_VALIDITY\x00" + semantic["semantic_id"].encode()
            ).digest(),
            target_key=None,
            carrier_enabled=False,
            bias=0.0,
            retain_operations=False,
        )
        parser_passes += int(result["parsed"]["valid"])
    parser_rate = parser_passes / len(development_semantics)
    validity = {
        "all_losses_finite": all(
            math.isfinite(item["mean_training_loss"]) and math.isfinite(item["development_nll"])
            for item in history
        ),
        "selected_development_nll_max": best["development_nll"]
        <= spec["generator"]["validity_gate"]["selected_development_nll_max"],
        "development_constrained_parser_success": parser_rate
        == spec["generator"]["validity_gate"]["development_constrained_parser_success"],
        "training_loss_decreased": history[-1]["mean_training_loss"]
        < history[0]["mean_training_loss"],
    }
    payload = {
        "schema": "exp33a.csw_a0.checkpoint.v1",
        "seed": seed,
        "embedding_dim": spec["generator"]["embedding_dim"],
        "hidden_dim": spec["generator"]["hidden_dim"],
        "tokenizer_sha256": hashlib.sha256(canonical(list(ALL_TOKENS))).hexdigest(),
        "corpus_sha256": sha256_file(CORPUS_DIR / "TRAINING_CORPUS.jsonl"),
        "selected_epoch": best["epoch"],
        "development_nll": best["development_nll"],
        "state_dict": best["state_dict"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    result = {
        "schema": "exp33a.csw_a0.training_attempt.v1",
        "seed": seed,
        "training_semantic_objects": len(training_ids),
        "epochs": history,
        "selected_epoch": best["epoch"],
        "selected_development_nll": best["development_nll"],
        "development_parser_rate": parser_rate,
        "validity_predicates": validity,
        "training_valid": all(validity.values()),
        "checkpoint_sha256": sha256_file(output),
        "elapsed_seconds": time.monotonic() - started,
        "authority_inputs": 0,
        "p4_objects": 0,
    }
    write_json(output.with_suffix(".json"), result)
    return result


def counter_uniform(stream_seed: bytes, step: int, model_history: Sequence[str]) -> float:
    digest = hmac.new(
        stream_seed,
        b"CSW_A0_COUNTER_UNIFORM_V1\x00"
        + step.to_bytes(8, "big")
        + hashlib.sha256(canonical(list(model_history))).digest(),
        hashlib.sha256,
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / (1 << 64)


def green_set(
    authority_key: bytes, eligible_position: int, public_history: Sequence[str], candidates: Sequence[str]
) -> set[str]:
    history_digest = hashlib.sha256(canonical(list(public_history))).digest()
    ranked = sorted(
        candidates,
        key=lambda token: (
            hmac.new(
                authority_key,
                b"CSW_A0_SYNTHETIC_AUTHORITY_GREENLIST_V1\x00"
                + eligible_position.to_bytes(4, "big")
                + history_digest
                + token.encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            token,
        ),
    )
    return set(ranked[: len(ranked) // 2])


def sample_ids(
    logits: torch.Tensor,
    candidates: Sequence[str],
    stream_seed: bytes,
    step: int,
    model_history: Sequence[str],
    *,
    green: set[str] | None,
    bias: float,
) -> tuple[str, dict[str, Any]]:
    ids = [TO_ID[token] for token in candidates]
    pre = torch.tensor([float(logits[index]) for index in ids], dtype=torch.float64)
    post = pre.clone()
    if green is not None:
        for index, token in enumerate(candidates):
            if token in green:
                post[index] += bias
    probabilities = torch.softmax(post, dim=0)
    pre_probabilities = torch.softmax(pre, dim=0)
    uniform = counter_uniform(stream_seed, step, model_history)
    cumulative = 0.0
    selected = len(candidates) - 1
    for index, probability in enumerate(probabilities.tolist()):
        cumulative += probability
        if uniform <= cumulative:
            selected = index
            break
    selected_token = candidates[selected]
    entropy = float(-(pre_probabilities * torch.log(pre_probabilities)).sum())
    kl = float(
        (
            probabilities
            * (torch.log(probabilities) - torch.log(pre_probabilities))
        ).sum()
    )
    trace = {
        "candidates": list(candidates),
        "pre_logits": pre.tolist(),
        "pre_probabilities": pre_probabilities.tolist(),
        "post_probabilities": probabilities.tolist(),
        "uniform": uniform,
        "selected_token": selected_token,
        "entropy_nats": entropy,
        "effective_support": math.exp(entropy),
        "max_probability": float(pre_probabilities.max()),
        "green_tokens": sorted(green) if green is not None else [],
        "green_mass": (
            sum(
                float(pre_probabilities[index])
                for index, token in enumerate(candidates)
                if token in green
            )
            if green is not None
            else None
        ),
        "red_mass": (
            sum(
                float(pre_probabilities[index])
                for index, token in enumerate(candidates)
                if token not in green
            )
            if green is not None
            else None
        ),
        "kl_nats": kl,
    }
    return selected_token, trace


def generate(
    model: SurfaceGRU,
    semantic: dict[str, Any],
    *,
    stream_seed: bytes,
    target_key: bytes | None,
    carrier_enabled: bool,
    bias: float,
    retain_operations: bool,
) -> dict[str, Any]:
    prompt = prompt_tokens(semantic)
    hidden, logits = model.initialize(encode(prompt))
    model_history = list(prompt)
    public_tokens: list[str] = []
    operations: list[dict[str, Any]] = []
    step = 0
    eligible_position = 0
    order_token, order_trace = sample_ids(
        logits,
        ("<ORDER_A>", "<ORDER_B>"),
        stream_seed,
        step,
        model_history,
        green=None,
        bias=0.0,
    )
    order = ORDERS[0 if order_token == "<ORDER_A>" else 1]
    if retain_operations:
        operations.append(
            {"kind": "ORDER", "model_step": step, "public": False, **order_trace}
        )
    model_history.append(order_token)
    hidden, logits = model.advance(hidden, TO_ID[order_token])
    step += 1
    for field in order:
        sequence: list[tuple[str, Sequence[str] | str]] = [
            ("eligible", GROUPS[0]),
            ("mandatory", "the"),
            ("eligible", GROUPS[1]),
            ("eligible", GROUPS[2]),
            ("eligible", GROUPS[3]),
            ("eligible", GROUPS[4]),
            ("mandatory", field),
            ("mandatory", "is"),
            ("mandatory", str(semantic[field])),
            ("eligible", GROUPS[5]),
            ("mandatory", "."),
            ("optional", OPTIONALS),
        ]
        for kind, allowed in sequence:
            candidates = (allowed,) if isinstance(allowed, str) else tuple(allowed)
            green = None
            if kind == "eligible" and carrier_enabled:
                if target_key is None:
                    raise A0Error("missing carrier key")
                green = green_set(target_key, eligible_position, public_tokens, candidates)
            selected, trace = sample_ids(
                logits,
                candidates,
                stream_seed,
                step,
                model_history,
                green=green,
                bias=bias if kind == "eligible" and carrier_enabled else 0.0,
            )
            if kind == "eligible":
                trace["eligible_position"] = eligible_position
                trace["public_history_before"] = list(public_tokens)
                trace["carrier_enabled"] = carrier_enabled
                eligible_position += 1
            if retain_operations:
                operations.append(
                    {
                        "kind": kind.upper(),
                        "model_step": step,
                        "field": field,
                        "public": selected not in ("<NONE>", "<ORDER_A>", "<ORDER_B>"),
                        **trace,
                    }
                )
            model_history.append(selected)
            if selected != "<NONE>":
                public_tokens.append(selected)
            hidden, logits = model.advance(hidden, TO_ID[selected])
            step += 1
    parsed = parse_output(public_tokens)
    return {
        "schema": "exp33a.csw_a0.generated_output.v1",
        "semantic_id": semantic["semantic_id"],
        "public_tokens": public_tokens,
        "literal_output": " ".join(public_tokens),
        "output_sha256": hashlib.sha256(" ".join(public_tokens).encode("utf-8")).hexdigest(),
        "parsed": parsed,
        "eligible_positions": eligible_position,
        "stochastic_draws": step,
        "operations": operations,
    }


def parse_output(tokens_or_text: Sequence[str] | str) -> dict[str, Any]:
    tokens = tokens_or_text.split() if isinstance(tokens_or_text, str) else list(tokens_or_text)
    cursor = 0
    values: dict[str, int] = {}
    order = []
    eligible = []
    public_history: list[str] = []
    try:
        for clause in range(4):
            selected = []
            for expected_kind in range(6):
                if expected_kind == 0:
                    token = tokens[cursor]
                    if token not in GROUPS[0]:
                        raise ValueError("group0")
                    cursor += 1
                    selected.append((0, token, list(public_history)))
                    public_history.append(token)
                    if tokens[cursor] != "the":
                        raise ValueError("the")
                    public_history.append(tokens[cursor])
                    cursor += 1
                else:
                    group_index = expected_kind
                    token = tokens[cursor]
                    if token not in GROUPS[group_index]:
                        raise ValueError(f"group{group_index}")
                    cursor += 1
                    selected.append((group_index, token, list(public_history)))
                    public_history.append(token)
                    if group_index == 4:
                        field = tokens[cursor]
                        if field not in FIELDS or field in values:
                            raise ValueError("field")
                        order.append(field)
                        public_history.append(field)
                        cursor += 1
                        if tokens[cursor] != "is":
                            raise ValueError("is")
                        public_history.append("is")
                        cursor += 1
                        value = tokens[cursor]
                        if value not in VALUES:
                            raise ValueError("value")
                        values[field] = int(value)
                        public_history.append(value)
                        cursor += 1
            if tokens[cursor] != ".":
                raise ValueError("period")
            public_history.append(".")
            cursor += 1
            if cursor < len(tokens) and tokens[cursor] in VISIBLE_OPTIONALS:
                public_history.append(tokens[cursor])
                cursor += 1
            for group_index, token, history in selected:
                eligible.append(
                    {
                        "eligible_position": len(eligible),
                        "group_index": group_index,
                        "selected_token": token,
                        "public_history_before": history,
                    }
                )
        if cursor != len(tokens):
            raise ValueError("trailing")
        if tuple(order) not in ORDERS:
            raise ValueError("order")
        if set(values) != set(FIELDS):
            raise ValueError("fields")
        canonical_values = {field: values[field] for field in FIELDS}
        if canonical_values["answer"] != canonical_values["z3"]:
            raise ValueError("answer")
        return {
            "valid": True,
            "values": canonical_values,
            "presentation_order": order,
            "canonical_order": list(FIELDS),
            "eligible": eligible,
            "eligible_count": len(eligible),
            "contradiction": False,
        }
    except (IndexError, ValueError) as exc:
        return {
            "valid": False,
            "reason": str(exc),
            "eligible": eligible,
            "eligible_count": len(eligible),
        }


def detector(output: str, authority_key: bytes, threshold: int, minimum: int = 24) -> dict[str, Any]:
    parsed = parse_output(output)
    if not parsed["valid"] or parsed["eligible_count"] < minimum:
        return {
            "decision": "ABSTAIN",
            "score": None,
            "eligible_count": parsed["eligible_count"],
            "parse_valid": parsed["valid"],
        }
    score = 0
    for item in parsed["eligible"]:
        candidates = GROUPS[item["group_index"]]
        green = green_set(
            authority_key,
            item["eligible_position"],
            item["public_history_before"],
            candidates,
        )
        score += int(item["selected_token"] in green)
    return {
        "decision": "ACCEPT" if score >= threshold else "REJECT",
        "score": score,
        "eligible_count": parsed["eligible_count"],
        "parse_valid": True,
    }


def authority_keys(master: bytes, family: str) -> dict[str, bytes]:
    return {
        f"A{index}": hmac.new(
            master,
            b"CSW_A0_AUTHORITY_KEY_V1\x00" + family.encode("ascii") + bytes([index]),
            hashlib.sha256,
        ).digest()
        for index in range(8)
    }


def stream_seed(domain: str, semantic_id: str, draw: int) -> bytes:
    return hashlib.sha256(
        b"CSW_A0_STREAM_V1\x00"
        + domain.encode("ascii")
        + b"\x00"
        + semantic_id.encode("ascii")
        + draw.to_bytes(4, "big")
    ).digest()


def assigned_authority(semantic_id: str, draw: int, family: str = "STATE") -> str:
    value = hashlib.sha256(
        b"CSW_A0_ASSIGNED_AUTHORITY_V1\x00"
        + family.encode("ascii")
        + b"\x00"
        + semantic_id.encode("ascii")
        + draw.to_bytes(4, "big")
    ).digest()[0] % 8
    return f"A{value}"


def semantic_match(parsed: dict[str, Any], semantic: dict[str, Any]) -> bool:
    return parsed["valid"] and parsed["values"] == {field: semantic[field] for field in FIELDS}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def development_calibration(checkpoint: Path) -> dict[str, Any]:
    spec = read_json(SPEC_PATH)
    model, payload = load_checkpoint(checkpoint)
    semantics = split_semantics()["development"]
    master = hashlib.sha256(b"CSW_A0_DEVELOPMENT_MASTER_V1").digest()
    keys = authority_keys(master, "STATE")
    output_path = DEVELOPMENT / "RAW_CALIBRATION.jsonl"
    operation_path = DEVELOPMENT / "OPERATION_LEDGER.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    operation_digest = hashlib.sha256()
    output_count = 0
    operation_count = 0
    settings: dict[str, list[dict[str, Any]]] = {"OFF": []}
    for bias in spec["channel"]["bias_grid"]:
        settings[str(bias)] = []
    with output_path.open("wb") as output_stream, operation_path.open("wb") as operation_stream:
        for semantic in semantics:
            for draw in range(spec["development_design"]["draws_per_context_per_condition"]):
                seed = stream_seed("DEVELOPMENT", semantic["semantic_id"], draw)
                target_id = assigned_authority(semantic["semantic_id"], draw)
                conditions = [("OFF", 0.0, None)] + [
                    (str(bias), float(bias), keys[target_id])
                    for bias in spec["channel"]["bias_grid"]
                ]
                for condition, bias, key in conditions:
                    generated = generate(
                        model,
                        semantic,
                        stream_seed=seed,
                        target_key=key,
                        carrier_enabled=condition != "OFF",
                        bias=bias,
                        retain_operations=True,
                    )
                    scores = {
                        authority: detector(generated["literal_output"], authority_key, 0)["score"]
                        for authority, authority_key in keys.items()
                    }
                    row = {
                        "schema": "exp33a.csw_a0.calibration_output.v1",
                        "ordinal": output_count,
                        "condition": condition,
                        "bias": bias,
                        "semantic": {field: semantic[field] for field in FIELDS},
                        "semantic_id": semantic["semantic_id"],
                        "draw": draw,
                        "target_authority": target_id,
                        "literal_output": generated["literal_output"],
                        "output_sha256": generated["output_sha256"],
                        "scores": scores,
                        "semantic_valid": semantic_match(generated["parsed"], semantic),
                        "eligible_count": generated["eligible_positions"],
                    }
                    append_jsonl(output_stream, row, output_digest)
                    setting_entry = {
                        **row,
                        "operations": generated["operations"],
                    }
                    settings[condition].append(setting_entry)
                    for operation in generated["operations"]:
                        op = {
                            "schema": "exp33a.csw_a0.calibration_operation.v1",
                            "output_ordinal": output_count,
                            "operation_ordinal": operation_count,
                            **operation,
                        }
                        append_jsonl(operation_stream, op, operation_digest)
                        operation_count += 1
                    output_count += 1
    off_maxima = [max(row["scores"].values()) for row in settings["OFF"]]
    threshold = None
    for candidate in range(0, 25):
        rate = sum(value >= candidate for value in off_maxima) / len(off_maxima)
        if rate <= spec["statistical_gates"]["nominal_family_wise_false_positive_max"]:
            threshold = candidate
            break
    if threshold is None:
        threshold = 25
    summaries = []
    selected = None
    for bias in spec["channel"]["bias_grid"]:
        rows = settings[str(bias)]
        accepted = 0
        semantics_valid = 0
        eligible = []
        entropy = []
        support = []
        max_probability = []
        green_mass = []
        red_mass = []
        kl = []
        margins = []
        for row in rows:
            target = row["target_authority"]
            correct = row["scores"][target]
            wrong = max(value for key, value in row["scores"].items() if key != target)
            accepted += int(correct >= threshold and wrong < threshold)
            semantics_valid += int(row["semantic_valid"])
            eligible.append(row["eligible_count"])
            margins.append(correct - wrong)
            for operation in row["operations"]:
                if operation["kind"] == "ELIGIBLE":
                    entropy.append(operation["entropy_nats"])
                    support.append(operation["effective_support"])
                    max_probability.append(operation["max_probability"])
                    green_mass.append(operation["green_mass"])
                    red_mass.append(operation["red_mass"])
                    kl.append(operation["kl_nats"])
        entropy_gate = sum(
            ent >= spec["architecture_contract"]["conditional_entropy_nats_min"]
            and sup >= spec["architecture_contract"]["effective_support_min"]
            and maximum <= spec["architecture_contract"]["max_token_probability_max"]
            and gm >= spec["architecture_contract"]["green_probability_mass_min"]
            and rm >= spec["architecture_contract"]["red_probability_mass_min"]
            for ent, sup, maximum, gm, rm in zip(
                entropy, support, max_probability, green_mass, red_mass
            )
        ) / len(entropy)
        summary = {
            "bias": bias,
            "outputs": len(rows),
            "power": accepted / len(rows),
            "semantic_rate": semantics_valid / len(rows),
            "eligible_min": min(eligible),
            "eligible_mean": sum(eligible) / len(eligible),
            "entropy_gate_fraction": entropy_gate,
            "entropy_mean": sum(entropy) / len(entropy),
            "effective_support_mean": sum(support) / len(support),
            "max_probability_worst": max(max_probability),
            "green_mass_min": min(green_mass),
            "red_mass_min": min(red_mass),
            "mean_kl_nats": sum(kl) / len(kl),
            "mean_margin": sum(margins) / len(margins),
        }
        summary["passes"] = (
            summary["power"] >= spec["statistical_gates"]["matching_authority_per_output_power_min"]
            and summary["semantic_rate"] == spec["semantic_gates"]["parser_success"]
            and summary["eligible_min"] >= spec["architecture_contract"]["public_eligible_positions_min"]
            and summary["entropy_gate_fraction"]
            >= spec["architecture_contract"]["positions_with_entropy_gate_fraction_min"]
            and summary["mean_kl_nats"] <= spec["semantic_gates"]["mean_eligible_kl_nats_max"]
            and summary["mean_margin"] >= spec["statistical_gates"]["paired_margin_mean_min"]
        )
        summaries.append(summary)
        if selected is None and summary["passes"]:
            selected = summary
    result = {
        "schema": "exp33a.csw_a0.calibration_results.v1",
        "checkpoint_sha256": sha256_file(checkpoint),
        "outputs": output_count,
        "operations": operation_count,
        "raw_outputs_sha256": output_digest.hexdigest(),
        "operation_ledger_sha256": operation_digest.hexdigest(),
        "off_outputs": len(settings["OFF"]),
        "empirical_threshold": threshold,
        "off_empirical_fwer": sum(value >= threshold for value in off_maxima) / len(off_maxima),
        "bias_results": summaries,
        "selected": selected,
        "qualification_accessed": False,
    }
    write_json(DEVELOPMENT / "CALIBRATION_RESULTS.json", result)
    if selected is not None:
        write_json(
            DEVELOPMENT / "FROZEN_CHANNEL.json",
            {
                "schema": "exp33a.csw_a0.frozen_channel.v1",
                "checkpoint_sha256": sha256_file(checkpoint),
                "bias": selected["bias"],
                "threshold": threshold,
                "minimum_eligible_length": spec["channel"]["minimum_eligible_length"],
                "authority_cardinality": spec["channel"]["authority_cardinality"],
                "green_fraction": spec["channel"]["green_fraction"],
                "qualification_manifest": spec["qualification_design"],
                "successor_authorized": False,
            },
        )
    return result


def strict_required_booleans(row: dict[str, Any]) -> bool:
    for name in REQUIRED_PREDICATES:
        if name not in row or type(row[name]) is not bool or row[name] is not True:
            return False
    return True


def init_lifecycle(private_path: Path) -> None:
    if private_path.exists():
        raise A0Error("lifecycle private exists")
    key = os.urandom(32)
    commitment = hashlib.sha256(b"CSW_A0_LIFECYCLE_V2_COMMIT\x00" + key).hexdigest()
    write_json(
        LIFECYCLE / "SPECIFICATION.json",
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_specification.v1",
            "namespace": "CSW_A0_LIFECYCLE_V2_TEST_ONLY",
            "rows": 4,
            "required_boolean_predicates": list(REQUIRED_PREDICATES),
            "nonboolean_truth_aggregation_forbidden": True,
            "scientific": False,
        },
    )
    write_json(
        LIFECYCLE / "AUTHORIZATION.json",
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_authorization.v1",
            "namespace": "CSW_A0_LIFECYCLE_V2_TEST_ONLY",
            "key_commitment_sha256": commitment,
            "scientific": False,
        },
    )
    write_json(
        private_path,
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_private.v1",
            "key_hex": key.hex(),
        },
    )


def lifecycle_raw(source_commit: str, private_path: Path) -> None:
    if git("rev-parse", "HEAD") != source_commit:
        raise A0Error("source commit")
    key = bytes.fromhex(read_json(private_path)["key_hex"])
    authorization = read_json(LIFECYCLE / "AUTHORIZATION.json")
    if authorization["key_commitment_sha256"] != hashlib.sha256(
        b"CSW_A0_LIFECYCLE_V2_COMMIT\x00" + key
    ).hexdigest():
        raise A0Error("lifecycle commitment")
    raw_rows = []
    operations = []
    for ordinal in range(4):
        payload = f"CSW-A0 lifecycle-v2 row {ordinal}".encode("ascii")
        body = {
            "ordinal": ordinal,
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        }
        raw_rows.append(
            {
                "schema": "exp33a.csw_a0.lifecycle_v2_raw.v1",
                **body,
                "mac_sha256": hmac.new(
                    key,
                    b"CSW_A0_LIFECYCLE_V2_RAW\x00" + canonical(body),
                    hashlib.sha256,
                ).hexdigest(),
            }
        )
        operations.append(
            {
                "schema": "exp33a.csw_a0.lifecycle_v2_operation.v1",
                "ordinal": ordinal,
                "completed": True,
            }
        )
    raw_count, raw_digest = write_jsonl(LIFECYCLE / "RAW_OUTPUTS.jsonl", raw_rows)
    operation_count, operation_digest = write_jsonl(
        LIFECYCLE / "OPERATION_LEDGER.jsonl", operations
    )
    write_json(
        LIFECYCLE / "RAW_MANIFEST.json",
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_raw_manifest.v1",
            "source_commit": source_commit,
            "raw_count": raw_count,
            "operation_count": operation_count,
            "raw_sha256": raw_digest,
            "operation_sha256": operation_digest,
            "key_reveal_present": False,
        },
    )


def lifecycle_reveal(raw_commit: str, private_path: Path) -> None:
    relative = (LIFECYCLE.relative_to(ROOT) / "KEY_REVEAL.json").as_posix()
    if relative in git("ls-tree", "-r", "--name-only", raw_commit).splitlines():
        raise A0Error("reveal in raw")
    write_json(
        LIFECYCLE / "KEY_REVEAL.json",
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_reveal.v1",
            "raw_commit": raw_commit,
            "key_hex": read_json(private_path)["key_hex"],
            "scientific": False,
        },
    )


def production_lifecycle(raw_commit: str, reveal_commit: str) -> dict[str, Any]:
    base = LIFECYCLE.relative_to(ROOT)
    authorization = json.loads(git_bytes(raw_commit, base / "AUTHORIZATION.json"))
    reveal = json.loads(git_bytes(reveal_commit, base / "KEY_REVEAL.json"))
    key = bytes.fromhex(reveal["key_hex"])
    authority_valid = (
        reveal["raw_commit"] == raw_commit
        and authorization["key_commitment_sha256"]
        == hashlib.sha256(b"CSW_A0_LIFECYCLE_V2_COMMIT\x00" + key).hexdigest()
    )
    rows = [
        json.loads(line)
        for line in git_bytes(raw_commit, base / "RAW_OUTPUTS.jsonl").splitlines()
    ]
    results = []
    for ordinal, row in enumerate(rows):
        body = {"ordinal": ordinal, "payload_base64": row.get("payload_base64")}
        expected = hmac.new(
            key,
            b"CSW_A0_LIFECYCLE_V2_RAW\x00" + canonical(body),
            hashlib.sha256,
        ).hexdigest()
        predicates = {
            "schema_valid": set(row)
            == {"schema", "ordinal", "payload_base64", "mac_sha256"},
            "ordinal_valid": row.get("ordinal") == ordinal,
            "payload_valid": isinstance(row.get("payload_base64"), str),
            "mac_valid": isinstance(row.get("mac_sha256"), str)
            and hmac.compare_digest(row["mac_sha256"], expected),
        }
        results.append({"ordinal": ordinal, **predicates, "passed": strict_required_booleans(predicates)})
    passed = authority_valid and len(rows) == 4 and all(row["passed"] for row in results)
    return {
        "schema": "exp33a.csw_a0.lifecycle_v2_production.v1",
        "raw_commit": raw_commit,
        "reveal_commit": reveal_commit,
        "authority_valid": authority_valid,
        "required_predicates": list(REQUIRED_PREDICATES),
        "rows": results,
        "passed": passed,
    }


def lifecycle_evaluate(raw_commit: str, reveal_commit: str) -> None:
    production = production_lifecycle(raw_commit, reveal_commit)
    write_json(LIFECYCLE / "PRODUCTION_RESULTS.json", production)
    subprocess.run(
        [
            sys.executable,
            str(REFERENCE),
            "lifecycle",
            "--raw-commit",
            raw_commit,
            "--reveal-commit",
            reveal_commit,
            "--output",
            str(LIFECYCLE / "REFERENCE_RESULTS.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    reference = read_json(LIFECYCLE / "REFERENCE_RESULTS.json")
    fields = ("raw_commit", "reveal_commit", "authority_valid", "required_predicates", "rows", "passed")
    agreement = all(production[field] == reference[field] for field in fields)
    write_json(
        LIFECYCLE / "TERMINAL.json",
        {
            "schema": "exp33a.csw_a0.lifecycle_v2_terminal.v1",
            "terminal": (
                "CSW_A0_LIFECYCLE_V2_READY"
                if production["passed"] and reference["passed"] and agreement
                else "CSW_A0_LIFECYCLE_NOT_READY"
            ),
            "production_reference_agreement": agreement,
            "raw_commit": raw_commit,
            "reveal_commit": reveal_commit,
            "scientific": False,
        },
    )


def initialize_qualification(private_path: Path, checkpoint: Path) -> None:
    if private_path.exists():
        raise A0Error("qualification private exists")
    frozen = read_json(DEVELOPMENT / "FROZEN_CHANNEL.json")
    master = os.urandom(32)
    family_masters = {
        family: hmac.new(
            master, b"CSW_A0_QUALIFICATION_FAMILY\x00" + family.encode(), hashlib.sha256
        ).digest()
        for family in ("STATE", "SEMANTIC_OBJECT", "ANSWER_DERIVED", "GENERIC_ORIGIN")
    }
    commitments = {
        family: hashlib.sha256(
            b"CSW_A0_QUALIFICATION_COMMIT\x00" + family.encode() + key
        ).hexdigest()
        for family, key in family_masters.items()
    }
    run_id = hashlib.sha256(
        b"CSW_A0_QUALIFICATION_RUN\x00" + os.urandom(32)
    ).hexdigest()
    write_json(
        QUALIFICATION / "AUTHORIZATION.json",
        {
            "schema": "exp33a.csw_a0.qualification_authorization.v1",
            "namespace": "CSW_A0_UNTOUCHED_QUALIFICATION_TEST_ONLY",
            "run_id": run_id,
            "family_commitments": commitments,
            "checkpoint_sha256": sha256_file(checkpoint),
            "frozen_channel_sha256": sha256_file(DEVELOPMENT / "FROZEN_CHANNEL.json"),
            "qualification_semantic_ids_sha256": hashlib.sha256(
                canonical([row["semantic_id"] for row in split_semantics()["qualification"]])
            ).hexdigest(),
            "planned_outputs": read_json(SPEC_PATH)["qualification_design"]["total_raw_outputs"],
            "scientific": False,
            "successor_authorized": False,
        },
    )
    write_json(
        QUALIFICATION / "MANIFEST.json",
        {
            "schema": "exp33a.csw_a0.qualification_manifest.v1",
            "semantic_ids": [row["semantic_id"] for row in split_semantics()["qualification"]],
            "semantic_count": 96,
            "watermarked_draws_per_context": 4,
            "carrier_off_draws_per_context": 4,
            "baseline_draws_per_context_per_family": 1,
            "condition_order": [
                "STATE",
                "OFF",
                "SEMANTIC_OBJECT",
                "ANSWER_DERIVED",
                "GENERIC_ORIGIN",
            ],
            "no_retry": True,
        },
    )
    write_json(
        private_path,
        {
            "schema": "exp33a.csw_a0.qualification_private.v1",
            "run_id": run_id,
            "family_master_hex": {
                family: key.hex() for family, key in family_masters.items()
            },
        },
    )


def baseline_authority(semantic: dict[str, Any], family: str) -> str:
    if family == "SEMANTIC_OBJECT":
        return f"A{int(semantic['semantic_id'][:2], 16) % 8}"
    if family == "ANSWER_DERIVED":
        return f"A{semantic['answer']}"
    if family == "GENERIC_ORIGIN":
        return "A0"
    raise A0Error("baseline family")


def qualification_raw(
    authorization_commit: str, private_path: Path, checkpoint: Path
) -> None:
    if git("rev-parse", "HEAD") != authorization_commit:
        raise A0Error("authorization commit")
    authorization = read_json(QUALIFICATION / "AUTHORIZATION.json")
    private = read_json(private_path)
    if private["run_id"] != authorization["run_id"]:
        raise A0Error("qualification run")
    family_masters = {
        family: bytes.fromhex(value)
        for family, value in private["family_master_hex"].items()
    }
    for family, key in family_masters.items():
        commitment = hashlib.sha256(
            b"CSW_A0_QUALIFICATION_COMMIT\x00" + family.encode() + key
        ).hexdigest()
        if commitment != authorization["family_commitments"][family]:
            raise A0Error("qualification commitment")
    model, _ = load_checkpoint(checkpoint)
    frozen = read_json(DEVELOPMENT / "FROZEN_CHANNEL.json")
    spec = read_json(SPEC_PATH)
    semantics = split_semantics()["qualification"]
    output_path = QUALIFICATION / "RAW_OUTPUTS.jsonl"
    operation_path = QUALIFICATION / "OPERATION_LEDGER.jsonl"
    output_digest = hashlib.sha256()
    operation_digest = hashlib.sha256()
    output_ordinal = 0
    operation_ordinal = 0
    started = 0
    completed = 0
    with output_path.open("wb") as out, operation_path.open("wb") as ops:
        for semantic in semantics:
            state_keys = authority_keys(family_masters["STATE"], "STATE")
            for condition in ("STATE", "OFF"):
                draws = (
                    spec["qualification_design"]["watermarked_draws_per_context"]
                    if condition == "STATE"
                    else spec["qualification_design"]["carrier_off_draws_per_context"]
                )
                for draw in range(draws):
                    started += 1
                    seed = stream_seed("QUALIFICATION", semantic["semantic_id"], draw)
                    target_id = assigned_authority(semantic["semantic_id"], draw)
                    enabled = condition == "STATE"
                    generated = generate(
                        model,
                        semantic,
                        stream_seed=seed,
                        target_key=state_keys[target_id] if enabled else None,
                        carrier_enabled=enabled,
                        bias=frozen["bias"] if enabled else 0.0,
                        retain_operations=True,
                    )
                    invariant_digests = []
                    if condition == "OFF":
                        for authority_index in range(8):
                            repeat = generate(
                                model,
                                semantic,
                                stream_seed=seed,
                                target_key=state_keys[f"A{authority_index}"],
                                carrier_enabled=False,
                                bias=0.0,
                                retain_operations=False,
                            )
                            invariant_digests.append(repeat["output_sha256"])
                    row = {
                        "schema": "exp33a.csw_a0.qualification_raw_output.v1",
                        "ordinal": output_ordinal,
                        "run_id": authorization["run_id"],
                        "condition": condition,
                        "family": "STATE",
                        "semantic": {field: semantic[field] for field in FIELDS},
                        "semantic_id": semantic["semantic_id"],
                        "draw": draw,
                        "target_authority": target_id,
                        "stream_seed_hex": seed.hex(),
                        "literal_output": generated["literal_output"],
                        "output_sha256": generated["output_sha256"],
                        "invariant_digests": invariant_digests,
                    }
                    append_jsonl(out, row, output_digest)
                    for operation in generated["operations"]:
                        op = {
                            "schema": "exp33a.csw_a0.qualification_operation.v1",
                            "output_ordinal": output_ordinal,
                            "operation_ordinal": operation_ordinal,
                            **operation,
                        }
                        append_jsonl(ops, op, operation_digest)
                        operation_ordinal += 1
                    output_ordinal += 1
                    completed += 1
            for family_index, family in enumerate(
                ("SEMANTIC_OBJECT", "ANSWER_DERIVED", "GENERIC_ORIGIN")
            ):
                started += 1
                seed = stream_seed(
                    f"QUALIFICATION_BASELINE_{family}", semantic["semantic_id"], 0
                )
                target_id = baseline_authority(semantic, family)
                keys = authority_keys(family_masters[family], family)
                generated = generate(
                    model,
                    semantic,
                    stream_seed=seed,
                    target_key=keys[target_id],
                    carrier_enabled=True,
                    bias=frozen["bias"],
                    retain_operations=True,
                )
                row = {
                    "schema": "exp33a.csw_a0.qualification_raw_output.v1",
                    "ordinal": output_ordinal,
                    "run_id": authorization["run_id"],
                    "condition": "BASELINE",
                    "family": family,
                    "semantic": {field: semantic[field] for field in FIELDS},
                    "semantic_id": semantic["semantic_id"],
                    "draw": 0,
                    "target_authority": target_id,
                    "stream_seed_hex": seed.hex(),
                    "literal_output": generated["literal_output"],
                    "output_sha256": generated["output_sha256"],
                    "invariant_digests": [],
                }
                append_jsonl(out, row, output_digest)
                for operation in generated["operations"]:
                    op = {
                        "schema": "exp33a.csw_a0.qualification_operation.v1",
                        "output_ordinal": output_ordinal,
                        "operation_ordinal": operation_ordinal,
                        **operation,
                    }
                    append_jsonl(ops, op, operation_digest)
                    operation_ordinal += 1
                output_ordinal += 1
                completed += 1
    if output_ordinal != spec["qualification_design"]["total_raw_outputs"]:
        raise A0Error("qualification output denominator")
    write_json(
        QUALIFICATION / "RAW_MANIFEST.json",
        {
            "schema": "exp33a.csw_a0.qualification_raw_manifest.v1",
            "authorization_commit": authorization_commit,
            "planned": spec["qualification_design"]["total_raw_outputs"],
            "started": started,
            "completed": completed,
            "failed": started - completed,
            "unrun": spec["qualification_design"]["total_raw_outputs"] - started,
            "outputs_sha256": output_digest.hexdigest(),
            "operations": operation_ordinal,
            "operations_sha256": operation_digest.hexdigest(),
            "reveal_present": False,
            "scientific": False,
        },
    )


def qualification_reveal(raw_commit: str, private_path: Path) -> None:
    relative = (QUALIFICATION.relative_to(ROOT) / "KEY_REVEAL.json").as_posix()
    if relative in git("ls-tree", "-r", "--name-only", raw_commit).splitlines():
        raise A0Error("qualification reveal in raw")
    private = read_json(private_path)
    write_json(
        QUALIFICATION / "KEY_REVEAL.json",
        {
            "schema": "exp33a.csw_a0.qualification_reveal.v1",
            "raw_commit": raw_commit,
            "run_id": private["run_id"],
            "family_master_hex": private["family_master_hex"],
            "scientific": False,
        },
    )


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocab = sorted(ELIGIBLE_TOKEN_SET)
    unigram_index = {token: index for index, token in enumerate(vocab)}
    bigrams = sorted(
        {
            (tokens[index], tokens[index + 1])
            for row in rows
            for tokens in [[token for token in row["literal_output"].split() if token in ELIGIBLE_TOKEN_SET]]
            for index in range(len(tokens) - 1)
        }
    )
    bigram_index = {pair: len(vocab) + index for index, pair in enumerate(bigrams)}
    matrix = np.zeros((len(rows), len(vocab) + len(bigrams) + 1), dtype=np.float64)
    labels = np.zeros(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        tokens = [token for token in row["literal_output"].split() if token in ELIGIBLE_TOKEN_SET]
        for token in tokens:
            matrix[row_index, unigram_index[token]] += 1
        for pair in zip(tokens, tokens[1:]):
            matrix[row_index, bigram_index[pair]] += 1
        matrix[row_index, -1] = len(row["literal_output"].split())
        labels[row_index] = int(row["target_authority"][1:])
    return matrix, labels, vocab + [f"{a}|{b}" for a, b in bigrams] + ["length"]


def logistic_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x, y, _ = feature_matrix(rows)
    predictions = np.zeros(len(rows), dtype=np.int64)
    for fold in range(5):
        test = np.array(
            [
                int(hashlib.sha256(row["semantic_id"].encode()).hexdigest(), 16) % 5 == fold
                for row in rows
            ],
            dtype=bool,
        )
        train = ~test
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale == 0] = 1
        train_x = (x[train] - mean) / scale
        test_x = (x[test] - mean) / scale
        weights = np.zeros((x.shape[1], 8), dtype=np.float64)
        for _ in range(80):
            logits = train_x @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            target = np.eye(8)[y[train]]
            gradient = train_x.T @ (probabilities - target) / max(1, train_x.shape[0])
            gradient += 0.01 * weights
            weights -= 0.15 * gradient
        predictions[test] = np.argmax(test_x @ weights, axis=1)
    correct = int((predictions == y).sum())
    permutation = []
    for index in range(200):
        shift = 1 + index % (len(y) - 1)
        permutation.append(float((predictions == np.roll(y, shift)).mean()))
    return {
        "classifier": "L2_MULTINOMIAL_LOGISTIC",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson95": wilson(correct, len(rows)),
        "permutation_mean": sum(permutation) / len(permutation),
        "permutation_max": max(permutation),
    }


def tree_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x, y, _ = feature_matrix(rows)
    predictions = np.zeros(len(rows), dtype=np.int64)
    for fold in range(5):
        test = np.array(
            [
                int(hashlib.sha256(row["semantic_id"].encode()).hexdigest(), 16) % 5 == fold
                for row in rows
            ],
            dtype=bool,
        )
        train_indices = np.where(~test)[0]
        test_indices = np.where(test)[0]

        def build(indices: np.ndarray, depth: int) -> Any:
            counts = np.bincount(y[indices], minlength=8)
            majority = int(np.argmax(counts))
            if depth == 3 or len(indices) < 8 or np.count_nonzero(counts) == 1:
                return ("leaf", majority)
            best = None
            for feature in range(x.shape[1]):
                values = x[indices, feature]
                threshold = float(np.median(values))
                left = indices[values <= threshold]
                right = indices[values > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                impurity = 0.0
                for child in (left, right):
                    child_counts = np.bincount(y[child], minlength=8) / len(child)
                    impurity += len(child) * (1 - float((child_counts * child_counts).sum()))
                candidate = (impurity, feature, threshold, left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            if best is None:
                return ("leaf", majority)
            _, feature, threshold, left, right = best
            return ("node", feature, threshold, build(left, depth + 1), build(right, depth + 1))

        tree = build(train_indices, 0)

        def predict(index: int, node: Any) -> int:
            if node[0] == "leaf":
                return node[1]
            return predict(index, node[3] if x[index, node[1]] <= node[2] else node[4])

        for index in test_indices:
            predictions[index] = predict(index, tree)
    correct = int((predictions == y).sum())
    permutation = [
        float((predictions == np.roll(y, 1 + index % (len(y) - 1))).mean())
        for index in range(200)
    ]
    return {
        "classifier": "DEPTH_3_DECISION_TREE",
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson95": wilson(correct, len(rows)),
        "permutation_mean": sum(permutation) / len(permutation),
        "permutation_max": max(permutation),
    }


def evaluate_qualification_rows(
    rows: list[dict[str, Any]], reveal: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
    family_masters = {
        family: bytes.fromhex(value)
        for family, value in reveal["family_master_hex"].items()
    }
    threshold = int(frozen["threshold"])
    evaluated = []
    semantic_success = 0
    invariant_violations = 0
    state_success = 0
    off_false = 0
    baseline_counts = {
        family: [0, 0] for family in ("SEMANTIC_OBJECT", "ANSWER_DERIVED", "GENERIC_ORIGIN")
    }
    state_by_context: dict[str, list[bool]] = {}
    off_by_context: dict[str, list[bool]] = {}
    margins = []
    off_rows = []
    for row in rows:
        keys = authority_keys(family_masters[row["family"]], row["family"])
        scores = {
            authority: detector(row["literal_output"], key, threshold)
            for authority, key in keys.items()
        }
        parsed = parse_output(row["literal_output"])
        semantic_ok = parsed["valid"] and parsed["values"] == row["semantic"]
        semantic_success += int(semantic_ok)
        target = row["target_authority"]
        correct = scores[target]
        wrong_scores = [
            result["score"]
            for authority, result in scores.items()
            if authority != target and result["score"] is not None
        ]
        matching = (
            correct["decision"] == "ACCEPT"
            and all(result["decision"] != "ACCEPT" for authority, result in scores.items() if authority != target)
        )
        if row["condition"] == "STATE":
            state_success += int(matching)
            state_by_context.setdefault(row["semantic_id"], []).append(matching)
            margins.append(correct["score"] - max(wrong_scores))
        elif row["condition"] == "OFF":
            any_accept = any(result["decision"] == "ACCEPT" for result in scores.values())
            off_false += int(any_accept)
            off_by_context.setdefault(row["semantic_id"], []).append(any_accept)
            off_rows.append(row)
            invariant_violations += int(len(set(row["invariant_digests"])) != 1)
        else:
            baseline_counts[row["family"]][0] += int(matching)
            baseline_counts[row["family"]][1] += 1
        evaluated.append(
            {
                "ordinal": row["ordinal"],
                "condition": row["condition"],
                "family": row["family"],
                "target_authority": target,
                "scores": {key: value["score"] for key, value in scores.items()},
                "decisions": {key: value["decision"] for key, value in scores.items()},
                "semantic_valid": semantic_ok,
                "matching_discrimination": matching,
            }
        )
    state_rows = sum(len(value) for value in state_by_context.values())
    off_count = sum(len(value) for value in off_by_context.values())
    context_successes = sum(sum(values) >= 3 for values in state_by_context.values())
    context_false = sum(any(values) for values in off_by_context.values())
    leakage = [logistic_cv(off_rows), tree_cv(off_rows)]
    leakage_unresolved = any(
        item["wilson95"][0] > read_json(SPEC_PATH)["leakage_audit"]["unresolved_rule"].split()[-1]
        if False
        else item["wilson95"][0] > 0.225
        for item in leakage
    )
    semantic_rate = semantic_success / len(rows)
    result = {
        "schema": "exp33a.csw_a0.qualification_results.v1",
        "rows": len(rows),
        "evaluated_rows": evaluated,
        "state": {
            "successes": state_success,
            "total": state_rows,
            "power": state_success / state_rows,
            "wilson95": wilson(state_success, state_rows),
            "context_successes": context_successes,
            "contexts": len(state_by_context),
            "context_success_rate": context_successes / len(state_by_context),
            "context_wilson95": wilson(context_successes, len(state_by_context)),
            "mean_margin": sum(margins) / len(margins),
        },
        "carrier_off": {
            "false_accepts": off_false,
            "total": off_count,
            "fwer_estimate": off_false / off_count,
            "wilson95": wilson(off_false, off_count),
            "context_false_accepts": context_false,
            "contexts": len(off_by_context),
            "context_rate": context_false / len(off_by_context),
            "context_wilson95": wilson(context_false, len(off_by_context)),
            "invariant_violations": invariant_violations,
        },
        "semantic": {
            "successes": semantic_success,
            "total": len(rows),
            "rate": semantic_rate,
        },
        "baselines": {
            family: {
                "successes": values[0],
                "total": values[1],
                "rate": values[0] / values[1],
            }
            for family, values in baseline_counts.items()
        },
        "leakage": leakage,
        "leakage_unresolved": leakage_unresolved,
    }
    spec = read_json(SPEC_PATH)
    gates = {
        "row_denominator": len(rows) == spec["qualification_design"]["total_raw_outputs"],
        "state_power": result["state"]["power"]
        >= spec["statistical_gates"]["matching_authority_per_output_power_min"],
        "context_power": result["state"]["context_success_rate"]
        >= spec["statistical_gates"]["context_success_fraction_min"],
        "nominal_fwer": result["carrier_off"]["fwer_estimate"]
        <= spec["statistical_gates"]["nominal_family_wise_false_positive_max"],
        "carrier_off_invariant": invariant_violations == 0,
        "semantics": semantic_rate == spec["semantic_gates"]["parser_success"],
        "paired_margin": result["state"]["mean_margin"]
        >= spec["statistical_gates"]["paired_margin_mean_min"],
        "leakage": not leakage_unresolved,
    }
    result["gates"] = gates
    result["passed"] = all(gates.values())
    return result


def qualification_evaluate(raw_commit: str, reveal_commit: str) -> None:
    base = QUALIFICATION.relative_to(ROOT)
    rows = [
        json.loads(line)
        for line in git_bytes(raw_commit, base / "RAW_OUTPUTS.jsonl").splitlines()
    ]
    reveal = json.loads(git_bytes(reveal_commit, base / "KEY_REVEAL.json"))
    frozen = json.loads(
        git_bytes(raw_commit, DEVELOPMENT.relative_to(ROOT) / "FROZEN_CHANNEL.json")
    )
    production = evaluate_qualification_rows(rows, reveal, frozen)
    production["raw_commit"] = raw_commit
    production["reveal_commit"] = reveal_commit
    write_json(QUALIFICATION / "PRODUCTION_RESULTS.json", production)
    subprocess.run(
        [
            sys.executable,
            str(REFERENCE),
            "qualification",
            "--raw-commit",
            raw_commit,
            "--reveal-commit",
            reveal_commit,
            "--output",
            str(QUALIFICATION / "REFERENCE_RESULTS.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    reference = read_json(QUALIFICATION / "REFERENCE_RESULTS.json")
    agreement = production == reference
    write_json(
        QUALIFICATION / "RECONCILIATION.json",
        {
            "schema": "exp33a.csw_a0.qualification_reconciliation.v1",
            "production_reference_exact_agreement": agreement,
            "row_count": len(rows),
        },
    )
    if not agreement:
        terminal = "CSW_A0_REFERENCE_EVALUATOR_DISAGREEMENT"
    elif not production["gates"]["semantics"]:
        terminal = "CSW_A0_SEMANTIC_PRESERVATION_FAILED"
    elif not production["gates"]["state_power"] or not production["gates"]["context_power"]:
        terminal = "CSW_A0_PER_OUTPUT_POWER_INSUFFICIENT"
    elif production["leakage_unresolved"]:
        terminal = "CSW_A0_LEAKAGE_ATTRIBUTION_UNRESOLVED"
    elif not production["passed"]:
        terminal = "CSW_A0_INVALID"
    else:
        terminal = "CSW_A0_CHANNEL_READY_FOR_SELECTED_MODEL_SCIENCE"
    write_json(
        QUALIFICATION / "TERMINAL.json",
        {
            "schema": "exp33a.csw_a0.terminal.v1",
            "terminal": terminal,
            "raw_commit": raw_commit,
            "reveal_commit": reveal_commit,
            "production_reference_agreement": agreement,
            "claim_bearing": False,
            "scientific": False,
            "successor_authorized": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-corpus")
    life_init = commands.add_parser("init-lifecycle")
    life_init.add_argument("--private-path", required=True, type=Path)
    life_raw = commands.add_parser("lifecycle-raw")
    life_raw.add_argument("--source-commit", required=True)
    life_raw.add_argument("--private-path", required=True, type=Path)
    life_reveal = commands.add_parser("lifecycle-reveal")
    life_reveal.add_argument("--raw-commit", required=True)
    life_reveal.add_argument("--private-path", required=True, type=Path)
    life_eval = commands.add_parser("lifecycle-evaluate")
    life_eval.add_argument("--raw-commit", required=True)
    life_eval.add_argument("--reveal-commit", required=True)
    training = commands.add_parser("train")
    training.add_argument("--seed", required=True, type=int)
    training.add_argument("--output", required=True, type=Path)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--checkpoint", required=True, type=Path)
    auth = commands.add_parser("authorize-qualification")
    auth.add_argument("--private-path", required=True, type=Path)
    auth.add_argument("--checkpoint", required=True, type=Path)
    qraw = commands.add_parser("qualification-raw")
    qraw.add_argument("--authorization-commit", required=True)
    qraw.add_argument("--private-path", required=True, type=Path)
    qraw.add_argument("--checkpoint", required=True, type=Path)
    qreveal = commands.add_parser("qualification-reveal")
    qreveal.add_argument("--raw-commit", required=True)
    qreveal.add_argument("--private-path", required=True, type=Path)
    qeval = commands.add_parser("qualification-evaluate")
    qeval.add_argument("--raw-commit", required=True)
    qeval.add_argument("--reveal-commit", required=True)
    args = parser.parse_args()
    if args.command == "prepare-corpus":
        prepare_corpus()
    elif args.command == "init-lifecycle":
        init_lifecycle(args.private_path)
    elif args.command == "lifecycle-raw":
        lifecycle_raw(args.source_commit, args.private_path)
    elif args.command == "lifecycle-reveal":
        lifecycle_reveal(args.raw_commit, args.private_path)
    elif args.command == "lifecycle-evaluate":
        lifecycle_evaluate(args.raw_commit, args.reveal_commit)
    elif args.command == "train":
        train(args.seed, args.output)
    elif args.command == "calibrate":
        development_calibration(args.checkpoint)
    elif args.command == "authorize-qualification":
        initialize_qualification(args.private_path, args.checkpoint)
    elif args.command == "qualification-raw":
        qualification_raw(args.authorization_commit, args.private_path, args.checkpoint)
    elif args.command == "qualification-reveal":
        qualification_reveal(args.raw_commit, args.private_path)
    elif args.command == "qualification-evaluate":
        qualification_evaluate(args.raw_commit, args.reveal_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
