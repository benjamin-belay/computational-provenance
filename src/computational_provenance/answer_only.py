"""Reproduce competence and frozen linear-localisation results for answer-only models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layer(hidden)


@dataclass(frozen=True)
class Trace:
    logits: torch.Tensor
    captures: dict[int, torch.Tensor]


class AnswerOnlyTransformer(nn.Module):
    """The paper's 540,808-parameter model, trained with answer labels only."""
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_mode = "sum"
        self.value_embeddings = nn.ModuleList([nn.Embedding(16, 128) for _ in range(4)])
        self.template_embedding = nn.Embedding(4, 128)
        self.cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.aggregate_mixer = None
        self.position_embedding = nn.Embedding(6, 128)
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.final_norm = nn.LayerNorm(128)
        self.answer_head = nn.Linear(128, 8)
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

    def forward(self, values: torch.Tensor, *, template: torch.Tensor | None = None,
                capture: bool = False, capture_position: int = 0) -> torch.Tensor | Trace:
        if template is None:
            template = (values[:, 0] + 3 * values[:, 1] + 5 * values[:, 2] + 7 * values[:, 3]) % 4
        pieces = [embedding(values[:, index]).unsqueeze(1) for index, embedding in enumerate(self.value_embeddings)]
        template_piece = self.template_embedding(template).unsqueeze(1)
        aggregate = self.cls.expand(len(values), -1, -1) + sum(pieces) + template_piece
        hidden = torch.cat([aggregate, *pieces, template_piece], dim=1)
        hidden = hidden + self.position_embedding(torch.arange(6, device=values.device))[None, :, :]
        captures = {}
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if capture:
                captures[index] = hidden[:, capture_position, :].clone()
        logits = self.answer_head(self.final_norm(hidden[:, 0, :]))
        return Trace(logits, captures) if capture else logits


def _all_rows() -> np.ndarray:
    return np.indices((16, 16, 16, 16), dtype=np.int64).reshape(4, -1).T


def _splits() -> dict[str, np.ndarray]:
    counts = {"model_train": 49152, "model_validation": 4096, "probe_fit": 6144, "site_selection": 3072, "intervention": 3072}
    rows = _all_rows()
    order = np.random.default_rng(742000).permutation(len(rows))
    result, offset = {}, 0
    for name, count in counts.items():
        result[name] = rows[order[offset:offset + count]]
        offset += count
    return result


def _facts(rows: np.ndarray) -> dict[str, np.ndarray]:
    a, b, c, d = (rows[:, index] for index in range(4))
    z2 = (a + b + c) % 16
    z3 = (5 * z2 + d) % 16
    return {"z2": z2, "quotient": z2 % 8, "alias_bit": z2 // 8, "z3": z3, "answer": z3 % 8}


def _templates(rows: np.ndarray) -> np.ndarray:
    return (rows[:, 0] + 3 * rows[:, 1] + 5 * rows[:, 2] + 7 * rows[:, 3]) % 4


def _load(index: int) -> AnswerOnlyTransformer:
    path = ROOT / f"checkpoints/answer_only/MODEL_{index}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_index") != index or payload.get("seed") != 772100 + index or payload.get("step") != 12000:
        raise ValueError(f"answer-only checkpoint identity: {path.name}")
    model = AnswerOnlyTransformer()
    model.load_state_dict(payload["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 540808:
        raise ValueError("answer-only parameter count")
    return model.eval()


@torch.no_grad()
def reproduce_competence() -> dict[str, Any]:
    rows = _splits()["model_validation"]
    facts, template = _facts(rows), _templates(rows)
    values = torch.as_tensor(rows)
    results = []
    gates = json.loads((ROOT / "protocols/answer_only_competence.json").read_text())["competence_gates"]
    for index in range(1, 4):
        model = _load(index)
        predictions = []
        for start in range(0, len(rows), 2048):
            output = model(values[start:start + 2048], template=torch.as_tensor(template[start:start + 2048]))
            assert isinstance(output, torch.Tensor)
            predictions.append(output.argmax(-1).numpy())
        prediction = np.concatenate(predictions)
        answer_accuracy = {str(value): float(np.mean(prediction[facts["answer"] == value] == value)) for value in range(8)}
        template_accuracy = {str(value): float(np.mean(prediction[template == value] == facts["answer"][template == value])) for value in range(4)}
        accuracy = float(np.mean(prediction == facts["answer"]))
        passed = (accuracy >= gates["overall_validation_accuracy"] and min(answer_accuracy.values()) >= gates["minimum_answer_class_accuracy"]
                  and min(template_accuracy.values()) >= gates["minimum_template_accuracy"])
        results.append({"model_index": index, "seed": 772100 + index, "rows": len(rows), "accuracy": accuracy,
                        "minimum_answer_accuracy": min(answer_accuracy.values()), "minimum_template_accuracy": min(template_accuracy.values()),
                        "per_answer_accuracy": answer_accuracy, "per_template_accuracy": template_accuracy, "passed": passed})
    return {"models": results, "models_passed": sum(row["passed"] for row in results), "three_of_three_passed": all(row["passed"] for row in results)}


@torch.no_grad()
def _collect(model: AnswerOnlyTransformer, rows: np.ndarray) -> dict[str, np.ndarray]:
    values, template = torch.as_tensor(rows), torch.as_tensor(_templates(rows))
    result: dict[str, np.ndarray] = {}
    for position in range(1, 6):
        stores = {layer: [] for layer in range(4)}
        for start in range(0, len(rows), 1024):
            trace = model(values[start:start + 1024], template=template[start:start + 1024], capture=True, capture_position=position)
            assert isinstance(trace, Trace)
            for layer in range(4):
                stores[layer].append(trace.captures[layer].numpy())
        for layer in range(4):
            result[f"L{layer}_P{position}"] = np.concatenate(stores[layer])
    return result


def _predict(npz: Any, prefix: str, values: np.ndarray) -> np.ndarray:
    coefficients = npz[prefix + "__coef"]
    scores = values @ coefficients.T + npz[prefix + "__intercept"]
    classes = npz[prefix + "__classes"]
    if coefficients.shape[0] == 1:
        return np.where(scores[:, 0] > 0, classes[1], classes[0])
    return classes[scores.argmax(axis=1)]


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for value in np.unique(truth):
        tp = np.sum((truth == value) & (prediction == value))
        fp = np.sum((truth != value) & (prediction == value))
        fn = np.sum((truth == value) & (prediction != value))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def _target_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    per_class = {str(int(value)): float(np.mean(prediction[truth == value] == value)) for value in np.unique(truth)}
    return {"accuracy": float(np.mean(prediction == truth)), "macro_f1": _macro_f1(truth, prediction),
            "minimum_class_accuracy": min(per_class.values()), "per_class_accuracy": per_class}


def reproduce_localisation() -> dict[str, Any]:
    """Re-evaluate the 260 pre-fitted probes on the disjoint 3,072-row selection split."""
    protocol = json.loads((ROOT / "protocols/answer_only_localisation.json").read_text())
    frozen = json.loads((ROOT / "evidence/answer_only/localisation/RESULT.json").read_text())
    rows = _splits()["site_selection"]
    labels = _facts(rows)
    activations = _collect(_load(1), rows)
    npz = np.load(ROOT / "evidence/answer_only/localisation/PROBE_COEFFICIENTS.npz")
    sites = []
    for order, site in enumerate(protocol["sites"]["site_order"]):
        targets = {}
        for target in protocol["probe"]["targets"]:
            targets[target] = _target_metrics(labels[target], _predict(npz, f"{site}__{target}", activations[site]))
        conditional_accuracy, conditional_balanced = {}, {}
        for quotient in range(8):
            mask = labels["quotient"] == quotient
            truth = labels["alias_bit"][mask]
            prediction = _predict(npz, f"{site}__conditional_alias_{quotient}", activations[site][mask])
            conditional_accuracy[str(quotient)] = float(np.mean(prediction == truth))
            recalls = [float(np.mean(prediction[truth == value] == value)) for value in np.unique(truth)]
            conditional_balanced[str(quotient)] = float(np.mean(recalls))
        targets["conditional_alias_bit"] = {
            "per_quotient_accuracy": conditional_accuracy,
            "per_quotient_balanced_accuracy": conditional_balanced,
            "macro_balanced_accuracy": float(np.mean(list(conditional_balanced.values()))),
            "minimum_accuracy": min(conditional_accuracy.values()),
        }
        gates = protocol["representation_gates"]
        q, z2, conditional = targets["quotient"], targets["z2"], targets["conditional_alias_bit"]
        quotient_qualified = q["accuracy"] >= gates["quotient_accuracy"] and q["macro_f1"] >= gates["quotient_macro_f1"] and q["minimum_class_accuracy"] >= gates["quotient_minimum_class_accuracy"]
        full_qualified = quotient_qualified and z2["accuracy"] >= gates["full_z2_accuracy"] and z2["macro_f1"] >= gates["full_z2_macro_f1"] and z2["minimum_class_accuracy"] >= gates["full_z2_minimum_class_accuracy"] and conditional["macro_balanced_accuracy"] >= gates["conditional_alias_macro_accuracy"] and conditional["minimum_accuracy"] >= gates["conditional_alias_minimum_accuracy"]
        digest = hashlib.sha256(activations[site].tobytes()).hexdigest()
        sites.append({"site": site, "order": order, "targets": targets, "quotient_qualified": quotient_qualified,
                      "full_qualified": full_qualified, "activation_sha256": digest,
                      "activation_digest_matches": digest == frozen["selection_activation_digests"][site]["sha256"]})
    best_z2 = max(sites, key=lambda row: row["targets"]["z2"]["accuracy"])
    best_quotient = max(sites, key=lambda row: row["targets"]["quotient"]["accuracy"])
    return {"development_model": 1, "selection_rows": len(rows), "candidate_sites": len(sites),
            "full_qualified_sites": sum(row["full_qualified"] for row in sites),
            "quotient_qualified_sites": sum(row["quotient_qualified"] for row in sites),
            "all_activation_digests_match": all(row["activation_digest_matches"] for row in sites),
            "best_z2": {"site": best_z2["site"], "accuracy": best_z2["targets"]["z2"]["accuracy"]},
            "best_quotient": {"site": best_quotient["site"], "accuracy": best_quotient["targets"]["quotient"]["accuracy"]},
            "terminal": protocol["terminals"]["no_site"] if not any(row["full_qualified"] or row["quotient_qualified"] for row in sites) else "SITE_QUALIFIED",
            "sites": sites}


def reproduce_answer_only() -> dict[str, Any]:
    return {"competence": reproduce_competence(), "localisation": reproduce_localisation()}


def train_answer_only(seed: int, *, steps: int = 12000) -> tuple[AnswerOnlyTransformer, dict[str, Any]]:
    """Run the registered answer-only recipe for one of the three paper seeds."""
    if seed not in (772101, 772102, 772103):
        raise ValueError("unregistered answer-only seed")
    import math
    from .training import configure
    configure(seed)
    model = AnswerOnlyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    rows = _splits()["model_train"]
    values = torch.as_tensor(rows)
    templates = torch.as_tensor(_templates(rows))
    answers = torch.as_tensor(_facts(rows)["answer"])
    trace = []
    model.train()
    for step in range(1, steps + 1):
        generator = torch.Generator().manual_seed(seed * 100_000 + step)
        index = torch.randint(len(rows), (512,), generator=generator)
        warmup = min(1.0, step / 800)
        progress = max(0.0, (step - 800) / (12000 - 800))
        rate = warmup * (5e-5 + 0.5 * (5e-4 - 5e-5) * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(values[index], template=templates[index]), answers[index])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == steps:
            trace.append({"step": step, "learning_rate": rate, "loss": float(loss.detach())})
    return model.eval(), {"seed": seed, "steps": steps, "training_rows": len(rows), "batch_size": 512,
                          "learning_rate": 5e-4, "minimum_learning_rate": 5e-5,
                          "warmup_steps": 800, "weight_decay": 1e-2, "trace": trace}


def fit_localisation_probes(output: Path) -> dict[str, Any]:
    """Fit all 260 registered probes from the released development checkpoint."""
    from sklearn.linear_model import LogisticRegression
    protocol = json.loads((ROOT / "protocols/answer_only_localisation.json").read_text())
    rows = _splits()["probe_fit"]
    labels = _facts(rows)
    activations = _collect(_load(1), rows)
    arrays: dict[str, np.ndarray] = {}
    for site in protocol["sites"]["site_order"]:
        values = activations[site]
        for target in protocol["probe"]["targets"]:
            probe = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", fit_intercept=True)
            probe.fit(values, labels[target])
            prefix = f"{site}__{target}"
            arrays[prefix + "__coef"] = probe.coef_
            arrays[prefix + "__intercept"] = probe.intercept_
            arrays[prefix + "__classes"] = probe.classes_
            arrays[prefix + "__n_iter"] = probe.n_iter_
        for quotient in range(8):
            mask = labels["quotient"] == quotient
            probe = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", fit_intercept=True)
            probe.fit(values[mask], labels["alias_bit"][mask])
            prefix = f"{site}__conditional_alias_{quotient}"
            arrays[prefix + "__coef"] = probe.coef_
            arrays[prefix + "__intercept"] = probe.intercept_
            arrays[prefix + "__classes"] = probe.classes_
            arrays[prefix + "__n_iter"] = probe.n_iter_
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return {"fit_rows": len(rows), "sites": 20, "probes": 260, "output": str(output)}
