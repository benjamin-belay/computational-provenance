"""Linear decoding probes fitted only on localization-fit records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import TinyProgrammeTransformer


@dataclass
class FittedProbe:
    layer: int
    position: int
    variable: str
    classifier: nn.Linear
    fit_correct: int
    fit_total: int
    selection_correct: int
    selection_total: int

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden).argmax(-1)

    def basis(self, rank: int) -> torch.Tensor:
        weight = self.classifier.weight.detach()
        _, _, vh = torch.linalg.svd(weight, full_matrices=False)
        columns = vh[: min(rank, vh.shape[0])].T
        if columns.shape[1] < rank:
            generator = torch.Generator().manual_seed(330700 + self.layer * 17 + rank)
            extra = torch.randn(weight.shape[1], rank - columns.shape[1], generator=generator)
            extra = extra - columns @ (columns.T @ extra)
            columns = torch.cat((columns, extra), dim=1)
        return torch.linalg.qr(columns, mode="reduced").Q[:, :rank]


@dataclass
class FittedSubspaceDecoder:
    layer: int
    position: int
    basis: torch.Tensor
    classifier: nn.Linear
    fit_correct: int
    fit_total: int
    selection_correct: int
    selection_total: int

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden @ self.basis).argmax(-1)


def extract_hidden(model: TinyProgrammeTransformer, rows, layer: int, position: int) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            tokens = torch.tensor([r.tokens for r in batch], dtype=torch.long)
            tables = torch.tensor([r.operation_tables for r in batch], dtype=torch.float32)
            _, acts, _ = model(tokens, tables, return_activations=True)
            chunks.append(acts[layer][:, position, :])
    return torch.cat(chunks)


def fit_probe(model, fit_rows, selection_rows, layer: int, position: int, variable: str, seed: int) -> FittedProbe:
    torch.manual_seed(seed)
    fit_hidden = extract_hidden(model, fit_rows, layer, position)
    select_hidden = extract_hidden(model, selection_rows, layer, position)
    fit_labels = torch.tensor([getattr(row.values, variable) for row in fit_rows], dtype=torch.long)
    select_labels = torch.tensor([getattr(row.values, variable) for row in selection_rows], dtype=torch.long)
    classifier = nn.Linear(fit_hidden.shape[1], 2)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(300):
        logits = classifier(fit_hidden)
        loss = nn.functional.cross_entropy(logits, fit_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        fit_correct = int((classifier(fit_hidden).argmax(-1) == fit_labels).sum())
        selection_correct = int((classifier(select_hidden).argmax(-1) == select_labels).sum())
    return FittedProbe(layer, position, variable, classifier, fit_correct, len(fit_rows), selection_correct, len(selection_rows))


def fit_subspace_decoder(
    model,
    fit_rows,
    selection_rows,
    layer: int,
    position: int,
    basis: torch.Tensor,
    variable: str,
    seed: int,
) -> FittedSubspaceDecoder:
    torch.manual_seed(seed)
    fit_hidden = extract_hidden(model, fit_rows, layer, position)
    select_hidden = extract_hidden(model, selection_rows, layer, position)
    fit_features = fit_hidden @ basis
    select_features = select_hidden @ basis
    fit_labels = torch.tensor([getattr(row.values, variable) for row in fit_rows], dtype=torch.long)
    select_labels = torch.tensor([getattr(row.values, variable) for row in selection_rows], dtype=torch.long)
    classifier = nn.Linear(basis.shape[1], 2)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(300):
        loss = nn.functional.cross_entropy(classifier(fit_features), fit_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        fit_correct = int((classifier(fit_features).argmax(-1) == fit_labels).sum())
        selection_correct = int((classifier(select_features).argmax(-1) == select_labels).sum())
    return FittedSubspaceDecoder(
        layer,
        position,
        basis.detach(),
        classifier,
        fit_correct,
        len(fit_rows),
        selection_correct,
        len(selection_rows),
    )
