"""Frozen localization search and declared IIT alignment materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time

import torch
from torch import nn

from .artifacts import sha256_file, write_json
from .catalog import records
from .interventions import evaluate_intervention, summarize
from .model import declared_iit_basis, random_basis
from .pairing import eligible_pairs
from .probes import fit_probe, fit_subspace_decoder
from .spec import IIT_LAYER, IIT_POSITION, IIT_RANK, IIT_STRENGTH, MODEL_SEEDS, SEARCH_GRID, Z2_POSITION
from .training import checkpoint_path, load_checkpoint


@dataclass(frozen=True)
class Candidate:
    method: str
    layer: int
    position: int
    rank: int
    strength: float

    @property
    def candidate_id(self) -> str:
        return f"{self.method}-L{self.layer}-P{self.position}-R{self.rank}-S{self.strength:g}"


def _pair_tensors(pairs):
    base_tokens = torch.tensor([p.base.tokens for p in pairs], dtype=torch.long)
    base_tables = torch.tensor([p.base.operation_tables for p in pairs], dtype=torch.float32)
    source_tokens = torch.tensor([p.source.tokens for p in pairs], dtype=torch.long)
    source_tables = torch.tensor([p.source.operation_tables for p in pairs], dtype=torch.float32)
    expected = torch.tensor([p.expected_joint for p in pairs], dtype=torch.long)
    return base_tokens, base_tables, source_tokens, source_tables, expected


def fit_das(model, pairs, candidate: Candidate, seed: int, steps: int = 60) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    raw = nn.Parameter(torch.randn(model.config.d_model, candidate.rank) / model.config.d_model**0.5)
    optimizer = torch.optim.Adam([raw], lr=0.04)
    base_tokens, base_tables, source_tokens, source_tables, expected = _pair_tensors(pairs)
    with torch.no_grad():
        source_activation = model.source_activation(source_tokens, source_tables, candidate.layer, candidate.position)
    original = [p.requires_grad for p in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trace = []
    try:
        for step in range(1, steps + 1):
            basis = torch.linalg.qr(raw, mode="reduced").Q
            logits = model(
                base_tokens,
                base_tables,
                intervention={
                    "layer": candidate.layer,
                    "position": candidate.position,
                    "source": source_activation,
                    "basis": basis,
                    "strength": candidate.strength,
                },
            )
            loss = nn.functional.cross_entropy(logits, expected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in {1, 10, 20, 40, 60}:
                trace.append(float(loss.detach()))
    finally:
        for parameter, requires_grad in zip(model.parameters(), original):
            parameter.requires_grad_(requires_grad)
    return torch.linalg.qr(raw.detach(), mode="reduced").Q, trace


def _candidate_score(row):
    return (
        row["control_gap"],
        row["causal_iia"],
        row["decode_selection_accuracy"],
        -row["layer"],
        -row["rank"],
        -row["strength"],
        row["method"] == "linear_probe",
    )


def search_model(root: Path, kind: str, seed: int) -> tuple[list[dict], dict]:
    model, _ = load_checkpoint(checkpoint_path(root, kind, seed))
    fit_rows = records("localization_fit")
    selection_rows = records("localization_selection")
    fit_pairs = eligible_pairs(fit_rows, "different_state", 64, seed + 12000)
    selection_pairs = eligible_pairs(selection_rows, "different_state", 64, seed + 13000)
    probes = {
        (layer, position): fit_probe(
            model,
            fit_rows,
            selection_rows,
            layer,
            position,
            "z2",
            seed + layer * 101 + position * 7,
        )
        for layer in SEARCH_GRID["layers"]
        for position in SEARCH_GRID["positions"]
    }
    rows = []
    bases = {}
    for method in SEARCH_GRID["methods"]:
        for layer in SEARCH_GRID["layers"]:
            for position in SEARCH_GRID["positions"]:
                for rank in SEARCH_GRID["ranks"]:
                    for strength in SEARCH_GRID["strengths"]:
                        candidate = Candidate(method, layer, position, rank, strength)
                        probe = probes[(layer, position)]
                        if method == "linear_probe":
                            basis, trace = probe.basis(rank), []
                        else:
                            basis, trace = fit_das(model, fit_pairs, candidate, seed * 1000 + layer * 100 + rank * 10 + int(strength * 10))
                        bases[candidate.candidate_id] = basis
                        causal_calls = evaluate_intervention(
                            model, selection_pairs, layer, position, basis, strength, probe, "causal_candidate"
                        )
                        control_basis = random_basis(model.config.d_model, rank, seed * 4000 + layer * 100 + rank)
                        random_calls = evaluate_intervention(
                            model,
                            selection_pairs,
                            layer,
                            position,
                            control_basis,
                            strength,
                            probe,
                            "matched_random_direction",
                            norm_reference_basis=basis,
                        )
                        causal, random_summary = summarize(causal_calls), summarize(random_calls)
                        rows.append(
                            {
                                "namespace": "DEV_ONLY",
                                "model_kind": kind,
                                "model_seed": seed,
                                "candidate_id": candidate.candidate_id,
                                "method": method,
                                "layer": layer,
                                "position": position,
                                "rank": rank,
                                "strength": strength,
                                "decode_fit_correct": probe.fit_correct,
                                "decode_fit_total": probe.fit_total,
                                "decode_selection_correct": probe.selection_correct,
                                "decode_selection_total": probe.selection_total,
                                "decode_selection_accuracy": probe.selection_correct / probe.selection_total,
                                "causal_correct": round(causal["iia"] * causal["attempted"]),
                                "causal_total": causal["attempted"],
                                "causal_iia": causal["iia"],
                                "random_correct": round(random_summary["iia"] * random_summary["attempted"]),
                                "random_total": random_summary["attempted"],
                                "random_iia": random_summary["iia"],
                                "control_gap": causal["iia"] - random_summary["iia"],
                                "das_loss_trace": trace,
                            }
                        )
    selected_row = max(rows, key=_candidate_score)
    selected = Candidate(
        selected_row["method"], selected_row["layer"], selected_row["position"], selected_row["rank"], selected_row["strength"]
    )
    selected_basis = bases[selected.candidate_id]
    selected_probe = probes[(selected.layer, selected.position)]
    if kind == "iit_primary":
        selected = Candidate("iit_declared", IIT_LAYER, IIT_POSITION, IIT_RANK, IIT_STRENGTH)
        selected_basis = declared_iit_basis(model.config)
        selected_probe = probes[(IIT_LAYER, IIT_POSITION)]
        declared_calls = evaluate_intervention(
            model,
            selection_pairs,
            selected.layer,
            selected.position,
            selected_basis,
            selected.strength,
            selected_probe,
            "causal_candidate",
        )
        declared_random_basis = random_basis(model.config.d_model, selected.rank, seed + 88000)
        declared_random = evaluate_intervention(
            model,
            selection_pairs,
            selected.layer,
            selected.position,
            declared_random_basis,
            selected.strength,
            selected_probe,
            "matched_random_direction",
            norm_reference_basis=selected_basis,
        )
        dc, dr = summarize(declared_calls), summarize(declared_random)
        rows.append(
            {
                "namespace": "DEV_ONLY",
                "model_kind": kind,
                "model_seed": seed,
                "candidate_id": selected.candidate_id,
                "method": selected.method,
                "layer": selected.layer,
                "position": selected.position,
                "rank": selected.rank,
                "strength": selected.strength,
                "decode_fit_correct": selected_probe.fit_correct,
                "decode_fit_total": selected_probe.fit_total,
                "decode_selection_correct": selected_probe.selection_correct,
                "decode_selection_total": selected_probe.selection_total,
                "decode_selection_accuracy": selected_probe.selection_correct / selected_probe.selection_total,
                "causal_correct": round(dc["iia"] * dc["attempted"]),
                "causal_total": dc["attempted"],
                "causal_iia": dc["iia"],
                "random_correct": round(dr["iia"] * dr["attempted"]),
                "random_total": dr["attempted"],
                "random_iia": dr["iia"],
                "control_gap": dc["iia"] - dr["iia"],
                "das_loss_trace": [],
                "declared_engineered_alignment": True,
            }
        )

    subspace_decoder = fit_subspace_decoder(
        model,
        fit_rows,
        selection_rows,
        selected.layer,
        selected.position,
        selected_basis,
        "z2",
        seed + 99000,
    )

    alignment_path = root / "artifacts/dev_only/alignments" / f"{kind}-seed-{seed}.pt"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "exp33ae.alignment.v2",
            "namespace": "DEV_ONLY",
            "model_kind": kind,
            "model_seed": seed,
            "selected_candidate": asdict(selected),
            "basis": selected_basis,
            "z2_probe_state": selected_probe.classifier.state_dict(),
            "z2_probe": {
                "layer": selected_probe.layer,
                "position": selected_probe.position,
                "fit_correct": selected_probe.fit_correct,
                "fit_total": selected_probe.fit_total,
                "selection_correct": selected_probe.selection_correct,
                "selection_total": selected_probe.selection_total,
            },
            "z2_subspace_decoder_state": subspace_decoder.classifier.state_dict(),
            "z2_subspace_decoder": {
                "layer": subspace_decoder.layer,
                "position": subspace_decoder.position,
                "rank": int(selected_basis.shape[1]),
                "fit_correct": subspace_decoder.fit_correct,
                "fit_total": subspace_decoder.fit_total,
                "selection_correct": subspace_decoder.selection_correct,
                "selection_total": subspace_decoder.selection_total,
                "definition": "linear decoder fitted only to selected-basis coordinates",
            },
            "selection_rule": "IIT uses prospectively declared alignment; comparisons maximize control_gap then IIA then decode with deterministic tie-breaks",
        },
        alignment_path,
    )
    return rows, {
        "model_kind": kind,
        "model_seed": seed,
        "selected_candidate": asdict(selected),
        "alignment_path": str(alignment_path.relative_to(root)),
        "alignment_sha256": sha256_file(alignment_path),
    }


def run_search(root: Path) -> dict:
    started = time.perf_counter()
    all_rows, selections = [], []
    for kind in ("ordinary", "iit_primary"):
        for seed in MODEL_SEEDS[kind]:
            rows, selection = search_model(root, kind, seed)
            all_rows.extend(rows)
            selections.append(selection)
    result = {
        "schema": "exp33ae.method_search.v1",
        "namespace": "DEV_ONLY",
        "claim_bearing": False,
        "grid": {key: list(value) for key, value in SEARCH_GRID.items()},
        "grid_rows_per_model": 64,
        "declared_iit_rows": 3,
        "total_rows": len(all_rows),
        "rows": all_rows,
        "selections": selections,
        "runtime_seconds": time.perf_counter() - started,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    write_json(root / "results/dev_only/method-search.json", result)
    csv_path = root / "results/dev_only/method-search.csv"
    fields = [key for key in all_rows[0] if key != "das_loss_trace"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row[key] for key in fields})
    return result
