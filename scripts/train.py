#!/usr/bin/env python3
"""Re-run a fixed training recipe (CPU; the transformer run is intentionally long)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from computational_provenance.training import TRAINING_PROFILES, read_rows, train


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("feedforward", "transformer"), required=True)
    parser.add_argument("--profile", choices=("designated", "robustness"), default="designated")
    parser.add_argument("--steps", type=int, help="smoke-test override; omit for the registered recipe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(ROOT / "data" / "training" / f"{args.architecture}.jsonl.gz")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    seeds = TRAINING_PROFILES[args.architecture][args.profile]
    for index, seed in enumerate(seeds, 1):
        model, record = train(args.architecture, rows, seed=seed, steps=args.steps)
        if args.profile == "designated":
            model_id = "R3_MODEL_1" if args.architecture == "feedforward" else "TR_ENG_PROVENANCE_MODEL_1"
        else:
            prefix = "R2_MODEL" if args.architecture == "feedforward" else "TR_ENG_ROBUSTNESS_MODEL"
            model_id = f"{prefix}_{index}"
        torch.save({"state_dict": model.state_dict(), "model_id": model_id, "step": record["steps"],
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()), **record},
                   args.output / f"{model_id}_STEP_{record['steps']}.pt")
        (args.output / f"seed_{seed}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append({key: record[key] for key in ("architecture", "seed", "steps", "training_rows")})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
