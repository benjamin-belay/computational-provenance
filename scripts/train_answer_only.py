#!/usr/bin/env python3
"""Train the three answer-only models, or refit the registered linear probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from computational_provenance.answer_only import fit_localisation_probes, train_answer_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-probes", action="store_true")
    parser.add_argument("--steps", type=int, default=12000, help="use a smaller value only for smoke testing")
    args = parser.parse_args()
    if args.fit_probes:
        print(json.dumps(fit_localisation_probes(args.output), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, seed in enumerate((772101, 772102, 772103), 1):
        model, record = train_answer_only(seed, steps=args.steps)
        torch.save({"model": model.state_dict(), "model_index": index, "seed": seed, "step": args.steps,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters())},
                   args.output / f"MODEL_{index}.pt")
        (args.output / f"MODEL_{index}_TRAINING.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
