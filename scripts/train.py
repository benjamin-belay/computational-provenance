#!/usr/bin/env python3
"""Re-run a fixed training recipe (CPU; the transformer run is intentionally long)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from computational_provenance.training import read_rows, train


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("feedforward", "transformer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(ROOT / "data" / "training" / f"{args.architecture}.jsonl.gz")
    model, record = train(args.architecture, rows)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict(), **record}, args.output / "checkpoint.pt")
    (args.output / "training.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: record[key] for key in ("architecture", "seed", "steps", "training_rows")}, indent=2))


if __name__ == "__main__":
    main()
