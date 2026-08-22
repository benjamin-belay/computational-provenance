#!/usr/bin/env python3
"""Replay the clean end-to-end pipeline on released public or protected rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from computational_provenance.pipeline import replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("feedforward", "transformer"), required=True)
    parser.add_argument("--split", choices=("public", "protected"), default="public")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path, help="checkpoint.pt emitted by scripts/train.py")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    result = replay(args.architecture, args.split, limit=args.limit, checkpoint_path=args.checkpoint)
    if not args.details:
        result = {key: result[key] for key in ("architecture", "split", "closures", "complete_successes")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
