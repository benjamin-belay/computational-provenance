#!/usr/bin/env python3
"""Independently reconstruct all four end-to-end paper evaluations."""
from __future__ import annotations

import argparse
import json

from computational_provenance.adjudication import reconstruct_all, reconstruct_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("feedforward", "transformer"))
    parser.add_argument("--split", choices=("public", "protected"))
    args = parser.parse_args()
    if bool(args.architecture) != bool(args.split):
        parser.error("--architecture and --split must be supplied together")
    result = reconstruct_evaluation(args.architecture, args.split) if args.architecture else reconstruct_all()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed", result.get("all_four_passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
