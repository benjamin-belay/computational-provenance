#!/usr/bin/env python3
"""Reproduce the paper's multi-seed robustness results."""
import argparse
import json

from computational_provenance.robustness import reproduce_robustness


parser = argparse.ArgumentParser()
parser.add_argument("architecture", choices=("feedforward", "transformer", "all"), default="all", nargs="?")
parser.add_argument("--limit", type=int, help="diagnostic subset; omit for the claim-bearing denominator")
parser.add_argument("--details", action="store_true")
args = parser.parse_args()
architectures = ("feedforward", "transformer") if args.architecture == "all" else (args.architecture,)
results = [reproduce_robustness(name, limit=args.limit) for name in architectures]
display = results if args.details else [{key: result[key] for key in ("architecture", "held_out_rows_per_model", "causal_cases_per_model", "models_passed", "expected_models", "all_models_passed")} for result in results]
print(json.dumps(display, indent=2, sort_keys=True))
if not all(result["all_models_passed"] for result in results):
    raise SystemExit(1)
