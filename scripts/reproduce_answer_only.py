#!/usr/bin/env python3
"""Reproduce answer-only competence and 20-site localisation."""
import argparse
import json

from computational_provenance.answer_only import reproduce_answer_only


parser = argparse.ArgumentParser()
parser.add_argument("--details", action="store_true")
parser.add_argument("--require-exact-activations", action="store_true",
                    help="fail unless activation bytes match the frozen numerical platform")
args = parser.parse_args()
result = reproduce_answer_only()
display = result if args.details else {
    "competence": {key: result["competence"][key] for key in ("models_passed", "three_of_three_passed")},
    "localisation": {key: result["localisation"][key] for key in ("selection_rows", "candidate_sites", "full_qualified_sites", "quotient_qualified_sites", "best_z2", "best_quotient", "all_activation_digests_match", "terminal")},
}
print(json.dumps(display, indent=2, sort_keys=True))
if not result["competence"]["three_of_three_passed"]:
    raise SystemExit(1)
if result["localisation"]["full_qualified_sites"] or result["localisation"]["quotient_qualified_sites"]:
    raise SystemExit(1)
if args.require_exact_activations and not result["localisation"]["all_activation_digests_match"]:
    raise SystemExit(1)
