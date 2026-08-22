#!/usr/bin/env python3
"""Verify released raw evidence, receipts, denominators, and adjudication commitments."""
import json

from computational_provenance.evidence import verify_all_evidence


result = verify_all_evidence()
print(json.dumps(result, indent=2, sort_keys=True))
if not result["all_four_passed"]:
    raise SystemExit(1)
