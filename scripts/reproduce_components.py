#!/usr/bin/env python3
from __future__ import annotations

import json
from computational_provenance.components import reproduce_components

result = reproduce_components()
print(json.dumps(result, indent=2, sort_keys=True))
if not result["all_passed"]:
    raise SystemExit(1)
