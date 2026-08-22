#!/usr/bin/env python3
"""Regenerate the byte-level release manifest from the current repository."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", ".pytest_cache", "__pycache__", "computational_provenance.egg-info"}


files = {}
for path in ROOT.rglob("*"):
    relative = path.relative_to(ROOT)
    if path.is_file() and path.name != "RELEASE_MANIFEST.json" and not set(relative.parts) & IGNORED:
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "schema": "computational-provenance.release-manifest.v3",
    "paper": "Towards Computational Provenance: Carrying Causal-State Evidence in Generated Text",
    "version": "1.1.0",
    "hash_algorithm": "sha256",
    "hash_note": "gzip artifacts are bound as released bytes; prospective raw commitments bind their uncompressed bytes",
    "files": dict(sorted(files.items())),
}
(ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"bound {len(files)} files")
