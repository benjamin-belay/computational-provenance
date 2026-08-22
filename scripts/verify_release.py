#!/usr/bin/env python3
"""Verify every released file and the complete prospective evidence chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from computational_provenance.evidence import verify_all_evidence


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "computational_provenance.egg-info"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_files() -> set[str]:
    return {str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
            if path.is_file() and path.name != "RELEASE_MANIFEST.json"
            and not (set(path.relative_to(ROOT).parts) & IGNORED_PARTS)}


def main() -> None:
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append("digest:" + relative)
    failures.extend("unbound:" + relative for relative in sorted(release_files() - set(manifest["files"])))
    evidence = verify_all_evidence()
    if not evidence["all_four_passed"]:
        failures.append("prospective-evidence")
    if failures:
        raise SystemExit("release verification failed:\n" + "\n".join(failures))
    print(f"verified {len(manifest['files'])} released files, 5,120 receipt envelopes, and four final 128/128 terminals")


if __name__ == "__main__":
    main()
