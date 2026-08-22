#!/usr/bin/env python3
"""Verify retained final terminals and every file bound by RELEASE_MANIFEST.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    for architecture in ("feedforward", "transformer"):
        for split in ("public", "protected"):
            terminal = json.loads((ROOT / "evidence" / architecture / split / "TERMINAL.json").read_text())
            audit = json.loads((ROOT / "evidence" / architecture / split / "MAIN_AUDIT.json").read_text())
            if split == "public":
                positive = terminal.get("valid") is True and audit.get("observed", {}).get("complete_closure_successes") == 128
            else:
                positive = terminal.get("scientific_positive") is True and terminal.get("complete_closure_successes") == 128
            if not positive:
                failures.append(f"terminal:{architecture}:{split}")
    if failures:
        raise SystemExit("release verification failed:\n" + "\n".join(failures))
    print(f"verified {len(manifest['files'])} files and four final 128/128 terminals")


if __name__ == "__main__":
    main()
