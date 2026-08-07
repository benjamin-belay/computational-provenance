"""Hash every candidate artifact without indexing Git internals or the index itself."""

from __future__ import annotations

from pathlib import Path

from .artifacts import sha256_file, write_json
from .spec import NAMESPACE


EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def build_artifact_index(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
            continue
        relative = str(path.relative_to(root))
        if relative == "results/dev_only/artifact-index.json":
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    report = {
        "schema": "exp33ae.artifact_index.v1", "namespace": NAMESPACE,
        "claim_bearing": False, "file_count": len(rows), "files": rows,
        "confirmation_artifacts_created": 0, "confirmation_artifacts_accessed": 0,
    }
    write_json(root / "results/dev_only/artifact-index.json", report)
    return report
