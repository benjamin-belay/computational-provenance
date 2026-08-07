"""Prospectively excluded pre-release and post-release artifact indexes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from .normative import DEV_NAMESPACE, protocol_hash
from .strict import canonical_json, sha256_file, strict_load_json, write_json


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
PRERELEASE_EXCLUDED = {
    "access/access.lock", "access/state.json", "access/journal.jsonl", "access/buffer.jsonl",
    "indexes/prerelease-index.json", "indexes/postrelease-index.json",
    "derived-status.json", "release-envelope.json",
}
POSTRELEASE_EXCLUDED = {"access/access.lock", "indexes/postrelease-index.json"}


def _build(output: Path, exclusions: set[str], schema: str) -> dict:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
            continue
        relative = str(path.relative_to(output))
        if relative in exclusions:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    value = {
        "schema": schema, "namespace": DEV_NAMESPACE, "claim_bearing": False,
        "protocol_sha256": protocol_hash(), "exclusions": sorted(exclusions), "files": files,
        "file_count": len(files), "payload_sha256": sha256(canonical_json(files).encode()).hexdigest(),
    }
    return value


def build_prerelease_index(output: Path) -> dict:
    value = _build(output, PRERELEASE_EXCLUDED, "exp33ae.dev_only_prerelease_index.v2")
    write_json(output / "indexes" / "prerelease-index.json", value)
    return value


def verify_prerelease_index(output: Path) -> dict:
    stored = strict_load_json(output / "indexes" / "prerelease-index.json")
    observed = _build(output, PRERELEASE_EXCLUDED, "exp33ae.dev_only_prerelease_index.v2")
    return {"matches": stored == observed, "expected": stored.get("payload_sha256"), "observed": observed["payload_sha256"], "files": observed["file_count"]}


def build_postrelease_index(output: Path) -> dict:
    value = _build(output, POSTRELEASE_EXCLUDED, "exp33ae.dev_only_postrelease_index.v2")
    write_json(output / "indexes" / "postrelease-index.json", value)
    return value


def verify_postrelease_index(output: Path) -> dict:
    stored = strict_load_json(output / "indexes" / "postrelease-index.json")
    observed = _build(output, POSTRELEASE_EXCLUDED, "exp33ae.dev_only_postrelease_index.v2")
    return {
        "matches": stored == observed,
        "expected": stored.get("payload_sha256"),
        "observed": observed["payload_sha256"],
        "files": observed["file_count"],
    }
