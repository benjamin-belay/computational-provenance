"""Actual index reconstruction and two-layer release boundary."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..artifacts import canonical_json, sha256_file, write_json
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash


EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
EXCLUDED_RELATIVE = {
    "results/dev_only/preconfirmation-revision-v1/artifact-index.json",
    "results/dev_only/preconfirmation-revision-v1/layer-1-finalizer-envelope.json",
    "results/dev_only/preconfirmation-revision-v1/layer-2-post-tag-attestation.json",
    "results/dev_only/preconfirmation-revision-v1/test-report.json",
    "results/dev_only/preconfirmation-revision-v1/hostile-audit.json",
}


def reconstruct_index(root: Path) -> dict:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
            continue
        relative = str(path.relative_to(root))
        if relative in EXCLUDED_RELATIVE:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload_hash = sha256(canonical_json(files).encode()).hexdigest()
    return {
        "schema": "exp33ae.dev_only_protocol_revision_index.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(), "exclusions": sorted(EXCLUDED_RELATIVE),
        "files": files, "file_count": len(files), "payload_index_sha256": payload_hash,
        "confirmation_payload_index_materialized": False,
    }


def finalize_layer_1(root: Path, output_directory: Path) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    index = reconstruct_index(root)
    write_json(output_directory / "artifact-index.json", index)
    reconstructed = reconstruct_index(root)
    if reconstructed != index:
        raise RuntimeError("actual index reconstruction changed after writing an excluded index")
    envelope = {
        "schema": "exp33ae.dev_only_layer_1_finalizer.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "actual_reconstructed_payload_index_sha256": reconstructed["payload_index_sha256"],
        "actual_reconstructed_file_count": reconstructed["file_count"],
        "index_file_sha256": sha256_file(output_directory / "artifact-index.json"),
        "consumed_placeholder": False, "layer_2_claimed_passed": False,
        "confirmation_release_finalized": False,
    }
    write_json(output_directory / "layer-1-finalizer-envelope.json", envelope)
    return envelope


def verify_index(root: Path, stored: Path) -> dict:
    expected = __import__("json").loads(stored.read_text())
    observed = reconstruct_index(root)
    return {
        "matches": expected == observed,
        "expected_payload_sha256": expected.get("payload_index_sha256"),
        "observed_payload_sha256": observed["payload_index_sha256"],
        "expected_files": expected.get("file_count"), "observed_files": observed["file_count"],
    }
