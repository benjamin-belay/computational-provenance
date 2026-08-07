"""Evaluator-derived DEV_ONLY atomic release; caller summaries are prohibited."""

from __future__ import annotations

import os
from pathlib import Path

from .access import DevOnlyAccessController, AccessViolation, _atomic_json, _fsync_directory
from .evaluator import evaluate, STATUS_INVALID
from .indexing import build_postrelease_index, verify_prerelease_index
from .normative import DEV_NAMESPACE, protocol_hash
from .strict import object_sha256, sha256_file


def evaluator_derived_release(root: Path, output: Path, controller: DevOnlyAccessController, caller_supplied=None) -> dict:
    if caller_supplied is not None:
        controller._invalidate_state_only("RELEASE_BINDING")
        raise AccessViolation("caller-supplied metrics, gates, evaluator results, or statuses are prohibited")
    snapshot = controller.validate_chain()
    if snapshot["state"]["state"] != "BUFFER_COMPLETE":
        raise AccessViolation("release requires BUFFER_COMPLETE")
    if not verify_prerelease_index(output)["matches"]:
        controller._invalidate_state_only("INDEX_BINDING")
        raise AccessViolation("prerelease index mismatch")
    result = evaluate(output, root, snapshot)
    status_path = output / "derived-status.json"
    _atomic_json(status_path, result)
    envelope = {
        "schema": "exp33ae.dev_only_release_envelope.v2", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "status": result["status"], "derived_status_sha256": sha256_file(status_path),
        "raw_buffer_sha256": sha256_file(controller.buffer_path),
        "registry_commitment": controller.registry_commitment,
        "access_journal_tail_before_release": snapshot["state"]["journal_tail_sha256"],
        "prerelease_index_sha256": verify_prerelease_index(output)["observed"],
        "caller_supplied_result_used": False, "terminal_confirmation_status_materialized": False,
        "confirmation_authorized": False, "protocol_sealed": False,
    }
    envelope["release_payload_sha256"] = object_sha256(envelope)
    release_path = output / "release-envelope.json"
    _atomic_json(release_path, envelope)
    controller.record_release(sha256_file(release_path), result["status"])
    post = build_postrelease_index(output)
    return {"status": result, "release": envelope, "postrelease_index": post}
