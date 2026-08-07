"""Real-file substitution polarity through the production full evaluator."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from ..artifacts import tensor_sha256
from .evaluator import evaluate, STATUS_INVALID
from .indexing import build_prerelease_index
from .strict import object_sha256, sha256_file, strict_load_json, strict_load_jsonl, write_json, write_jsonl


def _hardlink_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=os.link)


def _replace_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".replacement")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _replace_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".replacement")
    write_json(temporary, value)
    os.replace(temporary, path)


def _replace_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".replacement")
    write_jsonl(temporary, rows)
    os.replace(temporary, path)


def _snapshot(package: Path) -> dict:
    return {
        "state": {"state": "BUFFER_COMPLETE", "buffered_calls": 11760},
        "buffer_rows": strict_load_jsonl(package / "access/buffer.jsonl"),
    }


def _recommit_registry_and_raw(package: Path, change_expected: bool) -> None:
    registry_path = package / "registry/registry.jsonl"
    raw_path = package / "access/buffer.jsonl"
    rows = strict_load_jsonl(registry_path)
    raw = strict_load_jsonl(raw_path)
    if change_expected:
        rows[0]["expected_answer"] = (rows[0]["expected_answer"] + 1) % 4
    for row in rows:
        value = dict(row); value.pop("registry_row_sha256", None); value.pop("registry_payload_commitment", None)
        row["registry_row_sha256"] = object_sha256(value)
        row.pop("registry_payload_commitment", None)
    commitment = object_sha256(rows)
    for row in rows:
        row["registry_payload_commitment"] = commitment
    for index, observed in enumerate(raw):
        observed["registry_payload_commitment"] = commitment
        observed["registry_row_sha256"] = rows[index]["registry_row_sha256"]
        if change_expected and index == 0:
            observed["intervened_prediction"] = rows[0]["expected_answer"]
            observed["base_prediction"] = rows[0]["expected_answer"]
            observed["source_prediction"] = rows[0]["expected_answer"]
        value = dict(observed); value.pop("raw_row_sha256", None)
        observed["raw_row_sha256"] = object_sha256(value)
    _replace_jsonl(registry_path, rows)
    _replace_jsonl(raw_path, raw)
    manifest = strict_load_json(package / "registry/manifest.json")
    manifest["payload_commitment"] = commitment
    manifest["registry_file_sha256"] = sha256_file(registry_path)
    value = dict(manifest); value.pop("registry_manifest_sha256", None)
    manifest["registry_manifest_sha256"] = object_sha256(value)
    _replace_json(package / "registry/manifest.json", manifest)


def run_binding_polarity(root: Path, output: Path, report_path: Path | None = None) -> dict:
    scenarios = []

    def scenario(name: str, mutate, rebuild_index: bool = True, expected_reason: str | None = None):
        with tempfile.TemporaryDirectory(prefix="exp33ae-binding-polarity-") as temporary:
            package = Path(temporary) / "package"
            _hardlink_tree(output, package)
            mutate(package)
            if rebuild_index:
                index_path = package / "indexes/prerelease-index.json"
                if index_path.exists():
                    index_path.unlink()
                build_prerelease_index(package)
            result = evaluate(package, root, _snapshot(package))
            scenarios.append({
                "scenario": name, "upstream": "real_artifact_bytes", "expected_status": STATUS_INVALID,
                "actual_status": result["status"], "invalidity_reasons": result["invalidity_reasons"],
                "expected_reason": expected_reason,
                "passed": result["status"] == STATUS_INVALID and (expected_reason is None or expected_reason in result["invalidity_reasons"]),
                "direct_metric_or_status_edit": False,
            })

    def checkpoint_bytes(package):
        path = package / "checkpoints/model-0.pt"
        # PyTorch's ZIP reader tolerates trailing bytes; tensor content remains
        # valid while the independently bound checkpoint-file identity changes.
        _replace_bytes(path, path.read_bytes() + b"DEV_ONLY_CHECKPOINT_FILE_DRIFT")
    scenario("checkpoint_byte_drift", checkpoint_bytes, expected_reason="CHECKPOINT_FILE_BINDING")

    def tensor_state(package):
        path = package / "checkpoints/model-0.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = sorted(payload["state_dict"])[0]
        payload["state_dict"][key] = payload["state_dict"][key].clone()
        payload["state_dict"][key].view(-1)[0] += 0.125
        temporary = path.with_name(path.name + ".replacement")
        torch.save(payload, temporary); os.replace(temporary, path)
    scenario("canonical_tensor_state_drift", tensor_state, expected_reason="CHECKPOINT_TENSOR_BINDING")

    def basis_substitution(package):
        path = package / "representation.json"; value = strict_load_json(path)
        value["basis"][0][0] = 0.0; value["basis"][2][0] = 1.0
        value["basis_sha256"] = tensor_sha256(torch.tensor(value["basis"], dtype=torch.float32))
        copy_value = dict(value); copy_value.pop("representation_sha256", None); value["representation_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("basis_substitution", basis_substitution, expected_reason="BASIS_DECODER_BINDING")

    def decoder_substitution(package):
        path = package / "representation.json"; value = strict_load_json(path)
        value["decoder"]["coordinates"] = [1, 0]
        copy_value = dict(value); copy_value.pop("representation_sha256", None); value["representation_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("decoder_substitution", decoder_substitution, expected_reason="BASIS_DECODER_BINDING")

    def control_substitution(package):
        path = package / "controls/model-0.json"; value = strict_load_json(path)
        value["controls"][0]["basis"][0][0] += 0.125
        value["controls"][0]["basis_sha256"] = tensor_sha256(torch.tensor(value["controls"][0]["basis"], dtype=torch.float32))
        copy_value = dict(value); copy_value.pop("control_manifest_sha256", None); value["control_manifest_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("control_basis_substitution", control_substitution, expected_reason="CONTROL_BINDING")

    def training_drift(package):
        path = package / "training/model-0.json"; value = strict_load_json(path); value["steps"] = 1399
        copy_value = dict(value); copy_value.pop("scientific_manifest_sha256", None); value["scientific_manifest_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("training_manifest_drift", training_drift, expected_reason="TRAINING_BINDING")

    def environment_drift(package):
        path = package / "environment.json"; value = strict_load_json(path); value["torch"] = "substituted"
        copy_value = dict(value); copy_value.pop("manifest_sha256", None); value["manifest_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("dependency_environment_drift", environment_drift, expected_reason="ENVIRONMENT_BINDING")

    def source_drift(package):
        path = package / "operative-source-manifest.json"; value = strict_load_json(path); value["files"][0]["sha256"] = "0" * 64
        copy_value = dict(value); copy_value.pop("manifest_sha256", None); value["manifest_sha256"] = object_sha256(copy_value)
        _replace_json(path, value)
    scenario("source_or_threshold_commitment_drift", source_drift, expected_reason="SOURCE_BINDING")

    def live_source_scenario(name: str, relative: str):
        with tempfile.TemporaryDirectory(prefix="exp33ae-live-source-polarity-") as temporary:
            clone = Path(temporary) / "source"
            subprocess.run(["git", "clone", "--no-local", "--quiet", str(root), str(clone)], check=True)
            source_commit = strict_load_json(output / "operative-source-manifest.json")["source_commit"]
            subprocess.run(["git", "checkout", "--detach", "--quiet", source_commit], cwd=clone, check=True)
            path = clone / relative
            path.write_bytes(path.read_bytes() + b"\nDEV_ONLY_ACTUAL_SOURCE_DRIFT\n")
            result = evaluate(output, clone, _snapshot(output))
            scenarios.append({
                "scenario": name, "upstream": "actual_source_file_in_isolated_clone",
                "expected_status": STATUS_INVALID, "actual_status": result["status"],
                "invalidity_reasons": result["invalidity_reasons"], "expected_reason": "SOURCE_BINDING",
                "passed": result["status"] == STATUS_INVALID and "SOURCE_BINDING" in result["invalidity_reasons"],
                "direct_metric_or_status_edit": False,
            })

    live_source_scenario("actual_evaluator_source_file_drift", "src/exp33ae/remediation/evaluator.py")
    live_source_scenario("actual_dependency_lock_file_drift", "requirements-lock.txt")
    live_source_scenario("actual_threshold_protocol_file_drift", "protocol/dev_only/33a-e-remediation-v2.json")

    scenario("artifact_index_corruption", lambda package: _replace_bytes(package / "indexes/prerelease-index.json", b"{}\n"), rebuild_index=False, expected_reason="INDEX_BINDING")

    def raw_substitution(package):
        path = package / "access/buffer.jsonl"; rows = strict_load_jsonl(path)
        rows[0]["intervened_prediction"] = (rows[0]["intervened_prediction"] + 1) % 4
        value = dict(rows[0]); value.pop("raw_row_sha256", None); rows[0]["raw_row_sha256"] = object_sha256(value)
        _replace_jsonl(path, rows)
    scenario("coherent_raw_row_substitution", raw_substitution, expected_reason="RAW_BINDING")
    scenario("coherent_registry_plus_raw_substitution", lambda package: _recommit_registry_and_raw(package, True), expected_reason="REGISTRY_BINDING")

    report = {
        "schema": "exp33ae.dev_only_binding_polarity.v2", "claim_bearing": False,
        "scenario_count": len(scenarios), "passed_count": sum(row["passed"] for row in scenarios),
        "scenarios": scenarios, "same_production_evaluator": True,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
