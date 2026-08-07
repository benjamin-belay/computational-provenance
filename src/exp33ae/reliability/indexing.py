"""Separate canonical scientific and variable runtime-observation indexes."""

from __future__ import annotations

from pathlib import Path

from ..remediation.strict import object_sha256, sha256_file, write_json
from .normative import DEV_NAMESPACE


def _rows(output: Path, runtime: bool) -> list[dict]:
    values = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        rel = str(path.relative_to(output)); is_runtime = rel.startswith("runtime/")
        if is_runtime != runtime or rel.startswith("indexes/") or rel == "release-envelope.json": continue
        values.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return values


def build_indexes(output: Path) -> tuple[dict, dict]:
    scientific = {"schema": "exp33ae.dev_only_reliability_scientific_index.v1", "namespace": DEV_NAMESPACE,
                  "claim_bearing": False, "runtime_observations_included": False, "files": _rows(output, False)}
    scientific["canonical_scientific_identity"] = object_sha256(scientific)
    runtime = {"schema": "exp33ae.dev_only_reliability_runtime_index.v1", "namespace": DEV_NAMESPACE,
               "claim_bearing": False, "affects_scientific_identity": False, "files": _rows(output, True)}
    runtime["runtime_observation_identity"] = object_sha256(runtime)
    write_json(output / "indexes" / "scientific-index.json", scientific)
    write_json(output / "indexes" / "runtime-index.json", runtime)
    return scientific, runtime


def verify_indexes(output: Path) -> tuple[bool, bool]:
    from ..remediation.strict import strict_load_json
    observed_scientific = strict_load_json(output / "indexes" / "scientific-index.json")
    observed_runtime = strict_load_json(output / "indexes" / "runtime-index.json")
    expected_scientific = {"schema": observed_scientific["schema"], "namespace": DEV_NAMESPACE,
                           "claim_bearing": False, "runtime_observations_included": False, "files": _rows(output, False)}
    expected_scientific["canonical_scientific_identity"] = object_sha256(expected_scientific)
    expected_runtime = {"schema": observed_runtime["schema"], "namespace": DEV_NAMESPACE,
                        "claim_bearing": False, "affects_scientific_identity": False, "files": _rows(output, True)}
    expected_runtime["runtime_observation_identity"] = object_sha256(expected_runtime)
    return observed_scientific == expected_scientific, observed_runtime == expected_runtime

