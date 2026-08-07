"""Single normative source and frozen constants for confirmation protocol v1."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import pytest
import torch

from ..model import ModelConfig, TinyProgrammeTransformer
from ..remediation.strict import object_sha256, sha256_file, StrictValidationError


PROTOCOL_VERSION = 1
PROTOCOL_PATH = Path("protocol/freeze/33a-e-confirmation-protocol-v1.json")
PROTOCOL_NAMESPACE = "EXP33AE_CONFIRMATION_PROTOCOL_SPECIFICATION_V1"
DEV_NAMESPACE = "DEV_ONLY_EXP33AE_CONFIRMATION_PROTOCOL_FIXTURE_V1"
DEV_STAGE = "DEV_ONLY_PROTOCOL_FREEZE_TEST_V1"
CLAIM_NAMESPACE = "EXP33AE_CONFIRMATION_RUN_V1"
GOLDEN_MASTER_HEX = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
ACCEPTED_PREDECESSOR_TAG = "dev-only-33a-e-reliability-reproducibility-cleanup-candidate-v2"
ACCEPTED_PREDECESSOR_TAG_OBJECT = "1b189c4ef2f85521910017b27203812dbbdb7420"
ACCEPTED_PREDECESSOR_COMMIT = "cd22782b63234a403d6a870b182ef927c1acc630"
ACCEPTED_PREDECESSOR_TREE = "2a036d23761d091d0f10a8a510044b08d61024f2"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    value = json.loads((repository_root() / PROTOCOL_PATH).read_text())
    if (
        value.get("schema") != "exp33ae.confirmation_protocol.v1"
        or value.get("version") != PROTOCOL_VERSION
        or value.get("namespace") != PROTOCOL_NAMESPACE
        or value.get("claim_bearing") is not False
    ):
        raise StrictValidationError("PROTOCOL_SCHEMA", "wrong protocol identity")
    assert_consistency(value)
    return value


def protocol_hash() -> str:
    return object_sha256(load_protocol())


def assert_consistency(protocol: dict) -> None:
    model = protocol["model_and_training"]
    registry = protocol["registry"]
    controls = protocol["controls"]
    thresholds = protocol["thresholds_integer"]
    if (model["recipe_id"], model["d_model"], model["d_mlp"], model["steps"], model["population"]) != (
        "R3_MORE_STEPS_AND_WIDTH", 96, 192, 2800, 5
    ):
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "operative R3 recipe")
    config = ModelConfig(d_model=96, d_mlp=192, n_heads=4, n_layers=4)
    count = sum(parameter.numel() for parameter in TinyProgrammeTransformer(config).parameters())
    if count != model["parameter_count"] or model["parameter_count"] != 302788:
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "parameter count")
    if registry != {
        "created_and_committed_before_evaluation_access": True,
        "clean_target_per_model": 800,
        "clean_parallel_per_model": 800,
        "primary_per_model": 400,
        "controls_per_model": 1920,
        "total_per_model": 3920,
        "models": 5,
        "total": 19600,
        "planned_attempts_each": 1,
        "resume_policy": "same consumed session from exact first missing call; completed calls never re-executed",
        "row_linkage": registry["row_linkage"],
    }:
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "registry totals")
    if sum(item["total"] for item in (controls["random_direction"], controls["orthogonal_direction"], controls["alternate_location"])) + controls["alternate_early_layer"]["calls"] != 1920:
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "control total")
    if thresholds["each_individual_control_gap"] != [48, 160]:
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "control gap is not 0.30")
    if protocol["representation"]["primary_method"] != "fixed declared IIT alignment":
        raise StrictValidationError("PROTOCOL_CONSISTENCY", "primary method")


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def observed_dependencies() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pytest": pytest.__version__,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def offline_dependency_preflight() -> dict:
    protocol = load_protocol()["dependency_boundary"]
    observed = observed_dependencies()
    required = {name: protocol[name] for name in ("python", "torch", "numpy", "pytest")}
    mismatches = {name: {"required": required[name], "observed": observed[name]} for name in required if observed[name] != required[name]}
    return {
        "schema": "exp33ae.offline_dependency_preflight.v1",
        "network_used": False,
        "install_attempted": False,
        "required": required,
        "observed": observed,
        "passed": not mismatches,
        "mismatches": mismatches,
        "boundary": "locked externally supplied interpreter; independent dependency reconstruction is not claimed",
    }


def operative_source_paths(root: Path) -> list[str]:
    # The authoritative protocol is committed in the final candidate after the
    # executable-source commit is known.  It is bound separately by protocol
    # hash; this list is the operative executable/dependency source freeze.
    paths = ["pyproject.toml", "requirements-lock.txt"]
    paths += [str(path.relative_to(root)) for path in sorted((root / "src/exp33ae/confirmation_protocol").glob("*.py"))]
    paths += [str(path.relative_to(root)) for path in sorted((root / "schemas/confirmation-protocol-v1").glob("*.json"))]
    paths += [str(path.relative_to(root)) for path in sorted((root / "tests").glob("test_confirmation_protocol_freeze*.py"))]
    # These existing scientific primitives are invoked by the future path.
    paths += [
        "src/exp33ae/model.py", "src/exp33ae/spec.py", "src/exp33ae/task.py",
        "src/exp33ae/independent_oracle.py", "src/exp33ae/baselines.py", "src/exp33ae/artifacts.py",
        "src/exp33ae/remediation/strict.py", "src/exp33ae/remediation/access.py",
        "src/exp33ae/remediation/training.py",
    ]
    return sorted(set(paths))


def source_manifest(root: Path, source_commit: str) -> dict:
    files = []
    for relative in operative_source_paths(root):
        path = root / relative
        if not path.is_file():
            raise StrictValidationError("SOURCE_BINDING", f"missing {relative}")
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": git(root, "rev-parse", f"{source_commit}:{relative}"),
        })
    result = {
        "schema": "exp33ae.confirmation_source_manifest.v1",
        "claim_bearing": False,
        "source_commit": source_commit,
        "source_tree": git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "protocol_sha256": protocol_hash(),
        "files": files,
    }
    result["manifest_sha256"] = object_sha256(result)
    return result


def verify_live_source(root: Path, manifest: dict) -> None:
    copy = dict(manifest); observed = copy.pop("manifest_sha256", None)
    if observed != object_sha256(copy):
        raise StrictValidationError("SOURCE_BINDING", "source manifest self hash")
    if git(root, "rev-parse", f"{manifest['source_commit']}^{{tree}}") != manifest["source_tree"]:
        raise StrictValidationError("SOURCE_BINDING", "source tree")
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise StrictValidationError("SOURCE_BINDING", row["path"])
        if git(root, "hash-object", row["path"]) != row["git_blob"]:
            raise StrictValidationError("SOURCE_BINDING", f"blob {row['path']}")


def assert_dev_fixture(namespace: str, claim_bearing: bool, stage: str) -> None:
    if namespace != DEV_NAMESPACE or claim_bearing is not False or stage != DEV_STAGE:
        raise StrictValidationError("STAGE_GUARD", "fixture is not structurally DEV_ONLY")


def candidate_counters() -> dict:
    return dict(load_protocol()["candidate_counters"])
