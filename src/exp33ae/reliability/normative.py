"""Single normative protocol and frozen-source identity for the reliability study."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from ..remediation.strict import canonical_json, object_sha256, sha256_file


PROTOCOL_PATH = Path("protocol/dev_only/33a-e-training-reliability-v1.json")
DEV_NAMESPACE = "DEV_ONLY_33A_E_TRAINING_RELIABILITY"
DEV_STAGE_ID = "DEV_ONLY_33AE_TRAINING_RELIABILITY_V1"
FREEZE_TAG = "dev-only-33a-e-reliability-study-freeze-v1"
CANDIDATE_TAG = "dev-only-33a-e-training-reliability-candidate-v1"
BASELINE_COMMIT = "2abbfab5aeadbb247750e06f568afd60a8ca1597"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    value = json.loads((repository_root() / PROTOCOL_PATH).read_text())
    if value.get("namespace") != "DEV_ONLY_33A_E_TRAINING_RELIABILITY_SPECIFICATION":
        raise RuntimeError("wrong reliability protocol namespace")
    return value


def protocol_hash() -> str:
    return object_sha256(load_protocol())


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def require_dev_stage(stage_id: str) -> None:
    if stage_id != DEV_STAGE_ID:
        raise RuntimeError("non-DEV execution is structurally unavailable and derived nothing")


def operative_paths(root: Path) -> list[str]:
    paths = ["pyproject.toml", "requirements-lock.txt", str(PROTOCOL_PATH),
             "docs/HISTORICAL_NON_OPERATIVE_NOTICE.md"]
    paths += [str(p.relative_to(root)) for p in sorted((root / "src/exp33ae").rglob("*.py"))]
    paths += [str(p.relative_to(root)) for p in sorted((root / "tests").glob("test_*.py"))]
    paths += list(load_protocol()["development_exclusion_sources"])
    return sorted(set(paths))


def build_source_manifest(root: Path, freeze_commit: str, source_commit: str | None = None) -> dict:
    if git(root, "rev-parse", f"{FREEZE_TAG}^{{commit}}") != freeze_commit:
        raise RuntimeError("freeze tag does not resolve to supplied freeze commit")
    source_commit = source_commit or git(root, "rev-parse", "HEAD^{commit}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", freeze_commit, source_commit], cwd=root)
    if ancestor.returncode != 0:
        raise RuntimeError("operative source is not a descendant of the immutable preregistration")
    files = []
    for relative in operative_paths(root):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing frozen input {relative}")
        commit_blob = git(root, "rev-parse", f"{source_commit}:{relative}")
        working_blob = git(root, "hash-object", relative)
        if commit_blob != working_blob:
            raise RuntimeError(f"working operative source differs from study freeze: {relative}")
        files.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": sha256_file(path), "git_blob": commit_blob})
    value = {
        "schema": "exp33ae.dev_only_reliability_source_manifest.v1",
        "namespace": DEV_NAMESPACE, "claim_bearing": False,
        "baseline_commit": BASELINE_COMMIT,
        "freeze_tag": FREEZE_TAG,
        "freeze_commit": freeze_commit,
        "freeze_tree": git(root, "rev-parse", f"{freeze_commit}^{{tree}}"),
        "source_commit": source_commit,
        "source_tree": git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "postfreeze_mechanics_only_source": source_commit != freeze_commit,
        "protocol_sha256": protocol_hash(), "files": files,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    value["manifest_sha256"] = object_sha256(value)
    return value


def verify_source_manifest(root: Path, value: dict) -> None:
    expected_hash = value.get("manifest_sha256")
    copy = dict(value); copy.pop("manifest_sha256", None)
    if expected_hash != object_sha256(copy):
        raise RuntimeError("source manifest self hash mismatch")
    if value.get("freeze_tag") != FREEZE_TAG or value.get("baseline_commit") != BASELINE_COMMIT:
        raise RuntimeError("source manifest identity mismatch")
    if git(root, "rev-parse", f"{FREEZE_TAG}^{{commit}}") != value["freeze_commit"]:
        raise RuntimeError("freeze tag moved")
    if git(root, "rev-parse", f"{value['freeze_commit']}^{{tree}}") != value["freeze_tree"]:
        raise RuntimeError("freeze tree mismatch")
    if git(root, "rev-parse", f"{value['source_commit']}^{{tree}}") != value["source_tree"]:
        raise RuntimeError("operative source tree mismatch")
    for row in value["files"]:
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source drift: {row['path']}")
        if git(root, "rev-parse", f"{value['source_commit']}:{row['path']}") != row["git_blob"]:
            raise RuntimeError(f"frozen blob mismatch: {row['path']}")
