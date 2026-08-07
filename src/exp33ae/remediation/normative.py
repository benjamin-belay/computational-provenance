"""Single normative source, stage guard, and operative-source commitment."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from .strict import canonical_json, expect_exact_keys, expect_sha256, sha256_file


PROTOCOL_PATH = Path("protocol/dev_only/33a-e-remediation-v2.json")
DEV_NAMESPACE = "DEV_ONLY_33A_E_REMEDIATION"
DEV_STAGE_PREFIX = "DEV_ONLY_"
POSITIVE_ANALOGUE = "DEV_ONLY_SIMULATED_ENGINEERED_CAUSAL_ABSTRACTION_POSITIVE"


class StageGuardError(RuntimeError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    path = repository_root() / PROTOCOL_PATH
    value = json.loads(path.read_text())
    if value.get("namespace") != "DEV_ONLY_33A_E_REMEDIATION_SPECIFICATION":
        raise RuntimeError("remediation protocol is not the DEV_ONLY normative source")
    return value


def protocol_hash() -> str:
    return sha256(canonical_json(load_protocol()).encode()).hexdigest()


def require_dev_stage(stage_id: str) -> None:
    if type(stage_id) is not str or not stage_id.startswith(DEV_STAGE_PREFIX):
        raise StageGuardError(
            "non-DEV execution is unavailable: independent seal and authorization objects do not exist"
        )


OPERATIVE_PATHS = (
    "pyproject.toml",
    "requirements-lock.txt",
    "registries/dev_only/programme-manifest.jsonl",
    "registries/dev_only/example-index.jsonl",
    "registries/dev_only/evaluation-call-registry.jsonl",
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def operative_paths(root: Path) -> list[str]:
    values = list(OPERATIVE_PATHS)
    values.extend(str(path.relative_to(root)) for path in sorted((root / "src/exp33ae").rglob("*.py")))
    # Tests are part of the operative commitment: the candidate must bind the
    # complete executable verification surface, not only tests whose names were
    # introduced by this lineage.
    values.extend(str(path.relative_to(root)) for path in sorted((root / "tests").glob("test_*.py")))
    values.extend(str(path.relative_to(root)) for path in sorted((root / "protocol/dev_only").glob("*.json")))
    return sorted(set(values))


def build_operative_manifest(root: Path, source_commit: str | None = None) -> dict:
    commit = source_commit or _git(root, "rev-parse", "HEAD^{commit}")
    tree = _git(root, "rev-parse", f"{commit}^{{tree}}")
    files = []
    for relative in operative_paths(root):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing operative source {relative}")
        commit_blob = _git(root, "rev-parse", f"{commit}:{relative}")
        working_blob = _git(root, "hash-object", relative)
        if commit_blob != working_blob:
            raise RuntimeError(f"operative source differs from frozen commit: {relative}")
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob": commit_blob,
        })
    payload = {
        "schema": "exp33ae.operative_source_manifest.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "protocol_sha256": protocol_hash(),
        "source_commit": commit,
        "source_tree": tree,
        "files": files,
    }
    payload["manifest_sha256"] = sha256(canonical_json(payload).encode()).hexdigest()
    return payload


def verify_operative_manifest(root: Path, manifest: dict) -> list[str]:
    faults: list[str] = []
    try:
        expect_exact_keys(manifest, {"schema", "namespace", "claim_bearing", "protocol_sha256", "source_commit", "source_tree", "files", "manifest_sha256"}, "operative manifest")
        if manifest["schema"] != "exp33ae.operative_source_manifest.v2" or manifest["namespace"] != DEV_NAMESPACE or manifest["claim_bearing"] is not False or manifest["protocol_sha256"] != protocol_hash():
            faults.append("SOURCE_BINDING")
        if type(manifest["source_commit"]) is not str or len(manifest["source_commit"]) != 40:
            faults.append("SOURCE_BINDING")
        if type(manifest["source_tree"]) is not str or len(manifest["source_tree"]) != 40:
            faults.append("SOURCE_BINDING")
        expect_sha256(manifest["manifest_sha256"], "operative manifest hash")
        if type(manifest["files"]) is not list:
            faults.append("SOURCE_BINDING")
        else:
            for row in manifest["files"]:
                expect_exact_keys(row, {"path", "bytes", "sha256", "git_blob"}, "operative file")
                if type(row["path"]) is not str or type(row["bytes"]) is not int or row["bytes"] < 0:
                    faults.append("SOURCE_BINDING")
                expect_sha256(row["sha256"], "operative file sha256")
                if type(row["git_blob"]) is not str or len(row["git_blob"]) != 40:
                    faults.append("SOURCE_BINDING")
    except Exception:
        faults.append("SOURCE_BINDING")
    expected_hash = manifest.get("manifest_sha256")
    copy = dict(manifest)
    copy.pop("manifest_sha256", None)
    if expected_hash != sha256(canonical_json(copy).encode()).hexdigest():
        faults.append("SOURCE_BINDING")
    try:
        if _git(root, "rev-parse", f"{manifest['source_commit']}^{{tree}}") != manifest["source_tree"]:
            faults.append("SOURCE_BINDING")
        for row in manifest["files"]:
            path = root / row["path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                faults.append("SOURCE_BINDING")
                break
            if _git(root, "rev-parse", f"{manifest['source_commit']}:{row['path']}") != row["git_blob"]:
                faults.append("SOURCE_BINDING")
                break
    except Exception:
        faults.append("SOURCE_BINDING")
    return sorted(set(faults))
