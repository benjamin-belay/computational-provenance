#!/usr/bin/env python3
"""Verify the immutable evidence object supplied for independent review."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

TERMINAL = "MAIN_CSW_AM_SCB_M1_RNG2_INT_REVIEW_RELEASE_VERIFIED"


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes) -> dict[str, Any]:
    return json.loads(data, object_pairs_hook=no_duplicates)


def git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def git_file(repository: Path, commit: str, path: str) -> bytes:
    return git(repository, "show", f"{commit}:{path}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def verify(repository: Path) -> dict[str, Any]:
    review = (
        repository
        / "docs/programme_consolidation/scb-m1-rng2-int/review"
    )
    manifest_bytes = (review / "REVIEW_MANIFEST.json").read_bytes()
    manifest = parse_json(manifest_bytes)
    release = parse_json((review / "REVIEW_RELEASE.json").read_bytes())
    require(
        release["review_manifest"]["sha256"] == digest(manifest_bytes),
        "review manifest release digest",
    )
    require(release["status"] == "READY_FOR_INDEPENDENT_REVIEW", "release status")
    require(release["zero_new_scientific_evidence"] is True, "release evidence")
    commit = manifest["review_object"]["commit"]
    require(release["review_object_commit"] == commit, "release commit")
    tree = git(repository, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    subtree = git(
        repository,
        "rev-parse",
        f"{commit}:docs/programme_consolidation/scb-m1-rng2-int",
    ).decode().strip()
    require(tree == manifest["review_object"]["repository_tree"], "repository tree")
    require(subtree == manifest["review_object"]["handoff_subtree"], "handoff subtree")

    reviewed = 0
    for item in manifest["reviewed_files"]:
        data = git_file(repository, commit, item["path"])
        require(digest(data) == item["sha256"], f"reviewed file: {item['path']}")
        reviewed += 1

    for path, expected in (
        ("pyproject.toml", manifest["environment_binding"]["pyproject_sha256"]),
        (
            "requirements-lock.txt",
            manifest["environment_binding"]["requirements_lock_sha256"],
        ),
    ):
        require(digest(git_file(repository, commit, path)) == expected, path)

    public_manifest_path = (
        "docs/programme_consolidation/scb-m1-rng2-int/"
        "PUBLIC_ARTIFACT_MANIFEST.json"
    )
    public_manifest = parse_json(git_file(repository, commit, public_manifest_path))
    artifacts = 0
    for item in public_manifest["artifacts"]:
        require(
            digest(git_file(repository, commit, item["path"])) == item["sha256"],
            f"evidence artifact: {item['path']}",
        )
        artifacts += 1

    attestation_path = (
        "docs/programme_consolidation/scb-m1-rng2-int/"
        "SEALED_POPULATION_ATTESTATION.json"
    )
    attestation = parse_json(git_file(repository, commit, attestation_path))
    require(attestation["membership"]["closure_count"] == 256, "sealed closures")
    require(attestation["authorization"]["int_c1_authorized"] is False, "C1 authority")
    require(all(value == 0 for value in attestation["evidence_counts"].values()), "C1 use")

    commands = manifest["permitted_commands"]
    require(len(commands) == 3, "permitted command count")
    require(
        all("int-q1" not in item["command"] for item in commands),
        "protected Q1 command permitted",
    )
    require(
        all("controller" not in item["command"] for item in commands),
        "release controller permitted",
    )
    require(
        manifest["evidence_boundary"]["new_model_runs"] is False,
        "new model runs permitted",
    )
    require(
        manifest["evidence_boundary"]["protected_confirmation_access"] is False,
        "protected access permitted",
    )
    policy = parse_json((review / "REPORTING_POLICY.json").read_bytes())
    require(policy["evidence_object"]["commit"] == commit, "policy evidence commit")
    require(policy["evidence_object"]["immutable"] is True, "evidence not immutable")
    require(
        policy["evidence_object"]["review_manifest_sha256"] == digest(manifest_bytes),
        "policy manifest digest",
    )
    require(
        policy["reporting_layer"]["base_commit"]
        == "4552aa122ca9dd09abfead34efae1f0bf4cb9c18",
        "reporting base",
    )
    require(
        policy["reporting_layer"]["historical_evidence_modification_permitted"] is False,
        "historical modifications permitted",
    )
    require(
        policy["external_review"]["may_be_supplied_by_current_agent_lineage"] is False,
        "independence overstated",
    )
    require(
        {"protected_q1_rerun", "protected_confirmation_release", "new_model_runs_in_closed_lineage"}
        <= set(policy["prohibited_changes"]),
        "missing prohibited change",
    )

    return {
        "artifacts_verified_at_review_commit": artifacts,
        "evidence_object_immutable": True,
        "external_review_status": "PENDING_UNTIL_EXTERNAL_REPORT_RECEIVED",
        "protected_confirmation_authorized": False,
        "protected_confirmation_closures_sealed": 256,
        "review_commit": commit,
        "reviewed_files_verified": reviewed,
        "reporting_only_changes_permitted": True,
        "terminal": TERMINAL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = verify(args.repository.resolve())
    print(json.dumps(result, sort_keys=True))
    print(TERMINAL)


if __name__ == "__main__":
    main()
