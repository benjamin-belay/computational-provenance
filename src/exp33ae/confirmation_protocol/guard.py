"""Fail-closed claim launcher.  All checks precede entropy parsing or output creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from ..remediation.strict import StrictValidationError, expect_exact_keys, expect_sha256
from .entropy import EntropyContext, parse_master_hex
from .normative import CLAIM_NAMESPACE, git, load_protocol, offline_dependency_preflight, protocol_hash


def _git_oid(value, label):
    if type(value) is not str or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise StrictValidationError("SCHEMA", f"{label} is not a canonical SHA-1 Git object identity")
    return value


@dataclass(frozen=True)
class LaunchPlan:
    context: EntropyContext
    output: Path
    candidate_tag: str
    candidate_commit: str
    candidate_tree: str
    seal_identity: str
    authorization_identity: str


def _strict_seal(value: dict) -> dict:
    keys = {
        "schema", "claim_bearing", "independently_accepted", "candidate_tag",
        "candidate_commit", "candidate_tree", "operative_source_commit",
        "operative_source_tree", "protocol_sha256", "seal_identity",
    }
    expect_exact_keys(value, keys, "external protocol seal")
    if value["schema"] != "exp33ae.external_protocol_seal.v1" or value["claim_bearing"] is not True or value["independently_accepted"] is not True:
        raise StrictValidationError("AUTHORIZATION", "external seal is not independently accepted")
    for field in ("candidate_commit", "candidate_tree", "operative_source_commit", "operative_source_tree"):
        _git_oid(value[field], field)
    for field in ("protocol_sha256", "seal_identity"):
        expect_sha256(value[field], field)
    if type(value["candidate_tag"]) is not str or not value["candidate_tag"].startswith("33a-e-confirmation-protocol-freeze-"):
        raise StrictValidationError("AUTHORIZATION", "candidate tag not an accepted freeze naming class")
    return value


def _strict_authorization(value: dict, seal: dict) -> dict:
    keys = {"schema", "claim_bearing", "explicit_confirmation_authorized", "seal_identity", "authorization_identity", "authorized_stage"}
    expect_exact_keys(value, keys, "external confirmation authorization")
    if (value["schema"] != "exp33ae.external_confirmation_authorization.v1"
        or value["claim_bearing"] is not True
        or value["explicit_confirmation_authorized"] is not True
        or value["authorized_stage"] != "EXP33AE_CONFIRMATION_RUN_V1"
        or value["seal_identity"] != seal["seal_identity"]):
        raise StrictValidationError("AUTHORIZATION", "authorization does not bind accepted seal/stage")
    expect_sha256(value["authorization_identity"], "authorization_identity")
    return value


def _require_detached_clean(root: Path, seal: dict) -> None:
    if git(root, "rev-parse", "HEAD^{commit}") != seal["candidate_commit"]:
        raise StrictValidationError("FREEZE_IDENTITY", "wrong HEAD commit")
    if git(root, "rev-parse", "HEAD^{tree}") != seal["candidate_tree"]:
        raise StrictValidationError("FREEZE_IDENTITY", "wrong HEAD tree")
    if git(root, "rev-parse", f"refs/tags/{seal['candidate_tag']}^{{commit}}") != seal["candidate_commit"]:
        raise StrictValidationError("FREEZE_IDENTITY", "tag commit mismatch")
    if git(root, "rev-parse", f"refs/tags/{seal['candidate_tag']}^{{tree}}") != seal["candidate_tree"]:
        raise StrictValidationError("FREEZE_IDENTITY", "tag tree mismatch")
    branch = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if branch.returncode == 0:
        raise StrictValidationError("FREEZE_IDENTITY", "checkout is not detached")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StrictValidationError("SOURCE_BINDING", "checkout is dirty")


def _preflight_authority(root: Path, seal: dict | None, authorization: dict | None) -> tuple[dict, dict]:
    if seal is None or authorization is None:
        raise StrictValidationError("AUTHORIZATION", "external seal and separate authorization are both required")
    seal = _strict_seal(seal)
    authorization = _strict_authorization(authorization, seal)
    protocol = load_protocol()
    source = protocol["operative_source_freeze"]
    if source["commit"] in {"OPERATIVE_SOURCE_FREEZE_COMMIT", ""} or source["tree"] in {"OPERATIVE_SOURCE_FREEZE_TREE", ""}:
        raise StrictValidationError("FREEZE_IDENTITY", "protocol source identity has not been finalized")
    if (
        seal["operative_source_commit"] != source["commit"]
        or seal["operative_source_tree"] != source["tree"]
        or seal["protocol_sha256"] != protocol_hash()
    ):
        raise StrictValidationError("FREEZE_IDENTITY", "seal does not bind operative source/protocol")
    _require_detached_clean(root, seal)
    preflight = offline_dependency_preflight()
    if not preflight["passed"]:
        raise StrictValidationError("DEPENDENCY_BINDING", str(preflight["mismatches"]))
    return seal, authorization


def _plan(output: Path, seal: dict, authorization: dict, master_hex: str | None) -> LaunchPlan:
    if master_hex is None:
        raise StrictValidationError("ENTROPY_FORMAT", "external master not supplied")
    master = parse_master_hex(master_hex)  # deliberately last
    context = EntropyContext(master, seal["seal_identity"], CLAIM_NAMESPACE, "EXP33AE_CONFIRMATION_RUN_V1", True)
    return LaunchPlan(context, output, seal["candidate_tag"], seal["candidate_commit"], seal["candidate_tree"], seal["seal_identity"], authorization["authorization_identity"])


def preflight_claim_launch(root: Path, output: Path, seal: dict | None, authorization: dict | None, master_hex: str | None) -> LaunchPlan:
    """Validate authority/freeze/dependencies/output before touching master bytes."""
    seal, authorization = _preflight_authority(root, seal, authorization)
    if output.exists():
        raise StrictValidationError("OUTPUT_NAMESPACE", "confirmation output path must be initially absent")
    return _plan(output, seal, authorization, master_hex)


def preflight_claim_resume(root: Path, output: Path, seal: dict | None, authorization: dict | None, master_hex: str | None) -> LaunchPlan:
    """Resume only an existing consumed lineage; never create or re-open calls."""
    seal, authorization = _preflight_authority(root, seal, authorization)
    required = (output / "entropy-commitment.json", output / "registry/registry.jsonl", output / "access/state.json")
    if not output.is_dir() or not all(path.is_file() for path in required):
        raise StrictValidationError("OUTPUT_NAMESPACE", "resume requires an existing bound lineage")
    return _plan(output, seal, authorization, master_hex)


def assert_confirmation_disabled_in_candidate() -> None:
    protocol = load_protocol()
    if not protocol["external_launch_requirements"]["no_valid_seal_or_authorization_object_is_part_of_this_candidate"]:
        raise StrictValidationError("AUTHORIZATION", "candidate unexpectedly contains enabling authority")
