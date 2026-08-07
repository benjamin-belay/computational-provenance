"""Read-only hostile audit for the unsealed protocol-revision candidate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from ..artifacts import sha256_file, write_json
from .allocator import development_exclusions
from .build import _source_manifest
from .entropy import dev_golden_entropy
from .evaluator import FutureStatus, evaluate
from .normative import DEV_NAMESPACE, protocol_hash
from .polarity import load_fixture_package
from .release import verify_index
from .training_recipe import validate_training_recipe_against_implementation


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _check(condition: bool, evidence) -> dict:
    return {"passed": type(condition) is bool and condition, "evidence": evidence}


def run_audit(root: Path, output: Path, require_candidate_tag: bool = True, verify_live_predecessor: bool = True) -> dict:
    fixture = root / "artifacts/dev_only/preconfirmation-revision-v1"
    results = root / "results/dev_only/preconfirmation-revision-v1"
    schema_dir = root / "schemas/dev_only/preconfirmation-v1"
    checks = {}
    baseline = json.loads((root / "protocol/dev_only/protected-baseline-v1.json").read_text())

    protected = {}
    for tag, values in baseline["tags"].items():
        protected[tag] = {"object": _git(root, "rev-parse", tag), "peeled": _git(root, "rev-parse", f"{tag}^{{}}")}
    checks["01_protected_tags_unchanged"] = _check(protected == baseline["tags"], protected)

    if verify_live_predecessor:
        predecessor = root.parent / baseline["predecessor"]["path_name"]
        predecessor_state = {
            "mode": "live_source_workspace_check", "head": _git(predecessor, "rev-parse", "HEAD"),
            "tree": _git(predecessor, "rev-parse", "HEAD^{tree}"),
            "status": _git(predecessor, "status", "--porcelain", "--untracked-files=all"),
        }
        predecessor_ok = (
            predecessor_state["head"] == baseline["predecessor"]["head"]
            and predecessor_state["tree"] == baseline["predecessor"]["tree"] and predecessor_state["status"] == ""
        )
    else:
        predecessor_state = {
            "mode": "self_contained_recorded_baseline_check", "head": baseline["predecessor"]["head"],
            "tree": baseline["predecessor"]["tree"], "live_sibling_accessed": False,
        }
        predecessor_ok = all(
            type(baseline["predecessor"][field]) is str and len(baseline["predecessor"][field]) == 40
            for field in ("head", "tree")
        )
    checks["02_predecessor_unchanged_or_self_contained_baseline"] = _check(predecessor_ok, predecessor_state)

    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    candidate_tags = sorted(
        tag for tag in _git(root, "tag", "--points-at", "HEAD").splitlines()
        if tag.startswith("protocol-revision-candidate-v") and tag.removeprefix("protocol-revision-candidate-v").isdigit()
    )
    tag_ok = (not require_candidate_tag) or bool(candidate_tags) and all(
        _git(root, "rev-parse", f"{tag}^{{}}") == _git(root, "rev-parse", "HEAD") for tag in candidate_tags
    )
    checks["03_clean_candidate_git_state"] = _check(status == "" and tag_ok, {"status": status, "head": _git(root, "rev-parse", "HEAD"), "tree": _git(root, "rev-parse", "HEAD^{tree}"), "tag_required": require_candidate_tag, "candidate_tags_at_head": candidate_tags, "tag_ok": tag_ok})

    build = json.loads((fixture / "build-report.json").read_text())
    checks["04_protocol_hash_and_unsealed_boundary"] = _check(
        build["protocol_sha256"] == protocol_hash()
        and build["terminal_status_materialized"] is False
        and build["confirmation_objects_created"] == build["confirmation_objects_accessed"] == 0,
        {"protocol_sha256": protocol_hash(), "build_protocol_sha256": build["protocol_sha256"], "terminal_status_materialized": build["terminal_status_materialized"]},
    )

    source = _source_manifest(root, schema_dir)
    stored_source = json.loads((fixture / "source-schema-manifest.json").read_text())
    checks["05_source_and_schema_manifest_reconstructs"] = _check(source == stored_source, {"expected": stored_source["manifest_sha256"], "observed": source["manifest_sha256"], "files": len(source["files"])})

    entropy = dev_golden_entropy()
    golden = json.loads((fixture / "entropy-golden.json").read_text())
    entropy_ok = golden["commitment"] == entropy.commitment and golden["ordered_model_seed_identities"] == list(entropy.ordered_model_seeds()) and all(
        values == [entropy.derive(domain, index, 32).hex() for index in range(3)]
        for domain, values in golden["derived_golden_vectors"].items()
    )
    checks["06_entropy_golden_and_no_confirmation_entropy"] = _check(entropy_ok and golden["confirmation_master_entropy_generated"] is False, {"commitment": golden["commitment"], "model_seed_identities": golden["ordered_model_seed_identities"], "confirmation_master_entropy_generated": golden["confirmation_master_entropy_generated"]})

    package = load_fixture_package(root, fixture)
    allocation = package["allocation"]
    counts = {partition: sum(row.partition == partition for row in allocation["examples"]) for partition in {row.partition for row in allocation["examples"]}}
    checks["07_fresh_allocator_counts_and_exclusions"] = _check(
        counts == {"training": 8192, "model_selection": 480, "localization_fit": 480, "localization_selection": 480, "clean_target_axis": 800, "clean_parallel_axis": 800}
        and not ({row["programme_id"] for row in allocation["programmes"]} & development_exclusions(root)["programme_ids"]),
        {"programme_count": len(allocation["programmes"]), "example_counts": counts, "development_overlap": 0},
    )

    allocation_audit = json.loads((fixture / "allocation-audit.json").read_text())
    checks["08_within_run_axes_balance_and_shortcuts"] = _check(
        allocation_audit["within_run_heldout_only"] is True
        and allocation_audit["globally_unseen_compositions_claimed"] is False
        and set(allocation_audit["shortcut_baselines"]) == {"clean_target_axis", "clean_parallel_axis"}
        and all(len(values) == 10 for values in allocation_audit["shortcut_baselines"].values()),
        {"within_run_only": allocation_audit["within_run_heldout_only"], "shortcut_baselines": allocation_audit["shortcut_baselines"]},
    )

    training = validate_training_recipe_against_implementation()
    checks["09_normative_training_recipe_complete"] = _check(training["passed"] is True and all(training["checks"].values()), training)

    representation = json.loads((fixture / "representation-golden.json").read_text())
    checks["10_representation_golden_reconstructs"] = _check(
        representation["basis_orthonormal_max_error"] == 0.0
        and representation["swapped_coordinates"] == representation["source_coordinates"]
        and representation["delta_nonzero_indices"] == [0, 1],
        representation,
    )

    pairs = [json.loads(line) for line in (fixture / "primary-pairs.jsonl").read_text().splitlines() if line.strip()]
    pair_counts = {name: sum(row["stratum"] == name for row in pairs) for name in {row["stratum"] for row in pairs}}
    checks["11_exact_pair_strata"] = _check(pair_counts == {"A_answer_changing": 160, "B_outcome_collision": 80, "C_same_state": 80, "D_same_observable_answer": 80} and len({(row["base_id"], row["source_id"]) for row in pairs}) == 400, pair_counts)

    controls = json.loads((fixture / "controls.json").read_text())["controls"]
    control_counts = {name: sum(row["family"] == name for row in controls) for name in {row["family"] for row in controls}}
    checks["12_exact_control_construction"] = _check(control_counts == {"random_direction": 4, "orthogonal_direction": 4, "alternate_location": 3, "alternate_early_layer": 1}, control_counts)

    registry_manifest = json.loads((fixture / "registry-manifest.json").read_text())
    checks["13_registry_11760_and_commitment"] = _check(registry_manifest["row_count"] == 11760 and registry_manifest["registry_file_sha256"] == sha256_file(fixture / "registry.jsonl") and registry_manifest["confirmation_registry_materialized"] is False, registry_manifest)

    healthy = evaluate(package, root)
    checks["14_strict_evaluator_all_seed_positive_fixture"] = _check(healthy["simulated_status"] == FutureStatus.POSITIVE.value and len(healthy["simulated_seed_statuses"]) == 3 and healthy["terminal_status_materialized"] is False, {"simulated_status": healthy["simulated_status"], "seed_statuses": healthy.get("simulated_seed_statuses"), "terminal_status_materialized": healthy["terminal_status_materialized"]})

    access = json.loads((fixture / "access-happy-report.json").read_text())
    access_scenarios = sorted(fixture.glob("polarity-access-fixtures-shard-*/*/dev_only_access_journal.jsonl"))
    checks["15_one_access_and_buffered_release"] = _check(access["invalid"] is False and access["released"] is True and len(access_scenarios) >= 8, {"happy_events": [row["event"] for row in access["events"]], "polarity_access_journals": len(access_scenarios)})

    polarity = json.loads((fixture / "polarity-report.json").read_text())
    statuses = {row["actual_simulated_status"] for row in polarity["scenarios"]}
    checks["16_real_upstream_polarity_complete"] = _check(polarity["scenario_count"] == polarity["passed_count"] == 77 and polarity["all_real_upstream"] is True and statuses == {status.value for status in FutureStatus}, {"scenario_count": polarity["scenario_count"], "passed_count": polarity["passed_count"], "statuses": sorted(statuses)})

    test_report = json.loads((results / "test-report.json").read_text())
    checks["17_tests_all_outcomes_disclosed"] = _check(test_report["passed_suite"] is True and test_report["passed"] == test_report["collected"] and all(test_report[name] == 0 for name in ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed")), test_report)

    index_result = verify_index(root, results / "artifact-index.json")
    envelope = json.loads((results / "layer-1-finalizer-envelope.json").read_text())
    checks["18_actual_index_and_nonplaceholder_finalizer"] = _check(index_result["matches"] is True and envelope["consumed_placeholder"] is False and envelope["actual_reconstructed_payload_index_sha256"] == index_result["observed_payload_sha256"] and envelope["layer_2_claimed_passed"] is False, {"index": index_result, "envelope": envelope})

    new_names = _git(root, "diff", "--name-only", baseline["repository_baseline"]["head"], "HEAD").splitlines()
    deleted_names = _git(root, "log", "--diff-filter=D", "--name-only", "--pretty=format:", f"{baseline['repository_baseline']['head']}..HEAD").splitlines()
    allowed_new = all(
        name.startswith(("artifacts/dev_only/", "results/dev_only/", "schemas/dev_only/", "protocol/dev_only/", "docs/", "src/exp33ae/preconfirmation/", "tests/"))
        or name in {"README.md", "src/exp33ae/cli.py"}
        for name in new_names if name
    )
    checks["19_no_claim_bearing_confirmation_or_deleted_object"] = _check(allowed_new and not [name for name in deleted_names if name] and build["confirmation_objects_created"] == build["confirmation_objects_accessed"] == 0, {"new_files": len(new_names), "deleted_files": [name for name in deleted_names if name], "confirmation_objects_created": 0, "confirmation_objects_accessed": 0})

    verdict = json.loads((results / "revision-verdict.json").read_text())
    checks["20_pending_review_only_status"] = _check(verdict["status"] == "PROTOCOL_REVISION_IMPLEMENTED_PENDING_REVIEW" and verdict["protocol_sealing_authorized"] is False and verdict["confirmation_authorized"] is False, verdict)

    passed = all(row["passed"] for row in checks.values())
    report = {
        "schema": "exp33ae.protocol_revision_hostile_audit.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "reviewed_head": _git(root, "rev-parse", "HEAD"), "reviewed_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "checks": checks, "passed_checks": sum(row["passed"] for row in checks.values()), "total_checks": len(checks),
        "passed": passed, "status": "PROTOCOL_REVISION_IMPLEMENTED_PENDING_REVIEW" if passed else "PROTOCOL_REVISION_INCOMPLETE",
        "protocol_sealing_authorized": False, "confirmation_authorized": False,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    write_json(output, report)
    return report
