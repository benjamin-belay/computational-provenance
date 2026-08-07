"""Hostile read-only reconstruction of a frozen DEV_ONLY readiness candidate."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

import torch

from .artifacts import sha256_file, write_json
from .evaluate import read_jsonl
from .spec import GATES, MODEL_SEEDS, NAMESPACE, SEARCH_GRID
from .status import DevelopmentStatus, REQUIRED_GROUPS, evaluate_readiness
from .task_audit import run_task_audit
from .training import load_checkpoint


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _check(condition: bool, evidence) -> dict:
    return {"passed": type(condition) is bool and condition, "evidence": evidence}


def _json(path: Path):
    return json.loads(path.read_text())


def _json_normalize(value):
    """Compare in-memory structures exactly as their committed JSON encoding."""
    return json.loads(json.dumps(value, sort_keys=True))


def run_hostile_audit(root: Path, output: Path) -> dict:
    """Inspect *root* without writing beneath it; write only to explicit output."""
    root = root.resolve()
    checks: dict[str, dict] = {}

    governance = (root / "GOVERNANCE.md").read_text()
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    tags = set(_git(root, "tag", "--points-at", "HEAD").splitlines())
    checks["01_repository_governance_and_clean_git"] = _check(
        status == ""
        and "DEVELOPMENT_NOT_READY" in governance
        and "33a-N natural emergence is a separate future claim" in governance
        and "zero claim-bearing confirmation" in governance,
        {"head": _git(root, "rev-parse", "HEAD"), "tree": _git(root, "rev-parse", "HEAD^{tree}"), "status": status, "head_tags": sorted(tags)},
    )

    generated = [
        path for base in ("artifacts", "checkpoints", "data", "registries", "results")
        for path in (root / base).rglob("*") if path.is_file()
    ]
    wrong_namespace_paths = [
        str(path.relative_to(root)) for path in generated
        if "dev_only" not in {part.lower() for part in path.relative_to(root).parts}
    ]
    zero_fields, future_seeds = [], []
    for path in generated:
        if path.suffix not in {".json", ".jsonl"}:
            continue
        for line in path.read_text().splitlines() if path.suffix == ".jsonl" else [path.read_text()]:
            if not line.strip():
                continue
            value = json.loads(line)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, dict):
                    continue
                for key in ("confirmation_artifacts_created", "confirmation_artifacts_accessed"):
                    if key in item and item[key] != 0:
                        zero_fields.append((str(path.relative_to(root)), key, item[key]))
                if item.get("future_generation_seed") is not None:
                    future_seeds.append((str(path.relative_to(root)), item["future_generation_seed"]))
    checks["02_confirmation_isolation"] = _check(
        not wrong_namespace_paths and not zero_fields and not future_seeds,
        {"generated_files": len(generated), "wrong_namespace_paths": wrong_namespace_paths, "nonzero_access_fields": zero_fields, "future_seeds": future_seeds, "claim_bearing_confirmation_objects": 0},
    )

    saved_task = _json(root / "results/dev_only/task-design-audit.json")
    reconstructed_task = run_task_audit(materialize=False)
    task_keys = (
        "actual_programme_count", "unique_programme_ids", "unique_exhaustive_causal_signatures",
        "records", "exact_duplicate_count", "training_programme_overlap",
        "training_target_composition_overlap", "training_parallel_composition_overlap",
        "production_oracle", "counterfactual_oracle", "eligible_ordered_pair_counts",
        "best_shortcut_by_axis", "balance", "passed",
    )
    task_matches = all(
        saved_task[key] == _json_normalize(reconstructed_task[key])
        for key in task_keys
    )
    checks["03_task_and_independent_oracle_reconstruction"] = _check(
        task_matches and reconstructed_task["passed"]
        and reconstructed_task["production_oracle"]["mismatches"] == 0
        and reconstructed_task["counterfactual_oracle"]["mismatches"] == 0,
        {key: reconstructed_task[key] for key in task_keys},
    )
    checks["04_genuine_heldout_axes_and_split_identity"] = _check(
        reconstructed_task["exact_duplicate_count"] == 0
        and reconstructed_task["training_target_composition_overlap"]["causal_target_composition"] == 0
        and reconstructed_task["training_parallel_composition_overlap"]["causal_parallel_composition"] == 0
        and all(value == 0 for value in reconstructed_task["training_programme_overlap"].values()),
        {
            "exact_duplicates": reconstructed_task["exact_duplicate_count"],
            "programme_overlap": reconstructed_task["training_programme_overlap"],
            "target_overlap": reconstructed_task["training_target_composition_overlap"],
            "parallel_overlap": reconstructed_task["training_parallel_composition_overlap"],
            "primitive_transition_overlap": reconstructed_task["evaluation_primitive_semantic_transition_overlap_with_training"],
        },
    )
    shortcuts = reconstructed_task["best_shortcut_by_axis"]
    checks["05_shortcut_baselines_materially_below_competence"] = _check(
        shortcuts["heldout_target_composition"] <= 0.65
        and shortcuts["heldout_parallel_composition"] <= 0.65,
        {"best": shortcuts, "all": reconstructed_task["shortcut_baselines"]},
    )

    training = _json(root / "results/dev_only/training-full.json")
    expected_models = [(kind, seed) for kind in MODEL_SEEDS for seed in MODEL_SEEDS[kind]]
    observed_models = [(row["kind"], row["seed"]) for row in training["models"]]
    checkpoint_faults = []
    parameter_counts = set()
    for row in training["models"]:
        relative = f"checkpoints/dev_only/{row['kind']}-seed-{row['seed']}.pt"
        path = root / relative
        if not path.is_file() or sha256_file(path) != row["checkpoint_sha256"]:
            checkpoint_faults.append(relative)
            continue
        model, payload = load_checkpoint(path)
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        if payload.get("kind") != row["kind"] or payload.get("seed") != row["seed"]:
            checkpoint_faults.append(relative + ":identity")
    checks["06_model_population_and_checkpoint_identity"] = _check(
        observed_models == expected_models and not checkpoint_faults and parameter_counts == {136324},
        {"expected": expected_models, "observed": observed_models, "faults": checkpoint_faults, "parameter_counts": sorted(parameter_counts)},
    )

    evaluation = _json(root / "results/dev_only/development-evaluation.json")
    model_results = evaluation["model_results"]
    iit = [row for row in model_results if row["model_kind"] == "iit_primary"]
    ordinary = [row for row in model_results if row["model_kind"] == "ordinary"]
    checks["07_primary_competence_and_all_seed_replication"] = _check(
        [row["model_seed"] for row in iit] == list(MODEL_SEEDS["iit_primary"])
        and all(row["decision"]["status"] == DevelopmentStatus.OBSERVED.value for row in iit)
        and evaluation["iit_primary_aggregate"]["status"] == DevelopmentStatus.OBSERVED.value,
        {"iit_seed_statuses": [(row["model_seed"], row["decision"]["status"], row["competence"]["correct"], row["competence"]["total"]) for row in iit], "aggregate": evaluation["iit_primary_aggregate"]},
    )
    checks["08_ordinary_population_separated"] = _check(
        all(row["decision"]["status"] == DevelopmentStatus.COMPETENCE_FAILED.value for row in ordinary)
        and evaluation["ordinary_aggregate"]["status"] == DevelopmentStatus.COMPETENCE_FAILED.value,
        {"ordinary_seed_statuses": [(row["model_seed"], row["decision"]["status"], row["competence"]["correct"], row["competence"]["total"]) for row in ordinary], "aggregate": evaluation["ordinary_aggregate"]},
    )

    search = _json(root / "results/dev_only/method-search.json")
    with (root / "results/dev_only/method-search.csv").open(newline="") as handle:
        search_csv = list(csv.DictReader(handle))
    expected_grid = 2 * 4 * 2 * 2 * 2 * 6 + 3
    declared = [row for row in search["rows"] if row["method"] == "iit_declared"]
    checks["09_complete_method_search_denominator"] = _check(
        search["total_rows"] == expected_grid == len(search["rows"]) == len(search_csv)
        and len(declared) == 3,
        {"expected_rows": expected_grid, "json_rows": len(search["rows"]), "csv_rows": len(search_csv), "declared_rows": len(declared), "grid": SEARCH_GRID},
    )

    freeze = _json(root / "registries/dev_only/evaluation-freeze.json")
    registry_path = root / freeze["call_registry"]
    registry = read_jsonl(registry_path)
    registry_ids = [row["call_id"] for row in registry]
    plan_commit = _git(root, "rev-list", "-n", "1", "dev-evaluation-plan-v4")
    evaluation_commit = _git(root, "rev-list", "-n", "1", "development-evaluation-v5")
    committed_before = subprocess.run(["git", "merge-base", "--is-ancestor", plan_commit, evaluation_commit], cwd=root).returncode == 0
    checks["10_precall_registry_committed_and_complete"] = _check(
        len(registry) == 5760 == len(set(registry_ids))
        and sha256_file(registry_path) == freeze["call_registry_sha256"]
        and committed_before,
        {"rows": len(registry), "unique_ids": len(set(registry_ids)), "sha256": sha256_file(registry_path), "plan_commit": plan_commit, "evaluation_commit": evaluation_commit, "ancestor": committed_before},
    )

    ledger = read_jsonl(root / "registries/dev_only/evaluation-call-ledger.jsonl")
    ledger_ids = [row["call_id"] for row in ledger]
    observed_groups = Counter((row["model_kind"], row["model_seed"], row["control_family"], row["pair_kind"]) for row in ledger)
    per_seed_valid = all(
        observed_groups[(kind, seed, family, pair_kind)] == count
        for kind in ("ordinary", "iit_primary") for seed in MODEL_SEEDS[kind]
        for (family, pair_kind), count in REQUIRED_GROUPS.items()
    )
    checks["11_call_ledger_exact_reconciliation"] = _check(
        len(ledger) == evaluation["attempted_calls"] == evaluation["completed_calls"] == 5760
        and len(set(ledger_ids)) == 5760 and set(ledger_ids) == set(registry_ids)
        and per_seed_valid and all(row["completed"] is True for row in ledger),
        {"planned": len(registry), "attempted": len(ledger), "completed": sum(row["completed"] is True for row in ledger), "unique": len(set(ledger_ids)), "exact_group_counts": per_seed_valid},
    )

    reconstructed_metric = {}
    for result in model_results:
        key = (result["model_kind"], result["model_seed"])
        rows = [row for row in ledger if (row["model_kind"], row["model_seed"]) == key]
        causal = [row for row in rows if row["control_family"] == "causal_candidate" and row["pair_kind"] == "different_state"]
        reconstructed_metric[key] = {
            "iia": sum(row["intervened_prediction"] == row["expected_joint"] for row in causal) / len(causal),
            "state": sum(row["decoded_intervened_z2"] == row["source_z2"] for row in causal) / len(causal),
            "descendant": sum(row["intervened_z3"] == row["expected_z3"] for row in causal) / len(causal),
            "non_descendant": sum(row["intervened_n3"] == row["expected_n3"] for row in causal) / len(causal),
        }
    metric_match = all(
        reconstructed_metric[(row["model_kind"], row["model_seed"])][name] == row["decision"]["metrics"][stored]
        for row in model_results
        for name, stored in (("iia", "different_state_iia"), ("state", "state_transport"), ("descendant", "descendant_correctness"), ("non_descendant", "non_descendant_invariance"))
    )
    checks["12_raw_metric_reconstruction"] = _check(metric_match, {f"{kind}:{seed}": values for (kind, seed), values in reconstructed_metric.items()})

    norm_finite = all(
        isinstance(row["norm_ratio"], (int, float))
        and row["norm_ratio"] == row["norm_ratio"]
        and row["intervention_norm"] >= 0
        for row in ledger
    )
    checks["13_specificity_controls_and_intervention_accounting"] = _check(
        norm_finite and all(row["integrity"]["control_distinctness_match"] is True for row in model_results),
        {"norm_rows_finite": norm_finite, "control_families": sorted({row["control_family"] for row in ledger}), "all_distinctness_checks": all(row["integrity"]["control_distinctness_match"] is True for row in model_results)},
    )

    required_gate_names = {
        "task_accuracy_each_axis", "minimum_programme_accuracy", "shortcut_advantage", "decode_accuracy",
        "different_state_iia", "state_transport", "descendant_correctness", "non_descendant_invariance",
        "same_state_preservation", "same_answer_different_state_iia", "random_direction_gap",
        "orthogonal_direction_gap", "random_location_gap", "random_layer_gap", "call_base_source_accuracy",
        "call_completion", "required_primary_seeds",
    }
    checks["14_status_gate_completeness_and_precedence"] = _check(
        set(GATES) == required_gate_names
        and all(row["decision"]["faults"] == [] for row in model_results)
        and evaluation["iit_primary_aggregate"]["status"] == DevelopmentStatus.OBSERVED.value,
        {"gate_names": sorted(GATES), "primary_status": evaluation["iit_primary_aggregate"]["status"], "integrity_faults": {f"{row['model_kind']}:{row['model_seed']}": row["decision"]["faults"] for row in model_results}},
    )

    polarity = _json(root / "results/dev_only/real-upstream-polarity.json")
    substantive = all(
        row.get("substantive_check") is True
        and row.get("direct_summary_or_terminal_mutation") is False
        for row in polarity["scenarios"]
    )
    checks["15_real_upstream_polarity"] = _check(
        polarity["scenario_count"] == polarity["passed_count"] == 26 and substantive,
        {"scenario_count": polarity["scenario_count"], "passed_count": polarity["passed_count"], "substantive": substantive, "scenarios": [(row["scenario"], row["statuses"], row["passed"]) for row in polarity["scenarios"]]},
    )

    test_report = _json(root / "results/dev_only/test-report.json")
    checks["16_test_suite_all_outcomes_disclosed"] = _check(
        test_report["passed_suite"] is True
        and test_report["passed"] == test_report["collected"]
        and all(test_report[name] == 0 for name in ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed")),
        test_report,
    )

    index = _json(root / "results/dev_only/artifact-index.json")
    index_faults = [
        row["path"] for row in index["files"]
        if not (root / row["path"]).is_file()
        or (root / row["path"]).stat().st_size != row["bytes"]
        or sha256_file(root / row["path"]) != row["sha256"]
    ]
    required_repro = all((root / name).is_file() for name in (
        "requirements-lock.txt", "README.md", "docs/REPRODUCIBILITY.md", "results/dev_only/runtime-report.json"
    ))
    checks["17_artifact_index_and_reproducibility"] = _check(
        not index_faults and required_repro,
        {"indexed_files": index["file_count"], "index_faults": index_faults, "required_repro_files": required_repro},
    )

    protocol = (root / "docs/PRECONFIRMATION_PROTOCOL.md").read_text()
    required_phrases = (
        "fresh predetermined model seeds", "manual authorization", "no confirmation seed",
        "33a-E", "33a-N", "invalidity precedes", "all three",
    )
    checks["18_protocol_only_freeze_scope_complete"] = _check(
        all(phrase.lower() in protocol.lower() for phrase in required_phrases),
        {"required_phrases": list(required_phrases), "missing": [phrase for phrase in required_phrases if phrase.lower() not in protocol.lower()]},
    )

    required_checks = {name: check["passed"] for name, check in checks.items()}
    readiness = evaluate_readiness(evaluation["iit_primary_aggregate"], required_checks)
    passed = all(required_checks.values()) and readiness["status"] == DevelopmentStatus.READY.value
    report = {
        "schema": "exp33ae.hostile_readiness_audit.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "reviewed_root_name": root.name,
        "reviewed_head": _git(root, "rev-parse", "HEAD"),
        "reviewed_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "checks": checks,
        "passed_checks": sum(check["passed"] for check in checks.values()),
        "total_checks": len(checks),
        "passed": passed,
        "readiness": readiness,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
        "confirmation_authorized": False,
    }
    write_json(output, report)
    return report
