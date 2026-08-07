"""Replayable DEV_ONLY polarity over real ledgers and labelled synthetic boundaries.

Integrity cases mutate claim-shaped upstream registry/raw objects before the
normal strict validators.  Integer boundary cases mutate raw observations and
are explicitly labelled synthetic; they are not represented as model evidence.
"""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

from .evaluator import (
    _metrics_for_seed,
    _seed_status,
    POSITIVE_ANALOGUE,
    STATUS_COMPETENCE,
    STATUS_DECODE_ONLY,
    STATUS_INVALID,
    STATUS_NEGATIVE,
    STATUS_NO_ELIGIBLE,
)
from .normative import DEV_NAMESPACE
from .strict import object_sha256, strict_load_json, strict_load_jsonl, StrictValidationError, write_json
from .validation import validate_raw, validate_registry


def _aggregate(statuses: list[str]) -> str:
    for status in (STATUS_COMPETENCE, STATUS_NO_ELIGIBLE, STATUS_DECODE_ONLY, STATUS_NEGATIVE):
        if status in statuses:
            return status
    return POSITIVE_ANALOGUE if statuses == [POSITIVE_ANALOGUE] * 3 else STATUS_INVALID


def adjudicate_rows(registry: list[dict], manifest: dict, raw: list[dict], seeds: list[int], shortcuts: dict) -> dict:
    """The production row validators and scientific aggregator, fail closed."""
    try:
        validate_registry(registry, manifest, seeds)
        validate_raw(raw, registry)
        statuses, metrics = [], {}
        for order, seed in enumerate(seeds):
            values = _metrics_for_seed(
                registry[order * 3920:(order + 1) * 3920],
                raw[order * 3920:(order + 1) * 3920],
                shortcuts,
            )
            metrics[str(seed)] = values
            statuses.append(_seed_status(values))
        return {"status": _aggregate(statuses), "seed_statuses": statuses, "metrics_by_seed": metrics, "invalidity_reasons": []}
    except StrictValidationError as error:
        return {"status": STATUS_INVALID, "seed_statuses": [], "metrics_by_seed": {}, "invalidity_reasons": [error.reason]}
    except Exception as error:  # malformed packages never escape uncontrolled
        return {"status": STATUS_INVALID, "seed_statuses": [], "metrics_by_seed": {}, "invalidity_reasons": ["INTERNAL_VALIDATION_EXCEPTION"], "detail": type(error).__name__}


def _rehash(row: dict) -> dict:
    value = dict(row)
    value.pop("raw_row_sha256", None)
    value["raw_row_sha256"] = object_sha256(value)
    return value


def _replace(raw: list[dict], replacements: dict[int, dict]) -> list[dict]:
    values = list(raw)
    for index, row in replacements.items():
        values[index] = _rehash(row)
    return values


def _synthetic_positive(registry: list[dict], raw: list[dict]) -> list[dict]:
    values = []
    for plan, observed in zip(registry, raw):
        row = dict(observed)
        row["base_prediction"] = plan["base_answer"]
        row["source_prediction"] = plan["source_answer"]
        row["decoded_z2"] = plan["expected_z2"]
        row["intervened_z3"] = plan["expected_z3"]
        row["intervened_n3"] = plan["expected_n3"]
        if plan["method"] in {"clean_task_decode", "declared_iit_alignment"}:
            row["intervened_prediction"] = plan["expected_answer"]
        else:
            row["intervened_prediction"] = (plan["expected_answer"] + 1) % 4
        values.append(_rehash(row))
    return values


def _force_count(registry, raw, selector, raw_field: str, expected_field: str, correct: int, classes: int) -> list[dict]:
    indexes = [index for index, plan in enumerate(registry) if selector(plan)]
    if not 0 <= correct <= len(indexes):
        raise ValueError("synthetic correct count outside selected denominator")
    replacements = {}
    for offset, index in enumerate(indexes):
        row = dict(raw[index]); expected = registry[index][expected_field]
        row[raw_field] = expected if offset < correct else (expected + 1) % classes
        replacements[index] = row
    return _replace(raw, replacements)


def run_polarity(output: Path, report_path: Path | None = None) -> dict:
    registry = strict_load_jsonl(output / "registry/registry.jsonl")
    manifest = strict_load_json(output / "registry/manifest.json")
    raw = strict_load_jsonl(output / "access/buffer.jsonl")
    entropy = strict_load_json(output / "entropy.json")
    allocation = strict_load_json(output / "allocation/manifest.json")
    seeds = entropy["model_seed_identities"]
    shortcuts = allocation["shortcut_baselines"]
    scenarios = []

    def record(name, expected, altered_raw=None, altered_registry=None, altered_manifest=None, category="real_upstream_object", note=None):
        result = adjudicate_rows(
            registry if altered_registry is None else altered_registry,
            manifest if altered_manifest is None else altered_manifest,
            raw if altered_raw is None else altered_raw,
            seeds,
            shortcuts,
        )
        scenarios.append({
            "scenario": name, "category": category, "expected_status": expected,
            "actual_status": result["status"], "passed": result["status"] == expected,
            "invalidity_reasons": result["invalidity_reasons"], "note": note,
            "direct_metric_or_status_edit": False,
        })

    actual = adjudicate_rows(registry, manifest, raw, seeds, shortcuts)
    record("actual_three_model_rehearsal_replay", actual["status"], note="actual trained-model ledger")

    # Real ledger/schema objects enter the production validators.
    record("missing_call", STATUS_INVALID, raw[:-1])
    record("duplicate_call", STATUS_INVALID, raw + [raw[-1]])
    changed = dict(raw[0]); changed["call_id"] = "devonly-remcall-" + "0" * 32
    record("unplanned_call_identity", STATUS_INVALID, _replace(raw, {0: changed}))
    for name, field, value in (
        ("failed_call", "completed", False), ("dropped_call", "dropped", True),
        ("retried_call", "attempt_number", 2), ("wrong_type", "completed", 1),
        ("wrong_hash_format", "basis_sha256", "A" * 64), ("unknown_method", "method", "unknown"),
        ("norm_mismatch", "applied_norm", float(raw[0]["applied_norm"]) + 1.0),
    ):
        row = dict(raw[0]); row[field] = value
        record(name, STATUS_INVALID, _replace(raw, {0: row}))
    missing_field = dict(raw[0]); missing_field.pop("nonce")
    values = list(raw); values[0] = missing_field
    record("missing_json_field", STATUS_INVALID, values)
    extra_field = dict(raw[0]); extra_field["unexpected"] = 1
    record("extra_json_field", STATUS_INVALID, _replace(raw, {0: extra_field}))
    nan_row = dict(raw[0]); nan_row["applied_norm"] = float("nan")
    values = list(raw); values[0] = nan_row
    record("nonfinite_number", STATUS_INVALID, values)

    reordered = list(registry); reordered[0], reordered[1] = reordered[1], reordered[0]
    record("reordered_registry", STATUS_INVALID, altered_registry=reordered)
    duplicated = list(registry); duplicated[1] = duplicated[0]
    record("duplicated_registry_identity", STATUS_INVALID, altered_registry=duplicated)
    incomplete = [row for row in registry if row["control_id"] != "random_direction_0"]
    record("incomplete_control_family", STATUS_INVALID, altered_registry=incomplete)
    bad_manifest = dict(manifest); bad_manifest["model_seed_identities"] = [seeds[1], seeds[0], seeds[2]]
    record("reordered_seed_identities", STATUS_INVALID, altered_manifest=bad_manifest)
    bad_manifest = dict(manifest); bad_manifest["model_seed_identities"] = [seeds[0], seeds[1], seeds[1]]
    record("duplicated_seed_identity", STATUS_INVALID, altered_manifest=bad_manifest)

    # Complete synthetic observation scaffold for exact integer boundary tests.
    # These cases exercise normal raw validation and aggregation but are not
    # checkpoint-derived scientific evidence.
    synthetic = _synthetic_positive(registry, raw)
    synthetic_category = "synthetic_raw_threshold_fixture"
    record("synthetic_positive_analogue", POSITIVE_ANALOGUE, synthetic, category=synthetic_category)
    seed = seeds[0]
    target = lambda p: p["model_seed_identity"] == seed and p["method"] == "clean_task_decode" and p["axis"].endswith("target_composition")
    parallel = lambda p: p["model_seed_identity"] == seed and p["method"] == "clean_task_decode" and p["axis"].endswith("parallel_composition")
    clean = lambda p: p["model_seed_identity"] == seed and p["method"] == "clean_task_decode"
    primary = lambda stratum: lambda p: p["model_seed_identity"] == seed and p["method"] == "declared_iit_alignment" and p["pair_stratum"] == stratum
    record("target_task_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, target, "intervened_prediction", "expected_answer", 679, 4), category=synthetic_category)
    record("parallel_task_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, parallel, "intervened_prediction", "expected_answer", 679, 4), category=synthetic_category)
    target_programme = next(plan["programme_id"] for plan in registry if target(plan))
    programme = lambda p: target(p) and p["programme_id"] == target_programme
    record("worst_programme_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, programme, "intervened_prediction", "expected_answer", 11, 4), category=synthetic_category)
    best_target = max(value["correct"] for value in shortcuts["clean_target_axis"].values())
    best_parallel = max(value["correct"] for value in shortcuts["clean_parallel_axis"].values())
    record("target_shortcut_gap_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, target, "intervened_prediction", "expected_answer", best_target + 159, 4), category=synthetic_category)
    record("parallel_shortcut_gap_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, parallel, "intervened_prediction", "expected_answer", best_parallel + 159, 4), category=synthetic_category)
    record("decode_one_below", STATUS_NO_ELIGIBLE, _force_count(registry, synthetic, clean, "decoded_z2", "expected_z2", 1439, 2), category=synthetic_category)
    record("answer_changing_iia_one_below", STATUS_DECODE_ONLY, _force_count(registry, synthetic, primary("A_answer_changing"), "intervened_prediction", "expected_answer", 135, 4), category=synthetic_category)

    for stratum, total, minimum in (
        ("A_answer_changing", 160, 136), ("B_outcome_collision", 80, 68), ("D_same_observable_answer", 80, 68),
    ):
        record(f"transport_{stratum}_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, primary(stratum), "decoded_z2", "expected_z2", minimum - 1, 2), category=synthetic_category)
        record(f"descendant_{stratum}_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, primary(stratum), "intervened_z3", "expected_z3", minimum - 1, 2), category=synthetic_category)
    pooled = _force_count(registry, synthetic, primary("A_answer_changing"), "decoded_z2", "expected_z2", 151, 2)
    pooled = _force_count(registry, pooled, primary("B_outcome_collision"), "decoded_z2", "expected_z2", 68, 2)
    pooled = _force_count(registry, pooled, primary("D_same_observable_answer"), "decoded_z2", "expected_z2", 68, 2)
    record("transport_pooled_one_below", STATUS_NEGATIVE, pooled, category=synthetic_category)
    for stratum, minimum in (
        ("A_answer_changing", 152), ("B_outcome_collision", 76),
        ("C_same_state", 76), ("D_same_observable_answer", 76),
    ):
        record(f"non_descendant_{stratum}_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, primary(stratum), "intervened_n3", "expected_n3", minimum - 1, 2), category=synthetic_category)
    record("same_state_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, primary("C_same_state"), "intervened_prediction", "base_answer", 75, 4), category=synthetic_category)
    record("same_answer_different_state_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, primary("D_same_observable_answer"), "intervened_prediction", "expected_answer", 63, 4), category=synthetic_category)
    record("base_accuracy_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, primary("A_answer_changing"), "base_prediction", "base_answer", 135, 4), category=synthetic_category)
    record("source_accuracy_one_below", STATUS_COMPETENCE, _force_count(registry, synthetic, primary("A_answer_changing"), "source_prediction", "source_answer", 135, 4), category=synthetic_category)
    for control_id in sorted({row["control_id"] for row in registry if row["model_seed_identity"] == seed and row["control_id"] is not None}):
        selector = lambda p, control_id=control_id: p["model_seed_identity"] == seed and p["control_id"] == control_id
        # primary A is 160/160 in the scaffold, so 113 yields a 47/160 gap.
        record(f"control_gap_{control_id}_one_below", STATUS_NEGATIVE, _force_count(registry, synthetic, selector, "intervened_prediction", "expected_answer", 113, 4), category=synthetic_category)

    category_counts = Counter(row["category"] for row in scenarios)
    report = {
        "schema": "exp33ae.dev_only_remediation_polarity.v2", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "scenario_count": len(scenarios),
        "passed_count": sum(row["passed"] for row in scenarios), "category_counts": dict(sorted(category_counts.items())),
        "actual_rehearsal_status": actual["status"], "scenarios": scenarios,
        "all_direct_metric_or_status_edits_absent": all(not row["direct_metric_or_status_edit"] for row in scenarios),
        "synthetic_threshold_cases_are_model_evidence": False,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
