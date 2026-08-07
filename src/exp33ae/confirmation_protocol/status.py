"""Typed, integer-only terminal status derivation for the frozen five-seed run."""

from __future__ import annotations

from ..remediation.strict import StrictValidationError, expect_exact_keys, expect_int, expect_type


INVALID = "EXPERIMENT_INVALID"
RUNTIME_STOPPED = "EXPERIMENT_RUNTIME_STOPPED"
COMPETENCE_FAILED = "MODEL_COMPETENCE_FAILED"
NO_ELIGIBLE = "NO_ELIGIBLE_CAUSAL_REPRESENTATION"
DECODE_ONLY = "DECODE_ONLY_CAUSALITY_FAILED"
NEGATIVE = "EXPERIMENT_NEGATIVE"
POSITIVE = "CONFIRMED_ENGINEERED_CAUSAL_ABSTRACTION_POSITIVE"
STATUSES = (INVALID, RUNTIME_STOPPED, COMPETENCE_FAILED, NO_ELIGIBLE, DECODE_ONLY, NEGATIVE, POSITIVE)

STRATA = ("A", "B", "C", "D")
CONTROL_IDS = tuple(
    [f"random_direction_{index}" for index in range(4)]
    + [f"orthogonal_direction_{index}" for index in range(4)]
    + [f"alternate_location_{index}" for index in range(3)]
    + ["alternate_early_layer_0"]
)


def metric_keys() -> set[str]:
    return {
        "seed_order", "model_seed_identity", "task_target", "task_parallel",
        "worst_programme", "shortcut_target", "shortcut_parallel", "decode",
        "iia_A", "iia_D", "transport", "descendant", "non_descendant",
        "same_state", "control_iia", "base_correct", "source_correct",
        "group_denominators", "planned", "attempted", "completed",
    }


def validate_seed_metrics(value: dict) -> dict:
    expect_exact_keys(value, metric_keys(), "seed metrics")
    expect_int(value["seed_order"], "seed_order", 0, 4)
    expect_int(value["model_seed_identity"], "model_seed_identity", 1, (1 << 63) - 1)
    for field, maximum in (("task_target",800),("task_parallel",800),("worst_programme",16),
                           ("shortcut_target",800),("shortcut_parallel",800),("decode",1600),
                           ("iia_A",160),("iia_D",80),("same_state",80)):
        expect_int(value[field], field, 0, maximum)
    for field in ("transport", "descendant", "non_descendant"):
        expect_exact_keys(value[field], set(STRATA), field)
        for stratum, maximum in zip(STRATA, (160,80,80,80)):
            expect_int(value[field][stratum], f"{field}.{stratum}", 0, maximum)
    expect_exact_keys(value["control_iia"], set(CONTROL_IDS), "control_iia")
    for name in CONTROL_IDS:
        expect_int(value["control_iia"][name], name, 0, 160)
    for field in ("base_correct", "source_correct", "group_denominators"):
        expect_type(value[field], dict, field)
        if not value[field]:
            raise StrictValidationError("DENOMINATOR", f"empty {field}")
    if set(value["base_correct"]) != set(value["source_correct"]) or set(value["base_correct"]) != set(value["group_denominators"]):
        raise StrictValidationError("DENOMINATOR", "group identity mismatch")
    for group, denominator in value["group_denominators"].items():
        if type(group) is not str or not group:
            raise StrictValidationError("SCHEMA", "group identity")
        expect_int(denominator, f"denominator.{group}", 1)
        expect_int(value["base_correct"][group], f"base.{group}", 0, denominator)
        expect_int(value["source_correct"][group], f"source.{group}", 0, denominator)
    for field in ("planned", "attempted", "completed"):
        expect_int(value[field], field, 0, 3920)
    return value


def gates(value: dict) -> dict[str, bool]:
    value = validate_seed_metrics(value)
    competence = {
        "target_task": value["task_target"] >= 680,
        "parallel_task": value["task_parallel"] >= 680,
        "worst_programme": value["worst_programme"] >= 12,
        "shortcut_target": value["task_target"] - value["shortcut_target"] >= 160,
        "shortcut_parallel": value["task_parallel"] - value["shortcut_parallel"] >= 160,
        "base_accuracy": all(20 * value["base_correct"][group] >= 17 * denominator for group, denominator in value["group_denominators"].items()),
        "source_accuracy": all(20 * value["source_correct"][group] >= 17 * denominator for group, denominator in value["group_denominators"].items()),
    }
    causal = {
        "decode": value["decode"] >= 1440,
        "iia_A": value["iia_A"] >= 136,
        "transport_pooled": value["transport"]["A"] + value["transport"]["B"] + value["transport"]["D"] >= 288,
        "transport_A": value["transport"]["A"] >= 136,
        "transport_B": value["transport"]["B"] >= 68,
        "transport_D": value["transport"]["D"] >= 68,
        "descendant_A": value["descendant"]["A"] >= 136,
        "descendant_B": value["descendant"]["B"] >= 68,
        "descendant_D": value["descendant"]["D"] >= 68,
        "non_descendant_A": value["non_descendant"]["A"] >= 152,
        "non_descendant_B": value["non_descendant"]["B"] >= 76,
        "non_descendant_C": value["non_descendant"]["C"] >= 76,
        "non_descendant_D": value["non_descendant"]["D"] >= 76,
        "same_state": value["same_state"] >= 76,
        "iia_D": value["iia_D"] >= 64,
        "individual_control_gaps": all(value["iia_A"] - value["control_iia"][identity] >= 48 for identity in CONTROL_IDS),
        "call_reconciliation": value["planned"] == value["attempted"] == value["completed"] == 3920,
    }
    return {**competence, **causal}


def derive_status(seed_rows: list[dict], expected_seed_identities: list[int], integrity_reasons: list[str] | None = None, runtime_stopped_before_access: bool = False) -> dict:
    if integrity_reasons:
        return {"status": INVALID, "reasons": list(integrity_reasons), "per_seed": []}
    if runtime_stopped_before_access:
        if seed_rows:
            return {"status": INVALID, "reasons": ["runtime stop cannot accompany inspected seed outcomes"], "per_seed": []}
        return {"status": RUNTIME_STOPPED, "reasons": [], "per_seed": []}
    if type(seed_rows) is not list or len(seed_rows) != 5 or type(expected_seed_identities) is not list or len(expected_seed_identities) != 5:
        return {"status": INVALID, "reasons": ["exact five-seed population required"], "per_seed": []}
    if len(set(expected_seed_identities)) != 5:
        return {"status": INVALID, "reasons": ["duplicate expected seed identity"], "per_seed": []}
    try:
        for order, row in enumerate(seed_rows):
            validate_seed_metrics(row)
            if row["seed_order"] != order or row["model_seed_identity"] != expected_seed_identities[order]:
                raise StrictValidationError("SEED_IDENTITY", "wrong/reordered seed")
    except Exception as error:
        return {"status": INVALID, "reasons": [str(error)], "per_seed": []}
    per_seed = [{"seed_order": row["seed_order"], "gates": gates(row)} for row in seed_rows]
    competence_names = {"target_task", "parallel_task", "worst_programme", "shortcut_target", "shortcut_parallel", "base_accuracy", "source_accuracy"}
    if any(not all(item["gates"][name] for name in competence_names) for item in per_seed):
        status = COMPETENCE_FAILED
    elif any(not item["gates"]["decode"] for item in per_seed):
        status = NO_ELIGIBLE
    elif any(not item["gates"]["iia_A"] for item in per_seed):
        status = DECODE_ONLY
    elif any(not all(item["gates"].values()) for item in per_seed):
        status = NEGATIVE
    else:
        status = POSITIVE
    return {"status": status, "reasons": [], "per_seed": per_seed}


def qualifying_fixture(seed_order: int, seed_identity: int) -> dict:
    groups = {"clean_target":800,"clean_parallel":800,"primary_A":160,"primary_B":80,"primary_C":80,"primary_D":80,
              **{identity:160 for identity in CONTROL_IDS}}
    return {
        "seed_order": seed_order, "model_seed_identity": seed_identity,
        "task_target": 680, "task_parallel": 680, "worst_programme": 12,
        "shortcut_target": 520, "shortcut_parallel": 520, "decode": 1440,
        "iia_A": 136, "iia_D": 64,
        "transport": {"A":140,"B":68,"C":0,"D":80},
        "descendant": {"A":136,"B":68,"C":0,"D":68},
        "non_descendant": {"A":152,"B":76,"C":76,"D":76},
        "same_state": 76,
        "control_iia": {identity:88 for identity in CONTROL_IDS},
        "base_correct": {name:(17*denominator+19)//20 for name,denominator in groups.items()},
        "source_correct": {name:(17*denominator+19)//20 for name,denominator in groups.items()},
        "group_denominators": groups,
        "planned":3920,"attempted":3920,"completed":3920,
    }
