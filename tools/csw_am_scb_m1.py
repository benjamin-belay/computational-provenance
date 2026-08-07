#!/usr/bin/env python3
"""Static protocol, public development, and sealed confirmation for SCB-M1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1/CONFIG.json"
T1_CLOSURE_PATH = ROOT / "artifacts/dev_only/csw-am-scb-t1/closure/CLOSURE.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("DUPLICATE_KEY:" + key)
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line, object_pairs_hook=strict_pairs)
        if type(value) is not dict:
            raise ValueError("ROW_REQUIRED:" + path.name)
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)


def task(inputs: dict[str, int]) -> dict[str, int]:
    z1 = (inputs["a"] + inputs["b"]) % 16
    z2 = (z1 + inputs["c"]) % 16
    z3 = (5 * z2 + inputs["d"]) % 16
    return {"z1": z1, "z2": z2, "z3": z3, "answer": z3 % 8}


def template_id(inputs: dict[str, int]) -> int:
    return (
        inputs["a"] + 3 * inputs["b"] + 5 * inputs["c"] + 7 * inputs["d"]
    ) % 4


def validate_source(source_commit: str) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config["identity"] != "CSW-AM-SCB-M1":
        raise ValueError("CONFIG_IDENTITY")
    if (
        config["lineage"]["SCB_T1_closure_commit"]
        != subprocess.check_output(
            ["git", "rev-parse", config["lineage"]["SCB_T1_closure_commit"]],
            cwd=ROOT, text=True,
        ).strip()
    ):
        raise ValueError("T1_CLOSURE_COMMIT")
    closure = read_json(T1_CLOSURE_PATH)
    if (
        closure["lineage"]["T2_within_lineage_authorized"] is not False
        or closure["registered_denominator"]["models_passing"] != 2
        or closure["registered_denominator"]["strict_models"] != 3
    ):
        raise ValueError("T1_CLOSURE_BOUNDARY")
    if len(source_commit) != 40:
        raise ValueError("SOURCE_COMMIT")
    for relative in config["source_file_paths"]:
        if git_show(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BYTES:" + relative)
    return config


def validate_commit_file(path: Path, commit: str) -> dict[str, Any]:
    relative = str(path.resolve().relative_to(ROOT))
    if git_show(commit, relative) != path.read_bytes():
        raise ValueError("COMMIT_BINDING:" + relative)
    return read_json(path)


def hard_one_hot(categories: torch.Tensor) -> torch.Tensor:
    if categories.dtype != torch.long or categories.ndim != 1:
        raise ValueError("HARD_CATEGORY_TYPE_OR_SHAPE")
    if bool(((categories < 0) | (categories > 15)).any()):
        raise ValueError("HARD_CATEGORY_RANGE")
    return torch.nn.functional.one_hot(categories, 16).to(torch.float32)


def validate_one_hot(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 16 or not value.dtype.is_floating_point:
        raise ValueError("ONE_HOT_TYPE_OR_SHAPE")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError("ONE_HOT_VALUES")
    if not torch.all(value.sum(-1) == 1):
        raise ValueError("ONE_HOT_CARDINALITY")
    return value.argmax(-1)


class CausalStrictBottleneck(nn.Module):
    """Exact frozen T1 architecture with explicit hard-state entry points."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Embedding(16, 16)
        self.b = nn.Embedding(16, 16)
        self.c = nn.Embedding(16, 16)
        self.d = nn.Embedding(16, 16)
        self.z2_module = nn.Sequential(
            nn.Linear(48, 128), nn.GELU(), nn.Linear(128, 16)
        )
        self.z3_module = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16)
        )
        self.answer_module = nn.Linear(16, 8)

    def natural_z2(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.z2_module(torch.cat(
            (self.a(values[:, 0]), self.b(values[:, 1]), self.c(values[:, 2])),
            dim=-1,
        ))
        selected = logits.argmax(-1)
        return logits, hard_one_hot(selected)

    def transition(
        self, hard_z2: torch.Tensor, d_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_one_hot(hard_z2)
        logits = self.z3_module(torch.cat((hard_z2, self.d(d_values)), dim=-1))
        selected = logits.argmax(-1)
        return logits, hard_one_hot(selected)

    def answer(self, hard_z3: torch.Tensor) -> torch.Tensor:
        validate_one_hot(hard_z3)
        return self.answer_module(hard_z3)

    def run_natural(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        z2_logits, hard_z2 = self.natural_z2(values)
        z3_logits, hard_z3 = self.transition(hard_z2, values[:, 3])
        return {
            "z2_logits": z2_logits,
            "z2_selected": hard_z2.argmax(-1),
            "z2_consumed": validate_one_hot(hard_z2),
            "z3_logits": z3_logits,
            "z3_selected": hard_z3.argmax(-1),
            "z3_consumed": validate_one_hot(hard_z3),
            "answer_logits": self.answer(hard_z3),
        }

    def run_with_z2(
        self, values: torch.Tensor, target_z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hard_z2 = hard_one_hot(target_z2)
        z3_logits, hard_z3 = self.transition(hard_z2, values[:, 3])
        return {
            "z2_consumed": validate_one_hot(hard_z2),
            "z3_logits": z3_logits,
            "z3_selected": hard_z3.argmax(-1),
            "z3_consumed": validate_one_hot(hard_z3),
            "answer_logits": self.answer(hard_z3),
        }

    def run_with_z3(self, target_z3: torch.Tensor) -> dict[str, torch.Tensor]:
        hard_z3 = hard_one_hot(target_z3)
        return {
            "z3_consumed": validate_one_hot(hard_z3),
            "answer_logits": self.answer(hard_z3),
        }


def load_model(config: dict[str, Any]) -> CausalStrictBottleneck:
    path = ROOT / config["lineage"]["model_1_checkpoint_path"]
    if file_sha(path) != config["lineage"]["model_1_checkpoint_sha256"]:
        raise ValueError("CHECKPOINT_HASH")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["model_id"] != "MODEL_1"
        or payload["step"] != config["lineage"]["model_1_checkpoint_step"]
        or payload["comparator"] is not False
    ):
        raise ValueError("CHECKPOINT_IDENTITY")
    model = CausalStrictBottleneck()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def tensors(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([
        [row["inputs"][key] for key in ("a", "b", "c", "d")]
        for row in rows
    ], dtype=torch.long)


def render_prompt(inputs: dict[str, int]) -> str:
    templates = read_json(
        ROOT / "artifacts/dev_only/csw-am-t0-simple/authoritative/public/TASK_SPEC.json"
    )["task"]["prompt_templates"]
    return templates[template_id(inputs)].format(**inputs)


def protected_rows(
    sealed_rows: list[dict[str, Any]], role: str,
) -> list[dict[str, Any]]:
    result = []
    for row in sealed_rows:
        if row["role"] != role:
            continue
        base = row["inputs"]
        for c in (base["c"], base["c"] + 8):
            inputs = {"a": base["a"], "b": base["b"], "c": c, "d": base["d"]}
            result.append({
                "schema": "exp33a.csw_am_scb_m1.confirmation_row.v1",
                "role": role,
                "closure_id": row["closure_id"],
                "example_id": digest(
                    ["CSW_AM_SCB_M1_C1_ROW_V1", role, row["closure_id"], c]
                ),
                "orientation": (
                    "LOWER_C_ORIENTATION" if c < 8 else "UPPER_C_ORIENTATION"
                ),
                "inputs": inputs,
                "natural": task(inputs),
                "template_id": template_id(inputs),
                "prompt_text": render_prompt(inputs),
            })
    return result


def validate_rows(rows: list[dict[str, Any]], expected: int) -> None:
    if len(rows) != expected or len({row["example_id"] for row in rows}) != expected:
        raise ValueError("ROW_DENOMINATOR")
    for row in rows:
        if row["natural"] != task(row["inputs"]):
            raise ValueError("ROW_SEMANTICS")


def target_for(family: str, natural_z2: int, natural_z3: int) -> tuple[int | None, int]:
    if family == "ALIAS_Z2":
        target_z2 = (natural_z2 + 8) % 16
        return target_z2, -1
    if family == "NON_ALIAS_Z2":
        target_z2 = (natural_z2 + 1) % 16
        return target_z2, -1
    if family == "WRONG_Z2":
        target_z2 = (natural_z2 + 3) % 16
        return target_z2, -1
    if family == "SAME_Z2":
        return natural_z2, -1
    if family == "DIRECT_Z3":
        return None, (natural_z3 + 1) % 16
    raise ValueError("NO_TARGET:" + family)


def tensor_lists(result: dict[str, torch.Tensor]) -> dict[str, list[Any]]:
    converted = {}
    for key, value in result.items():
        if key == "answer_logits":
            converted["answer"] = value.argmax(-1).tolist()
        elif key.endswith("_logits"):
            continue
        else:
            converted[key] = value.tolist()
    return converted


def execute_cases(
    model: CausalStrictBottleneck, rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    values = tensors(rows)
    batch_size = config["budgets"]["batch_size"]
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch_values = values[start:start + batch_size]
            natural_raw = model.run_natural(batch_values)
            natural = tensor_lists(natural_raw)
            family_outputs: dict[str, dict[str, list[Any]]] = {}
            for family in ("ALIAS_Z2", "NON_ALIAS_Z2", "WRONG_Z2", "SAME_Z2"):
                targets = torch.tensor([
                    target_for(
                        family, row["natural"]["z2"], row["natural"]["z3"]
                    )[0]
                    for row in batch_rows
                ], dtype=torch.long)
                family_outputs[family] = tensor_lists(
                    model.run_with_z2(batch_values, targets)
                )
            family_outputs["SHAM"] = tensor_lists(
                model.run_with_z2(
                    batch_values, torch.tensor(natural["z2_selected"], dtype=torch.long)
                )
            )
            direct_targets = torch.tensor([
                target_for(
                    "DIRECT_Z3", row["natural"]["z2"], row["natural"]["z3"]
                )[1]
                for row in batch_rows
            ], dtype=torch.long)
            family_outputs["DIRECT_Z3"] = tensor_lists(
                model.run_with_z3(direct_targets)
            )
            for index, row in enumerate(batch_rows):
                truth = row["natural"]
                natural_observed = {
                    key: natural[key][index]
                    for key in (
                        "z2_selected", "z2_consumed", "z3_selected",
                        "z3_consumed", "answer",
                    )
                }
                natural_correct = (
                    natural_observed["z2_selected"] == truth["z2"]
                    and natural_observed["z3_selected"] == truth["z3"]
                    and natural_observed["answer"] == truth["answer"]
                )
                cases.append({
                    "schema": "exp33a.csw_am_scb_m1.case.v1",
                    "example_id": row["example_id"],
                    "closure_id": row["closure_id"],
                    "orientation": row["orientation"],
                    "template_id": row["template_id"],
                    "inputs": row["inputs"],
                    "family": "NATURAL",
                    "registered_natural": truth,
                    "natural_observed": natural_observed,
                    "target_z2": None,
                    "target_z3": None,
                    "observed": natural_observed,
                    "predicates": {
                        "natural_z2": natural_observed["z2_selected"] == truth["z2"],
                        "natural_z3": natural_observed["z3_selected"] == truth["z3"],
                        "natural_answer": natural_observed["answer"] == truth["answer"],
                        "selected_equals_consumed": (
                            natural_observed["z2_selected"]
                            == natural_observed["z2_consumed"]
                            and natural_observed["z3_selected"]
                            == natural_observed["z3_consumed"]
                        ),
                        "natural_joint": natural_correct,
                    },
                })
                for family, output in family_outputs.items():
                    target_z2, target_z3 = (
                        (natural_observed["z2_selected"], truth["z3"])
                        if family == "SHAM"
                        else target_for(family, truth["z2"], truth["z3"])
                    )
                    if target_z2 is not None:
                        expected_z3 = (5 * target_z2 + row["inputs"]["d"]) % 16
                    else:
                        expected_z3 = target_z3
                    expected_answer = expected_z3 % 8
                    observed = {
                        key: output[key][index] for key in output
                    }
                    target_consumed = (
                        observed["z2_consumed"] == target_z2
                        if target_z2 is not None
                        else observed["z3_consumed"] == target_z3
                    )
                    generated_mapping = (
                        observed.get("z3_selected", expected_z3) == expected_z3
                        and observed["answer"] == expected_answer
                    )
                    if family == "SHAM":
                        trace_identity = (
                            observed["z3_selected"] == natural_observed["z3_selected"]
                            and observed["answer"] == natural_observed["answer"]
                        )
                    elif family == "SAME_Z2":
                        trace_identity = (
                            not natural_correct
                            or (
                                observed["z3_selected"]
                                == natural_observed["z3_selected"]
                                and observed["answer"] == natural_observed["answer"]
                            )
                        )
                    else:
                        trace_identity = None
                    cases.append({
                        "schema": "exp33a.csw_am_scb_m1.case.v1",
                        "example_id": row["example_id"],
                        "closure_id": row["closure_id"],
                        "orientation": row["orientation"],
                        "template_id": row["template_id"],
                        "inputs": row["inputs"],
                        "family": family,
                        "registered_natural": truth,
                        "natural_observed": natural_observed,
                        "target_z2": target_z2,
                        "target_z3": target_z3 if target_z2 is None else expected_z3,
                        "expected_answer": expected_answer,
                        "observed": observed,
                        "predicates": {
                            "natural_joint": natural_correct,
                            "target_consumed": target_consumed,
                            "generated_mapping": generated_mapping,
                            "trace_identity": trace_identity,
                            "joint_success": natural_correct
                            and target_consumed and generated_mapping,
                            "conditional_transport": target_consumed and generated_mapping,
                        },
                    })
    return cases


def accuracy(values: list[bool]) -> float:
    return sum(values) / len(values)


def subgroup_minimum(
    rows: list[dict[str, Any]], predicate: str, minimum: int,
) -> tuple[float, list[dict[str, Any]]]:
    groups = {
        "natural_z2": [row["registered_natural"]["z2"] for row in rows],
        "answer": [row["registered_natural"]["answer"] for row in rows],
        "template": [row["template_id"] for row in rows],
        "orientation": [row["orientation"] for row in rows],
    }
    records = []
    for family, values in groups.items():
        for value, denominator in sorted(Counter(values).items(), key=lambda x: str(x[0])):
            if denominator < minimum:
                continue
            selected = [
                row["predicates"][predicate]
                for row, group_value in zip(rows, values)
                if group_value == value
            ]
            records.append({
                "family": family, "value": str(value),
                "correct": sum(selected), "denominator": denominator,
                "accuracy": accuracy(selected),
            })
    return min(row["accuracy"] for row in records), records


def aggregate(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_family = {
        family: [row for row in cases if row["family"] == family]
        for family in config["families"]
    }
    gates = config["gates"]
    natural = by_family["NATURAL"]
    natural_metrics = {
        name: accuracy([row["predicates"][name] for row in natural])
        for name in ("natural_z2", "natural_z3", "natural_answer", "natural_joint")
    }
    natural_min, natural_subgroups = subgroup_minimum(
        natural, "natural_joint", gates["minimum_powered_subgroup_denominator"]
    )
    family_metrics = {}
    for family in config["families"][1:]:
        rows = by_family[family]
        conditional_rows = [
            row for row in rows if row["predicates"]["natural_joint"]
        ]
        family_min, subgroup_rows = subgroup_minimum(
            rows, "generated_mapping",
            gates["minimum_powered_subgroup_denominator"],
        )
        family_metrics[family] = {
            "cases": len(rows),
            "target_consumption_accuracy": accuracy([
                row["predicates"]["target_consumed"] for row in rows
            ]),
            "generated_mapping_accuracy": accuracy([
                row["predicates"]["generated_mapping"] for row in rows
            ]),
            "minimum_powered_subgroup_accuracy": family_min,
            "subgroups": subgroup_rows,
            "complete_denominator_joint_success": accuracy([
                row["predicates"]["joint_success"] for row in rows
            ]),
            "conditional_denominator": len(conditional_rows),
            "conditional_transport_accuracy": accuracy([
                row["predicates"]["conditional_transport"]
                for row in conditional_rows
            ]),
            "trace_identity_accuracy": (
                accuracy([
                    row["predicates"]["trace_identity"] for row in rows
                ]) if family == "SHAM" else
                accuracy([
                    row["predicates"]["trace_identity"]
                    for row in conditional_rows
                ]) if family == "SAME_Z2" else None
            ),
        }
    predicates = {
        "complete_denominator": all(
            len(by_family[family]) == len(natural)
            for family in config["families"]
        ),
        "natural_answer":
            natural_metrics["natural_answer"] >= gates["natural_answer_accuracy"],
        "natural_z2":
            natural_metrics["natural_z2"] >= gates["natural_z2_accuracy"],
        "natural_z3":
            natural_metrics["natural_z3"] >= gates["natural_z3_accuracy"],
        "natural_subgroups":
            natural_min >= gates["minimum_powered_subgroup_accuracy"],
        "exact_target_consumption": all(
            value["target_consumption_accuracy"]
            == gates["target_consumption_accuracy"]
            for value in family_metrics.values()
        ),
        "generated_mapping": all(
            value["generated_mapping_accuracy"]
            >= gates["generated_mapping_accuracy"]
            and value["minimum_powered_subgroup_accuracy"]
            >= gates["minimum_powered_subgroup_accuracy"]
            for value in family_metrics.values()
        ),
        "conditional_transport": all(
            value["conditional_transport_accuracy"]
            >= gates["conditional_transport_accuracy"]
            for value in family_metrics.values()
        ),
        "complete_joint": all(
            value["complete_denominator_joint_success"]
            >= gates["complete_denominator_joint_success"]
            for value in family_metrics.values()
        ),
        "sham_identity":
            family_metrics["SHAM"]["trace_identity_accuracy"]
            == gates["sham_trace_identity"],
        "same_state_identity":
            family_metrics["SAME_Z2"]["trace_identity_accuracy"]
            == gates["sham_trace_identity"],
    }
    return {
        "schema": "exp33a.csw_am_scb_m1.metrics.v1",
        "cases": len(cases),
        "examples": len(natural),
        "natural": {
            **natural_metrics,
            "minimum_powered_subgroup_accuracy": natural_min,
            "subgroups": natural_subgroups,
        },
        "families": family_metrics,
        "predicates": predicates,
        "passed": all(predicates.values()),
    }


def target_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for natural_z2 in range(16):
        for d in range(16):
            natural_z3 = (5 * natural_z2 + d) % 16
            for family in ("ALIAS_Z2", "NON_ALIAS_Z2", "WRONG_Z2", "SAME_Z2"):
                target_z2, _ = target_for(family, natural_z2, natural_z3)
                expected_z3 = (5 * target_z2 + d) % 16
                rows.append({
                    "natural_z2": natural_z2, "d": d,
                    "natural_z3": natural_z3,
                    "natural_answer": natural_z3 % 8,
                    "family": family, "target_z2": target_z2,
                    "expected_z3": expected_z3,
                    "expected_answer": expected_z3 % 8,
                })
            _, direct_z3 = target_for("DIRECT_Z3", natural_z2, natural_z3)
            rows.append({
                "natural_z2": natural_z2, "d": d,
                "natural_z3": natural_z3,
                "natural_answer": natural_z3 % 8,
                "family": "DIRECT_Z3", "target_z2": None,
                "expected_z3": direct_z3,
                "expected_answer": direct_z3 % 8,
            })
    return rows


def s0(source_commit: str, output: Path) -> dict[str, Any]:
    config = validate_source(source_commit)
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    output.mkdir(parents=True)
    manifest = target_manifest(config)
    relation_checks = {
        "alias_answer_always_unchanged": all(
            row["expected_answer"] == row["natural_answer"]
            for row in manifest if row["family"] == "ALIAS_Z2"
        ),
        "non_alias_answer_always_changed": all(
            row["expected_answer"] != row["natural_answer"]
            for row in manifest if row["family"] == "NON_ALIAS_Z2"
        ),
        "wrong_answer_always_changed": all(
            row["expected_answer"] != row["natural_answer"]
            for row in manifest if row["family"] == "WRONG_Z2"
        ),
        "same_answer_always_unchanged": all(
            row["expected_answer"] == row["natural_answer"]
            for row in manifest if row["family"] == "SAME_Z2"
        ),
        "direct_z3_answer_always_changed": all(
            row["expected_answer"] != row["natural_answer"]
            for row in manifest if row["family"] == "DIRECT_Z3"
        ),
        "all_16_natural_z2": len({row["natural_z2"] for row in manifest}) == 16,
        "all_16_d": len({row["d"] for row in manifest}) == 16,
        "T1_result_preserved": config["lineage"]["SCB_T1_result_superseded"] is False,
        "zero_S0_interventions": config["chronology"]["S0_interventions"] == 0,
    }
    write_rows(output / "TARGET_MANIFEST.jsonl", manifest)
    write_json(output / "SUPPORT_AND_RULES.json", {
        "schema": "exp33a.csw_am_scb_m1.s0_support.v1",
        "target_rows": len(manifest), "relation_checks": relation_checks,
        "gates": config["gates"], "budgets": config["budgets"],
    })
    terminal = (
        config["terminals"]["S0_ready"]
        if all(relation_checks.values()) else config["terminals"]["invalid"]
    )
    result = {
        "schema": "exp33a.csw_am_scb_m1.s0_terminal.v1",
        "terminal": terminal, "source_commit": source_commit,
        "interventions": 0, "model_loads": 0, "model_forwards": 0,
        "protected_identities_released": 0,
        "D0_authorized": terminal == config["terminals"]["S0_ready"],
        "C1_authorized": False, "receipts_authorized": False,
        "carrier_operations_authorized": False,
    }
    write_json(output / "TERMINAL.json", result)
    return result


def validate_authorization(
    path: Path, commit: str, source_commit: str, expected_phase: str,
) -> dict[str, Any]:
    value = validate_commit_file(path, commit)
    if (
        value["schema"] != "exp33a.csw_am_scb_m1.authorization.v1"
        or value["identity"] != "CSW-AM-SCB-M1"
        or value["source_commit"] != source_commit
        or value["phase"] != expected_phase
        or value["receipts_authorized"] is not False
        or value["carrier_operations_authorized"] is not False
    ):
        raise ValueError("AUTHORIZATION_CONTENT")
    return value


def execute_phase(
    phase: str, source_commit: str, authorization: Path,
    authorization_commit: str, output: Path,
    sealed_allocation: Path | None = None, sealed_manifest: Path | None = None,
) -> dict[str, Any]:
    config = validate_source(source_commit)
    auth = validate_authorization(
        authorization, authorization_commit, source_commit, phase
    )
    if output.exists():
        raise FileExistsError("OUTPUT_EXISTS")
    if phase == "D0":
        s0_terminal = ROOT / auth["S0_terminal_path"]
        s0_value = validate_commit_file(s0_terminal, auth["S0_commit"])
        if s0_value["terminal"] != config["terminals"]["S0_ready"]:
            raise ValueError("S0_NOT_READY")
        public_path = ROOT / config["populations"]["D0_public_path"]
        if file_sha(public_path) != config["populations"]["D0_public_sha256"]:
            raise ValueError("D0_DATA_HASH")
        rows = read_rows(public_path)
        validate_rows(rows, config["populations"]["D0_oriented_rows"])
        protected_released = 0
    else:
        d0_terminal = ROOT / auth["D0_terminal_path"]
        d0_value = validate_commit_file(d0_terminal, auth["D0_commit"])
        if d0_value["terminal"] != config["terminals"]["D0_pass"]:
            raise ValueError("D0_NOT_READY")
        if sealed_allocation is None or sealed_manifest is None:
            raise ValueError("SEALED_PATH_REQUIRED")
        if (
            file_sha(sealed_allocation)
            != config["populations"]["R0_full_allocation_sha256"]
            or file_sha(sealed_manifest)
            != config["populations"]["R0_sealed_manifest_sha256"]
        ):
            raise ValueError("SEALED_BINDING")
        rows = protected_rows(
            read_rows(sealed_allocation), config["populations"]["C1_role"]
        )
        validate_rows(rows, config["populations"]["C1_oriented_rows"])
        if len({row["closure_id"] for row in rows}) != config["populations"]["C1_closures"]:
            raise ValueError("C1_CLOSURE_DENOMINATOR")
        protected_released = config["populations"]["C1_closures"]
    model = load_model(config)
    cases = execute_cases(model, rows, config)
    expected_cases = (
        config["budgets"]["D0_total_cases"]
        if phase == "D0" else config["budgets"]["C1_total_cases"]
    )
    if len(cases) != expected_cases:
        raise ValueError("CASE_DENOMINATOR")
    metrics = aggregate(cases, config)
    output.mkdir(parents=True)
    write_rows(output / "INPUT_ROWS.jsonl", rows)
    write_rows(output / "CASES.jsonl", cases)
    write_json(output / "METRICS.json", metrics)
    passed_terminal = (
        config["terminals"]["D0_pass"]
        if phase == "D0" else config["terminals"]["C1_pass"]
    )
    failed_terminal = (
        config["terminals"]["D0_fail"]
        if phase == "D0" else config["terminals"]["C1_fail"]
    )
    terminal = {
        "schema": "exp33a.csw_am_scb_m1.phase_terminal.v1",
        "phase": phase,
        "terminal": passed_terminal if metrics["passed"] else failed_terminal,
        "passed": metrics["passed"],
        "examples": len(rows), "cases": len(cases),
        "protected_closures_released": protected_released,
        "model_2_transfer_closures_released": 0,
        "model_3_transfer_closures_released": 0,
        "reserve_closures_released": 0,
        "receipts": 0, "carrier_operations": 0,
        "next_phase_authorized": phase == "D0" and metrics["passed"],
    }
    write_json(output / "TERMINAL.json", terminal)
    write_json(output / "RUN_BINDING.json", {
        "schema": "exp33a.csw_am_scb_m1.run_binding.v1",
        "phase": phase, "source_commit": source_commit,
        "authorization_commit": authorization_commit,
        "checkpoint_sha256": config["lineage"]["model_1_checkpoint_sha256"],
        "input_rows_sha256": file_sha(output / "INPUT_ROWS.jsonl"),
        "cases_sha256": file_sha(output / "CASES.jsonl"),
        "metrics_sha256": file_sha(output / "METRICS.json"),
    })
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("s0")
    static.add_argument("--source-commit", required=True)
    static.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--phase", choices=["D0", "C1"], required=True)
    execute.add_argument("--source-commit", required=True)
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--authorization-commit", required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--sealed-allocation", type=Path)
    execute.add_argument("--sealed-manifest", type=Path)
    args = parser.parse_args()
    try:
        result = (
            s0(args.source_commit, args.output)
            if args.command == "s0" else
            execute_phase(
                args.phase, args.source_commit, args.authorization,
                args.authorization_commit, args.output,
                args.sealed_allocation, args.sealed_manifest,
            )
        )
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
