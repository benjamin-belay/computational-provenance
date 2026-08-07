"""Pre-transformer task-design audit and frozen DEV_ONLY split materialization."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .artifacts import canonical_json, sha256_bytes, sha256_file, write_json, write_jsonl
from .baselines import shallow_mlp_baseline, shortcut_suite
from .catalog import programme_catalog, records
from .independent_oracle import oracle, oracle_do_z2
from .spec import (
    ALL_OPERATIONS,
    GATES,
    HELDOUT_OPERATION_IDS,
    INTEGRITY_OPERATION_IDS,
    K,
    NAMESPACE,
    PARTITIONS,
    PROGRAM_COUNTS,
    TRAIN_OPERATION_IDS,
)
from .task import do_z2, programme_examples


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _programme_signature(programme) -> str:
    values = [
        (row.values.z1, row.values.z2, row.values.z3, row.values.n1, row.values.n2, row.values.n3)
        for row in programme_examples(programme)
    ]
    return sha256_bytes(canonical_json(values).encode())


def run_task_audit(materialize: bool = True) -> dict:
    catalog = programme_catalog()
    split_rows = {partition: records(partition) for partition in PARTITIONS}
    all_programs = [program for programs in catalog.values() for program in programs]

    production_mismatches = 0
    oracle_count = 0
    for rows in split_rows.values():
        for row in rows:
            expected = oracle(row.programme.operations, (row.a, row.b, row.c, row.d))
            actual = row.values
            oracle_count += 1
            if expected != {
                "z1": actual.z1,
                "z2": actual.z2,
                "z3": actual.z3,
                "n1": actual.n1,
                "n2": actual.n2,
                "n3": actual.n3,
                "joint_answer": actual.joint_answer,
            }:
                production_mismatches += 1

    do_count = do_mismatches = 0
    for program in all_programs:
        for d in range(K):
            for value in range(K):
                do_count += 1
                if do_z2(program, d, value) != oracle_do_z2(program.operations, d, value)["z3"]:
                    do_mismatches += 1

    programme_ids = [p.programme_id for p in all_programs]
    target_compositions = {name: {p.target_composition_id for p in programs} for name, programs in catalog.items()}
    parallel_compositions = {name: {p.parallel_composition_id for p in programs} for name, programs in catalog.items()}
    train_ops = {ALL_OPERATIONS[i] for i in TRAIN_OPERATION_IDS}
    held_ops = {ALL_OPERATIONS[i] for i in HELDOUT_OPERATION_IDS}
    integrity_ops = {ALL_OPERATIONS[i] for i in INTEGRITY_OPERATION_IDS}
    signatures = {_programme_signature(program): program.programme_id for program in all_programs}

    exact_ids = {}
    exact_duplicates = []
    for partition, rows in split_rows.items():
        for row in rows:
            previous = exact_ids.setdefault(row.example_id, partition)
            if previous != partition:
                exact_duplicates.append((row.example_id, previous, partition))

    balance = {}
    for partition, rows in split_rows.items():
        balance[partition] = {
            "z2": dict(sorted(Counter(row.values.z2 for row in rows).items())),
            "z3": dict(sorted(Counter(row.values.z3 for row in rows).items())),
            "n3": dict(sorted(Counter(row.values.n3 for row in rows).items())),
            "joint_answer_minmax": [
                min(Counter(row.joint_answer for row in rows).values()),
                max(Counter(row.joint_answer for row in rows).values()),
            ],
        }

    causal_rows = split_rows["causal_evaluation"]
    train_transition_keys = {
        (op, x, u, op[2 * x + u])
        for program in catalog["training"]
        for op in program.operations
        for x in range(K)
        for u in range(K)
    }
    evaluation_transition_keys = {
        (op, x, u, op[2 * x + u])
        for program in catalog["causal_target_composition"] + catalog["causal_parallel_composition"]
        for op in program.operations
        for x in range(K)
        for u in range(K)
    }
    pair_counts = Counter()
    for base in causal_rows:
        for source in causal_rows:
            if base.example_id == source.example_id:
                continue
            if base.values.z2 == source.values.z2:
                pair_counts["same_state_different_input"] += 1
                continue
            pair_counts["different_state"] += 1
            expected_z3 = do_z2(base.programme, base.d, source.values.z2)
            expected_joint = expected_z3 * K + base.values.n3
            if base.joint_answer == source.joint_answer:
                pair_counts["same_answer_different_state"] += 1
            if expected_joint != base.joint_answer:
                pair_counts["changed_state_different_answer"] += 1
    eval_sets = {
        "heldout_target_composition": tuple(
            r for r in causal_rows if r.axis == "heldout_target_chain_composition"
        ),
        "heldout_parallel_composition": tuple(
            r for r in causal_rows if r.axis == "heldout_parallel_chain_composition"
        ),
        "model_selection": split_rows["model_selection"],
    }
    shortcuts = shortcut_suite(split_rows["training"], eval_sets)
    shallow = shallow_mlp_baseline(split_rows["training"], eval_sets)
    for axis in shortcuts:
        shortcuts[axis]["shallow_position_mlp"] = shallow["accuracy"][axis]

    best_shortcut = {
        axis: max(value for name, value in metrics.items() if name != "chance")
        for axis, metrics in shortcuts.items()
    }
    target_composition_overlap = {
        name: len(target_compositions["training"] & target_compositions[name])
        for name in target_compositions
        if name != "training"
    }
    parallel_composition_overlap = {
        name: len(parallel_compositions["training"] & parallel_compositions[name])
        for name in parallel_compositions
        if name != "training"
    }
    programme_overlap = {
        name: len({p.programme_id for p in catalog["training"]} & {p.programme_id for p in programs})
        for name, programs in catalog.items()
        if name != "training"
    }
    report = {
        "schema": "exp33ae.task_design_audit.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
        "future_generation_seed": None,
        "modulus": K,
        "operation_function_count": len(ALL_OPERATIONS),
        "operation_banks": {
            "training": list(TRAIN_OPERATION_IDS),
            "heldout": list(HELDOUT_OPERATION_IDS),
            "integrity": list(INTEGRITY_OPERATION_IDS),
            "pairwise_disjoint": not (train_ops & held_ops or train_ops & integrity_ops or held_ops & integrity_ops),
        },
        "programme_counts": {name: len(programs) for name, programs in catalog.items()},
        "actual_programme_count": len(all_programs),
        "unique_programme_ids": len(set(programme_ids)),
        "unique_exhaustive_causal_signatures": len(signatures),
        "records": {partition: len(rows) for partition, rows in split_rows.items()},
        "exact_duplicate_count": len(exact_duplicates),
        "training_programme_overlap": programme_overlap,
        "training_target_composition_overlap": target_composition_overlap,
        "training_parallel_composition_overlap": parallel_composition_overlap,
        "evaluation_operation_overlap_with_training": len(
            train_ops
            & {
                op
                for p in catalog["causal_target_composition"] + catalog["causal_parallel_composition"]
                for op in p.operations
            }
        ),
        "evaluation_primitive_semantic_transition_overlap_with_training": len(
            train_transition_keys & evaluation_transition_keys
        ),
        "heldout_axes": {
            "target": {
                "axis": "heldout_target_chain_composition",
                "training_target_composition_overlap": target_composition_overlap["causal_target_composition"],
                "note": "parallel-chain compositions may repeat; target-chain compositions do not",
            },
            "parallel": {
                "axis": "heldout_parallel_chain_composition",
                "training_parallel_composition_overlap": parallel_composition_overlap["causal_parallel_composition"],
                "note": "target-chain compositions may repeat; parallel-chain compositions do not",
            },
        },
        "eligible_ordered_pair_counts": dict(sorted(pair_counts.items())),
        "production_oracle": {"comparisons": oracle_count, "mismatches": production_mismatches},
        "counterfactual_oracle": {"comparisons": do_count, "mismatches": do_mismatches},
        "balance": balance,
        "shortcut_baselines": shortcuts,
        "best_shortcut_by_axis": best_shortcut,
        "shallow_baseline": shallow,
        "development_gates_current_versioned_after_v2": GATES,
        "passed": (
            len(set(programme_ids)) == len(all_programs)
            and len(signatures) == len(all_programs)
            and not exact_duplicates
            and all(value == 0 for value in programme_overlap.values())
            and target_composition_overlap["causal_target_composition"] == 0
            and parallel_composition_overlap["causal_parallel_composition"] == 0
            and production_mismatches == 0
            and do_mismatches == 0
            and best_shortcut["heldout_target_composition"] <= 0.65
            and best_shortcut["heldout_parallel_composition"] <= 0.65
        ),
    }

    if materialize:
        root = _root()
        programme_rows = []
        for group, programs in catalog.items():
            for program in programs:
                programme_rows.append(
                    {
                        "namespace": NAMESPACE,
                        "catalog_group": group,
                        "partition": program.partition,
                        "axis": program.axis,
                        "programme_id": program.programme_id,
                        "target_composition_id": program.target_composition_id,
                        "parallel_composition_id": program.parallel_composition_id,
                        "operations": list(program.operations),
                    }
                )
        write_jsonl(root / "registries/dev_only/programme-manifest.jsonl", programme_rows)
        example_rows = (
            {
                "namespace": NAMESPACE,
                "example_id": row.example_id,
                "partition": row.partition,
                "axis": row.axis,
                "programme_id": row.programme.programme_id,
                "a": row.a,
                "b": row.b,
                "c": row.c,
                "d": row.d,
                "z1": row.values.z1,
                "z2": row.values.z2,
                "z3": row.values.z3,
                "n1": row.values.n1,
                "n2": row.values.n2,
                "n3": row.values.n3,
                "joint_answer": row.joint_answer,
                "included": True,
                "exclusion_reason": None,
            }
            for partition in PARTITIONS
            for row in split_rows[partition]
        )
        write_jsonl(root / "registries/dev_only/example-index.jsonl", example_rows)
        write_jsonl(root / "registries/dev_only/exclusions.jsonl", [])
        write_json(root / "results/dev_only/task-design-audit.json", report)
        programme_path = root / "registries/dev_only/programme-manifest.jsonl"
        example_path = root / "registries/dev_only/example-index.jsonl"
        split_manifest = {
                "schema": "exp33ae.split_manifest.v1",
                "namespace": NAMESPACE,
                "input_policy": "all 2^4=16 input tuples included for every registered programme",
                "programme_counts": report["programme_counts"],
                "record_counts": report["records"],
                "programme_manifest_sha256": sha256_file(programme_path),
                "example_index_sha256": sha256_file(example_path),
                "future_generation_seed": None,
                "confirmation_artifacts_created": 0,
                "confirmation_artifacts_accessed": 0,
        }
        write_json(root / "registries/dev_only/split-manifest.json", split_manifest)
    return report
