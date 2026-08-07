"""Deterministic balanced Boolean programme catalog with two held-out axes."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product

from .spec import ALL_OPERATIONS, HELDOUT_OPERATION_IDS, PROGRAM_COUNTS, TRAIN_OPERATION_IDS, stable_digest
from .task import Programme, interpret, programme_examples


def _ordered_triples(operation_ids: tuple[int, ...], salt: str) -> list[tuple[int, int, int]]:
    triples = list(product(operation_ids, repeat=3))
    triples.sort(key=lambda triple: stable_digest(f"{salt}:{triple}"))
    return triples


def _is_joint_balanced(programme: Programme) -> bool:
    values = [interpret(programme, a, b, c, d) for a, b, c, d in product(range(2), repeat=4)]
    joint = Counter(value.joint_answer for value in values)
    state_balanced = all(
        Counter(getattr(value, variable) for value in values) == Counter({0: 8, 1: 8})
        for variable in ("z1", "z2", "z3", "n1", "n2", "n3")
    )
    return state_balanced and [joint[i] for i in range(4)] == [4, 4, 4, 4]


def _programmes(target_pool, parallel_pool, count: int, partition: str, axis: str, salt: str):
    candidates = []
    for target_ids in target_pool:
        for parallel_ids in parallel_pool:
            ops = tuple(ALL_OPERATIONS[i] for i in (*target_ids, *parallel_ids))
            programme = Programme(*ops, partition=partition, axis=axis)
            if _is_joint_balanced(programme):
                candidates.append(programme)
    candidates.sort(key=lambda p: stable_digest(f"{salt}:{p.programme_id}"))
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} balanced programmes available for {salt}")
    return tuple(candidates[:count])


@lru_cache(maxsize=1)
def programme_catalog() -> dict[str, tuple[Programme, ...]]:
    target = _ordered_triples(TRAIN_OPERATION_IDS, "boolean-target-v4")
    parallel = _ordered_triples(TRAIN_OPERATION_IDS, "boolean-parallel-v4")
    catalogue_specs = {
        "training": (target[:40], parallel[:40], "training", "training_compositions"),
        "model_selection": (target[40:46], parallel[40:46], "model_selection", "heldout_joint_compositions"),
        "localization_fit": (target[46:52], parallel[46:52], "localization_fit", "heldout_joint_compositions"),
        "localization_selection": (target[52:58], parallel[52:58], "localization_selection", "heldout_joint_compositions"),
        "causal_target_composition": (
            target[58:64], parallel[:40], "causal_evaluation", "heldout_target_chain_composition"
        ),
        "causal_parallel_composition": (
            target[:40], parallel[58:64], "causal_evaluation", "heldout_parallel_chain_composition"
        ),
        "stress_integrity": (
            target[58:64], parallel[58:64], "stress_integrity", "dev_only_doubly_heldout_compositions"
        ),
    }
    catalog = {}
    for name, (target_pool, parallel_pool, partition, axis) in catalogue_specs.items():
        catalog[name] = _programmes(
            target_pool,
            parallel_pool,
            PROGRAM_COUNTS[name],
            partition,
            axis,
            f"boolean-{name}-v4-fixed-table",
        )
    return catalog


@lru_cache(maxsize=None)
def records(split: str) -> tuple:
    catalog = programme_catalog()
    if split == "causal_evaluation":
        programs = catalog["causal_target_composition"] + catalog["causal_parallel_composition"]
    elif split in catalog:
        programs = catalog[split]
    else:
        raise ValueError(f"unknown DEV_ONLY split {split!r}")
    return tuple(example for program in programs for example in programme_examples(program))
