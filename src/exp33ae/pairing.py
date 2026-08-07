"""Prospectively balanced, outcome-independent source/base pairing."""

from __future__ import annotations

from dataclasses import dataclass

from .independent_oracle import oracle_do_z2
from .spec import K, stable_digest


@dataclass(frozen=True)
class Pair:
    pair_id: str
    kind: str
    base: object
    source: object
    expected_z3: int
    expected_joint: int
    no_op: bool


def _ordered(rows, salt: str):
    return sorted(rows, key=lambda row: stable_digest(f"{salt}:{row.example_id}"))


def _select_unique(candidates, quota: int, used: set[tuple[str, str]], salt: str):
    ordered = sorted(candidates, key=lambda pair: stable_digest(f"{salt}:{pair[0].example_id}:{pair[1].example_id}"))
    selected = []
    for base, source in ordered:
        identity = (base.example_id, source.example_id)
        if identity in used:
            continue
        used.add(identity)
        selected.append((base, source))
        if len(selected) == quota:
            return selected
    raise RuntimeError(f"only {len(selected)}/{quota} unique pairs for stratum {salt}")


def eligible_pairs(rows, kind: str, count: int, seed: int) -> list[Pair]:
    if kind not in {"different_state", "same_state", "same_answer_different_state"}:
        raise ValueError(kind)
    axes = sorted({row.axis for row in rows})
    strata = []
    if kind == "different_state":
        for axis in axes:
            for base_state in range(K):
                for expected_joint in range(K * K):
                    strata.append((axis, base_state, expected_joint))
    else:
        for axis in axes:
            for base_state in range(K):
                for base_joint in range(K * K):
                    strata.append((axis, base_state, base_joint))
    if count % len(strata):
        raise ValueError(f"count {count} is not divisible by {len(strata)} pairing strata")
    quota = count // len(strata)
    used: set[tuple[str, str]] = set()
    chosen = []
    for stratum_index, stratum in enumerate(strata):
        axis, base_state, label = stratum
        bases = [
            row for row in rows
            if row.axis == axis and row.values.z2 == base_state
            and (kind == "different_state" or row.joint_answer == label)
        ]
        for source_axis_index, source_axis in enumerate(axes):
            # If a per-stratum quota is odd, alternate the extra source-axis
            # assignment by stratum. There are an even number of strata, so
            # aggregate source-axis counts remain exact and outcome-independent.
            source_quota = quota // len(axes)
            if source_axis_index == (stratum_index + seed) % len(axes):
                source_quota += quota % len(axes)
            sources = [
                row
                for row in rows
                if row.axis == source_axis and row.example_id not in {base.example_id for base in bases}
            ]
            candidates = []
            for base in _ordered(bases, f"base:{seed}:{stratum}:{source_axis}"):
                for source in _ordered(sources, f"source:{seed}:{stratum}:{source_axis}:{base.example_id}"):
                    if base.example_id == source.example_id:
                        continue
                    if kind == "different_state" and base.values.z2 == source.values.z2:
                        continue
                    if kind == "same_state" and base.values.z2 != source.values.z2:
                        continue
                    if kind == "same_answer_different_state" and not (
                        base.values.z2 != source.values.z2 and base.joint_answer == source.joint_answer
                    ):
                        continue
                    expected_z3 = oracle_do_z2(base.programme.operations, base.d, source.values.z2)["z3"]
                    expected_joint = expected_z3 * K + base.values.n3
                    if kind == "different_state" and expected_joint != label:
                        continue
                    candidates.append((base, source))
            chosen.extend(
                _select_unique(
                    candidates,
                    source_quota,
                    used,
                    f"{seed}:{kind}:{stratum}:source-axis:{source_axis}",
                )
            )
    pairs = []
    for index, (base, source) in enumerate(chosen):
        expected_z3 = oracle_do_z2(base.programme.operations, base.d, source.values.z2)["z3"]
        expected_joint = expected_z3 * K + base.values.n3
        pairs.append(
            Pair(
                pair_id=f"devpair-{kind}-{seed}-{index:05d}",
                kind=kind,
                base=base,
                source=source,
                expected_z3=expected_z3,
                expected_joint=expected_joint,
                no_op=base.values.z2 == source.values.z2,
            )
        )
    return pairs
