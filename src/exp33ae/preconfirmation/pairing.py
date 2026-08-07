"""Exact deterministic primary pair strata for the DEV_ONLY protocol fixture."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256

from ..artifacts import canonical_json
from ..independent_oracle import oracle_do_z2
from ..spec import ALL_OPERATIONS
from .entropy import DevEntropy
from .normative import DEV_NAMESPACE, load_protocol


STRATA = ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")


def _axis_code(axis: str) -> int:
    return 0 if axis == "blinded_within_run_target_composition" else 1


def _category(row) -> tuple[int, int, int, int]:
    return _axis_code(row.axis), row.z2, row.z3, row.n3


def _score(entropy: DevEntropy, label: str, value) -> bytes:
    return sha256(entropy.derive("source-base-pair-allocation", 0) + canonical_json([label, value]).encode()).digest()


def _counterfactual(base, source_z2: int) -> dict:
    operations = tuple(ALL_OPERATIONS[index] for index in base.operation_ids)
    z3 = oracle_do_z2(operations, base.d, source_z2)["z3"]
    return {"z2": source_z2, "z3": z3, "n3": base.n3, "answer": 2 * z3 + base.n3}


def _base_eligible(stratum: str, base) -> bool:
    other = 1 - base.z2
    cf = _counterfactual(base, other)
    if stratum in {"A_answer_changing", "D_same_observable_answer"}:
        return cf["answer"] != base.answer
    if stratum == "B_outcome_collision":
        return cf["answer"] == base.answer
    return True


def _source_category(stratum: str, base_category: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    axis, z2, z3, n3 = base_category
    if stratum == "A_answer_changing":
        return 1 - axis, 1 - z2, 1 - z3, n3
    if stratum in {"B_outcome_collision", "D_same_observable_answer"}:
        return 1 - axis, 1 - z2, z3, n3
    return 1 - axis, z2, z3, n3


def build_pairs(allocation: dict, entropy: DevEntropy, counts: dict[str, int] | None = None, namespace: str = DEV_NAMESPACE) -> list[dict]:
    counts = counts or {name: load_protocol()["pair_strata"][name]["count"] for name in STRATA}
    examples = [row for row in allocation["examples"] if row.partition in {"clean_target_axis", "clean_parallel_axis"}]
    by_category = {}
    for category in [(axis, z2, z3, n3) for axis in range(2) for z2 in range(2) for z3 in range(2) for n3 in range(2)]:
        by_category[category] = [row for row in examples if _category(row) == category]
    pairs, identities = [], set()
    ordinal = 0
    for stratum in STRATA:
        count = counts[stratum]
        per_category = count // 16
        if per_category * 16 != count:
            raise RuntimeError("pair count must support exact 16-cell balance")
        for category in sorted(by_category):
            bases = [row for row in by_category[category] if _base_eligible(stratum, row)]
            bases.sort(key=lambda row: _score(entropy, stratum + ":base", row.example_id))
            sources = list(by_category[_source_category(stratum, category)])
            sources.sort(key=lambda row: _score(entropy, stratum + ":source", row.example_id))
            if len(bases) < per_category or len(sources) < per_category:
                raise RuntimeError(f"insufficient exact pair cell {stratum} {category}")
            for offset in range(per_category):
                base = bases[offset]
                source = sources[offset]
                if base.example_id == source.example_id:
                    raise RuntimeError("source and base must be distinct")
                cf = _counterfactual(base, source.z2)
                predicate = {
                    "A_answer_changing": base.z2 != source.z2 and cf["answer"] != base.answer and base.answer != source.answer,
                    "B_outcome_collision": base.z2 != source.z2 and cf["answer"] == base.answer,
                    "C_same_state": base.z2 == source.z2,
                    "D_same_observable_answer": base.z2 != source.z2 and base.answer == source.answer and cf["answer"] != base.answer,
                }[stratum]
                if not predicate:
                    raise RuntimeError(f"formal eligibility failed for {stratum}")
                identity = (base.example_id, source.example_id)
                if identity in identities:
                    raise RuntimeError("pair identity reused across primary strata")
                identities.add(identity)
                raw = entropy.derive("source-base-pair-allocation", ordinal, 16)
                pair_id = "devonly-pcrpair-" + raw.hex()
                ordinal += 1
                pairs.append({
                    "schema": "exp33ae.dev_only_pair.v1", "namespace": namespace, "stage_id": entropy.stage_id,
                    "pair_id": pair_id, "stratum": stratum, "eligible": True,
                    "base_id": base.example_id, "source_id": source.example_id,
                    "base_programme_id": base.programme_id, "source_programme_id": source.programme_id,
                    "base_axis": base.axis, "source_axis": source.axis,
                    "base_z2": base.z2, "base_z3": base.z3, "base_n3": base.n3, "base_answer": base.answer,
                    "source_z2": source.z2, "source_z3": source.z3, "source_n3": source.n3, "source_answer": source.answer,
                    "collision": cf["answer"] == base.answer, "same_observable_answer": base.answer == source.answer,
                    "no_op": base.z2 == source.z2, "expected_z2": cf["z2"], "expected_z3": cf["z3"],
                    "expected_n3": cf["n3"], "expected_answer": cf["answer"],
                })
    if Counter(row["stratum"] for row in pairs) != Counter(counts):
        raise RuntimeError("pair stratum count mismatch")
    for stratum in STRATA:
        rows = [row for row in pairs if row["stratum"] == stratum]
        expected_each = len(rows) // 16
        for side in ("base", "source"):
            counts = Counter((0 if row[f"{side}_axis"].endswith("target_composition") else 1, row[f"{side}_z2"], row[f"{side}_z3"], row[f"{side}_n3"]) for row in rows)
            if len(counts) != 16 or set(counts.values()) != {expected_each}:
                raise RuntimeError(f"nonuniform {side} pair marginals in {stratum}: {counts}")
    return pairs
