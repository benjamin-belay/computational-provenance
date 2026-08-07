"""Deterministic confirmation-shaped allocator, barred to DEV_ONLY entropy."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from ..artifacts import canonical_json
from ..independent_oracle import oracle
from ..spec import ALL_OPERATIONS
from .entropy import DevEntropy
from .normative import DEV_NAMESPACE, load_protocol, require_dev_stage


def _operation_ids(operations) -> tuple[int, ...]:
    lookup = {tuple(op): index for index, op in enumerate(ALL_OPERATIONS)}
    return tuple(lookup[tuple(op)] for op in operations)


def programme_id(ids: tuple[int, ...]) -> str:
    payload = ";".join("".join(map(str, ALL_OPERATIONS[index])) for index in ids)
    return "pgm-" + sha256(payload.encode()).hexdigest()[:16]


def composition_id(prefix: str, ids: tuple[int, int, int]) -> str:
    payload = ";".join("".join(map(str, ALL_OPERATIONS[index])) for index in ids)
    return prefix + sha256(payload.encode()).hexdigest()[:16]


def causal_signature(ids: tuple[int, ...]) -> str:
    operations = tuple(ALL_OPERATIONS[index] for index in ids)
    rows = [oracle(operations, inputs) for inputs in product(range(2), repeat=4)]
    return sha256(canonical_json(rows).encode()).hexdigest()


def is_balanced(ids: tuple[int, ...]) -> bool:
    operations = tuple(ALL_OPERATIONS[index] for index in ids)
    rows = [oracle(operations, inputs) for inputs in product(range(2), repeat=4)]
    return (
        all(Counter(row[name] for row in rows) == Counter({0: 8, 1: 8}) for name in ("z2", "z3", "n3"))
        and Counter(row["joint_answer"] for row in rows) == Counter({0: 4, 1: 4, 2: 4, 3: 4})
    )


@dataclass(frozen=True)
class ProtocolExample:
    namespace: str
    stage_id: str
    partition: str
    axis: str
    programme_id: str
    causal_signature: str
    target_composition_id: str
    parallel_composition_id: str
    operation_ids: tuple[int, ...]
    a: int
    b: int
    c: int
    d: int
    z1: int
    z2: int
    z3: int
    n1: int
    n2: int
    n3: int
    answer: int
    example_id: str

    def as_dict(self) -> dict:
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in self.__dict__.items()}

    @property
    def tokens(self) -> tuple[int, ...]:
        return (0, 1, 2, 3, 4, 5, 6, 7, 13 + self.a, 13 + self.b, 13 + self.c, 13 + self.d, 9, 10, 11, 12, 8)

    @property
    def operation_tables(self):
        return tuple(ALL_OPERATIONS[index] for index in self.operation_ids)

    @property
    def joint_answer(self) -> int:
        return self.answer


def _example(stage_id: str, partition: str, axis: str, ids: tuple[int, ...], inputs: tuple[int, int, int, int], namespace: str = DEV_NAMESPACE) -> ProtocolExample:
    pid = programme_id(ids)
    values = oracle(tuple(ALL_OPERATIONS[index] for index in ids), inputs)
    raw = canonical_json([stage_id, pid, *inputs])
    return ProtocolExample(
        namespace, stage_id, partition, axis, pid, causal_signature(ids),
        composition_id("tcomp-", ids[:3]), composition_id("ncomp-", ids[3:]), ids,
        *inputs, values["z1"], values["z2"], values["z3"], values["n1"], values["n2"], values["n3"],
        values["joint_answer"], "devonly-pcrex-" + sha256(raw.encode()).hexdigest()[:24],
    )


def development_exclusions(root: Path) -> dict:
    programmes = []
    with (root / "registries/dev_only/programme-manifest.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                programmes.append(json.loads(line))
    examples = []
    with (root / "registries/dev_only/example-index.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
    calls = []
    with (root / "registries/dev_only/evaluation-call-registry.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                calls.append(json.loads(line))
    return {
        "programme_ids": {row["programme_id"] for row in programmes},
        "example_ids": {row["example_id"] for row in examples},
        "pair_ids": {row["pair_id"] for row in calls},
        "call_ids": {row["call_id"] for row in calls},
        "nonces": {row["nonce"] for row in calls if "nonce" in row},
    }


def _score(entropy: DevEntropy, domain: str, value) -> bytes:
    seed = entropy.derive(domain, 0, 32)
    return sha256(seed + canonical_json(value).encode()).digest()


def allocate(
    root: Path,
    entropy: DevEntropy,
    counts_programmes: dict[str, int] | None = None,
    namespace: str = DEV_NAMESPACE,
    extra_exclusions: dict[str, set[str]] | None = None,
) -> dict:
    require_dev_stage(entropy.stage_id)
    counts_programmes = counts_programmes or load_protocol()["allocation"]["counts_programmes"]
    exclusions = development_exclusions(root)
    if extra_exclusions is not None:
        unknown = set(extra_exclusions) - set(exclusions)
        if unknown:
            raise ValueError(f"unknown exclusion categories: {sorted(unknown)}")
        exclusions = {
            key: set(values) | set(extra_exclusions.get(key, set()))
            for key, values in exclusions.items()
        }
    triples = list(product(range(4), repeat=3))
    target = sorted(triples, key=lambda value: _score(entropy, "programme-allocation", ["target", value]))
    parallel = sorted(triples, key=lambda value: _score(entropy, "programme-allocation", ["parallel", value]))
    train_t, held_t = target[:40], target[40:]
    train_n, held_n = parallel[:40], parallel[40:]
    specifications = (
        ("training", "revealed_training", train_t, train_n, 512),
        ("model_selection", "revealed_model_selection", train_t, train_n, 30),
        ("localization_fit", "revealed_localization_fit", train_t, train_n, 30),
        ("localization_selection", "revealed_localization_selection", train_t, train_n, 30),
        ("clean_target_axis", "blinded_within_run_target_composition", held_t, train_n, 50),
        ("clean_parallel_axis", "blinded_within_run_parallel_composition", train_t, held_n, 50),
    )
    used, allocations = set(), {}
    for partition, axis, target_pool, parallel_pool, count in specifications:
        candidates = []
        for t_ids in target_pool:
            for n_ids in parallel_pool:
                ids = (*t_ids, *n_ids)
                pid = programme_id(ids)
                if pid in exclusions["programme_ids"] or pid in used or not is_balanced(ids):
                    continue
                candidates.append(ids)
        candidates.sort(key=lambda value: _score(entropy, "split-allocation", [partition, value]))
        if len(candidates) < count:
            raise RuntimeError(f"allocation constraint failure for {partition}: {len(candidates)} < {count}")
        selected = tuple(candidates[:count])
        used.update(programme_id(ids) for ids in selected)
        allocations[partition] = selected

    examples = []
    programme_rows = []
    for partition, axis, *_ in specifications:
        for ids in allocations[partition]:
            programme_rows.append({
                "namespace": namespace, "stage_id": entropy.stage_id, "partition": partition, "axis": axis,
                "programme_id": programme_id(ids), "causal_signature": causal_signature(ids),
                "operation_ids": list(ids), "operations": [list(ALL_OPERATIONS[index]) for index in ids],
                "target_composition_id": composition_id("tcomp-", ids[:3]),
                "parallel_composition_id": composition_id("ncomp-", ids[3:]),
            })
            examples.extend(_example(entropy.stage_id, partition, axis, ids, inputs, namespace) for inputs in product(range(2), repeat=4))

    ids = [row["programme_id"] for row in programme_rows]
    signatures = [row["causal_signature"] for row in programme_rows]
    example_ids = [row.example_id for row in examples]
    if len(ids) != len(set(ids)) or len(signatures) != len(set(signatures)):
        raise RuntimeError("functional programme collision")
    if set(ids) & exclusions["programme_ids"] or set(example_ids) & exclusions["example_ids"]:
        raise RuntimeError("development allocation overlap")
    counts = Counter(row.partition for row in examples)
    expected = {name: count * 16 for name, count in counts_programmes.items()}
    if dict(counts) != expected:
        raise RuntimeError(f"example counts mismatch: {counts}")
    balance = {}
    for partition in expected:
        rows = [row for row in examples if row.partition == partition]
        balance[partition] = {
            "z2": dict(Counter(row.z2 for row in rows)), "z3": dict(Counter(row.z3 for row in rows)),
            "n3": dict(Counter(row.n3 for row in rows)), "answer": dict(Counter(row.answer for row in rows)),
        }
        if balance[partition] != {
            "z2": {0: len(rows)//2, 1: len(rows)//2}, "z3": {0: len(rows)//2, 1: len(rows)//2},
            "n3": {0: len(rows)//2, 1: len(rows)//2},
            "answer": {0: len(rows)//4, 1: len(rows)//4, 2: len(rows)//4, 3: len(rows)//4},
        }:
            raise RuntimeError(f"allocation imbalance in {partition}")
    return {
        "schema": "exp33ae.dev_only_allocation.v1", "namespace": namespace, "stage_id": entropy.stage_id,
        "claim_bearing": False, "programmes": programme_rows, "examples": examples,
        "target_training_compositions": [composition_id("tcomp-", value) for value in train_t],
        "target_heldout_compositions": [composition_id("tcomp-", value) for value in held_t],
        "parallel_training_compositions": [composition_id("ncomp-", value) for value in train_n],
        "parallel_heldout_compositions": [composition_id("ncomp-", value) for value in held_n],
        "balance": balance, "exclusion_counts": {key: len(value) for key, value in exclusions.items()},
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }


def shortcut_baselines(allocation: dict, entropy: DevEntropy | None = None) -> dict:
    train = [row for row in allocation["examples"] if row.partition == "training"]
    sets = {name: [row for row in allocation["examples"] if row.partition == name] for name in ("clean_target_axis", "clean_parallel_axis")}

    def accuracy(rows, key):
        tables = defaultdict(Counter)
        for row in train:
            tables[key(row)][row.answer] += 1
        correct = 0
        for row in rows:
            counts = tables.get(key(row), Counter({0: 1}))
            prediction = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            correct += prediction == row.answer
        return {"correct": correct, "total": len(rows)}

    keys = {
        "exact_programme_input_lookup": lambda r: (r.programme_id, r.a, r.b, r.c, r.d),
        "programme_only_lookup": lambda r: r.programme_id,
        "target_composition_input_lookup": lambda r: (r.target_composition_id, r.a, r.b, r.c, r.d),
        "parallel_composition_input_lookup": lambda r: (r.parallel_composition_id, r.a, r.b, r.c, r.d),
        "input_only_lookup": lambda r: (r.a, r.b, r.c, r.d),
        "last_two_inputs_lookup": lambda r: (r.c, r.d),
        "operation_unigram_bag_lookup": lambda r: tuple(sorted(r.operation_ids)),
    }
    result = {
        axis: {name: accuracy(rows, key) for name, key in keys.items()}
        for axis, rows in sets.items()
    }
    train_programmes = {row.programme_id: row.operation_ids for row in train}
    for axis, rows in sets.items():
        correct = 0
        for row in rows:
            nearest = max(
                train_programmes.items(),
                key=lambda item: (sum(left == right for left, right in zip(row.operation_ids, item[1])), item[0]),
            )[1]
            predicted = oracle(tuple(ALL_OPERATIONS[index] for index in nearest), (row.a, row.b, row.c, row.d))["joint_answer"]
            correct += predicted == row.answer
        result[axis]["nearest_programme_transition"] = {"correct": correct, "total": len(rows)}
        result[axis]["direct_output_marginal"] = {"correct": len(rows) // 4, "total": len(rows)}
    if entropy is not None:
        from ..baselines import shallow_mlp_baseline
        seed = int.from_bytes(entropy.derive("programme-allocation", 999, 8), "big") & ((1 << 63) - 1)
        shallow = shallow_mlp_baseline(tuple(train), {name: tuple(rows) for name, rows in sets.items()}, seed=seed)
        for axis, value in shallow["accuracy"].items():
            result[axis]["shallow_position_mlp"] = {"correct": round(value * len(sets[axis])), "total": len(sets[axis])}
    return result
