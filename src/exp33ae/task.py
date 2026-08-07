"""Production causal programme and tokenization; DEV_ONLY records only."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from .spec import (
    BOS,
    INPUT_MARKER,
    K,
    NAMESPACE,
    N1_SLOT,
    N2_SLOT,
    NUMBER_OFFSET,
    PARTITIONS,
    QUERY,
    ROLE_TOKENS,
    SEQ_LEN,
    Z1_SLOT,
    Z2_SLOT,
)

Operation = tuple[int, int, int, int]


class ForbiddenNamespace(RuntimeError):
    pass


def require_dev_partition(partition: str) -> None:
    if partition not in PARTITIONS:
        raise ForbiddenNamespace(f"refusing non-DEV_ONLY partition {partition!r}")


def apply_operation(op: Operation, x: int, u: int) -> int:
    return op[2 * x + u]


@dataclass(frozen=True)
class Programme:
    t1: Operation
    t2: Operation
    t3: Operation
    n1: Operation
    n2: Operation
    n3: Operation
    partition: str
    axis: str

    @property
    def operations(self) -> tuple[Operation, ...]:
        return self.t1, self.t2, self.t3, self.n1, self.n2, self.n3

    @property
    def programme_id(self) -> str:
        payload = ";".join("".join(map(str, op)) for op in self.operations)
        return "pgm-" + sha256(payload.encode()).hexdigest()[:16]

    @property
    def target_composition_id(self) -> str:
        payload = ";".join("".join(map(str, op)) for op in (self.t1, self.t2, self.t3))
        return "tcomp-" + sha256(payload.encode()).hexdigest()[:16]

    @property
    def parallel_composition_id(self) -> str:
        payload = ";".join("".join(map(str, op)) for op in (self.n1, self.n2, self.n3))
        return "ncomp-" + sha256(payload.encode()).hexdigest()[:16]

    def tokens(self, a: int, b: int, c: int, d: int) -> tuple[int, ...]:
        values: list[int] = [BOS]
        for role in ROLE_TOKENS:
            values.append(ROLE_TOKENS[role])
        values.extend(
            (
                INPUT_MARKER,
                NUMBER_OFFSET + a,
                NUMBER_OFFSET + b,
                NUMBER_OFFSET + c,
                NUMBER_OFFSET + d,
                Z1_SLOT,
                N1_SLOT,
                Z2_SLOT,
                N2_SLOT,
                QUERY,
            )
        )
        assert len(values) == SEQ_LEN
        return tuple(values)


@dataclass(frozen=True)
class CausalValues:
    z1: int
    z2: int
    z3: int
    n1: int
    n2: int
    n3: int

    @property
    def joint_answer(self) -> int:
        return self.z3 * K + self.n3


def interpret(programme: Programme, a: int, b: int, c: int, d: int) -> CausalValues:
    z1 = apply_operation(programme.t1, a, b)
    z2 = apply_operation(programme.t2, z1, c)
    z3 = apply_operation(programme.t3, z2, d)
    n1 = apply_operation(programme.n1, a, d)
    n2 = apply_operation(programme.n2, n1, b)
    n3 = apply_operation(programme.n3, n2, c)
    return CausalValues(z1, z2, z3, n1, n2, n3)


def do_z2(programme: Programme, d: int, value: int) -> int:
    return apply_operation(programme.t3, value, d)


@dataclass(frozen=True)
class Example:
    namespace: str
    example_id: str
    partition: str
    axis: str
    programme: Programme
    a: int
    b: int
    c: int
    d: int
    values: CausalValues

    @property
    def tokens(self) -> tuple[int, ...]:
        return self.programme.tokens(self.a, self.b, self.c, self.d)

    @property
    def operation_tables(self) -> tuple[tuple[int, int, int, int], ...]:
        return self.programme.operations

    @property
    def joint_answer(self) -> int:
        return self.values.joint_answer


def make_example(programme: Programme, a: int, b: int, c: int, d: int) -> Example:
    require_dev_partition(programme.partition)
    raw = f"{programme.programme_id}:{a}:{b}:{c}:{d}"
    return Example(
        NAMESPACE,
        "devex-" + sha256(raw.encode()).hexdigest()[:20],
        programme.partition,
        programme.axis,
        programme,
        a,
        b,
        c,
        d,
        interpret(programme, a, b, c, d),
    )


def programme_examples(programme: Programme) -> Iterable[Example]:
    for a in range(K):
        for b in range(K):
            for c in range(K):
                for d in range(K):
                    yield make_example(programme, a, b, c, d)
