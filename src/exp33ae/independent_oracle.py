"""Independent non-importing oracle for the fixed modulus-five programme."""

from __future__ import annotations


def _step(op: tuple[int, int, int, int], x: int, u: int) -> int:
    return op[(x << 1) | u]


def oracle(ops: tuple[tuple[int, int, int, int], ...], inputs: tuple[int, int, int, int]) -> dict[str, int]:
    if len(ops) != 6:
        raise ValueError("oracle requires six operation tables")
    a, b, c, d = inputs
    z1 = _step(ops[0], a, b)
    z2 = _step(ops[1], z1, c)
    z3 = _step(ops[2], z2, d)
    n1 = _step(ops[3], a, d)
    n2 = _step(ops[4], n1, b)
    n3 = _step(ops[5], n2, c)
    return {"z1": z1, "z2": z2, "z3": z3, "n1": n1, "n2": n2, "n3": n3, "joint_answer": 2 * z3 + n3}


def oracle_do_z2(ops: tuple[tuple[int, int, int, int], ...], d: int, value: int) -> dict[str, int]:
    z3 = _step(ops[2], value, d)
    return {"z2": value, "z3": z3}
