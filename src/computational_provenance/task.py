"""Registered arithmetic task and answer-preserving intervention."""
from __future__ import annotations


def task(a: int, b: int, c: int, d: int) -> dict[str, int]:
    values = (a, b, c, d)
    if any(type(value) is not int or not 0 <= value < 16 for value in values):
        raise ValueError("inputs must be integers in [0, 15]")
    z1 = (a + b) % 16
    z2 = (z1 + c) % 16
    z3 = (5 * z2 + d) % 16
    return {"z1": z1, "z2": z2, "z3": z3, "answer": z3 % 8}


def alias_state(state: dict[str, int]) -> dict[str, int]:
    """Return the registered alternative path with the same final answer."""
    return {"z1": state["z1"], "z2": (state["z2"] + 8) % 16,
            "z3": (state["z3"] + 8) % 16, "answer": state["answer"]}
