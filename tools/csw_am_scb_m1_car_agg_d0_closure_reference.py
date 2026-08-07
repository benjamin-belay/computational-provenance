#!/usr/bin/env python3
"""Independent read-only reconstruction of the AGG-D0 closure evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED")
    return value


def assess(evidence: Path) -> dict[str, Any]:
    predecessor = ROOT / "artifacts/dev_only/csw-am-scb-m1-car-agg-d0"
    source_terminal = load(predecessor / "TERMINAL.json")
    source_audit = load(predecessor / "MAIN_AUDIT.json")
    closure = load(evidence / "CLOSURE.json")
    claims = load(evidence / "CLAIMS_MATRIX.json")
    resources = load(evidence / "RESOURCE_STATUS.json")
    successor = load(evidence / "RNG2_SUCCESSOR_BOUNDARY.json")
    official = load(evidence / "OFFICIAL_RESULT.json")
    draw_rows = [
        json.loads(line)
        for line in (predecessor / "PRIMITIVE_DRAW_COMPARISONS.jsonl")
        .read_text().splitlines()
    ]
    token_mismatch = min(
        row["model_step"] for row in draw_rows
        if row["natural_selected_token"] != row["alias_selected_token"]
    )
    uniform_mismatch = min(
        row["model_step"] for row in draw_rows
        if row["natural_base_uniform"] != row["alias_base_uniform"]
    )
    statuses = {row["claim"]: row["status"] for row in claims["claims"]}
    gates = {
        "source_terminal_preserved": (
            closure["predecessor_scientific_terminal_preserved"]
            == source_terminal["terminal"]
            == source_audit["main_verdict"]
        ),
        "divergence_reconstructed": (
            token_mismatch == 3
            and uniform_mismatch == 4
            and closure["diagnostic_fixture"]["first_selected_token_divergence_step"] == 3
            and closure["diagnostic_fixture"]["first_primitive_draw_divergence_step"] == 4
        ),
        "performance_unclaimed": statuses["Aggregate detector performance"] == "UNTESTED",
        "thresholds_uninstantiated": (
            statuses["Threshold T"] == "NOT_INSTANTIATED"
            and statuses["Margin M"] == "NOT_INSTANTIATED"
        ),
        "future_roles_untouched": resources["untouched"] == {
            "calibration_closures": 512,
            "protected_confirmation_closures": 256,
            "protected_qualification_closures": 128,
            "public_reserve_closures": 640,
            "validation_closures": 512,
        },
        "agg_cal1_prohibited": closure["AGG_CAL1"] == "PROHIBITED",
        "successor_non_authorizing": (
            successor["status"] == "DRAFT_NON_AUTHORIZING"
            and successor["successor_authorized"] is False
            and successor["new_stochastic_execution_authorized"] is False
            and successor["protected_release_authorized"] is False
        ),
        "official_terminal": (
            official["terminal"]
            == "MAIN_CSW_AM_SCB_M1_CAR_AGG_D0_CLOSURE_CONFIRMED"
            and official["passed"] is True
        ),
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_car_agg_d0_closure.reference.v1",
        "gates": gates,
        "valid": all(gates.values()),
        "first_selected_token_divergence_step": token_mismatch,
        "first_primitive_draw_divergence_step": uniform_mismatch,
        "predecessor_terminal": source_terminal["terminal"],
        "terminal": (
            "MAIN_CSW_AM_SCB_M1_CAR_AGG_D0_CLOSURE_CONFIRMED"
            if all(gates.values())
            else "MAIN_CSW_AM_SCB_M1_CAR_AGG_D0_CLOSURE_INVALID"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.evidence)
    data = canonical(result) + b"\n"
    if args.output:
        if args.output.exists():
            raise ValueError("OUTPUT_BOUNDARY")
        args.output.write_bytes(data)
    else:
        print(data.decode(), end="")


if __name__ == "__main__":
    main()
