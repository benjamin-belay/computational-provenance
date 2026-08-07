#!/usr/bin/env python3
"""Independent static reconstruction for RNG2-INT-S0."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "protocol/csw-am-scb-m1-rng2-int-s0"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def reconstruct(official_dir: Path, output: Path) -> dict[str, Any]:
    config = read(BASE / "CONFIG.json")
    protocol = read(BASE / "INTEGRATION_PROTOCOL.json")
    matrix = read(BASE / "VERDICT_MATRIX.json")
    role = read(BASE / "PUBLIC_ROLE_BINDING.json")
    official = read(official_dir / "OFFICIAL_RESULT.json")
    receipt = read(ROOT / config["bindings"]["receipt_main_path"])
    rng2 = read(ROOT / config["bindings"]["rng2_v1_main_path"])
    reserve = read(ROOT / config["bindings"]["reserve_commitments_path"])
    gates = {
        "receipt_confirmed": receipt["main_verdict"] == config["bindings"]["receipt_terminal"],
        "rng2_validated": (
            rng2["main_verdict"] == config["bindings"]["rng2_v1_terminal"]
            and rng2["thresholds"]["T"] == protocol["unchanged_channel"]["T"]
            and rng2["thresholds"]["M"] == protocol["unchanged_channel"]["M"]
        ),
        "gating_order": protocol["claim_bearing_generation_order"].index(
            "VERIFY_CAUSAL_CHAIN"
        ) < protocol["claim_bearing_generation_order"].index("DERIVE_RNG2_AUTHORITY"),
        "five_decisions": len(matrix["channels"]) == 5,
        "role_root": (
            reserve["roles"]["RNG2_REMAINING_PUBLIC_RESERVE"]["merkle_root"]
            == role["source_membership_root"] == role["target_membership_root"]
        ),
        "no_redraw_or_release": (
            role["redraw"] is False and role["manual_swap"] is False
            and role["identity_release"] == 0
        ),
        "error_claim_bounded": (
            config["error_interpretation"]["INT_V1_role"]
            == "INTEGRATION_REGRESSION_AUDIT"
            and config["error_interpretation"]["INT_V1_independently_reestimates_one_percent_rate"]
            is False
            and math.isclose(
                config["error_interpretation"]["zero_event_one_sided_95_upper_bound"],
                1 - 0.05 ** (1 / 128),
            )
        ),
        "zero_work": all(
            value == 0 for value in official["zero_scientific_work"].values()
        ),
        "official_pass": official["passed"] is True
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_s0.reference.v1",
        "identity": config["identity"],
        "gates": gates,
        "passed": all(gates.values()),
        "official_sha256": digest(official_dir / "OFFICIAL_RESULT.json"),
        "reconstructed": {
            "T": protocol["unchanged_channel"]["T"],
            "M": protocol["unchanged_channel"]["M"],
            "validation_closures": role["closures"],
            "validation_root": role["target_membership_root"],
            "identity_releases": role["identity_release"],
            "generic_error_control_inherited": True,
            "independent_one_percent_reestimation": False
        }
    }
    write(output / "REFERENCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.official, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
