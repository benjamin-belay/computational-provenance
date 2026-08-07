#!/usr/bin/env python3
"""Independent audit of the sealed RNG2 clean-calibration reserve split."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "protocol/csw-am-scb-m1-rng2-ccal-r0/CONFIG.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def decrypt(envelope: dict[str, Any], key: bytes, aad: bytes) -> dict[str, Any]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    assert envelope["aad_sha256"] == sha(aad)
    return json.loads(AESGCM(key).decrypt(
        bytes.fromhex(envelope["nonce_hex"]),
        bytes.fromhex(envelope["ciphertext_hex"]),
        aad,
    ))


def merkle(leaves: list[str]) -> str:
    values = list(leaves)
    while len(values) > 1:
        if len(values) % 2:
            values.append(values[-1])
        values = [
            sha(b"NODE\x00" + bytes.fromhex(values[index])
                + bytes.fromhex(values[index + 1]))
            for index in range(0, len(values), 2)
        ]
    return values[0]


def audit(
    old_secret_path: Path, new_secret_path: Path, allocation: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config = read_json(CONFIG)
    old_secret, new_secret = read_json(old_secret_path), read_json(new_secret_path)
    old_envelope = read_json(ROOT / config["lineage"]["old_full_allocation_path"])
    old = decrypt(
        old_envelope, bytes.fromhex(old_secret["encryption_key"]),
        config["lineage"]["old_aad"].encode(),
    )
    new = decrypt(
        read_json(allocation / "FULL_ALLOCATION.sealed"),
        bytes.fromhex(new_secret["encryption_key"]),
        config["new_controller"]["sealed_aad"].encode(),
    )
    public = read_json(allocation / "ROLE_COMMITMENTS.json")
    official = read_json(allocation / "OFFICIAL_RESULT.json")
    reserve = old["roles"]["PUBLIC_RESERVE"]
    seed = bytes.fromhex(new_secret["allocation_seed"])
    domain = config["new_controller"]["allocation_domain"].encode()
    expected = sorted(
        reserve,
        key=lambda row: (
            hmac.new(seed, domain + b"\x00" + row["closure_id"].encode(), hashlib.sha256).digest(),
            row["closure_id"],
        ),
    )
    expected_roles = {
        "RNG2_CLEAN_CALIBRATION": expected[:512],
        "RNG2_REMAINING_PUBLIC_RESERVE": expected[512:],
    }
    roots = {
        role: merkle([record["leaf_sha256"] for record in new["roles"][role]])
        for role in expected_roles
    }
    checks = {
        "identity": new["identity"] == config["identity"],
        "seed_commitment": new["seed_commitment"] == public["seed_commitment"],
        "source_reserve": (
            len(reserve) == 640
            and new["source_reserve_root"] == public["source_reserve_root"]
        ),
        "exact_rank_split": all(
            [row["closure_id"] for row in new["roles"][role]]
            == [row["closure_id"] for row in expected_roles[role]]
            for role in expected_roles
        ),
        "counts": (
            len(new["roles"]["RNG2_CLEAN_CALIBRATION"]) == 512
            and len(new["roles"]["RNG2_REMAINING_PUBLIC_RESERVE"]) == 128
        ),
        "roots": roots == official["role_roots"]
            == {role: public["roles"][role]["merkle_root"] for role in roots},
        "full_root": new["full_allocation_root"] == public["full_allocation_root"]
            == official["full_allocation_root"],
        "validation_unchanged": (
            public["validation_role_unchanged"]["membership_root"]
            == "cf121cb23d0d2fc85aaa01c2ec30cb9617efe4e748789dd2a6726d7c18ef11e1"
            and public["validation_role_unchanged"]["authorized"] is False
        ),
        "no_release": (
            public["identity_release_count"] == 0
            and official["new_identity_releases"] == 0
            and official["calibration_release_authorized"] is False
        ),
        "official": official["passed"] is True,
    }
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_ccal_r0.reference.v1",
        "identity": config["identity"],
        "checks": checks,
        "role_counts": {role: len(rows) for role, rows in expected_roles.items()},
        "role_roots": roots,
        "passed": all(checks.values()),
        "terminal": (
            config["terminals"]["ready"] if all(checks.values())
            else config["terminals"]["allocation_invalid"]
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "REFERENCE_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-secret", type=Path, required=True)
    parser.add_argument("--new-secret", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.old_secret, args.new_secret, args.allocation, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
