#!/usr/bin/env python3
"""Independent closure audit for committed RNG2-D0 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/dev_only/csw-am-scb-m1-rng2-d0"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError(path.name)
    return value


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def committed(commit: str, path: Path) -> bool:
    relative = str(path.relative_to(ROOT))
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    ) == path.read_bytes()


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit(source_commit: str, evidence_commit: str) -> dict[str, Any]:
    official_path = ARTIFACT / "OFFICIAL_RESULT.json"
    reference_path = ARTIFACT / "REFERENCE_RESULT.json"
    official = read_json(official_path)
    reference = read_json(reference_path)
    manifest = read_json(ARTIFACT / "EVIDENCE_MANIFEST.json")
    manifest_checks = {
        item["path"]: (
            file_sha(ARTIFACT / item["path"]) == item["sha256"]
            and (ARTIFACT / item["path"]).stat().st_size == item["bytes"]
        )
        for item in manifest["files"]
    }
    committed_files = [
        path for path in ARTIFACT.iterdir()
        if path.is_file() and path.name not in ("MAIN_AUDIT.json", "TERMINAL.json")
    ]
    selection = rows(ARTIFACT / "SELECTED_DEVELOPMENT_ROWS.jsonl")
    distribution = rows(ARTIFACT / "DISTRIBUTIONAL_RESULTS.jsonl")
    operations = rows(ARTIFACT / "PAIRED_OPERATION_TRACES.jsonl")
    outputs = rows(ARTIFACT / "GENERIC_OUTPUTS.jsonl")
    deterministic = rows(ARTIFACT / "DETERMINISTIC_FIXTURES.jsonl")
    boundary = rows(ARTIFACT / "BOUNDARY_FIXTURES.jsonl")
    malformed = rows(ARTIFACT / "MALFORMED_FIXTURES.jsonl")

    distribution_reconstruction = True
    distribution_contexts = 0
    for fixture in distribution:
        distribution_contexts += fixture["contexts"]
        if sum(fixture["counts"]) != fixture["contexts"]:
            distribution_reconstruction = False
        for category in fixture["categories"]:
            probability = category["probability"]
            expected = fixture["contexts"] * probability
            bound = 0.0 if probability == 0.0 else (
                6.0 * math.sqrt(
                    fixture["contexts"] * probability * (1.0 - probability)
                ) + 1.0
            )
            passed = (
                category["count"] == 0 if probability == 0.0
                else abs(category["count"] - expected) <= bound
            )
            distribution_reconstruction &= (
                category["expected"] == expected
                and category["bound"] == bound
                and category["passed"] == passed
            )

    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    reference_agreement = 0
    none_count = 0
    none_valid = 0
    for operation in operations:
        by_pair[(operation["closure_id"], operation["model_step"])][operation["side"]] = operation
        reference_agreement += int(operation["reference_agreement"])
        if operation["selected_token"] == "<NONE>":
            none_count += 1
            none_valid += int(operation["public"] is False)
    paired_equal = 0
    post_divergence_equal = 0
    post_divergence_total = 0
    diverged = 0
    by_closure: dict[str, list[tuple[int, dict[str, dict[str, Any]]]]] = defaultdict(list)
    for (closure, step), pair in by_pair.items():
        by_closure[closure].append((step, pair))
    for closure, step_pairs in by_closure.items():
        step_pairs.sort()
        first = None
        for index, (_, pair) in enumerate(step_pairs):
            if pair["NATURAL"]["model_history_before"] != pair["ALIAS"]["model_history_before"]:
                first = index
                break
        if first is not None:
            diverged += 1
        for index, (_, pair) in enumerate(step_pairs):
            equal = (
                pair["NATURAL"]["uniform_bytes_hex"]
                == pair["ALIAS"]["uniform_bytes_hex"]
            )
            paired_equal += int(equal)
            if first is not None and index >= first:
                post_divergence_total += 1
                post_divergence_equal += int(equal)

    checks = {
        "source_commit_bound": (
            type(official["source_commit"]) is str
            and len(official["source_commit"]) >= 7
            and source_commit.startswith(official["source_commit"])
        ),
        "evidence_commit_bound": all(
            committed(evidence_commit, path) for path in committed_files
        ),
        "manifest_complete": all(manifest_checks.values()) and len(manifest_checks) == 11,
        "official_reference_terminal_agree": (
            official["passed"] is True
            and reference["passed"] is True
            and official["terminal"] == reference["terminal"]
        ),
        "selection_denominator": len(selection) == 128,
        "selection_support": (
            {(row["natural"]["z2"], row["template_id"]) for row in selection}
            == {(z2, template) for z2 in range(16) for template in range(4)}
            and {row["visible_answer"] for row in selection} == set(range(8))
        ),
        "deterministic_fixtures": (
            len(deterministic) == 6
            and all(row["exact_agreement"] for row in deterministic)
        ),
        "boundary_fixtures": len(boundary) == 8 and all(row["passed"] for row in boundary),
        "malformed_fixtures": len(malformed) == 9 and all(row["passed"] for row in malformed),
        "distributional_reconstruction": (
            len(distribution) == 8
            and distribution_contexts == 2_097_152
            and distribution_reconstruction
        ),
        "output_denominator": len(outputs) == 128,
        "operation_denominator": len(operations) == 6272,
        "production_reference_operation_agreement": reference_agreement == 6272,
        "paired_uniform_reconstruction": paired_equal == 3136,
        "post_divergence_pairing_reconstruction": (
            diverged == 64
            and post_divergence_equal == post_divergence_total
            and post_divergence_total == 2925
        ),
        "NONE_reconstruction": none_count == 58 and none_valid == 58,
        "future_roles_unaccessed": all(
            official["release_status"][field] == 0
            for field in (
                "calibration_identities_accessed", "validation_identities_accessed",
                "reserve_identities_accessed", "protected_identities_accessed",
            )
        ),
        "CAL1_not_authorized": official["release_status"]["RNG2_CAL1_authorized"] is False,
    }
    passed = all(checks.values())
    terminal = (
        "MAIN_CSW_AM_SCB_M1_RNG2_D0_IMPLEMENTATION_READY"
        if passed else "MAIN_CSW_AM_SCB_M1_RNG2_D0_AUDIT_FAILED"
    )
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.main_audit.v1",
        "identity": "CSW-AM-SCB-M1-RNG2-D0",
        "source_freeze_commit": source_commit,
        "raw_evidence_commit": evidence_commit,
        "passed": passed,
        "main_verdict": terminal,
        "checks": checks,
        "reconstructed_metrics": {
            "selected_closures": len({row["closure_id"] for row in selection}),
            "selected_oriented_rows": len(selection),
            "distributional_contexts": distribution_contexts,
            "generic_outputs": len(outputs),
            "operation_rows": len(operations),
            "paired_uniform_agreements": paired_equal,
            "diverged_closures": diverged,
            "post_divergence_uniform_agreements": post_divergence_equal,
            "post_divergence_uniform_comparisons": post_divergence_total,
            "NONE_operations": none_count,
        },
        "evidence_bindings": {
            "official_result_sha256": file_sha(official_path),
            "reference_result_sha256": file_sha(reference_path),
            "evidence_manifest_sha256": file_sha(ARTIFACT / "EVIDENCE_MANIFEST.json"),
        },
        "claim_boundary": {
            "sampler_implementation_ready": passed,
            "generic_channel_power_confirmed": False,
            "receipt_bound_output_provenance_confirmed": False,
            "RNG2_CAL1_release_authorized": False,
            "separate_CAL1_request_permitted": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evidence-commit", required=True)
    args = parser.parse_args()
    result = audit(args.source_commit, args.evidence_commit)
    write_json(ARTIFACT / "MAIN_AUDIT.json", result)
    write_json(ARTIFACT / "TERMINAL.json", {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.terminal.v1",
        "identity": result["identity"],
        "terminal": result["main_verdict"],
        "passed": result["passed"],
        "main_audit_sha256": file_sha(ARTIFACT / "MAIN_AUDIT.json"),
        "CAL1_release_authorized": False,
        "separate_CAL1_request_permitted": result["passed"],
    })
    print(canonical(result).decode())


if __name__ == "__main__":
    main()
