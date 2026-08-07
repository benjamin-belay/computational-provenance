#!/usr/bin/env python3
"""Read-only verifier for the closed RNG2 integration handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_TERMINAL = "MAIN_CSW_AM_SCB_M1_RNG2_INT_HANDOFF_VERIFIED"


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=no_duplicates)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(repository: Path) -> dict[str, Any]:
    package = repository / "docs/programme_consolidation/scb-m1-rng2-int"
    manifest = read_json(package / "PUBLIC_ARTIFACT_MANIFEST.json")
    checked = 0
    for item in manifest["artifacts"]:
        path = repository / item["path"]
        require(path.is_file(), f"missing artifact: {item['path']}")
        require(sha256(path) == item["sha256"], f"digest mismatch: {item['path']}")
        read_json(path)
        checked += 1

    final = read_json(
        repository / "artifacts/dev_only/csw-am-scb-m1-rng2-int-final/MAIN_AUDIT.json"
    )
    require(final["valid"] is True, "final audit invalid")
    require(final["lineage_closed"] is True, "lineage not closed")
    require(final["protected_confirmation_authorized"] is False, "C1 authorized")
    require(
        final["main_verdict"]
        == "MAIN_CSW_AM_SCB_M1_RNG2_INT_FINAL_PROGRAMME_CONSOLIDATED",
        "wrong final terminal",
    )

    public = read_json(
        repository / "artifacts/dev_only/csw-am-scb-m1-rng2-int-v1/MAIN_AUDIT.json"
    )
    require(public["valid"] is True, "public INT-V1 invalid")
    require(public["observed"]["complete_closure_successes"] == 128, "public numerator")
    require(public["observed"]["complete_closure_fraction"] == 1.0, "public fraction")
    require(public["observed"]["attack_failures"] == 0, "public attacks")
    require(public["observed"]["carrier_off_regression_events"] == 0, "public off")
    require(public["observed"]["wrong_confusion_events"] == 0, "public confusion")
    require(all(public["gates"].values()), "public gate failure")
    require(all(public["independent_agreement"].values()), "public disagreement")

    protected = read_json(
        repository / "artifacts/dev_only/csw-am-scb-m1-rng2-int-q1/MAIN_AUDIT.json"
    )
    require(protected["valid"] is False, "protected Q1 incorrectly valid")
    require(
        protected["observed"]["complete_closure_successes"] == 127,
        "protected numerator",
    )
    require(protected["gates"]["computation_fidelity"] is False, "Q1 computation gate")
    require(protected["observed"]["attack_failures"] == 2, "Q1 attack failures")
    require(protected["gates"]["audit_chronology"] is True, "Q1 chronology")
    require(protected["independent_agreement"]["counts"] is True, "Q1 counts")

    calibration = read_json(
        repository / "artifacts/dev_only/csw-am-scb-m1-rng2-ccal1/MAIN_AUDIT.json"
    )
    require(calibration["thresholds"]["T"] == 4.041451884327381, "wrong T")
    require(calibration["thresholds"]["M"] == 0.0, "wrong M")

    attestation = read_json(package / "SEALED_POPULATION_ATTESTATION.json")
    require(attestation["membership"]["closure_count"] == 256, "sealed count")
    require(
        attestation["membership"]["merkle_root"]
        == "bff3ab2452559b3af2c7279330143ec1a4357cd62015cf42161071fd7bb0c4e6",
        "sealed root",
    )
    require(attestation["authorization"]["int_c1_authorized"] is False, "C1 authority")
    require(all(value == 0 for value in attestation["evidence_counts"].values()), "C1 use")

    with (package / "results_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row["population"]: row for row in csv.DictReader(stream)}
    require(rows["INT-V1"]["complete_successes"] == "128", "CSV public")
    require(rows["INT-Q1"]["complete_successes"] == "127", "CSV protected")
    require(rows["INT-C1"]["formal_status"] == "NOT_TESTED_SEALED", "CSV sealed")

    required_docs = {
        "README.md",
        "HANDOFF.json",
        "PROGRAMME_RESULT.md",
        "CLAIMS_MATRIX.md",
        "EXPERIMENT_CHRONOLOGY.md",
        "REPRODUCIBILITY.md",
        "ENVIRONMENT.md",
        "LIMITATIONS_AND_NON_CLAIMS.md",
        "INDEPENDENT_REVIEW_TEMPLATE.md",
        "chronology.csv",
    }
    require(all((package / name).is_file() for name in required_docs), "missing handoff doc")
    handoff = read_json(package / "HANDOFF.json")
    require(handoff["zero_new_scientific_work"] is True, "handoff created evidence")
    require(
        handoff["independent_review"]["status"] == "PENDING_EXTERNAL_REVIEW",
        "review status overstated",
    )

    return {
        "artifacts_verified": checked,
        "final_programme_closed": True,
        "protected_confirmation_authorized": False,
        "protected_confirmation_closures_sealed": 256,
        "protected_qualification_complete_successes": 127,
        "public_validation_complete_successes": 128,
        "terminal": EXPECTED_TERMINAL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = verify(args.repository.resolve())
    print(json.dumps(result, sort_keys=True))
    print(EXPECTED_TERMINAL)


if __name__ == "__main__":
    main()
