"""Run the actual test suite and record every pytest outcome category."""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from .artifacts import write_json
from .spec import NAMESPACE


class _OutcomeCounter:
    def __init__(self):
        self.counts = {
            "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
            "deselected": 0, "xfailed": 0, "xpassed": 0,
        }
        self.collected = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_deselected(self, items):
        self.counts["deselected"] += len(items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.counts["errors"] += 1

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            if report.failed:
                self.counts["errors"] += 1
            return
        was_xfail = hasattr(report, "wasxfail")
        if report.passed and was_xfail:
            self.counts["xpassed"] += 1
        elif report.skipped and was_xfail:
            self.counts["xfailed"] += 1
        elif report.passed:
            self.counts["passed"] += 1
        elif report.failed:
            self.counts["failed"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1


def run_test_report(root: Path, output: Path) -> dict:
    counter = _OutcomeCounter()
    started = time.perf_counter()
    exit_code = int(pytest.main([
        str(root / "tests"), "-q", "-ra", "--strict-markers", "-p", "no:cacheprovider",
    ], plugins=[counter]))
    report = {
        "schema": "exp33ae.test_report.v1",
        "namespace": NAMESPACE,
        "claim_bearing": False,
        "collected": counter.collected,
        **counter.counts,
        "exit_code": exit_code,
        "runtime_seconds": time.perf_counter() - started,
        "passed_suite": (
            exit_code == 0
            and counter.counts["passed"] == counter.collected
            and all(counter.counts[name] == 0 for name in (
                "failed", "errors", "skipped", "deselected", "xfailed", "xpassed"
            ))
        ),
        "confirmation_artifacts_created": 0,
        "confirmation_artifacts_accessed": 0,
    }
    write_json(output, report)
    return report
