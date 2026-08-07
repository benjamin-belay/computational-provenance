"""Append-only one-access and buffered-release mechanism for DEV_ONLY fixtures."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from ..artifacts import canonical_json
from .normative import DEV_NAMESPACE, require_dev_stage


class AccessViolation(RuntimeError):
    pass


class DevOnlyAccessSession:
    def __init__(self, directory: Path, stage_id: str, expected_call_ids: list[str]):
        require_dev_stage(stage_id)
        self.directory = directory
        self.stage_id = stage_id
        self.expected = tuple(expected_call_ids)
        self.journal = directory / "dev_only_access_journal.jsonl"
        self.buffer = directory / "dev_only_unreleased_buffer.jsonl"
        self.release = directory / "dev_only_atomic_release.json"
        directory.mkdir(parents=True, exist_ok=True)

    def _events(self) -> list[dict]:
        if not self.journal.exists():
            return []
        return [json.loads(line) for line in self.journal.read_text().splitlines() if line.strip()]

    def _append(self, event: str, details: dict | None = None) -> None:
        events = self._events()
        previous = events[-1]["event_hash"] if events else "0" * 64
        row = {
            "schema": "exp33ae.dev_only_access_event.v1", "namespace": DEV_NAMESPACE,
            "stage_id": self.stage_id, "ordinal": len(events), "event": event,
            "details": details or {}, "previous_event_hash": previous,
        }
        row["event_hash"] = sha256(canonical_json(row).encode()).hexdigest()
        with self.journal.open("a") as handle:
            handle.write(canonical_json(row) + "\n")

    def authorize_once(self) -> None:
        events = self._events()
        if events:
            self._append("SECOND_ACCESS_ATTEMPT_INVALIDATES")
            raise AccessViolation("second access fails closed and permanently invalidates the fixture lineage")
        self._append("ONE_ACCESS_CONSUMED", {"expected_calls": len(self.expected)})

    def buffer_outputs(self, rows: list[dict]) -> None:
        events = self._events()
        if not events or events[0]["event"] != "ONE_ACCESS_CONSUMED":
            raise AccessViolation("outputs require the consumed access state")
        if any(event["event"].endswith("INVALIDATES") for event in events):
            raise AccessViolation("invalid access lineage")
        with self.buffer.open("w") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        self._append("OUTPUTS_BUFFERED", {"rows": len(rows), "buffer_sha256": sha256(self.buffer.read_bytes()).hexdigest()})

    def inspect_partial(self) -> None:
        self._append("PARTIAL_OUTPUT_INSPECTION_INVALIDATES")
        raise AccessViolation("partial buffered-output inspection invalidates")

    def atomic_release(self, evaluator_result: dict) -> dict:
        events = self._events()
        if any(event["event"].endswith("INVALIDATES") for event in events):
            raise AccessViolation("invalid access lineage cannot release")
        rows = [json.loads(line) for line in self.buffer.read_text().splitlines() if line.strip()]
        observed = [row.get("call_id") for row in rows]
        if tuple(observed) != self.expected or len(set(observed)) != len(observed):
            self._append("RECONCILIATION_FAILURE_INVALIDATES", {"observed": len(observed)})
            raise AccessViolation("buffer is incomplete, reordered, or duplicated")
        envelope = {
            "schema": "exp33ae.dev_only_atomic_release.v1", "namespace": DEV_NAMESPACE,
            "stage_id": self.stage_id, "claim_bearing": False, "raw_rows": len(rows),
            "raw_sha256": sha256(self.buffer.read_bytes()).hexdigest(), "simulated_evaluator_result": evaluator_result,
            "confirmation_terminal_status_materialized": False,
        }
        temporary = self.directory / "dev_only_atomic_release.pending"
        temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.release)
        self._append("ATOMIC_DEV_ONLY_RELEASED", {"release_sha256": sha256(self.release.read_bytes()).hexdigest()})
        return envelope

    def report(self) -> dict:
        events = self._events()
        invalid = any(event["event"].endswith("INVALIDATES") for event in events)
        return {
            "schema": "exp33ae.dev_only_access_report.v1", "namespace": DEV_NAMESPACE,
            "stage_id": self.stage_id, "events": events, "invalid": invalid,
            "access_consumed": bool(events), "released": self.release.exists(),
            "confirmation_access_record_materialized": False,
        }
