"""Bounded one-access controller with locking, durable journal, and exact resume."""

from __future__ import annotations

from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path

from .normative import DEV_NAMESPACE, require_dev_stage
from .strict import canonical_json, object_sha256, strict_load_json, strict_load_jsonl


class AccessViolation(RuntimeError):
    pass


class ConcurrentAccess(AccessViolation):
    pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class DevOnlyAccessController:
    def __init__(self, directory: Path, stage_id: str, access_token: bytes, registry_commitment: str, expected_call_ids: list[str]):
        require_dev_stage(stage_id)
        if type(access_token) is not bytes or len(access_token) != 32:
            raise ValueError("access token must be 32 bytes")
        self.directory = directory
        self.stage_id = stage_id
        self.token_hash = sha256(access_token).hexdigest()
        self.registry_commitment = registry_commitment
        self.expected = tuple(expected_call_ids)
        self.lock_path = directory / "access.lock"
        self.state_path = directory / "state.json"
        self.journal_path = directory / "journal.jsonl"
        self.buffer_path = directory / "buffer.jsonl"
        self._lock_descriptor: int | None = None

    def acquire(self, initialize: bool = False) -> "DevOnlyAccessController":
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise ConcurrentAccess("concurrent access rejected") from error
        self._lock_descriptor = descriptor
        if initialize:
            if self.state_path.exists() or self.journal_path.exists() or self.buffer_path.exists():
                self.close()
                raise AccessViolation("exclusive access state already exists")
            state = {
                "schema": "exp33ae.dev_only_access_state.v2", "namespace": DEV_NAMESPACE,
                "stage_id": self.stage_id, "claim_bearing": False, "state": "UNOPENED",
                "token_sha256": self.token_hash, "registry_commitment": self.registry_commitment,
                "expected_calls": len(self.expected), "buffered_calls": 0,
                "journal_events": 0, "journal_tail_sha256": "0" * 64,
            }
            descriptor_state = os.open(self.state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor_state, (json.dumps(state, indent=2, sort_keys=True) + "\n").encode())
                os.fsync(descriptor_state)
            finally:
                os.close(descriptor_state)
            _fsync_directory(self.directory)
        self.validate_chain()
        return self

    def close(self) -> None:
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def __enter__(self):
        if self._lock_descriptor is None:
            raise AccessViolation("controller must be acquired explicitly")
        return self

    def __exit__(self, *_):
        self.close()

    def _state(self) -> dict:
        value = strict_load_json(self.state_path)
        base_keys = {"schema", "namespace", "stage_id", "claim_bearing", "state", "token_sha256", "registry_commitment", "expected_calls", "buffered_calls", "journal_events", "journal_tail_sha256"}
        observed_keys = frozenset(value)
        if observed_keys not in {frozenset(base_keys), frozenset(base_keys | {"invalid_reason"})}:
            raise AccessViolation("access state has missing or extra fields")
        if (
            value["schema"] != "exp33ae.dev_only_access_state.v2"
            or value["namespace"] != DEV_NAMESPACE or value["stage_id"] != self.stage_id
            or value["claim_bearing"] is not False
            or value["state"] not in {"UNOPENED", "BUFFERING", "BUFFER_COMPLETE", "RELEASED", "INVALID"}
            or type(value["expected_calls"]) is not int or value["expected_calls"] != len(self.expected)
            or type(value["buffered_calls"]) is not int or not 0 <= value["buffered_calls"] <= len(self.expected)
            or type(value["journal_events"]) is not int or value["journal_events"] < 0
        ):
            raise AccessViolation("access state schema invalid")
        if value.get("token_sha256") != self.token_hash or value.get("registry_commitment") != self.registry_commitment:
            raise AccessViolation("access token or registry commitment mismatch")
        return value

    def _write_state(self, value: dict) -> None:
        _atomic_json(self.state_path, value)

    def _journal_rows(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return strict_load_jsonl(self.journal_path)

    def validate_chain(self) -> dict:
        state = self._state()
        rows = self._journal_rows()
        previous = "0" * 64
        for ordinal, row in enumerate(rows):
            copy = dict(row)
            observed = copy.pop("event_sha256", None)
            if (
                set(copy) != {"schema", "namespace", "stage_id", "claim_bearing", "ordinal", "event", "details", "previous_event_sha256"}
                or copy["schema"] != "exp33ae.dev_only_access_event.v2"
                or copy["namespace"] != DEV_NAMESPACE or copy["stage_id"] != self.stage_id
                or copy["claim_bearing"] is not False or copy["ordinal"] != ordinal
                or copy["previous_event_sha256"] != previous or observed != object_sha256(copy)
            ):
                self._invalidate_state_only("ACCESS_CHAIN")
                raise AccessViolation("journal chain invalid")
            detail_keys = {
                "ONE_ACCESS_CONSUMED": {"registry_commitment", "expected_calls"},
                "RESUME_FROM_FIRST_MISSING": {"first_missing_order"},
                "CALLS_BUFFERED": {"start", "count", "end", "buffer_sha256"},
                "BUFFER_COMPLETE": {"calls", "buffer_sha256"},
                "EVALUATOR_DERIVED_ATOMIC_RELEASE": {"release_sha256", "status"},
                "SECOND_OR_INVALID_ACCESS_REJECTED": {"requested_resume"},
            }
            if row["event"] not in detail_keys or type(row["details"]) is not dict or set(row["details"]) != detail_keys[row["event"]]:
                self._invalidate_state_only("ACCESS_CHAIN")
                raise AccessViolation("journal event schema invalid")
            previous = observed
        if state["journal_events"] != len(rows) or state["journal_tail_sha256"] != previous:
            self._invalidate_state_only("ACCESS_CHAIN")
            raise AccessViolation("journal truncation, deletion, or tail mismatch")
        buffer_rows = strict_load_jsonl(self.buffer_path) if self.buffer_path.exists() else []
        observed_ids = [row.get("call_id") for row in buffer_rows]
        if tuple(observed_ids) != self.expected[:len(observed_ids)] or len(observed_ids) != len(set(observed_ids)):
            self._invalidate_state_only("CALL_RECONCILIATION")
            raise AccessViolation("buffer is not the exact registry prefix")
        if state["buffered_calls"] != len(buffer_rows):
            self._invalidate_state_only("ACCESS_STATE")
            raise AccessViolation("buffer/state interruption mismatch")
        return {"state": state, "events": rows, "buffer_rows": buffer_rows}

    def _invalidate_state_only(self, reason: str) -> None:
        try:
            state = strict_load_json(self.state_path)
            state["state"] = "INVALID"
            state["invalid_reason"] = reason
            _atomic_json(self.state_path, state)
        except Exception:
            pass

    def _append_event(self, event: str, details: dict) -> None:
        state = self._state()
        row = {
            "schema": "exp33ae.dev_only_access_event.v2", "namespace": DEV_NAMESPACE,
            "stage_id": self.stage_id, "claim_bearing": False,
            "ordinal": state["journal_events"], "event": event, "details": details,
            "previous_event_sha256": state["journal_tail_sha256"],
        }
        row["event_sha256"] = object_sha256(row)
        descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (canonical_json(row) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.directory)
        state["journal_events"] += 1
        state["journal_tail_sha256"] = row["event_sha256"]
        self._write_state(state)

    def begin(self, resume: bool = False) -> int:
        snapshot = self.validate_chain()
        state = snapshot["state"]
        if state["state"] == "UNOPENED":
            if resume:
                self._invalidate_state_only("ACCESS_STATE")
                raise AccessViolation("cannot resume unopened access")
            self._append_event("ONE_ACCESS_CONSUMED", {"registry_commitment": self.registry_commitment, "expected_calls": len(self.expected)})
            state = self._state(); state["state"] = "BUFFERING"; self._write_state(state)
            return 0
        if state["state"] == "BUFFERING" and resume:
            self._append_event("RESUME_FROM_FIRST_MISSING", {"first_missing_order": state["buffered_calls"]})
            return state["buffered_calls"]
        self._append_event("SECOND_OR_INVALID_ACCESS_REJECTED", {"requested_resume": resume})
        self._invalidate_state_only("ACCESS_STATE")
        raise AccessViolation("second access or invalid state rejected")

    def append_rows(self, rows: list[dict]) -> None:
        snapshot = self.validate_chain()
        state = snapshot["state"]
        if state["state"] != "BUFFERING" or not rows:
            raise AccessViolation("rows require nonempty buffering state")
        start = state["buffered_calls"]
        ids = [row.get("call_id") for row in rows]
        if tuple(ids) != self.expected[start:start+len(rows)] or len(ids) != len(set(ids)):
            self._invalidate_state_only("CALL_RECONCILIATION")
            raise AccessViolation("attempted calls are not exact next registry segment")
        descriptor = os.open(self.buffer_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            for row in rows:
                os.write(descriptor, (canonical_json(row) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.directory)
        state["buffered_calls"] += len(rows)
        self._write_state(state)
        self._append_event("CALLS_BUFFERED", {"start": start, "count": len(rows), "end": start + len(rows), "buffer_sha256": sha256(self.buffer_path.read_bytes()).hexdigest()})

    def mark_buffer_complete(self) -> None:
        snapshot = self.validate_chain()
        state = snapshot["state"]
        if state["state"] != "BUFFERING" or state["buffered_calls"] != len(self.expected):
            raise AccessViolation("cannot complete an incomplete buffer")
        self._append_event("BUFFER_COMPLETE", {"calls": len(self.expected), "buffer_sha256": sha256(self.buffer_path.read_bytes()).hexdigest()})
        state = self._state(); state["state"] = "BUFFER_COMPLETE"; self._write_state(state)

    def record_release(self, release_sha256: str, status: str) -> None:
        snapshot = self.validate_chain()
        if snapshot["state"]["state"] != "BUFFER_COMPLETE":
            raise AccessViolation("release requires complete buffer")
        self._append_event("EVALUATOR_DERIVED_ATOMIC_RELEASE", {"release_sha256": release_sha256, "status": status})
        state = self._state(); state["state"] = "RELEASED"; self._write_state(state)

    def report(self) -> dict:
        snapshot = self.validate_chain()
        return {
            "schema": "exp33ae.dev_only_access_report.v2", "namespace": DEV_NAMESPACE,
            "stage_id": self.stage_id, "claim_bearing": False,
            "state": snapshot["state"]["state"], "events": len(snapshot["events"]),
            "journal_tail_sha256": snapshot["state"]["journal_tail_sha256"],
            "buffered_calls": snapshot["state"]["buffered_calls"],
            "expected_calls": len(self.expected), "confirmation_access_record_materialized": False,
        }
