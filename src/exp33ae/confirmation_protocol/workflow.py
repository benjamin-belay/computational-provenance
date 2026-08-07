"""Order-enforcing lifecycle used by both future execution and DEV_ONLY tests."""

from __future__ import annotations

from dataclasses import dataclass

from ..remediation.strict import StrictValidationError


PHASES = (
    "AUTHORIZED", "ENTROPY_COMMITTED", "ALLOCATION_COMMITTED",
    "TRAINING_COMPLETE", "CHECKPOINTS_FROZEN", "REGISTRY_COMMITTED",
    "ACCESS_CONSUMED", "BUFFER_COMPLETE", "RELEASED",
)


@dataclass(frozen=True)
class WorkflowState:
    phase: str
    access_consumed: bool = False
    released: bool = False

    def __post_init__(self):
        if self.phase not in PHASES:
            raise StrictValidationError("ACCESS_STATE", "unknown phase")

    def advance(self, target: str) -> "WorkflowState":
        if target not in PHASES:
            raise StrictValidationError("ACCESS_STATE", "unknown target")
        current = PHASES.index(self.phase)
        if PHASES.index(target) != current + 1:
            raise StrictValidationError("ACCESS_STATE", f"out-of-order {self.phase}->{target}")
        if self.released:
            raise StrictValidationError("ACCESS_STATE", "already released")
        return WorkflowState(target, self.access_consumed or target == "ACCESS_CONSUMED", target == "RELEASED")

    def runtime_stop_status(self) -> str:
        return "EXPERIMENT_RUNTIME_STOPPED" if not self.access_consumed else "EXPERIMENT_INVALID"
