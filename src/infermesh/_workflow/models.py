"""Workflow-internal data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckpointKey:
    """One logical workflow item in checkpoint storage."""

    record_fingerprint: bytes
    occurrence: int

    def sql_params(self) -> tuple[bytes, int]:
        """Return the key in the shape expected by SQLite parameter binding."""

        return (self.record_fingerprint, self.occurrence)


@dataclass(slots=True)
class _WorkItem:
    """One unit of work in the generate workflow."""

    output_index: int
    checkpoint_key: CheckpointKey
    mapped_input: Any
    metadata: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _PreparedWorkItem:
    """One source row after resume/mapping validation."""

    output_index: int
    checkpoint_key: CheckpointKey
    work_item: _WorkItem | None
    immediate_error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _SourceExhausted:
    """Sentinel returned when a blocking preparer is out of rows."""


_SOURCE_EXHAUSTED = _SourceExhausted()


@dataclass(slots=True)
class _SourceRow:
    """One source row and its parse outcome."""

    source_index: int
    raw_line: str
    raw_record: dict[str, Any] | None
    error: Exception | None


@dataclass(frozen=True, slots=True)
class _CheckpointItem:
    """One logical checkpoint item loaded from SQLite."""

    output_index: int
    status: int
    error: str | None


@dataclass(frozen=True, slots=True)
class _ResumePlan:
    """Ephemeral planner DB that drives resumed file-backed source reads."""

    planner_path: Path
