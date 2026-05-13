"""Workflow-internal data models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CheckpointKey:
    """One logical workflow item in checkpoint storage."""

    record_fingerprint: bytes
    occurrence: int

    def sql_params(self) -> tuple[bytes, int]:
        """Return SQLite parameters for this key.

        Returns
        -------
        tuple[bytes, int]
            Record fingerprint and occurrence in database column order.
        """

        return (self.record_fingerprint, self.occurrence)


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One schedulable endpoint-neutral workflow item."""

    output_index: int
    checkpoint_key: CheckpointKey
    mapped_input: Any
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreparedItem:
    """One source row after mapping and resume selection."""

    output_index: int
    checkpoint_key: CheckpointKey
    item: SourceItem | None
    immediate_error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class SettledRecord:
    """One settled output row ready for persistence."""

    output_index: int
    checkpoint_key: CheckpointKey
    record: dict[str, Any]
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class BatchWorkflowSummary:
    """Observable summary for one batch workflow invocation."""

    total: int
    completed: int
    errored: int
    skipped: int
    output_path: Path | None = None


Operation = Callable[[SourceItem], Awaitable[tuple[Any, BaseException | None]]]
RecordBuilder = Callable[[SourceItem, Any, BaseException | None], dict[str, Any]]


class WorkStream(Protocol):
    """Blocking stream consumed by the async batch workflow runner."""

    def next_prepared(self) -> PreparedItem | _SourceExhausted:
        """Return the next prepared item.

        Returns
        -------
        PreparedItem | _SourceExhausted
            Prepared item to schedule or an exhaustion sentinel.
        """

    def close(self) -> None:
        """Release stream resources."""


class SourceStream(Protocol):
    """Endpoint-neutral source of workflow items."""

    def iter_all(self) -> WorkStream:
        """Return a stream over all source rows.

        Returns
        -------
        WorkStream
            Blocking stream over source rows in stable order.
        """

    def close(self) -> None:
        """Release source resources."""


class RunStore(Protocol):
    """Durable progress store for a batch workflow."""

    @property
    def skipped(self) -> int:
        """Rows already settled before this invocation started.

        Returns
        -------
        int
            Count of rows omitted from this run because resume found them
            already terminal.
        """

    @property
    def output_path(self) -> Path | None:
        """Return the user-facing output path.

        Returns
        -------
        Path | None
            Output path for file-backed stores, otherwise ``None``.
        """

    def open(
        self,
        source: SourceStream,
        *,
        resume: bool,
        mapping_fingerprint: str,
    ) -> WorkStream:
        """Prepare run state and return a blocking work stream.

        Parameters
        ----------
        source
            Source stream for the invocation.
        resume
            Whether to reuse durable state from a previous invocation.
        mapping_fingerprint
            Stable mapper identity used to reject incompatible resumes.

        Returns
        -------
        WorkStream
            Blocking stream selected by the store.
        """

    async def settle(self, settled: SettledRecord) -> None:
        """Durably settle one output row.

        Parameters
        ----------
        settled
            Record and checkpoint key to persist.
        """

    def close(self) -> None:
        """Release store resources."""


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
