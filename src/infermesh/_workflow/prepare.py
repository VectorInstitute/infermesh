"""Work-item preparation helpers for the generate workflow."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Protocol, cast

from .checkpoint import (
    _SETTLED_STATUSES,
    _connect_checkpoint_db_read_only,
    _load_checkpoint_item,
)
from .mapping import _apply_mapper_or_builtin, _validate_metadata
from .models import (
    _SOURCE_EXHAUSTED,
    CheckpointKey,
    _PreparedWorkItem,
    _ResumePlan,
    _SourceExhausted,
    _SourceRow,
    _WorkItem,
)
from .resume import ResumePlanner
from .source import _iter_source_rows_with_keys


class Preparer(Protocol):
    """Small blocking interface used by the rolling scheduler."""

    def next_prepared(self) -> _PreparedWorkItem | _SourceExhausted:
        """Return the next schedulable item or the exhaustion sentinel."""

    def close(self) -> None:
        """Release preparer resources."""


class SequentialPreparer:
    """Prepare source rows sequentially on one blocking worker thread."""

    def __init__(
        self,
        *,
        prompt: str | None,
        input_jsonl: str | None,
        resume: bool,
        checkpoint_path: Path | None,
        mapper: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        self._prompt = prompt
        self._input_jsonl = input_jsonl
        self._resume = resume
        self._checkpoint_path = checkpoint_path
        self._mapper = mapper
        self._source_rows: (
            Generator[tuple[_SourceRow, CheckpointKey], None, None] | None
        ) = None
        self._checkpoint_connection: sqlite3.Connection | None = None

    def next_prepared(self) -> _PreparedWorkItem | _SourceExhausted:
        """Return the next schedulable work item or source exhaustion sentinel."""

        self._ensure_open()
        assert self._source_rows is not None
        for source_row, checkpoint_key in self._source_rows:
            prepared = _prepare_generate_work_item(
                source_row=source_row,
                checkpoint_key=checkpoint_key,
                resume=self._resume,
                checkpoint_connection=self._checkpoint_connection,
                mapper=self._mapper,
            )
            if prepared is not None:
                return prepared
        return _SOURCE_EXHAUSTED

    def close(self) -> None:
        """Release any blocking-thread resources held by the preparer."""

        if self._source_rows is not None:
            self._source_rows.close()
            self._source_rows = None
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def _ensure_open(self) -> None:
        if self._source_rows is None:
            self._source_rows = cast(
                Generator[tuple[_SourceRow, CheckpointKey], None, None],
                _iter_source_rows_with_keys(
                    prompt=self._prompt,
                    input_jsonl=self._input_jsonl,
                ),
            )
        if not self._resume or self._checkpoint_connection is not None:
            return
        if self._checkpoint_path is None:
            raise RuntimeError("Resume path requires a checkpoint file path.")
        self._checkpoint_connection = _connect_checkpoint_db_read_only(
            self._checkpoint_path
        )


class PlannedResumePreparer:
    """Prepare only pending rows from a precomputed resume plan."""

    def __init__(
        self,
        *,
        input_jsonl: str,
        resume_plan: _ResumePlan,
        mapper: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        self._input_jsonl = input_jsonl
        self._resume_plan = resume_plan
        self._mapper = mapper
        self._planned_rows: (
            Generator[tuple[_SourceRow, int, CheckpointKey], None, None] | None
        ) = None

    def next_prepared(self) -> _PreparedWorkItem | _SourceExhausted:
        """Return the next pending row from the precomputed resume plan."""

        self._ensure_open()
        assert self._planned_rows is not None
        for source_row, output_index, checkpoint_key in self._planned_rows:
            return _prepare_mapped_work_item(
                source_row=source_row,
                output_index=output_index,
                checkpoint_key=checkpoint_key,
                mapper=self._mapper,
            )
        return _SOURCE_EXHAUSTED

    def close(self) -> None:
        """Release any blocking-thread resources held by the preparer."""

        if self._planned_rows is not None:
            self._planned_rows.close()
            self._planned_rows = None

    def _ensure_open(self) -> None:
        if self._planned_rows is not None:
            return
        self._planned_rows = cast(
            Generator[tuple[_SourceRow, int, CheckpointKey], None, None],
            ResumePlanner.iter_rows(
                self._resume_plan,
                input_jsonl=self._input_jsonl,
            ),
        )


def _prepare_generate_work_item(
    *,
    source_row: _SourceRow,
    checkpoint_key: CheckpointKey,
    resume: bool,
    checkpoint_connection: sqlite3.Connection | None,
    mapper: Callable[[dict[str, Any]], Any] | None,
) -> _PreparedWorkItem | None:
    """Convert one source row into schedulable work or an immediate error."""

    output_index = _generate_output_index_for_resume_row(
        resume=resume,
        checkpoint_connection=checkpoint_connection,
        checkpoint_key=checkpoint_key,
        source_index=source_row.source_index,
    )
    if output_index is None:
        return None
    return _prepare_mapped_work_item(
        source_row=source_row,
        output_index=output_index,
        checkpoint_key=checkpoint_key,
        mapper=mapper,
    )


def _prepare_mapped_work_item(
    *,
    source_row: _SourceRow,
    output_index: int,
    checkpoint_key: CheckpointKey,
    mapper: Callable[[dict[str, Any]], Any] | None,
) -> _PreparedWorkItem:
    """Prepare one already-selected row for mapping/generation."""

    if source_row.error is not None:
        return _PreparedWorkItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            work_item=None,
            immediate_error=source_row.error,
        )

    raw_record = source_row.raw_record
    if raw_record is None:
        raise RuntimeError(
            "Invariant violated: source_row.raw_record is None after error check."
        )

    mapping_result = _apply_mapper_or_builtin(raw_record, mapper)
    if isinstance(mapping_result, Exception):
        return _PreparedWorkItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            work_item=None,
            immediate_error=mapping_result,
        )

    mapped_input, metadata = mapping_result
    metadata_result = _validate_metadata(metadata)
    if isinstance(metadata_result, Exception):
        return _PreparedWorkItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            work_item=None,
            immediate_error=metadata_result,
        )

    return _PreparedWorkItem(
        output_index=output_index,
        checkpoint_key=checkpoint_key,
        work_item=_WorkItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            mapped_input=mapped_input,
            metadata=metadata_result,
        ),
    )


def _generate_output_index_for_resume_row(
    *,
    resume: bool,
    checkpoint_connection: sqlite3.Connection | None,
    checkpoint_key: CheckpointKey,
    source_index: int,
) -> int | None:
    """Return the output index for this row, or ``None`` if it is settled."""

    if not resume:
        return source_index
    if checkpoint_connection is None:
        raise RuntimeError("Resume path requires an open checkpoint connection.")
    checkpoint_item = _load_checkpoint_item(checkpoint_connection, checkpoint_key)
    if checkpoint_item is None:
        raise ValueError(
            "Resume source does not match the checkpoint file. Added, removed, or "
            "modified row occurrences are not supported."
        )
    if checkpoint_item.status in _SETTLED_STATUSES:
        # Returning ``None`` lets sequential resume skip settled rows without
        # emitting duplicate output or consuming a scheduler slot for them.
        return None
    return checkpoint_item.output_index
