"""Source parsing and fingerprinting helpers for the workflow engine."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import IO, Any, cast

from .models import (
    _SOURCE_EXHAUSTED,
    CheckpointKey,
    PreparedItem,
    SourceItem,
    WorkStream,
    _SourceExhausted,
    _SourceRow,
)


def _compute_record_fingerprint(raw_record: dict[str, object]) -> bytes:
    """Return a stable SHA-256 digest of the canonical JSON representation."""

    canonical = json.dumps(
        raw_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _compute_parse_error_fingerprint(raw_line: str) -> bytes:
    """Return a stable fingerprint for one malformed JSONL source line."""

    return hashlib.sha256(f"__parse_error__{raw_line}".encode()).digest()


def _parse_source_line(*, source_index: int, stripped: str) -> _SourceRow:
    """Parse one non-empty source line into a workflow source row."""

    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return _SourceRow(
            source_index=source_index,
            raw_line=stripped,
            raw_record=None,
            error=exc,
        )
    if not isinstance(record, dict):
        return _SourceRow(
            source_index=source_index,
            raw_line=stripped,
            raw_record=None,
            error=ValueError("Generation rows must be JSON objects."),
        )
    return _SourceRow(
        source_index=source_index,
        raw_line=stripped,
        raw_record=record,
        error=None,
    )


def _iter_source_rows(
    *, prompt: str | None, input_jsonl: str | None
) -> Iterator[_SourceRow]:
    """Yield source rows one line at a time."""

    if prompt is not None:
        yield _SourceRow(
            source_index=0,
            raw_line=prompt,
            raw_record={"prompt": prompt},
            error=None,
        )
        return

    ctx = (
        open(input_jsonl, encoding="utf-8")  # noqa: SIM115
        if input_jsonl is not None
        else contextlib.nullcontext(sys.stdin)
    )
    with ctx as source:
        index = 0
        for raw_line in source:
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield _parse_source_line(source_index=index, stripped=stripped)
            index += 1


def _iter_binary_source_rows_with_offsets(
    input_jsonl: str,
) -> Iterator[tuple[_SourceRow, int]]:
    """Yield file-backed source rows alongside their byte offsets."""

    with open(input_jsonl, "rb") as source:
        index = 0
        while True:
            offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                return
            stripped_bytes = raw_line.strip()
            if not stripped_bytes:
                continue
            yield (
                _parse_source_line(
                    source_index=index,
                    stripped=stripped_bytes.decode("utf-8"),
                ),
                offset,
            )
            index += 1


def _load_source_row_at_offset(
    source_file: IO[bytes] | None, *, offset: int, source_index: int
) -> _SourceRow:
    """Seek to ``offset`` and parse one source row from a binary JSONL file."""

    if source_file is None:
        raise RuntimeError("Resume planner requires an open source file.")
    source_file.seek(offset)
    raw_line = source_file.readline()
    if not raw_line:
        raise RuntimeError("Resume planner source offset points past EOF.")
    stripped = raw_line.strip().decode("utf-8")
    if not stripped:
        raise RuntimeError("Resume planner source offset points to a blank line.")
    return _parse_source_line(source_index=source_index, stripped=stripped)


def _compute_source_row_fingerprint(source_row: _SourceRow) -> bytes:
    """Return the checkpoint fingerprint for one parsed source row."""

    if source_row.raw_record is not None:
        return _compute_record_fingerprint(source_row.raw_record)
    return _compute_parse_error_fingerprint(source_row.raw_line)


def _materialize_stdin_source() -> Path:
    """Copy stdin to a temporary JSONL file so file-backed runs can replay it."""

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".jsonl",
        delete=False,
    ) as file_handle:
        for raw_line in sys.stdin:
            file_handle.write(raw_line)
        return Path(file_handle.name)


def _resume_key_for_source_row(
    source_row: _SourceRow, fingerprint_counts: dict[bytes, int]
) -> CheckpointKey:
    """Return the occurrence-aware resume key for one source row."""

    fingerprint = _compute_source_row_fingerprint(source_row)
    occurrence = fingerprint_counts.get(fingerprint, 0)
    fingerprint_counts[fingerprint] = occurrence + 1
    return CheckpointKey(record_fingerprint=fingerprint, occurrence=occurrence)


def _iter_source_rows_with_keys(
    *, prompt: str | None, input_jsonl: str | None
) -> Iterator[tuple[_SourceRow, CheckpointKey]]:
    """Yield ``(source_row, checkpoint_key)`` pairs with occurrence-aware keys."""

    fingerprint_counts: dict[bytes, int] = {}
    for source_row in _iter_source_rows(prompt=prompt, input_jsonl=input_jsonl):
        yield source_row, _resume_key_for_source_row(source_row, fingerprint_counts)


def _paths_reference_same_file(input_path: Path, output_path: Path) -> bool:
    """Return whether two paths resolve to the same file target."""

    if input_path.exists() and output_path.exists():
        try:
            if input_path.samefile(output_path):
                return True
        except OSError:
            pass
    return input_path.resolve(strict=False) == output_path.resolve(strict=False)


def _validate_distinct_input_output_paths(
    *, input_jsonl: str | None, output_jsonl: str | None
) -> None:
    """Reject file-backed runs that reuse the same path for input and output."""

    if input_jsonl is None or output_jsonl is None:
        return
    if _paths_reference_same_file(Path(input_jsonl), Path(output_jsonl)):
        raise ValueError("--input-jsonl and --output-jsonl must be different files.")


class _FileSourceWorkStream:
    """Blocking stream over every file source row."""

    def __init__(
        self,
        *,
        prompt: str | None,
        input_jsonl: str | None,
        mapper: Callable[[dict[str, Any]], Any] | None,
    ) -> None:
        self._prompt = prompt
        self._input_jsonl = input_jsonl
        self._mapper = mapper
        self._source_rows: (
            Generator[tuple[_SourceRow, CheckpointKey], None, None] | None
        ) = None

    def next_prepared(self) -> PreparedItem | _SourceExhausted:
        self._ensure_open()
        assert self._source_rows is not None
        for source_row, checkpoint_key in self._source_rows:
            return _prepare_source_item(
                source_row=source_row,
                output_index=source_row.source_index,
                checkpoint_key=checkpoint_key,
                mapper=self._mapper,
            )
        return _SOURCE_EXHAUSTED

    def close(self) -> None:
        if self._source_rows is not None:
            self._source_rows.close()
            self._source_rows = None

    def _ensure_open(self) -> None:
        if self._source_rows is None:
            self._source_rows = cast(
                Generator[tuple[_SourceRow, CheckpointKey], None, None],
                _iter_source_rows_with_keys(
                    prompt=self._prompt,
                    input_jsonl=self._input_jsonl,
                ),
            )


class FileSourceStream:
    """Prompt, JSONL, or stdin source stream for workflow rows."""

    def __init__(
        self,
        *,
        prompt: str | None,
        input_jsonl: str | None,
        mapper: Callable[[dict[str, Any]], Any] | None,
        staged_stdin_path: Path | None = None,
    ) -> None:
        """Create a file-backed source stream.

        Parameters
        ----------
        prompt
            Single prompt to run, if this source is prompt-backed.
        input_jsonl
            JSONL source path, or a staged stdin path for replayable runs.
        mapper
            Optional mapper applied to raw JSONL records before scheduling.
        staged_stdin_path
            Temporary file created from stdin and owned by this source stream.
        """

        self.prompt = prompt
        self.input_jsonl = input_jsonl
        self.mapper = mapper
        self._staged_stdin_path = staged_stdin_path

    @classmethod
    def from_options(
        cls,
        *,
        prompt: str | None,
        input_jsonl: str | None,
        output_jsonl: str | None,
        mapper: Callable[[dict[str, Any]], Any] | None,
    ) -> FileSourceStream:
        """Build a source stream from CLI-style input options.

        Parameters
        ----------
        prompt
            Optional single prompt.
        input_jsonl
            Optional JSONL input path. If omitted, stdin is used.
        output_jsonl
            Optional output path. File-backed output requires stdin to be
            materialized so checkpoint bootstrap and resume can replay it.
        mapper
            Optional mapper applied to JSONL records.

        Returns
        -------
        FileSourceStream
            Source stream with any needed stdin staging owned by the instance.

        Raises
        ------
        ValueError
            If input and output paths reference the same file.
        """

        staged_stdin_path: Path | None = None
        effective_input_jsonl = input_jsonl
        if output_jsonl and prompt is None and input_jsonl is None:
            # File-backed runs scan input more than once: once for checkpoint
            # bootstrap/resume validation and again for scheduling work.
            staged_stdin_path = _materialize_stdin_source()
            effective_input_jsonl = str(staged_stdin_path)
        _validate_distinct_input_output_paths(
            input_jsonl=effective_input_jsonl,
            output_jsonl=output_jsonl,
        )
        return cls(
            prompt=prompt,
            input_jsonl=effective_input_jsonl,
            mapper=mapper,
            staged_stdin_path=staged_stdin_path,
        )

    def iter_all(self) -> WorkStream:
        """Return a blocking stream over all source rows.

        Returns
        -------
        WorkStream
            Stream consumed by the batch workflow runner.
        """

        return _FileSourceWorkStream(
            prompt=self.prompt,
            input_jsonl=self.input_jsonl,
            mapper=self.mapper,
        )

    def close(self) -> None:
        """Release source-owned temporary files."""

        if self._staged_stdin_path is not None:
            self._staged_stdin_path.unlink(missing_ok=True)
            self._staged_stdin_path = None


def _map_source_record(
    raw_record: dict[str, Any], mapper: Callable[[dict[str, Any]], Any] | None
) -> tuple[Any, dict[str, Any] | None] | Exception:
    """Apply explicit or built-in mapping to one source record."""

    if mapper is not None:
        try:
            mapping_result = mapper(raw_record)
        except Exception as exc:  # noqa: BLE001
            return exc
        if not isinstance(mapping_result, dict):
            return ValueError(
                f"Mapper must return a dict, got {type(mapping_result).__name__!r}"
            )
        if "input" not in mapping_result:
            return KeyError("Mapper return value is missing required key 'input'")
        return mapping_result["input"], mapping_result.get("metadata")

    missing = object()
    for key in ("responses_input", "messages", "prompt"):
        mapped_input = raw_record.get(key, missing)
        if mapped_input is not missing and mapped_input is not None:
            return mapped_input, None
    return ValueError(
        "Generation rows require 'prompt', 'messages', or 'responses_input'."
    )


def _validate_mapped_metadata(metadata: Any) -> dict[str, Any] | None | Exception:
    """Validate mapper metadata before output records copy it through."""

    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return TypeError("Mapper 'metadata' must be a dict when provided.")
    try:
        json.dumps(metadata)
    except TypeError as exc:
        return TypeError(f"Mapper 'metadata' must be JSON serializable: {exc}")
    return metadata


def _prepare_source_item(
    *,
    source_row: _SourceRow,
    output_index: int,
    checkpoint_key: CheckpointKey,
    mapper: Callable[[dict[str, Any]], Any] | None,
) -> PreparedItem:
    """Convert one source row into an endpoint-neutral prepared item."""

    if source_row.error is not None:
        return PreparedItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            item=None,
            immediate_error=source_row.error,
        )

    raw_record = source_row.raw_record
    if raw_record is None:
        raise RuntimeError(
            "Invariant violated: source_row.raw_record is None after error check."
        )

    mapping_result = _map_source_record(raw_record, mapper)
    if isinstance(mapping_result, Exception):
        return PreparedItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            item=None,
            immediate_error=mapping_result,
        )

    mapped_input, metadata = mapping_result
    metadata_result = _validate_mapped_metadata(metadata)
    if isinstance(metadata_result, Exception):
        return PreparedItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            item=None,
            immediate_error=metadata_result,
        )

    return PreparedItem(
        output_index=output_index,
        checkpoint_key=checkpoint_key,
        item=SourceItem(
            output_index=output_index,
            checkpoint_key=checkpoint_key,
            mapped_input=mapped_input,
            metadata=metadata_result,
        ),
    )
