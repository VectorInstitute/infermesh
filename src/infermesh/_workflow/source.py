"""Source parsing and fingerprinting helpers for the workflow engine."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from .models import CheckpointKey, _SourceRow


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
