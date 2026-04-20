"""Internal workflow engine for the CLI."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import inspect
import json
import queue
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from infermesh._batch_utils import cancel_tasks
from infermesh._cli_support import _build_generation_record, _write_jsonl

if TYPE_CHECKING:
    from infermesh.client import LMClient
    from infermesh.types import EndpointType


@dataclass
class _WorkItem:
    """One unit of work in the generate workflow."""

    output_index: int
    checkpoint_key: tuple[bytes, int]
    mapped_input: Any
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class _PreparedWorkItem:
    """One source row after resume/mapping validation."""

    output_index: int
    checkpoint_key: tuple[bytes, int]
    work_item: _WorkItem | None
    immediate_error: BaseException | None = None


@dataclass(frozen=True)
class _SourceExhausted:
    """Sentinel returned when the blocking source preparer is out of rows."""


_SOURCE_EXHAUSTED = _SourceExhausted()


@dataclass
class _SourceRow:
    """One source row and its parse outcome."""

    source_index: int
    raw_line: str
    raw_record: dict[str, Any] | None
    error: Exception | None


@dataclass(frozen=True)
class _CheckpointItem:
    """One logical checkpoint item loaded from SQLite."""

    output_index: int
    status: int
    error: str | None


@dataclass
class _PersistenceRequest:
    """One settled row that must be durably written by the sink thread."""

    record: dict[str, Any]
    checkpoint_key: tuple[bytes, int]
    status: int
    error: str | None
    done: threading.Event
    failure: BaseException | None = None


@dataclass
class _PersistenceShutdown:
    """Signal the sink thread to flush and stop."""

    done: threading.Event


class _BlockingWorkItemPreparer:
    """Prepare source rows on one blocking worker thread."""

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
            Generator[tuple[_SourceRow, tuple[bytes, int]], None, None] | None
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
        """Lazily initialize iterator and resume connection on the worker thread."""

        if self._source_rows is None:
            self._source_rows = cast(
                Generator[tuple[_SourceRow, tuple[bytes, int]], None, None],
                _iter_source_rows_with_keys(
                    prompt=self._prompt,
                    input_jsonl=self._input_jsonl,
                ),
            )
        if not self._resume or self._checkpoint_connection is not None:
            return
        if self._checkpoint_path is None:
            raise RuntimeError("Resume path requires a checkpoint file path.")
        self._checkpoint_connection = _connect_checkpoint_db(self._checkpoint_path)


_SCHEMA_VERSION = 1
_RUN_METADATA_SINGLETON = 1
_ITEM_INSERT_BATCH_SIZE = 1000
_STATUS_LOG_INTERVAL = 100_000
_CHECKPOINT_PATH_HASH_LENGTH = 8

_PENDING_STATUS = 0
_SUCCESS_STATUS = 1
_ERROR_STATUS = 2
_SETTLED_STATUSES = frozenset({_SUCCESS_STATUS, _ERROR_STATUS})
_STATUS_NAMES = {
    _PENDING_STATUS: "pending",
    _SUCCESS_STATUS: "success",
    _ERROR_STATUS: "error",
}
_STATUS_VALUES = {name: value for value, name in _STATUS_NAMES.items()}
_RESUME_SOURCE_MISMATCH_ERROR = (
    "Resume source does not match the checkpoint file. Added, removed, or "
    "modified row occurrences are not supported."
)

# Encodes v1 built-in field-extraction semantics. Bump the literal to invalidate
# existing checkpoints whenever the built-in mapping logic changes.
_BUILTIN_MAPPING_FINGERPRINT = hashlib.sha256(
    b"infermesh.generate.builtin_mapping.v1"
).hexdigest()


class _FileBackedPersistenceSink:
    """Serialize output/checkpoint writes onto one dedicated thread.

    ``generate_batch`` callbacks may fire from a non-caller thread. Routing all
    I/O through a single dedicated thread is simpler than holding a mutex across
    two interdependent file handles (the JSONL output and the SQLite checkpoint).
    """

    def __init__(self, *, output_path: Path, checkpoint_path: Path) -> None:
        self._output_path = output_path
        self._checkpoint_path = checkpoint_path
        self._queue: queue.Queue[_PersistenceRequest | _PersistenceShutdown] = (
            queue.Queue()
        )
        self._started = threading.Event()
        self._failure: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="infermesh-generate-persistence",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()  # block until the thread has opened both file handles
        self._raise_if_failed()

    def write_record(
        self,
        record: dict[str, Any],
        checkpoint_key: tuple[bytes, int],
        *,
        status: int,
        error: str | None,
    ) -> None:
        """Persist one settled record and checkpoint update."""

        if self._closed:
            raise RuntimeError("Cannot write to a closed persistence sink.")
        self._raise_if_failed()
        request = _PersistenceRequest(
            record=record,
            checkpoint_key=checkpoint_key,
            status=status,
            error=error,
            done=threading.Event(),
        )
        self._queue.put(request)
        self._wait_for_event(request.done)
        if request.failure is not None:
            raise request.failure
        self._raise_if_failed()

    def close(self) -> None:
        """Stop the sink thread and re-raise any background failure."""

        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        if self._thread.is_alive():
            shutdown = _PersistenceShutdown(done=threading.Event())
            self._queue.put(shutdown)
            self._wait_for_event(shutdown.done)
            self._thread.join()
        self._raise_if_failed()

    def _run(self) -> None:
        """Own the output file and checkpoint connection on one thread."""

        out_file: IO[str] | None = None
        connection: sqlite3.Connection | None = None
        try:
            out_file = open(self._output_path, "a", encoding="utf-8")  # noqa: SIM115
            connection = _connect_checkpoint_db(self._checkpoint_path)
            self._started.set()
            while True:
                item = self._queue.get()
                if isinstance(item, _PersistenceShutdown):
                    item.done.set()
                    return
                try:
                    out_file.write(json.dumps(item.record) + "\n")
                    out_file.flush()
                    _mark_checkpoint_item_settled(
                        connection,
                        item.checkpoint_key,
                        status=item.status,
                        error=item.error,
                    )
                except BaseException as exc:  # noqa: BLE001
                    self._set_failure(exc)
                    item.failure = exc
                    item.done.set()
                    self._fail_pending_items(exc)
                    return
                item.done.set()
        except BaseException as exc:  # noqa: BLE001
            self._set_failure(exc)
            self._fail_pending_items(exc)
        finally:
            self._started.set()  # unblock __init__ even if file-open failed
            if connection is not None:
                connection.close()
            if out_file is not None:
                out_file.close()

    def _wait_for_event(self, event: threading.Event) -> None:
        """Wait for a request acknowledgement or fail if the thread dies."""

        # Poll with a short timeout so we detect a dead thread that never set
        # the event (e.g. if it was killed by an unhandled signal).
        while not event.wait(timeout=0.1):
            if not self._thread.is_alive():
                break
        if event.is_set():
            return
        self._raise_if_failed()
        raise RuntimeError("Persistence sink stopped before acknowledging a write.")

    def _set_failure(self, exc: BaseException) -> None:
        """Remember the first background failure."""

        if self._failure is None:
            self._failure = exc

    def _raise_if_failed(self) -> None:
        """Raise the first background failure, if any."""

        if self._failure is not None:
            raise self._failure

    def _fail_pending_items(self, exc: BaseException) -> None:
        """Unblock any queued callers when the sink cannot continue."""

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _PersistenceShutdown):
                item.done.set()
                continue
            item.failure = exc
            item.done.set()


def _compute_record_fingerprint(raw_record: dict[str, Any]) -> bytes:
    """Return a stable SHA-256 digest of the canonical JSON representation."""

    canonical = json.dumps(
        raw_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _compute_parse_error_fingerprint(raw_line: str) -> bytes:
    """Return a stable fingerprint for one malformed JSONL source line."""

    return hashlib.sha256(f"__parse_error__{raw_line}".encode()).digest()


def _checkpoint_path_for(
    output_jsonl: str, *, checkpoint_dir: str | None = None
) -> Path:
    """Derive the checkpoint file path from the output JSONL path.

    ``results.jsonl``  ->  ``results.checkpoint.sqlite``
    ``results``        ->  ``results.checkpoint.sqlite``
    """
    path = Path(output_jsonl)
    checkpoint_stem = path.stem if path.suffix == ".jsonl" else path.name
    if checkpoint_dir is None:
        return path.with_name(checkpoint_stem + ".checkpoint.sqlite")

    override_dir = Path(checkpoint_dir).expanduser()
    override_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_path = path.expanduser().resolve(strict=False)
    path_hash = hashlib.sha256(str(resolved_output_path).encode("utf-8")).hexdigest()[
        :_CHECKPOINT_PATH_HASH_LENGTH
    ]
    return override_dir / f"{checkpoint_stem}.{path_hash}.checkpoint.sqlite"


def _configure_checkpoint_journal_mode(connection: sqlite3.Connection) -> str:
    """Configure the checkpoint DB for portable rollback journaling."""

    persist_mode = connection.execute("PRAGMA journal_mode=PERSIST").fetchone()
    journal_mode = str(persist_mode[0]).lower() if persist_mode is not None else ""
    if journal_mode == "persist":
        return journal_mode

    delete_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    journal_mode = str(delete_mode[0]).lower() if delete_mode is not None else ""
    if journal_mode == "delete":
        return journal_mode

    raise RuntimeError(
        "Checkpoint DB could not be configured for rollback journaling. "
        f"SQLite reported journal_mode={journal_mode!r}."
    )


def _connect_checkpoint_db(checkpoint_path: Path) -> sqlite3.Connection:
    """Open a checkpoint database connection."""

    connection = sqlite3.connect(checkpoint_path)
    _configure_checkpoint_journal_mode(connection)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "PRAGMA busy_timeout=5000"
    )  # retry briefly on write-lock contention
    return connection


def _initialize_checkpoint_db(
    connection: sqlite3.Connection, mapping_fingerprint: str
) -> None:
    """Create the checkpoint schema and write the run metadata row."""

    connection.executescript(
        """
        CREATE TABLE run_metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            schema_version INTEGER NOT NULL,
            mapping_fingerprint TEXT NOT NULL
        );

        CREATE TABLE items (
            record_fingerprint BLOB NOT NULL,
            occurrence INTEGER NOT NULL,
            output_index INTEGER NOT NULL,
            status INTEGER NOT NULL,
            error TEXT,
            PRIMARY KEY (record_fingerprint, occurrence)
        );

        CREATE INDEX idx_items_status_output_index
        ON items(status, output_index);
        """
    )
    connection.execute(
        """
        INSERT INTO run_metadata (singleton, schema_version, mapping_fingerprint)
        VALUES (?, ?, ?)
        """,
        (_RUN_METADATA_SINGLETON, _SCHEMA_VERSION, mapping_fingerprint),
    )


def _load_run_metadata(connection: sqlite3.Connection) -> tuple[int, str]:
    """Load the singleton run metadata row."""

    row = connection.execute(
        """
        SELECT schema_version, mapping_fingerprint
        FROM run_metadata
        WHERE singleton = ?
        """,
        (_RUN_METADATA_SINGLETON,),
    ).fetchone()
    if row is None:
        raise ValueError("Checkpoint file is invalid: missing run metadata.")
    schema_version = int(row[0])
    mapping_fingerprint = str(row[1])
    return schema_version, mapping_fingerprint


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
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield _SourceRow(
                    source_index=index,
                    raw_line=stripped,
                    raw_record=None,
                    error=exc,
                )
            else:
                yield _SourceRow(
                    source_index=index,
                    raw_line=stripped,
                    raw_record=record,
                    error=None,
                )
            index += 1


def _iter_source_rows_with_keys(
    *, prompt: str | None, input_jsonl: str | None
) -> Iterator[tuple[_SourceRow, tuple[bytes, int]]]:
    """Yield ``(source_row, checkpoint_key)`` pairs with occurrence-aware keys."""

    fingerprint_counts: dict[bytes, int] = {}
    for source_row in _iter_source_rows(prompt=prompt, input_jsonl=input_jsonl):
        yield source_row, _resume_key_for_source_row(source_row, fingerprint_counts)


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
) -> tuple[bytes, int]:
    """Return the occurrence-aware resume key for one source row."""

    if source_row.raw_record is not None:
        fingerprint = _compute_record_fingerprint(source_row.raw_record)
    else:
        fingerprint = _compute_parse_error_fingerprint(source_row.raw_line)
    occurrence = fingerprint_counts.get(fingerprint, 0)
    fingerprint_counts[fingerprint] = occurrence + 1
    return fingerprint, occurrence


def _compute_mapper_implementation_fingerprint(
    mapper: Callable[[dict[str, Any]], Any],
) -> str:
    """Return a stable fingerprint for the mapper implementation."""

    # Prefer the full module source over the function alone: this captures
    # changes to helper functions that the mapper calls but did not change itself.
    module = inspect.getmodule(mapper)
    if module is not None:
        with contextlib.suppress(OSError, TypeError):
            return hashlib.sha256(inspect.getsource(module).encode("utf-8")).hexdigest()

    with contextlib.suppress(OSError, TypeError):
        return hashlib.sha256(inspect.getsource(mapper).encode("utf-8")).hexdigest()

    # Last resort for callables whose source cannot be inspected (built-ins,
    # extension modules, dynamically created functions).
    code = getattr(mapper, "__code__", None)
    fallback_payload = repr(
        {
            "co_code": getattr(code, "co_code", None),
            "co_consts": getattr(code, "co_consts", None),
            "co_names": getattr(code, "co_names", None),
            "defaults": getattr(mapper, "__defaults__", None),
            "kwdefaults": getattr(mapper, "__kwdefaults__", None),
        }
    )
    return hashlib.sha256(fallback_payload.encode("utf-8")).hexdigest()


def _compute_mapping_fingerprint(
    *, mapper_spec: str | None, mapper: Callable[[dict[str, Any]], Any] | None
) -> str:
    """Return the fingerprint that ties a run to its mapping strategy."""

    if mapper is None:
        return _BUILTIN_MAPPING_FINGERPRINT

    module_name = getattr(mapper, "__module__", type(mapper).__module__)
    qualname = getattr(mapper, "__qualname__", type(mapper).__qualname__)
    payload = json.dumps(
        {
            "mapper_spec": mapper_spec,
            "module_name": module_name,
            "qualname": qualname,
            "implementation_fingerprint": _compute_mapper_implementation_fingerprint(
                mapper
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _insert_pending_checkpoint_items(
    connection: sqlite3.Connection,
    *,
    prompt: str | None,
    input_jsonl: str | None,
) -> None:
    """Insert one pending checkpoint item per source row."""

    batch: list[tuple[bytes, int, int, int, None]] = []
    insert_sql = """
        INSERT INTO items (
            record_fingerprint,
            occurrence,
            output_index,
            status,
            error
        )
        VALUES (?, ?, ?, ?, ?)
    """

    for source_row, checkpoint_key in _iter_source_rows_with_keys(
        prompt=prompt, input_jsonl=input_jsonl
    ):
        batch.append(
            (
                checkpoint_key[0],
                checkpoint_key[1],
                source_row.source_index,
                _PENDING_STATUS,
                None,
            )
        )
        if len(batch) >= _ITEM_INSERT_BATCH_SIZE:
            connection.executemany(insert_sql, batch)
            batch.clear()

    if batch:
        connection.executemany(insert_sql, batch)


def _bootstrap_checkpoint(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    checkpoint_path: Path,
    mapping_fingerprint: str,
) -> None:
    """Create the checkpoint DB and bootstrap one pending row per source item."""

    connection = _connect_checkpoint_db(checkpoint_path)
    try:
        _initialize_checkpoint_db(connection, mapping_fingerprint)
        _insert_pending_checkpoint_items(
            connection,
            prompt=prompt,
            input_jsonl=input_jsonl,
        )
        connection.commit()
    finally:
        connection.close()


def _stage_fresh_workflow_files(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_path: Path,
    checkpoint_path: Path,
    mapping_fingerprint: str,
) -> None:
    """Stage fresh workflow artifacts and replace existing ones after bootstrap."""

    # Write to temp files first so that a failure during bootstrap never
    # overwrites valid existing artifacts. The finally block removes any
    # temps that were not successfully promoted via replace().
    staged_output_path: Path | None = None
    staged_checkpoint_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            staged_checkpoint_path = Path(file_handle.name)
        _bootstrap_checkpoint(
            prompt=prompt,
            input_jsonl=input_jsonl,
            checkpoint_path=staged_checkpoint_path,
            mapping_fingerprint=mapping_fingerprint,
        )

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            staged_output_path = Path(file_handle.name)

        # Checkpoint is promoted before output so that the only ambiguous
        # crash state (checkpoint present, output absent) is detectable by
        # --resume validation rather than silently unrecoverable.
        staged_checkpoint_path.replace(checkpoint_path)
        staged_output_path.replace(output_path)
    finally:
        if staged_checkpoint_path is not None and staged_checkpoint_path.exists():
            staged_checkpoint_path.unlink()
        if staged_output_path is not None and staged_output_path.exists():
            staged_output_path.unlink()


def _load_checkpoint_item(
    connection: sqlite3.Connection, checkpoint_key: tuple[bytes, int]
) -> _CheckpointItem | None:
    """Load one checkpoint item by its occurrence-aware key."""

    row = connection.execute(
        """
        SELECT output_index, status, error
        FROM items
        WHERE record_fingerprint = ? AND occurrence = ?
        """,
        checkpoint_key,
    ).fetchone()
    if row is None:
        return None
    error_value = row[2]
    if error_value is not None and not isinstance(error_value, str):
        raise ValueError("Checkpoint file is invalid: item error column must be text.")
    return _CheckpointItem(
        output_index=int(row[0]),
        status=int(row[1]),
        error=error_value,
    )


def _validate_resume_mapping_fingerprint(
    connection: sqlite3.Connection, *, mapping_fingerprint: str
) -> None:
    """Validate mapping consistency between the checkpoint and current run."""

    schema_version, checkpoint_mapping_fingerprint = _load_run_metadata(connection)
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            "Checkpoint file uses an unsupported schema version. Restart the run "
            "without --resume."
        )
    if checkpoint_mapping_fingerprint != mapping_fingerprint:
        raise ValueError(
            "Resume mapping does not match the checkpoint file. Use the original "
            "mapper implementation or restart without --resume."
        )


def _load_output_indices_with_status(
    output_path: Path, on_status: Callable[[str], Any] | None
) -> set[int]:
    """Return output indices while emitting coarse resume-scan status lines."""

    observed_indices: set[int] = set()
    with output_path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            output_index = row.get("_index")
            if isinstance(output_index, int):
                observed_indices.add(output_index)
            if on_status is not None and line_number % _STATUS_LOG_INTERVAL == 0:
                on_status(f"Resume: scanned {line_number:,} output rows...")
    return observed_indices


def _load_checkpoint_fingerprint_counts(
    connection: sqlite3.Connection,
) -> dict[bytes, int]:
    """Return remaining per-fingerprint source occurrences from the checkpoint."""

    return {
        cast(bytes, row[0]): int(row[1])
        for row in connection.execute(
            """
            SELECT record_fingerprint, COUNT(*)
            FROM items
            GROUP BY record_fingerprint
            """
        )
    }


def _validate_resume_output_rows(
    connection: sqlite3.Connection,
    output_path: Path,
    on_status: Callable[[str], Any] | None = None,
) -> None:
    """Ensure every settled checkpoint item already exists in the output file."""

    if on_status is not None:
        on_status("Resume: validating output artifact...")
    output_indices = _load_output_indices_with_status(output_path, on_status)
    missing_indices: list[int] = []

    for row in connection.execute(
        """
        SELECT output_index
        FROM items
        WHERE status IN (?, ?)
        ORDER BY output_index
        """,
        (_SUCCESS_STATUS, _ERROR_STATUS),
    ):
        output_index = int(row[0])
        if output_index not in output_indices:
            missing_indices.append(output_index)
            if len(missing_indices) >= 11:
                break

    if missing_indices:
        missing_text = ", ".join(str(index) for index in missing_indices[:10])
        if len(missing_indices) > 10:
            missing_text += ", ..."
        raise ValueError(
            "Output file is missing settled checkpoint rows for _index values "
            f"{missing_text}. Restore the output artifact or restart the run "
            "without --resume."
        )


def _validate_resume_source(
    connection: sqlite3.Connection,
    *,
    prompt: str | None,
    input_jsonl: str | None,
    on_status: Callable[[str], Any] | None = None,
) -> None:
    """Validate that the current source matches the checkpoint occurrence set."""

    if on_status is not None:
        on_status("Resume: validating input source...")

    remaining_counts = _load_checkpoint_fingerprint_counts(connection)
    for seen_count, source_row in enumerate(
        _iter_source_rows(prompt=prompt, input_jsonl=input_jsonl), start=1
    ):
        if source_row.raw_record is not None:
            fingerprint = _compute_record_fingerprint(source_row.raw_record)
        else:
            fingerprint = _compute_parse_error_fingerprint(source_row.raw_line)
        remaining = remaining_counts.get(fingerprint)
        if remaining is None:
            raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)
        if remaining == 1:
            del remaining_counts[fingerprint]
        else:
            remaining_counts[fingerprint] = remaining - 1
        if on_status is not None and seen_count % _STATUS_LOG_INTERVAL == 0:
            on_status(f"Resume: scanned {seen_count:,} source rows...")

    if remaining_counts:
        raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)


def _validate_resume_checkpoint(
    output_path: Path,
    checkpoint_path: Path,
    *,
    mapping_fingerprint: str,
    prompt: str | None,
    input_jsonl: str | None,
    on_status: Callable[[str], Any] | None = None,
) -> None:
    """Validate the resume checkpoint for a file-backed workflow."""

    if not checkpoint_path.exists():
        raise ValueError(
            f"--resume requires checkpoint file {checkpoint_path}. "
            "Start a fresh file-backed run first."
        )
    if not output_path.exists():
        raise ValueError(
            f"--resume requires output file {output_path} because checkpoint file "
            f"{checkpoint_path} already exists."
        )

    connection = _connect_checkpoint_db(checkpoint_path)
    try:
        if on_status is not None:
            on_status("Resume: validating checkpoint file...")
        _validate_resume_mapping_fingerprint(
            connection,
            mapping_fingerprint=mapping_fingerprint,
        )
        _validate_resume_output_rows(connection, output_path, on_status=on_status)
        _validate_resume_source(
            connection,
            prompt=prompt,
            input_jsonl=input_jsonl,
            on_status=on_status,
        )
    finally:
        connection.close()


def _load_mapper(mapper_spec: str) -> Callable[[dict[str, Any]], Any]:
    """Load a mapper function from a ``"package.module:function"`` spec."""

    module_path, sep, func_name = mapper_spec.rpartition(":")
    if not sep or not module_path or not func_name:
        raise ValueError(
            f"--mapper must be 'package.module:function', got {mapper_spec!r}"
        )
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None:
        raise ValueError(f"--mapper: {module_path!r} has no attribute {func_name!r}")
    if not callable(func):
        raise ValueError(
            f"--mapper: {mapper_spec!r} resolved to a non-callable {type(func).__name__!r}"
        )
    return cast(Callable[[dict[str, Any]], Any], func)


def _apply_mapper_or_builtin(
    raw_record: dict[str, Any], mapper: Callable[[dict[str, Any]], Any] | None
) -> tuple[Any, dict[str, Any] | None] | Exception:
    """Apply the mapper (or built-in field extraction) to ``raw_record``."""

    if mapper is not None:
        try:
            result = mapper(raw_record)
        except Exception as exc:  # noqa: BLE001
            return exc
        if not isinstance(result, dict):
            return ValueError(
                f"Mapper must return a dict, got {type(result).__name__!r}"
            )
        if "input" not in result:
            return KeyError("Mapper return value is missing required key 'input'")
        return result["input"], result.get("metadata")

    input_data = (
        raw_record.get("responses_input")
        or raw_record.get("messages")
        or raw_record.get("prompt")
    )
    if input_data is None:
        return ValueError(
            "Generation rows require 'prompt', 'messages', or 'responses_input'."
        )
    return input_data, None


def _validate_metadata(metadata: Any) -> dict[str, Any] | None | Exception:
    """Validate mapper metadata before it reaches the sink."""

    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return TypeError("Mapper 'metadata' must be a dict when provided.")
    try:
        json.dumps(metadata)
    except TypeError as exc:
        return TypeError(f"Mapper 'metadata' must be JSON serializable: {exc}")
    return metadata


def _mark_checkpoint_item_settled(
    connection: sqlite3.Connection,
    checkpoint_key: tuple[bytes, int],
    *,
    status: int,
    error: str | None,
) -> None:
    """Update one checkpoint item from pending to a terminal state."""

    cursor = connection.execute(
        """
        UPDATE items
        SET status = ?, error = ?
        WHERE record_fingerprint = ? AND occurrence = ?
        """,
        (status, error, checkpoint_key[0], checkpoint_key[1]),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Checkpoint item update failed for settled workflow row.")
    connection.commit()


def _write_item_result(
    persistence_sink: _FileBackedPersistenceSink | None,
    output_index: int,
    checkpoint_key: tuple[bytes, int],
    result: Any,
    error: BaseException | None,
    *,
    metadata: dict[str, Any] | None,
    parse_json: bool,
) -> dict[str, Any]:
    """Build one settled item and optionally persist it through the sink."""

    record = _build_generation_record(
        output_index, result, error, parse_json=parse_json
    )
    if metadata is not None:
        record["metadata"] = metadata
    if persistence_sink is not None:
        persistence_sink.write_record(
            record,
            checkpoint_key,
            status=_ERROR_STATUS if error else _SUCCESS_STATUS,
            error=str(error) if error else None,
        )
    return record


async def _agenerate_work_item(
    client: LMClient,
    item: _WorkItem,
    *,
    endpoint: EndpointType,
) -> tuple[Any, Exception | None]:
    """Run one workflow item and return its settled outcome."""

    try:
        result = await client.agenerate(
            item.mapped_input,
            endpoint=endpoint,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return None, exc
    return result, None


def _emit_settled_work_item(
    persistence_sink: _FileBackedPersistenceSink | None,
    item: _WorkItem,
    result: Any,
    error: BaseException | None,
    *,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Persist or print one settled workflow item, then tick progress."""

    record = _write_item_result(
        persistence_sink,
        item.output_index,
        item.checkpoint_key,
        result,
        error,
        metadata=item.metadata,
        parse_json=parse_json,
    )
    if persistence_sink is None:
        _write_jsonl([record], None)
    if on_progress is not None:
        on_progress()


def _emit_immediate_error(
    persistence_sink: _FileBackedPersistenceSink | None,
    output_index: int,
    checkpoint_key: tuple[bytes, int],
    error: BaseException,
    *,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Write one per-item error row without aborting the run, then invoke progress."""

    record = _write_item_result(
        persistence_sink,
        output_index,
        checkpoint_key,
        None,
        error,
        metadata=None,
        parse_json=parse_json,
    )
    if persistence_sink is None:
        _write_jsonl([record], None)
    if on_progress is not None:
        on_progress()


def _prepare_generate_work_item(
    *,
    source_row: _SourceRow,
    checkpoint_key: tuple[bytes, int],
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
    checkpoint_key: tuple[bytes, int],
    source_index: int,
) -> int | None:
    """Return the output index for this row, or ``None`` if it is settled."""

    if not resume:
        return source_index
    if checkpoint_connection is None:
        raise RuntimeError("Resume path requires an open checkpoint connection.")
    checkpoint_item = _load_checkpoint_item(checkpoint_connection, checkpoint_key)
    if checkpoint_item is None:
        raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)
    if checkpoint_item.status in _SETTLED_STATUSES:
        return None
    return checkpoint_item.output_index


async def _arun_generate_source_rows(
    client: LMClient,
    *,
    preparer: _BlockingWorkItemPreparer,
    preparer_executor: ThreadPoolExecutor,
    resume: bool,
    persistence_sink: _FileBackedPersistenceSink | None,
    window_size: int,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Stream mapped rows through a rolling in-flight generation window."""

    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    loop = asyncio.get_running_loop()
    active_tasks: dict[asyncio.Task[tuple[Any, Exception | None]], _WorkItem] = {}
    source_exhausted = False
    any_work = False

    async def fill_window() -> None:
        nonlocal source_exhausted
        nonlocal any_work

        while len(active_tasks) < window_size and not source_exhausted:
            prepared = await loop.run_in_executor(
                preparer_executor,
                preparer.next_prepared,
            )
            if isinstance(prepared, _SourceExhausted):
                source_exhausted = True
                return
            any_work = True
            if prepared.immediate_error is not None:
                _emit_immediate_error(
                    persistence_sink,
                    prepared.output_index,
                    prepared.checkpoint_key,
                    prepared.immediate_error,
                    parse_json=parse_json,
                    on_progress=on_progress,
                )
                continue

            work_item = prepared.work_item
            assert work_item is not None
            # This helper runs inside `_arun_generate_source_rows`, so the
            # current event loop is already active when we admit a new task.
            task = asyncio.create_task(
                _agenerate_work_item(
                    client,
                    work_item,
                    endpoint=endpoint,
                )
            )
            active_tasks[task] = work_item

    try:
        await fill_window()
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                item = active_tasks.pop(task)
                result, error = task.result()
                _emit_settled_work_item(
                    persistence_sink,
                    item,
                    result,
                    error,
                    parse_json=parse_json,
                    on_progress=on_progress,
                )
            await fill_window()
    except BaseException:
        await cancel_tasks(list(active_tasks))
        raise
    finally:
        await loop.run_in_executor(preparer_executor, preparer.close)

    if resume and not any_work:
        sys.stderr.write("Nothing to do — all rows already completed.\n")


def _run_generate_source_rows(
    client: LMClient,
    *,
    prompt: str | None,
    effective_input_jsonl: str | None,
    resume: bool,
    checkpoint_path: Path | None,
    persistence_sink: _FileBackedPersistenceSink | None,
    mapper: Callable[[dict[str, Any]], Any] | None,
    window_size: int,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Run the rolling generate scheduler on the client's background loop."""

    preparer = _BlockingWorkItemPreparer(
        prompt=prompt,
        input_jsonl=effective_input_jsonl,
        resume=resume,
        checkpoint_path=checkpoint_path,
        mapper=mapper,
    )
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="infermesh-generate-prep",
    ) as preparer_executor:
        client._run_sync(
            _arun_generate_source_rows(
                client,
                preparer=preparer,
                preparer_executor=preparer_executor,
                resume=resume,
                persistence_sink=persistence_sink,
                window_size=window_size,
                endpoint=endpoint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
        )


def _cleanup_generate_workflow_resources(
    *,
    persistence_sink: _FileBackedPersistenceSink | None,
    staged_stdin_path: Path | None,
) -> None:
    """Release workflow resources and re-raise the first cleanup failure."""

    cleanup_error: BaseException | None = None
    steps: list[Callable[[], None]] = []
    if persistence_sink is not None:
        steps.append(persistence_sink.close)
    if staged_stdin_path is not None:
        steps.append(lambda: staged_stdin_path.unlink(missing_ok=True))
    for step in steps:
        try:
            step()
        except BaseException as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


def run_generate_workflow(
    client: LMClient,
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_jsonl: str | None,
    checkpoint_dir: str | None,
    mapper_spec: str | None,
    resume: bool,
    endpoint: EndpointType,
    window_size: int,
    parse_json: bool,
    on_progress: Callable[[], Any] | None = None,
    on_status: Callable[[str], Any] | None = None,
) -> None:
    """Run the generate workflow engine."""

    mapper = _load_mapper(mapper_spec) if mapper_spec else None
    mapping_fingerprint = _compute_mapping_fingerprint(
        mapper_spec=mapper_spec, mapper=mapper
    )

    effective_input_jsonl = input_jsonl
    staged_stdin_path: Path | None = None
    if output_jsonl and prompt is None and input_jsonl is None:
        # File-backed runs must re-read the source for resume validation, but
        # stdin can only be consumed once — spool it to a temp file up front.
        staged_stdin_path = _materialize_stdin_source()
        effective_input_jsonl = str(staged_stdin_path)

    output_path = Path(output_jsonl) if output_jsonl else None
    checkpoint_path = (
        _checkpoint_path_for(output_jsonl, checkpoint_dir=checkpoint_dir)
        if output_jsonl
        else None
    )

    persistence_sink: _FileBackedPersistenceSink | None = None

    try:
        _validate_distinct_input_output_paths(
            input_jsonl=effective_input_jsonl, output_jsonl=output_jsonl
        )

        if output_path is not None and not resume:
            assert checkpoint_path is not None
            if on_status is not None:
                on_status("Preparing fresh workflow artifacts...")
            _stage_fresh_workflow_files(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
            )

        if resume and output_path is not None:
            assert checkpoint_path is not None
            _validate_resume_checkpoint(
                output_path,
                checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                on_status=on_status,
            )
        if output_path is not None:
            assert checkpoint_path is not None
            if on_status is not None:
                on_status("Opening output and checkpoint files...")
            persistence_sink = _FileBackedPersistenceSink(
                output_path=output_path, checkpoint_path=checkpoint_path
            )

        _run_generate_source_rows(
            client,
            prompt=prompt,
            effective_input_jsonl=effective_input_jsonl,
            resume=resume,
            checkpoint_path=checkpoint_path,
            persistence_sink=persistence_sink,
            mapper=mapper,
            window_size=window_size,
            endpoint=endpoint,
            parse_json=parse_json,
            on_progress=on_progress,
        )

    finally:
        _cleanup_generate_workflow_resources(
            persistence_sink=persistence_sink,
            staged_stdin_path=staged_stdin_path,
        )
