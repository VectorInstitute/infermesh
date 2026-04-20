"""Internal workflow engine for the CLI."""

from __future__ import annotations

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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

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


_SCHEMA_VERSION = 1
_RUN_METADATA_SINGLETON = 1
_ITEM_INSERT_BATCH_SIZE = 1000

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


def _checkpoint_path_for(output_jsonl: str) -> Path:
    """Derive the checkpoint file path from the output JSONL path.

    ``results.jsonl``  ->  ``results.checkpoint.sqlite``
    ``results``        ->  ``results.checkpoint.sqlite``
    """
    path = Path(output_jsonl)
    if path.suffix == ".jsonl":
        return path.with_name(path.stem + ".checkpoint.sqlite")
    return path.with_name(path.name + ".checkpoint.sqlite")


def _connect_checkpoint_db(checkpoint_path: Path) -> sqlite3.Connection:
    """Open a checkpoint database connection."""

    connection = sqlite3.connect(checkpoint_path)
    connection.execute("PRAGMA journal_mode=WAL")  # crash-safe; allows concurrent reads
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


def _load_output_indices(output_path: Path) -> set[int]:
    """Return the set of integer ``_index`` values present in the output file."""

    observed_indices: set[int] = set()
    with output_path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
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
    return observed_indices


def _validate_resume_output_rows(
    connection: sqlite3.Connection, output_path: Path
) -> None:
    """Ensure every settled checkpoint item already exists in the output file."""

    output_indices = _load_output_indices(output_path)
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
) -> None:
    """Validate that the current source matches the checkpoint occurrence set."""

    seen_count = 0
    for _, checkpoint_key in _iter_source_rows_with_keys(
        prompt=prompt, input_jsonl=input_jsonl
    ):
        if _load_checkpoint_item(connection, checkpoint_key) is None:
            raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)
        seen_count += 1

    # The per-key lookup above catches added rows; the count comparison catches
    # removed rows (source has fewer occurrences than the checkpoint recorded).
    expected_count = int(connection.execute("SELECT COUNT(*) FROM items").fetchone()[0])
    if seen_count != expected_count:
        raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)


def _open_resume_checkpoint(
    output_path: Path,
    checkpoint_path: Path,
    *,
    mapping_fingerprint: str,
    prompt: str | None,
    input_jsonl: str | None,
) -> sqlite3.Connection:
    """Open and validate the resume checkpoint for a file-backed workflow."""

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
        _validate_resume_mapping_fingerprint(
            connection,
            mapping_fingerprint=mapping_fingerprint,
        )
        _validate_resume_output_rows(connection, output_path)
        _validate_resume_source(
            connection,
            prompt=prompt,
            input_jsonl=input_jsonl,
        )
    except BaseException:
        connection.close()
        raise
    return connection


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


def _flush_window(
    client: LMClient,
    window: list[_WorkItem],
    persistence_sink: _FileBackedPersistenceSink | None,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> list[dict[str, Any]]:
    """Execute one window of work through ``generate_batch`` and write results."""

    inputs = [item.mapped_input for item in window]
    stdout_rows: list[dict[str, Any]] = []

    def on_result(batch_idx: int, result: Any, error: BaseException | None) -> None:
        item = window[batch_idx]
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
            stdout_rows.append(record)
        if on_progress is not None:
            on_progress()

    client.generate_batch(
        inputs,
        endpoint=endpoint,
        return_exceptions=True,
        on_result=on_result,
    )
    return stdout_rows


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


def _run_generate_source_rows(
    client: LMClient,
    *,
    prompt: str | None,
    effective_input_jsonl: str | None,
    resume: bool,
    checkpoint_connection: sqlite3.Connection | None,
    persistence_sink: _FileBackedPersistenceSink | None,
    mapper: Callable[[dict[str, Any]], Any] | None,
    window_size: int,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Stream mapped rows into ``generate_batch`` windows."""

    window: list[_WorkItem] = []
    any_work = False

    for source_row, checkpoint_key in _iter_source_rows_with_keys(
        prompt=prompt, input_jsonl=effective_input_jsonl
    ):
        output_index = _generate_output_index_for_resume_row(
            resume=resume,
            checkpoint_connection=checkpoint_connection,
            checkpoint_key=checkpoint_key,
            source_index=source_row.source_index,
        )
        if output_index is None:
            continue

        any_work = True

        if source_row.error is not None:
            _emit_immediate_error(
                persistence_sink,
                output_index,
                checkpoint_key,
                source_row.error,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            continue

        raw_record = source_row.raw_record
        if raw_record is None:
            raise RuntimeError(
                "Invariant violated: source_row.raw_record is None after error check."
            )

        mapping_result = _apply_mapper_or_builtin(raw_record, mapper)
        if isinstance(mapping_result, Exception):
            _emit_immediate_error(
                persistence_sink,
                output_index,
                checkpoint_key,
                mapping_result,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            continue

        mapped_input, metadata = mapping_result
        metadata_result = _validate_metadata(metadata)
        if isinstance(metadata_result, Exception):
            _emit_immediate_error(
                persistence_sink,
                output_index,
                checkpoint_key,
                metadata_result,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            continue
        window.append(
            _WorkItem(
                output_index=output_index,
                checkpoint_key=checkpoint_key,
                mapped_input=mapped_input,
                metadata=metadata_result,
            )
        )

        if len(window) >= window_size:
            stdout_rows = _flush_window(
                client,
                window,
                persistence_sink,
                endpoint,
                parse_json,
                on_progress,
            )
            if stdout_rows:
                _write_jsonl(stdout_rows, None)
            window.clear()

    if window:
        stdout_rows = _flush_window(
            client,
            window,
            persistence_sink,
            endpoint,
            parse_json,
            on_progress,
        )
        if stdout_rows:
            _write_jsonl(stdout_rows, None)

    if resume and not any_work:
        sys.stderr.write("Nothing to do — all rows already completed.\n")


def _cleanup_generate_workflow_resources(
    *,
    persistence_sink: _FileBackedPersistenceSink | None,
    checkpoint_connection: sqlite3.Connection | None,
    staged_stdin_path: Path | None,
) -> None:
    """Release workflow resources and re-raise the first cleanup failure."""

    cleanup_error: BaseException | None = None
    steps: list[Callable[[], None]] = []
    if persistence_sink is not None:
        steps.append(persistence_sink.close)
    if checkpoint_connection is not None:
        steps.append(checkpoint_connection.close)
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
    mapper_spec: str | None,
    resume: bool,
    endpoint: EndpointType,
    window_size: int,
    parse_json: bool,
    on_progress: Callable[[], Any] | None = None,
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
    checkpoint_path = _checkpoint_path_for(output_jsonl) if output_jsonl else None

    checkpoint_connection: sqlite3.Connection | None = None
    persistence_sink: _FileBackedPersistenceSink | None = None

    try:
        _validate_distinct_input_output_paths(
            input_jsonl=effective_input_jsonl, output_jsonl=output_jsonl
        )

        if output_path is not None and not resume:
            assert checkpoint_path is not None
            _stage_fresh_workflow_files(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
            )

        if resume and output_path is not None:
            assert checkpoint_path is not None
            checkpoint_connection = _open_resume_checkpoint(
                output_path,
                checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
            )
        if output_path is not None:
            assert checkpoint_path is not None
            persistence_sink = _FileBackedPersistenceSink(
                output_path=output_path, checkpoint_path=checkpoint_path
            )

        _run_generate_source_rows(
            client,
            prompt=prompt,
            effective_input_jsonl=effective_input_jsonl,
            resume=resume,
            checkpoint_connection=checkpoint_connection,
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
            checkpoint_connection=checkpoint_connection,
            staged_stdin_path=staged_stdin_path,
        )
