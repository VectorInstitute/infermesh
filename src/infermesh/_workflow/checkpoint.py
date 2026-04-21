"""Checkpoint storage and persistence helpers for the workflow engine."""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .models import CheckpointKey, _CheckpointItem
from .source import _iter_source_rows_with_keys

_SCHEMA_VERSION = 1
_RUN_METADATA_SINGLETON = 1
_ITEM_INSERT_BATCH_SIZE = 1000
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


def _checkpoint_path_for(
    output_jsonl: str, *, checkpoint_dir: str | None = None
) -> Path:
    """Derive the checkpoint file path from the output JSONL path."""

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
    """Open a read-write checkpoint database connection."""

    connection = sqlite3.connect(checkpoint_path)
    _configure_checkpoint_journal_mode(connection)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _connect_checkpoint_db_read_only(checkpoint_path: Path) -> sqlite3.Connection:
    """Open a read-only checkpoint connection for resume validation."""

    connection = sqlite3.connect(
        checkpoint_path.expanduser().resolve(strict=False).as_uri() + "?mode=ro",
        uri=True,
    )
    connection.execute("PRAGMA query_only=ON")
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
    return int(row[0]), str(row[1])


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
        prompt=prompt,
        input_jsonl=input_jsonl,
    ):
        batch.append(
            (
                checkpoint_key.record_fingerprint,
                checkpoint_key.occurrence,
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

        # Replace the visible artifacts only after checkpoint bootstrap has
        # succeeded, so old run files survive bootstrap failures intact.
        staged_checkpoint_path.replace(checkpoint_path)
        staged_output_path.replace(output_path)
    finally:
        if staged_checkpoint_path is not None and staged_checkpoint_path.exists():
            staged_checkpoint_path.unlink()
        if staged_output_path is not None and staged_output_path.exists():
            staged_output_path.unlink()


def _load_checkpoint_item(
    connection: sqlite3.Connection, checkpoint_key: CheckpointKey
) -> _CheckpointItem | None:
    """Load one checkpoint item by its occurrence-aware key."""

    row = connection.execute(
        """
        SELECT output_index, status, error
        FROM items
        WHERE record_fingerprint = ? AND occurrence = ?
        """,
        checkpoint_key.sql_params(),
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


def _mark_checkpoint_item_settled(
    connection: sqlite3.Connection,
    checkpoint_key: CheckpointKey,
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
        (status, error, *checkpoint_key.sql_params()),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Checkpoint item update failed for settled workflow row.")
    connection.commit()


@dataclass
class _PersistenceRequest:
    """One settled row that must be durably written by the sink thread."""

    record: dict[str, Any]
    checkpoint_key: CheckpointKey
    status: int
    error: str | None
    done: threading.Event
    failure: BaseException | None = None


@dataclass
class _PersistenceShutdown:
    """Signal the sink thread to flush and stop."""

    done: threading.Event


class _FileBackedPersistenceSink:
    """Serialize output/checkpoint writes onto one dedicated thread."""

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
        self._started.wait()
        self._raise_if_failed()

    def write_record(
        self,
        record: dict[str, Any],
        checkpoint_key: CheckpointKey,
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
                    # Write the user-facing output first, then mark the
                    # checkpoint item settled. A crash between the two can cause
                    # duplicate work on resume, but not silent row loss.
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
            self._started.set()
            if connection is not None:
                connection.close()
            if out_file is not None:
                out_file.close()

    def _wait_for_event(self, event: threading.Event) -> None:
        while not event.wait(timeout=0.1):
            if not self._thread.is_alive():
                break
        if event.is_set():
            return
        self._raise_if_failed()
        raise RuntimeError("Persistence sink stopped before acknowledging a write.")

    def _set_failure(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _fail_pending_items(self, exc: BaseException) -> None:
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
