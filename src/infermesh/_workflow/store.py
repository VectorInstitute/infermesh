"""Run stores for batch workflow durability and resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from .models import (
    _SOURCE_EXHAUSTED,
    CheckpointKey,
    PreparedItem,
    SettledRecord,
    SourceStream,
    WorkStream,
    _CheckpointItem,
    _ResumePlan,
    _SourceExhausted,
    _SourceRow,
)
from .source import (
    FileSourceStream,
    _compute_source_row_fingerprint,
    _iter_binary_source_rows_with_offsets,
    _iter_source_rows,
    _iter_source_rows_with_keys,
    _load_source_row_at_offset,
    _prepare_source_item,
)

_SCHEMA_VERSION = 1
_RUN_METADATA_SINGLETON = 1
_ITEM_INSERT_BATCH_SIZE = 1000
_CHECKPOINT_PATH_HASH_LENGTH = 8
_STATUS_LOG_INTERVAL = 100_000

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

    delete_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    journal_mode = str(delete_mode[0]).lower() if delete_mode is not None else ""
    if journal_mode == "delete":
        return journal_mode

    persist_mode = connection.execute("PRAGMA journal_mode=PERSIST").fetchone()
    journal_mode = str(persist_mode[0]).lower() if persist_mode is not None else ""
    if journal_mode == "persist":
        return journal_mode

    raise RuntimeError(
        "Checkpoint DB could not be configured for rollback journaling. "
        f"SQLite reported journal_mode={journal_mode!r}."
    )


def _checkpoint_journal_path(checkpoint_path: Path) -> Path:
    """Return SQLite's rollback journal sidecar path for ``checkpoint_path``."""

    return checkpoint_path.with_name(f"{checkpoint_path.name}-journal")


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

        # Replace visible files only after checkpoint bootstrap succeeds. The
        # empty output goes first so a crash cannot leave a new all-pending
        # checkpoint pointing at stale output rows from an older run.
        staged_output_path.replace(output_path)
        staged_checkpoint_path.replace(checkpoint_path)
    finally:
        if staged_checkpoint_path is not None and staged_checkpoint_path.exists():
            staged_checkpoint_path.unlink()
        if staged_checkpoint_path is not None:
            _checkpoint_journal_path(staged_checkpoint_path).unlink(missing_ok=True)
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
        """Persist one settled record and checkpoint update.

        Parameters
        ----------
        record
            User-facing JSONL output row.
        checkpoint_key
            Checkpoint item to mark settled after the output row is written.
        status
            Terminal checkpoint status.
        error
            Error text to persist in the checkpoint, if any.
        """

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
                    # Output is the user-visible artifact. Write it before the
                    # checkpoint transition so a crash can duplicate work on
                    # resume, but cannot silently lose a completed row.
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


class OutputIndexBitmap:
    """Compact presence bitmap for observed output rows."""

    def __init__(self) -> None:
        self._bits = bytearray()

    def add(self, output_index: int) -> None:
        """Mark one observed output index.

        Parameters
        ----------
        output_index
            Output row index observed in the JSONL artifact.

        Raises
        ------
        ValueError
            If ``output_index`` is negative.
        """

        if output_index < 0:
            raise ValueError("Output rows must not use negative _index values.")
        byte_index = output_index // 8
        if byte_index >= len(self._bits):
            self._bits.extend(b"\x00" * (byte_index + 1 - len(self._bits)))
        self._bits[byte_index] |= 1 << (output_index % 8)

    def contains(self, output_index: int) -> bool:
        """Return whether the bitmap contains ``output_index``.

        Parameters
        ----------
        output_index
            Output row index to check.

        Returns
        -------
        bool
            Whether the index was seen in the output artifact.
        """

        if output_index < 0:
            return False
        byte_index = output_index // 8
        if byte_index >= len(self._bits):
            return False
        return bool(self._bits[byte_index] & (1 << (output_index % 8)))

    @classmethod
    def load(
        cls,
        output_path: Path,
        *,
        on_status: Callable[[str], Any] | None = None,
    ) -> OutputIndexBitmap:
        """Load observed output indices from a JSONL output artifact.

        Parameters
        ----------
        output_path
            Output JSONL file to scan.
        on_status
            Optional callback for progress messages during large scans.

        Returns
        -------
        OutputIndexBitmap
            Bitmap containing each observed ``_index`` value.
        """

        bitmap = cls()
        with output_path.open(encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    if on_status is not None:
                        on_status(
                            "Resume: ignored malformed output row "
                            f"{line_number:,} while validating artifact."
                        )
                    continue
                if not isinstance(row, dict):
                    if on_status is not None:
                        on_status(
                            "Resume: ignored non-object output row "
                            f"{line_number:,} while validating artifact."
                        )
                    continue
                output_index = row.get("_index")
                if isinstance(output_index, int):
                    bitmap.add(output_index)
                elif on_status is not None:
                    on_status(
                        "Resume: ignored output row "
                        f"{line_number:,} without integer _index."
                    )
                if on_status is not None and line_number % _STATUS_LOG_INTERVAL == 0:
                    on_status(f"Resume: scanned {line_number:,} output rows...")
        return bitmap


class ResumeValidator:
    """Validate resume state and optionally build a file-backed resume plan."""

    def __init__(
        self,
        *,
        output_path: Path,
        checkpoint_path: Path,
        mapping_fingerprint: str,
        prompt: str | None,
        input_jsonl: str | None,
        on_status: Callable[[str], Any] | None = None,
    ) -> None:
        self._output_path = output_path
        self._checkpoint_path = checkpoint_path
        self._mapping_fingerprint = mapping_fingerprint
        self._prompt = prompt
        self._input_jsonl = input_jsonl
        self._on_status = on_status

    def validate(self) -> _ResumePlan | None:
        """Validate checkpoint, output, and source compatibility.

        Returns
        -------
        _ResumePlan | None
            Planner-backed resume plan for seekable JSONL sources, otherwise
            ``None`` when sequential validation is enough.

        Raises
        ------
        ValueError
            If checkpoint metadata, output rows, or source rows are incompatible
            with the requested resume.
        """

        if not self._checkpoint_path.exists():
            raise ValueError(
                f"--resume requires checkpoint file {self._checkpoint_path}. "
                "Start a fresh file-backed run first."
            )
        if not self._output_path.exists():
            raise ValueError(
                f"--resume requires output file {self._output_path} because "
                f"checkpoint file {self._checkpoint_path} already exists."
            )

        connection = _connect_checkpoint_db_read_only(self._checkpoint_path)
        try:
            if self._on_status is not None:
                self._on_status("Resume: validating checkpoint file...")
            self._validate_mapping_fingerprint(connection)
            self._validate_output_rows(connection)
            if self._prompt is None and self._input_jsonl is not None:
                # File-backed JSONL can be indexed once into a planner DB so
                # large resumes jump straight to pending rows instead of
                # rewalking a long settled prefix in Python.
                return ResumePlanner(
                    checkpoint_connection=connection,
                    input_jsonl=self._input_jsonl,
                    on_status=self._on_status,
                ).build()
            self._validate_source(connection)
            return None
        finally:
            connection.close()

    def _validate_mapping_fingerprint(self, connection: sqlite3.Connection) -> None:
        schema_version, checkpoint_mapping_fingerprint = _load_run_metadata(connection)
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint file uses an unsupported schema version. Restart the run "
                "without --resume."
            )
        if checkpoint_mapping_fingerprint != self._mapping_fingerprint:
            raise ValueError(
                "Resume mapping does not match the checkpoint file. Use the original "
                "mapper implementation or restart without --resume."
            )

    def _validate_output_rows(self, connection: sqlite3.Connection) -> None:
        if self._on_status is not None:
            self._on_status("Resume: validating output artifact...")
        output_indices = OutputIndexBitmap.load(
            self._output_path,
            on_status=self._on_status,
        )
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
            if not output_indices.contains(output_index):
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

    def _validate_source(self, connection: sqlite3.Connection) -> None:
        if self._on_status is not None:
            self._on_status("Resume: validating input source...")

        remaining_counts = self._load_checkpoint_fingerprint_counts(connection)
        for seen_count, source_row in enumerate(
            _iter_source_rows(prompt=self._prompt, input_jsonl=self._input_jsonl),
            start=1,
        ):
            fingerprint = _compute_source_row_fingerprint(source_row)
            remaining = remaining_counts.get(fingerprint)
            if remaining is None:
                raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)
            if remaining == 1:
                del remaining_counts[fingerprint]
            else:
                remaining_counts[fingerprint] = remaining - 1
            if self._on_status is not None and seen_count % _STATUS_LOG_INTERVAL == 0:
                self._on_status(f"Resume: scanned {seen_count:,} source rows...")

        if remaining_counts:
            raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)

    @staticmethod
    def _load_checkpoint_fingerprint_counts(
        connection: sqlite3.Connection,
    ) -> dict[bytes, int]:
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


class ResumePlanner:
    """Own the temporary SQLite database used to plan resumed file-backed runs."""

    def __init__(
        self,
        *,
        checkpoint_connection: sqlite3.Connection,
        input_jsonl: str,
        on_status: Callable[[str], Any] | None = None,
    ) -> None:
        self._checkpoint_connection = checkpoint_connection
        self._input_jsonl = input_jsonl
        self._on_status = on_status

    def build(self) -> _ResumePlan:
        """Build the ephemeral planner DB for a resumed file-backed workflow.

        Returns
        -------
        _ResumePlan
            Handle to the temporary planner DB.

        Raises
        ------
        ValueError
            If source rows do not exactly match checkpoint rows.
        BaseException
            Propagates SQLite and filesystem errors after cleaning up partial
            planner artifacts.
        """

        planner_path = self._create_path()
        planner_connection: sqlite3.Connection | None = None
        cleanup_planner_path = False
        try:
            planner_connection = self._connect_planner_db(planner_path)
            self._initialize_db(planner_connection)
            self._copy_checkpoint_items(planner_connection)
            if self._on_status is not None:
                self._on_status("Resume: building resume plan...")
            self._index_source_rows(planner_connection)
            self._materialize_source_items(planner_connection)
            if self._on_status is not None:
                self._on_status("Resume: locating pending rows...")
            self._validate_source_plan(planner_connection)
            self._materialize_pending_work(planner_connection)
            planner_connection.commit()
            return _ResumePlan(planner_path=planner_path)
        except BaseException:
            cleanup_planner_path = True
            raise
        finally:
            if planner_connection is not None:
                planner_connection.close()
            if cleanup_planner_path:
                planner_path.unlink(missing_ok=True)

    @staticmethod
    def iter_rows(
        resume_plan: _ResumePlan, *, input_jsonl: str
    ) -> Iterator[tuple[_SourceRow, int, CheckpointKey]]:
        """Yield pending source rows in source order using the built plan.

        Parameters
        ----------
        resume_plan
            Planner DB handle returned by :meth:`build`.
        input_jsonl
            Original JSONL source path.

        Yields
        ------
        tuple[_SourceRow, int, CheckpointKey]
            Parsed source row, original output index, and checkpoint key.
        """

        planner_connection = sqlite3.connect(resume_plan.planner_path)
        source_file = open(input_jsonl, "rb")  # noqa: SIM115
        try:
            for row in planner_connection.execute(
                """
                SELECT
                    source_order,
                    output_index,
                    byte_offset,
                    record_fingerprint,
                    occurrence
                FROM pending_work
                ORDER BY source_order
                """
            ):
                source_order, output_index, byte_offset, fingerprint, occurrence = row
                yield (
                    _load_source_row_at_offset(
                        source_file,
                        offset=int(byte_offset),
                        source_index=int(source_order),
                    ),
                    int(output_index),
                    CheckpointKey(bytes(fingerprint), int(occurrence)),
                )
        finally:
            source_file.close()
            planner_connection.close()

    @staticmethod
    def cleanup(resume_plan: _ResumePlan | None) -> None:
        """Remove the ephemeral planner DB if one exists.

        Parameters
        ----------
        resume_plan
            Planner DB handle to remove, or ``None``.
        """

        if resume_plan is not None:
            resume_plan.planner_path.unlink(missing_ok=True)

    @staticmethod
    def _temp_dir() -> Path:
        """Return the directory used for ephemeral resume planner databases."""

        return Path(os.getenv("TMPDIR") or tempfile.gettempdir())

    @classmethod
    def _create_path(cls) -> Path:
        planner_dir = cls._temp_dir()
        planner_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=planner_dir,
            prefix=".infermesh-resume-plan.",
            suffix=".sqlite",
            delete=False,
        ) as file_handle:
            return Path(file_handle.name)

    @staticmethod
    def _connect_planner_db(planner_path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(planner_path)
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    @staticmethod
    def _initialize_db(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE checkpoint_items (
                record_fingerprint BLOB NOT NULL,
                occurrence INTEGER NOT NULL,
                output_index INTEGER NOT NULL,
                status INTEGER NOT NULL,
                PRIMARY KEY (record_fingerprint, occurrence)
            );

            CREATE TABLE source_rows (
                source_order INTEGER PRIMARY KEY,
                byte_offset INTEGER NOT NULL,
                record_fingerprint BLOB NOT NULL
            );
            """
        )

    def _copy_checkpoint_items(self, planner_connection: sqlite3.Connection) -> None:
        batch: list[tuple[bytes, int, int, int]] = []
        for row in self._checkpoint_connection.execute(
            """
            SELECT record_fingerprint, occurrence, output_index, status
            FROM items
            """
        ):
            batch.append((bytes(row[0]), int(row[1]), int(row[2]), int(row[3])))
            if len(batch) >= _ITEM_INSERT_BATCH_SIZE:
                planner_connection.executemany(
                    """
                    INSERT INTO checkpoint_items (
                        record_fingerprint,
                        occurrence,
                        output_index,
                        status
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
        if batch:
            planner_connection.executemany(
                """
                INSERT INTO checkpoint_items (
                    record_fingerprint,
                    occurrence,
                    output_index,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

    def _index_source_rows(self, planner_connection: sqlite3.Connection) -> None:
        batch: list[tuple[int, int, bytes]] = []
        for seen_count, (source_row, byte_offset) in enumerate(
            _iter_binary_source_rows_with_offsets(self._input_jsonl),
            start=1,
        ):
            fingerprint = _compute_source_row_fingerprint(source_row)
            batch.append((source_row.source_index, byte_offset, fingerprint))
            if len(batch) >= _ITEM_INSERT_BATCH_SIZE:
                planner_connection.executemany(
                    """
                    INSERT INTO source_rows (
                        source_order,
                        byte_offset,
                        record_fingerprint
                    )
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
            if self._on_status is not None and seen_count % _STATUS_LOG_INTERVAL == 0:
                self._on_status(f"Resume: indexed {seen_count:,} source rows...")
        if batch:
            planner_connection.executemany(
                """
                INSERT INTO source_rows (source_order, byte_offset, record_fingerprint)
                VALUES (?, ?, ?)
                """,
                batch,
            )

    @staticmethod
    def _materialize_source_items(planner_connection: sqlite3.Connection) -> None:
        # SQLite derives duplicate occurrences without keeping a Python
        # fingerprint->count map for million-row resume inputs.
        planner_connection.executescript(
            """
            CREATE TABLE source_items AS
            SELECT
                source_order,
                byte_offset,
                record_fingerprint,
                row_number() OVER (
                    PARTITION BY record_fingerprint
                    ORDER BY source_order
                ) - 1 AS occurrence
            FROM source_rows;

            DROP TABLE source_rows;

            CREATE INDEX idx_source_items_key
            ON source_items(record_fingerprint, occurrence);

            CREATE INDEX idx_checkpoint_items_status_output_index
            ON checkpoint_items(status, output_index);
            """
        )

    @staticmethod
    def _validate_source_plan(planner_connection: sqlite3.Connection) -> None:
        # Exact source/checkpoint equivalence is a set relationship. Anti-joins
        # catch both added source rows and checkpoint rows missing from source.
        source_extra = planner_connection.execute(
            """
            SELECT 1
            FROM source_items AS source
            LEFT JOIN checkpoint_items AS checkpoint
            USING (record_fingerprint, occurrence)
            WHERE checkpoint.output_index IS NULL
            LIMIT 1
            """
        ).fetchone()
        if source_extra is not None:
            raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)

        checkpoint_extra = planner_connection.execute(
            """
            SELECT 1
            FROM checkpoint_items AS checkpoint
            LEFT JOIN source_items AS source
            USING (record_fingerprint, occurrence)
            WHERE source.source_order IS NULL
            LIMIT 1
            """
        ).fetchone()
        if checkpoint_extra is not None:
            raise ValueError(_RESUME_SOURCE_MISMATCH_ERROR)

    @staticmethod
    def _materialize_pending_work(planner_connection: sqlite3.Connection) -> None:
        # Materializing pending rows once makes the scheduler's read path cheap
        # and deterministic after validation has paid the indexing cost.
        planner_connection.executescript(
            f"""
            CREATE TABLE pending_work AS
            SELECT
                source.source_order,
                source.byte_offset,
                checkpoint.output_index,
                checkpoint.record_fingerprint,
                checkpoint.occurrence
            FROM source_items AS source
            INNER JOIN checkpoint_items AS checkpoint
            USING (record_fingerprint, occurrence)
            WHERE checkpoint.status = {_PENDING_STATUS};

            CREATE INDEX idx_pending_work_source_order
            ON pending_work(source_order);
            """
        )


class _SequentialWorkStream:
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

    def next_prepared(self) -> PreparedItem | _SourceExhausted:
        self._ensure_open()
        assert self._source_rows is not None
        for source_row, checkpoint_key in self._source_rows:
            output_index = self._output_index_for_row(
                source_row=source_row,
                checkpoint_key=checkpoint_key,
            )
            if output_index is None:
                continue
            return _prepare_source_item(
                source_row=source_row,
                output_index=output_index,
                checkpoint_key=checkpoint_key,
                mapper=self._mapper,
            )
        return _SOURCE_EXHAUSTED

    def close(self) -> None:
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

    def _output_index_for_row(
        self,
        *,
        source_row: _SourceRow,
        checkpoint_key: CheckpointKey,
    ) -> int | None:
        if not self._resume:
            return source_row.source_index
        if self._checkpoint_connection is None:
            raise RuntimeError("Resume path requires an open checkpoint connection.")
        checkpoint_item = _load_checkpoint_item(
            self._checkpoint_connection,
            checkpoint_key,
        )
        if checkpoint_item is None:
            raise ValueError(
                "Resume source does not match the checkpoint file. Added, removed, "
                "or modified row occurrences are not supported."
            )
        if checkpoint_item.status in _SETTLED_STATUSES:
            return None
        return checkpoint_item.output_index


class _PlannedResumeWorkStream:
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

    def next_prepared(self) -> PreparedItem | _SourceExhausted:
        self._ensure_open()
        assert self._planned_rows is not None
        for source_row, output_index, checkpoint_key in self._planned_rows:
            return _prepare_source_item(
                source_row=source_row,
                output_index=output_index,
                checkpoint_key=checkpoint_key,
                mapper=self._mapper,
            )
        return _SOURCE_EXHAUSTED

    def close(self) -> None:
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


class StdoutRunStore:
    """Run store for non-checkpointed stdout workflows."""

    @property
    def skipped(self) -> int:
        """Return already-settled rows skipped by this invocation.

        Returns
        -------
        int
            Always zero because stdout workflows do not persist resume state.
        """

        return 0

    @property
    def output_path(self) -> Path | None:
        """Return the user-facing output path.

        Returns
        -------
        None
            Stdout workflows have no output path.
        """

        return None

    def open(
        self,
        source: SourceStream,
        *,
        resume: bool,
        mapping_fingerprint: str,
    ) -> WorkStream:
        """Open a non-checkpointed work stream.

        Parameters
        ----------
        source
            Source stream to consume.
        resume
            Must be ``False`` because stdout workflows cannot resume.
        mapping_fingerprint
            Ignored; no checkpoint metadata exists for stdout workflows.

        Returns
        -------
        WorkStream
            Blocking stream over all source rows.

        Raises
        ------
        ValueError
            If ``resume`` is requested.
        """

        del mapping_fingerprint
        if resume:
            raise ValueError(
                "--resume requires --output-jsonl because resumed runs need a "
                "checkpoint file."
            )
        return source.iter_all()

    async def settle(self, settled: SettledRecord) -> None:
        """Write one settled record to stdout.

        Parameters
        ----------
        settled
            Settled workflow record.
        """

        sys.stdout.write(json.dumps(settled.record) + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        """Release store resources."""

        return


class SqliteFileRunStore:
    """File/SQLite run store for resumable workflows."""

    def __init__(
        self,
        *,
        output_jsonl: str,
        checkpoint_dir: str | None,
        on_status: Callable[[str], Any] | None = None,
    ) -> None:
        """Create a file-backed run store.

        Parameters
        ----------
        output_jsonl
            Output JSONL path.
        checkpoint_dir
            Optional directory for checkpoint files.
        on_status
            Optional callback for setup and resume progress messages.
        """

        self._output_path = Path(output_jsonl)
        self._checkpoint_path = _checkpoint_path_for(
            output_jsonl,
            checkpoint_dir=checkpoint_dir,
        )
        self._on_status = on_status
        self._sink: _FileBackedPersistenceSink | None = None
        self._resume_plan: _ResumePlan | None = None
        self._skipped = 0

    @property
    def checkpoint_path(self) -> Path:
        """Return the checkpoint DB path.

        Returns
        -------
        Path
            SQLite checkpoint path derived from the output path and optional
            checkpoint directory.
        """

        return self._checkpoint_path

    @property
    def skipped(self) -> int:
        """Return rows skipped because they were already settled.

        Returns
        -------
        int
            Number of terminal checkpoint rows found at resume start.
        """

        return self._skipped

    @property
    def output_path(self) -> Path | None:
        """Return the user-facing output path.

        Returns
        -------
        Path | None
            Output JSONL path.
        """

        return self._output_path

    def open(
        self,
        source: SourceStream,
        *,
        resume: bool,
        mapping_fingerprint: str,
    ) -> WorkStream:
        """Open checkpoint/output artifacts and choose a work stream.

        Parameters
        ----------
        source
            File-backed source stream to schedule.
        resume
            Whether to resume from existing checkpoint/output artifacts.
        mapping_fingerprint
            Fingerprint used to validate mapper compatibility on resume.

        Returns
        -------
        WorkStream
            Sequential or planner-backed blocking work stream.

        Raises
        ------
        TypeError
            If ``source`` is not file-backed.
        ValueError
            If resume artifacts are missing or incompatible.
        """

        if not isinstance(source, FileSourceStream):
            raise TypeError("SqliteFileRunStore requires FileSourceStream.")

        prompt = source.prompt
        input_jsonl = source.input_jsonl
        if not resume:
            if self._on_status is not None:
                self._on_status("Preparing fresh workflow artifacts...")
            _stage_fresh_workflow_files(
                prompt=prompt,
                input_jsonl=input_jsonl,
                output_path=self._output_path,
                checkpoint_path=self._checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
            )

        if resume:
            self._resume_plan = ResumeValidator(
                output_path=self._output_path,
                checkpoint_path=self._checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
                prompt=prompt,
                input_jsonl=input_jsonl,
                on_status=self._on_status,
            ).validate()
            self._skipped = _count_settled_items(self._checkpoint_path)

        if self._on_status is not None:
            self._on_status("Opening output and checkpoint files...")
        self._sink = _FileBackedPersistenceSink(
            output_path=self._output_path,
            checkpoint_path=self._checkpoint_path,
        )

        if self._resume_plan is not None:
            assert input_jsonl is not None
            return _PlannedResumeWorkStream(
                input_jsonl=input_jsonl,
                resume_plan=self._resume_plan,
                mapper=source.mapper,
            )
        return _SequentialWorkStream(
            prompt=prompt,
            input_jsonl=input_jsonl,
            resume=resume,
            checkpoint_path=self._checkpoint_path,
            mapper=source.mapper,
        )

    async def settle(self, settled: SettledRecord) -> None:
        """Durably write one settled record and checkpoint transition.

        Parameters
        ----------
        settled
            Settled workflow row to persist.

        Raises
        ------
        RuntimeError
            If the store has not been opened.
        """

        if self._sink is None:
            raise RuntimeError("Run store must be opened before settling records.")

        sink = self._sink

        def write_record() -> None:
            sink.write_record(
                settled.record,
                settled.checkpoint_key,
                status=_ERROR_STATUS if settled.error else _SUCCESS_STATUS,
                error=str(settled.error) if settled.error else None,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, write_record)

    def close(self) -> None:
        """Close persistence resources and remove temporary planner state.

        Raises
        ------
        BaseException
            Re-raises the first sink or planner cleanup error.
        """

        cleanup_error: BaseException | None = None
        try:
            if self._sink is not None:
                self._sink.close()
                self._sink = None
        except BaseException as exc:  # noqa: BLE001
            cleanup_error = exc

        try:
            ResumePlanner.cleanup(self._resume_plan)
            self._resume_plan = None
        except BaseException as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        if cleanup_error is not None:
            raise cleanup_error


def _count_settled_items(checkpoint_path: Path) -> int:
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect_checkpoint_db_read_only(checkpoint_path)
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM items
            WHERE status IN (?, ?)
            """,
            (_SUCCESS_STATUS, _ERROR_STATUS),
        ).fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        if connection is not None:
            connection.close()


__all__ = [
    "SqliteFileRunStore",
    "StdoutRunStore",
]
