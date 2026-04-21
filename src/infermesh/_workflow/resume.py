"""Resume validation and planning helpers for the workflow engine."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from .checkpoint import (
    _ERROR_STATUS,
    _ITEM_INSERT_BATCH_SIZE,
    _PENDING_STATUS,
    _SCHEMA_VERSION,
    _SUCCESS_STATUS,
    _connect_checkpoint_db_read_only,
    _load_run_metadata,
)
from .models import CheckpointKey, _ResumePlan, _SourceRow
from .source import (
    _compute_source_row_fingerprint,
    _iter_binary_source_rows_with_offsets,
    _iter_source_rows,
    _load_source_row_at_offset,
)

_STATUS_LOG_INTERVAL = 100_000
_RESUME_SOURCE_MISMATCH_ERROR = (
    "Resume source does not match the checkpoint file. Added, removed, or "
    "modified row occurrences are not supported."
)


class OutputIndexBitmap:
    """Compact presence bitmap for observed output rows."""

    def __init__(self) -> None:
        self._bits = bytearray()

    def add(self, output_index: int) -> None:
        """Mark one observed output index."""

        if output_index < 0:
            raise ValueError("Output rows must not use negative _index values.")
        byte_index = output_index // 8
        if byte_index >= len(self._bits):
            self._bits.extend(b"\x00" * (byte_index + 1 - len(self._bits)))
        self._bits[byte_index] |= 1 << (output_index % 8)

    def contains(self, output_index: int) -> bool:
        """Return whether the bitmap contains ``output_index``."""

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
        """Load a bitmap of observed output indices from the output artifact."""

        bitmap = cls()
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
                    bitmap.add(output_index)
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
        """Validate the checkpoint and return a resume plan when needed."""

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
                # Only file-backed resume needs the planner; prompt runs have no
                # seekable source file and can validate inline.
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
        # A bitmap keeps this validation compact even when _index values are
        # sparse or the run has already settled a very large number of rows.
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
        """Build the ephemeral planner DB for a resumed file-backed workflow."""

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
        """Yield pending source rows in source order using the built plan."""

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
        """Remove the ephemeral planner DB if one exists."""

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
        # The planner DB is disposable scratch rebuilt on every resume, so we
        # optimize for planning throughput rather than planner durability.
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
        # Derive duplicate occurrences on disk so million-row resumes do not
        # need a Python fingerprint->count map just to align with checkpoint
        # occurrence keys.
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
        # These two anti-joins enforce exact source/checkpoint equivalence
        # without re-running row-by-row checkpoint lookups in Python.
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
        # Materializing pending rows once lets the scheduler jump straight to
        # unfinished work instead of rewalking the settled prefix on resume.
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


def validate_resume_checkpoint(
    output_path: Path,
    checkpoint_path: Path,
    *,
    mapping_fingerprint: str,
    prompt: str | None,
    input_jsonl: str | None,
    on_status: Callable[[str], Any] | None = None,
) -> _ResumePlan | None:
    """Validate the resume checkpoint and optionally build a resume plan."""

    return ResumeValidator(
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        mapping_fingerprint=mapping_fingerprint,
        prompt=prompt,
        input_jsonl=input_jsonl,
        on_status=on_status,
    ).validate()
