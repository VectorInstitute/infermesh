"""Unit tests for the workflow engine (infermesh._workflow)."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import sys
import threading
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import infermesh._workflow.checkpoint as checkpoint_module
from infermesh._workflow import run_generate_workflow
from infermesh._workflow.checkpoint import (
    _checkpoint_path_for,
    _connect_checkpoint_db,
    _load_run_metadata,
)
from infermesh._workflow.mapping import _compute_mapping_fingerprint
from infermesh._workflow.resume import ResumePlanner
from infermesh.sync_runner import SyncRunner
from tests.fakes import (
    checkpoint_item_for_parse_error,
    checkpoint_item_for_record,
    load_resume_state,
    write_checkpoint_db,
)

# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    output_text: str
    request_id: str = "req-0"
    finish_reason: str = "stop"
    token_usage: None = None


class _FakeClient:
    """Async fake that records workflow admissions one item at a time."""

    def __init__(self) -> None:
        self._sync_runner = SyncRunner()
        self.inputs: list[Any] = []
        self.active = 0
        self.peak_active = 0

    def _run_sync(self, coroutine: Any) -> Any:
        return self._sync_runner.run(coroutine)

    async def agenerate(self, input_data: Any, **kwargs: Any) -> _FakeResult:
        self.inputs.append(input_data)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            return _FakeResult(output_text=f"out:{input_data}")
        finally:
            self.active -= 1

    def close(self) -> None:
        self._sync_runner.close()


class _RollingWindowFakeClient(_FakeClient):
    """Fake client that exposes whether the workflow refilled before a slow item ended."""

    def __init__(self) -> None:
        super().__init__()
        self.tail_started_before_slow_finished = False
        self._allow_slow_finish: asyncio.Event | None = None

    async def agenerate(self, input_data: Any, **kwargs: Any) -> _FakeResult:
        self.inputs.append(input_data)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if self._allow_slow_finish is None:
                self._allow_slow_finish = asyncio.Event()
            if input_data == "slow":
                await asyncio.wait_for(self._allow_slow_finish.wait(), timeout=1.0)
            else:
                await asyncio.sleep(0)
                if input_data == "tail-1":
                    self.tail_started_before_slow_finished = True
                    self._allow_slow_finish.set()
            return _FakeResult(output_text=f"out:{input_data}")
        finally:
            self.active -= 1


class _SelectiveFailingFakeClient(_FakeClient):
    """Fake client that fails selected prompts while letting siblings continue."""

    def __init__(self, *, failing_inputs: set[Any]) -> None:
        super().__init__()
        self._failing_inputs = set(failing_inputs)

    async def agenerate(self, input_data: Any, **kwargs: Any) -> _FakeResult:
        self.inputs.append(input_data)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            if input_data in self._failing_inputs:
                raise RuntimeError(f"boom:{input_data}")
            return _FakeResult(output_text=f"out:{input_data}")
        finally:
            self.active -= 1


class _MapperSignalFakeClient(_FakeClient):
    """Fake client that signals when a specific mapped input has completed."""

    def __init__(self, *, release_event: threading.Event) -> None:
        super().__init__()
        self._release_event = release_event

    async def agenerate(self, input_data: Any, **kwargs: Any) -> _FakeResult:
        self.inputs.append(input_data)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(0)
            if input_data == "first":
                self._release_event.set()
            return _FakeResult(output_text=f"out:{input_data}")
        finally:
            self.active -= 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_input(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "input.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _read_output(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(
    client: _FakeClient,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    checkpoint_dir: str | None = None,
    prompt: str | None = None,
    mapper_spec: str | None = None,
    resume: bool = False,
    window_size: int = 128,
    parse_json: bool = False,
    on_status: Any = None,
) -> None:
    try:
        run_generate_workflow(
            client,  # type: ignore[arg-type]
            prompt=prompt,
            input_jsonl=str(input_path) if input_path is not None else None,
            output_jsonl=str(output_path) if output_path is not None else None,
            checkpoint_dir=checkpoint_dir,
            mapper_spec=mapper_spec,
            resume=resume,
            endpoint="chat_completion",
            window_size=window_size,
            parse_json=parse_json,
            on_status=on_status,
        )
    finally:
        client.close()


def _load_mapping_fingerprint(checkpoint_path: Path) -> str:
    connection = _connect_checkpoint_db(checkpoint_path)
    try:
        _, mapping_fingerprint = _load_run_metadata(connection)
        return mapping_fingerprint
    finally:
        connection.close()


def _load_checkpoint_journal_mode(checkpoint_path: Path) -> str:
    connection = sqlite3.connect(checkpoint_path)
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower() if row is not None else ""
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_file_backed_generate_streams_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"prompt": f"p{i}"} for i in range(7)]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()
    monkeypatch.setattr(ResumePlanner, "_temp_dir", lambda: tmp_path)

    _run(client, input_path=input_path, output_path=output_path, window_size=3)

    assert client.inputs == [f"p{i}" for i in range(7)]
    assert client.peak_active <= 3
    assert not list(tmp_path.glob(".infermesh-resume-plan.*.sqlite"))

    out_rows = _read_output(output_path)
    assert len(out_rows) == 7


def test_default_checkpoint_uses_portable_rollback_journaling(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    assert checkpoint_path.exists()
    assert _load_checkpoint_journal_mode(checkpoint_path) in {"persist", "delete"}
    assert not Path(f"{checkpoint_path}-wal").exists()
    assert not Path(f"{checkpoint_path}-shm").exists()


def test_checkpoint_override_disambiguates_same_output_basename(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    output_a = tmp_path / "run-a" / "out.jsonl"
    output_b = tmp_path / "run-b" / "out.jsonl"

    checkpoint_a = _checkpoint_path_for(
        str(output_a), checkpoint_dir=str(checkpoint_dir)
    )
    checkpoint_b = _checkpoint_path_for(
        str(output_b), checkpoint_dir=str(checkpoint_dir)
    )

    assert checkpoint_a.parent == checkpoint_dir
    assert checkpoint_b.parent == checkpoint_dir
    assert checkpoint_a != checkpoint_b
    assert checkpoint_a.name.startswith("out.")
    assert checkpoint_b.name.startswith("out.")


def test_generate_workflow_supports_running_event_loop(tmp_path: Path) -> None:
    rows = [{"prompt": "hello"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    async def invoke_workflow() -> None:
        _run(client, input_path=input_path, output_path=output_path)

    asyncio.run(invoke_workflow())

    assert _read_output(output_path)[0]["output_text"] == "out:hello"


def test_file_backed_generate_refills_window_as_items_finish(tmp_path: Path) -> None:
    rows = [
        {"prompt": "slow"},
        {"prompt": "fast-1"},
        {"prompt": "fast-2"},
        {"prompt": "tail-1"},
        {"prompt": "tail-2"},
    ]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))
    client = _RollingWindowFakeClient()

    _run(client, input_path=input_path, output_path=output_path, window_size=3)

    assert client.peak_active <= 3
    assert client.tail_started_before_slow_finished

    out_rows = _read_output(output_path)
    assert len(out_rows) == 5

    state = load_resume_state(checkpoint_path)
    assert len(state) == 5
    assert {row["status"] for row in state.values()} == {"success"}


def test_resume_skips_settled_rows(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))

    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0], occurrence=0, index=0, status="success"
            ),
            checkpoint_item_for_record(
                rows[1], occurrence=0, index=1, status="pending"
            ),
            checkpoint_item_for_record(
                rows[2], occurrence=0, index=2, status="pending"
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == ["b", "c"]

    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:b"
    assert rows_by_index[2]["output_text"] == "out:c"

    state = load_resume_state(checkpoint_path)
    assert {row["status"] for row in state.values()} == {"success"}


def test_resume_reports_planner_status_and_cleans_temp_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0], occurrence=0, index=0, status="success"
            ),
            checkpoint_item_for_record(
                rows[1], occurrence=0, index=1, status="pending"
            ),
        ],
    )
    monkeypatch.setattr(ResumePlanner, "_temp_dir", lambda: tmp_path)
    statuses: list[str] = []

    client = _FakeClient()
    _run(
        client,
        input_path=input_path,
        output_path=output_path,
        resume=True,
        on_status=statuses.append,
    )

    assert "Resume: validating checkpoint file..." in statuses
    assert "Resume: validating output artifact..." in statuses
    assert "Resume: building resume plan..." in statuses
    assert "Resume: locating pending rows..." in statuses
    assert "Opening output and checkpoint files..." in statuses
    assert not list(tmp_path.glob(".infermesh-resume-plan.*.sqlite"))


def test_resume_reuses_checkpoint_dir_override(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_dir = tmp_path / "scratch-checkpoints"
    checkpoint_path = _checkpoint_path_for(
        str(output_path), checkpoint_dir=str(checkpoint_dir)
    )

    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0], occurrence=0, index=0, status="success"
            ),
            checkpoint_item_for_record(
                rows[1], occurrence=0, index=1, status="pending"
            ),
            checkpoint_item_for_record(
                rows[2], occurrence=0, index=2, status="pending"
            ),
        ],
        checkpoint_dir=str(checkpoint_dir),
    )

    client = _FakeClient()
    _run(
        client,
        input_path=input_path,
        output_path=output_path,
        checkpoint_dir=str(checkpoint_dir),
        resume=True,
    )

    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[0]["output_text"] == "cached-a"
    assert rows_by_index[1]["output_text"] == "out:b"
    assert rows_by_index[2]["output_text"] == "out:c"

    state = load_resume_state(checkpoint_path)
    assert {row["status"] for row in state.values()} == {"success"}


def test_resume_requires_same_checkpoint_dir_override(tmp_path: Path) -> None:
    row = {"prompt": "a"}
    input_path = _write_input(tmp_path, [row])
    output_path = tmp_path / "out.jsonl"
    checkpoint_dir = tmp_path / "scratch-checkpoints"

    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [checkpoint_item_for_record(row, occurrence=0, index=0, status="success")],
        checkpoint_dir=str(checkpoint_dir),
    )

    client = _FakeClient()
    with pytest.raises(ValueError, match="requires checkpoint file"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_builtin_row_conventions(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"prompt": "hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
        {"responses_input": [{"role": "user", "content": "hey"}]},
        {"prompt": ""},
        {"messages": []},
        {"responses_input": []},
    ]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    assert client.inputs == [
        "hello",
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "hey"}],
        "",
        [],
        [],
    ]
    out_rows = _read_output(output_path)
    assert len(out_rows) == 6
    assert all(r["error"] is None for r in out_rows)


def test_non_object_source_rows_become_error_rows_without_aborting_siblings(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "good"}),
                json.dumps([]),
                json.dumps({"prompt": "later"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    assert client.inputs == ["good", "later"]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert len(rows_by_index) == 3
    assert rows_by_index[0]["error"] is None
    assert rows_by_index[1]["output_text"] is None
    assert "JSON objects" in rows_by_index[1]["error"]
    assert rows_by_index[2]["error"] is None


def test_mapper_import_and_metadata(tmp_path: Path) -> None:
    fake_mod = types.ModuleType("_test_wf_mapper_mod")

    def my_mapper(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": record["prompt"].upper(),
            "metadata": {"original": record["prompt"]},
        }

    fake_mod.my_mapper = my_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_mapper_mod"] = fake_mod
    try:
        rows = [{"prompt": "hello"}]
        input_path = _write_input(tmp_path, rows)
        output_path = tmp_path / "out.jsonl"
        client = _FakeClient()

        _run(
            client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_mapper_mod:my_mapper",
        )

        assert client.inputs == ["HELLO"]
        out_rows = _read_output(output_path)
        assert out_rows[0]["metadata"] == {"original": "hello"}
        assert out_rows[0]["error"] is None
    finally:
        del sys.modules["_test_wf_mapper_mod"]


def test_mapper_waiting_on_generation_progress_does_not_block_loop(
    tmp_path: Path,
) -> None:
    fake_mod = types.ModuleType("_test_wf_waiting_mapper_mod")
    release_event = threading.Event()
    client = _MapperSignalFakeClient(release_event=release_event)

    def waiting_mapper(record: dict[str, Any]) -> dict[str, Any]:
        if record["prompt"] == "second" and not release_event.wait(timeout=1.0):
            raise RuntimeError("mapper never observed first completion")
        return {"input": record["prompt"]}

    fake_mod.waiting_mapper = waiting_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_waiting_mapper_mod"] = fake_mod
    try:
        rows = [{"prompt": "first"}, {"prompt": "second"}]
        input_path = _write_input(tmp_path, rows)
        output_path = tmp_path / "out.jsonl"

        _run(
            client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_waiting_mapper_mod:waiting_mapper",
            window_size=2,
        )

        rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
        assert rows_by_index[0]["output_text"] == "out:first"
        assert rows_by_index[0]["error"] is None
        assert rows_by_index[1]["output_text"] == "out:second"
        assert rows_by_index[1]["error"] is None
    finally:
        del sys.modules["_test_wf_waiting_mapper_mod"]


def test_mapper_ignores_extra_keys(tmp_path: Path) -> None:
    fake_mod = types.ModuleType("_test_wf_extra_mod")

    def my_mapper(record: dict[str, Any]) -> dict[str, Any]:
        return {"input": record["prompt"], "metadata": None, "extra_ignored_key": 42}

    fake_mod.my_mapper = my_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_extra_mod"] = fake_mod
    try:
        rows = [{"prompt": "hello"}]
        input_path = _write_input(tmp_path, rows)
        output_path = tmp_path / "out.jsonl"
        client = _FakeClient()

        _run(
            client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_extra_mod:my_mapper",
        )

        out_rows = _read_output(output_path)
        assert len(out_rows) == 1
        assert out_rows[0]["error"] is None
    finally:
        del sys.modules["_test_wf_extra_mod"]


def test_mapper_validation_failure_becomes_error_row(tmp_path: Path) -> None:
    fake_mod = types.ModuleType("_test_wf_bad_mapper_mod")

    def bad_mapper(record: dict[str, Any]) -> dict[str, Any]:
        return {"no_input_key": "oops"}  # missing required "input"

    fake_mod.bad_mapper = bad_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_bad_mapper_mod"] = fake_mod
    try:
        rows = [{"prompt": "first"}, {"prompt": "second"}]
        input_path = _write_input(tmp_path, rows)
        output_path = tmp_path / "out.jsonl"
        client = _FakeClient()

        _run(
            client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_bad_mapper_mod:bad_mapper",
        )

        # Both rows become error rows; no generation request is started.
        assert client.inputs == []
        out_rows = _read_output(output_path)
        assert len(out_rows) == 2
        assert all(r["output_text"] is None for r in out_rows)
        assert all(r["error"] is not None for r in out_rows)
    finally:
        del sys.modules["_test_wf_bad_mapper_mod"]


def test_builtin_mapping_failure_becomes_error_row(tmp_path: Path) -> None:
    rows = [{"text": "wrong_field"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    assert client.inputs == []
    out_rows = _read_output(output_path)
    assert len(out_rows) == 1
    assert out_rows[0]["output_text"] is None
    assert out_rows[0]["error"] is not None
    assert (
        "require" in out_rows[0]["error"].lower()
        or "prompt" in out_rows[0]["error"].lower()
    )


def test_malformed_json_line_becomes_error_row(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "not valid json\n" + json.dumps({"prompt": "good"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    out_rows = _read_output(output_path)
    assert len(out_rows) == 2

    error_rows = [r for r in out_rows if r["error"] is not None]
    success_rows = [r for r in out_rows if r["error"] is None]
    assert len(error_rows) == 1
    assert len(success_rows) == 1
    assert success_rows[0]["output_text"] is not None


def test_provider_failure_becomes_error_row_and_settles_checkpoint(
    tmp_path: Path,
) -> None:
    rows = [{"prompt": "good"}, {"prompt": "bad"}, {"prompt": "also-good"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))
    client = _SelectiveFailingFakeClient(failing_inputs={"bad"})

    _run(client, input_path=input_path, output_path=output_path, window_size=2)

    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[0]["output_text"] == "out:good"
    assert rows_by_index[0]["error"] is None
    assert rows_by_index[1]["output_text"] is None
    assert rows_by_index[1]["error"] == "boom:bad"
    assert rows_by_index[2]["output_text"] == "out:also-good"
    assert rows_by_index[2]["error"] is None

    state = load_resume_state(checkpoint_path)
    assert {row["status"] for row in state.values()} == {"success", "error"}
    failed_item = next(row for row in state.values() if row["_index"] == 1)
    assert failed_item["status"] == "error"
    assert failed_item["error"] == "boom:bad"


def test_fresh_run_bootstrap_failure_preserves_existing_artifacts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))
    old_output = json.dumps({"_index": 99, "output_text": "keep-me"}) + "\n"
    output_path.write_text(old_output, encoding="utf-8")
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                {"prompt": "old"},
                occurrence=0,
                index=99,
                status="success",
            )
        ],
    )
    old_checkpoint = checkpoint_path.read_bytes()
    client = _FakeClient()

    with pytest.raises(FileNotFoundError):
        _run(client, input_path=input_path, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == old_output
    assert checkpoint_path.read_bytes() == old_checkpoint
    assert client.inputs == []


def test_resume_skips_items_from_checkpoint_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        "\n".join(
            json.dumps(
                {"_index": index, "output_text": f"cached-{rows[index]['prompt']}"}
            )
            for index in (0, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                row,
                occurrence=0,
                index=source_index,
                status=status,
            )
            for source_index, row, status in [
                (0, rows[0], "success"),
                (1, rows[1], "pending"),
                (2, rows[2], "success"),
            ]
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == ["b"]


def test_resume_requires_state_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="checkpoint file"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_rejects_missing_output_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    write_checkpoint_db(
        output_path,
        [checkpoint_item_for_record(rows[0], occurrence=0, index=0, status="success")],
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="requires output file"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_rejects_truncated_output_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                row,
                occurrence=0,
                index=index,
                status="success",
            )
            for index, row in enumerate(rows)
        ],
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="missing settled checkpoint rows"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_output_bitmap_handles_sparse_high_indexes(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"_index": 0, "output_text": "cached-a"}),
                json.dumps({"_index": 1000, "output_text": "cached-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0], occurrence=0, index=0, status="success"
            ),
            checkpoint_item_for_record(
                rows[1], occurrence=0, index=1000, status="success"
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == []


def test_completed_run_updates_checkpoint_rows_in_place(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    checkpoint_path = _checkpoint_path_for(str(output_path))
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    state = load_resume_state(checkpoint_path)
    assert len(state) == 3
    assert {row["status"] for row in state.values()} == {"success"}
    assert _load_mapping_fingerprint(checkpoint_path) == _compute_mapping_fingerprint(
        mapper_spec=None,
        mapper=None,
    )


def test_resume_tracks_duplicate_records_independently(tmp_path: Path) -> None:
    rows = [{"prompt": "dup"}, {"prompt": "dup"}, {"prompt": "tail"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-dup"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0],
                occurrence=0,
                index=0,
                status="success",
            ),
            checkpoint_item_for_record(
                rows[1],
                occurrence=1,
                index=1,
                status="pending",
            ),
            checkpoint_item_for_record(
                rows[2],
                occurrence=0,
                index=2,
                status="pending",
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == ["dup", "tail"]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:dup"
    assert rows_by_index[2]["output_text"] == "out:tail"


def test_resume_rejects_changed_mapper_implementation(tmp_path: Path) -> None:
    rows = [{"prompt": "first"}, {"prompt": "second"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    module_name = "_test_wf_changed_mapper_mod"
    module_path = tmp_path / f"{module_name}.py"

    def write_mapper_module(prefix: str) -> None:
        module_path.write_text(
            "\n".join(
                [
                    f'HELPER_PREFIX = "{prefix}"',
                    "",
                    "def helper(text):",
                    '    return f"{HELPER_PREFIX}:{text}"',
                    "",
                    "def my_mapper(record):",
                    '    return {"input": helper(record["prompt"])}',
                    "",
                ]
            ),
            encoding="utf-8",
        )

    sys.path.insert(0, str(tmp_path))
    try:
        write_mapper_module("v1")
        sys.modules.pop(module_name, None)
        initial_client = _FakeClient()
        _run(
            initial_client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec=f"{module_name}:my_mapper",
        )

        checkpoint_path = _checkpoint_path_for(str(output_path))
        assert _load_mapping_fingerprint(checkpoint_path)

        write_mapper_module("v2")
        sys.modules.pop(module_name, None)

        resume_client = _FakeClient()
        with pytest.raises(ValueError, match="Resume mapping does not match"):
            _run(
                resume_client,
                input_path=input_path,
                output_path=output_path,
                mapper_spec=f"{module_name}:my_mapper",
                resume=True,
            )
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(tmp_path))


def test_resume_allows_empty_custom_mapper_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"

    fake_mod = types.ModuleType("_test_wf_empty_mapper_mod")

    def my_mapper(record: dict[str, Any]) -> dict[str, Any]:
        return {"input": record["prompt"].upper()}

    my_mapper.__module__ = "_test_wf_empty_mapper_mod"
    fake_mod.my_mapper = my_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_empty_mapper_mod"] = fake_mod
    try:
        initial_client = _FakeClient()
        _run(
            initial_client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_empty_mapper_mod:my_mapper",
        )

        resume_client = _FakeClient()
        _run(
            resume_client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_empty_mapper_mod:my_mapper",
            resume=True,
        )

        checkpoint_path = _checkpoint_path_for(str(output_path))
        assert output_path.read_text(encoding="utf-8") == ""
        assert load_resume_state(checkpoint_path) == {}
        assert resume_client.inputs == []
    finally:
        del sys.modules["_test_wf_empty_mapper_mod"]


def test_resume_preserves_original_indexes_after_reorder(tmp_path: Path) -> None:
    original_rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    reordered_rows = [original_rows[2], original_rows[0], original_rows[1]]
    input_path = _write_input(tmp_path, reordered_rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                original_rows[0],
                occurrence=0,
                index=0,
                status="success",
            ),
            checkpoint_item_for_record(
                original_rows[1],
                occurrence=0,
                index=1,
                status="pending",
            ),
            checkpoint_item_for_record(
                original_rows[2],
                occurrence=0,
                index=2,
                status="pending",
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == ["c", "b"]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:b"
    assert rows_by_index[2]["output_text"] == "out:c"


@pytest.mark.parametrize(
    ("checkpoint_rows", "input_rows"),
    [
        ([{"prompt": "a"}, {"prompt": "a"}], [{"prompt": "a"}]),
        ([{"prompt": "a"}], [{"prompt": "a"}, {"prompt": "a"}]),
    ],
)
def test_resume_rejects_mismatched_occurrences(
    tmp_path: Path,
    checkpoint_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> None:
    input_path = _write_input(tmp_path, input_rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("", encoding="utf-8")
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                row,
                occurrence=occurrence,
                index=occurrence,
                status="pending",
            )
            for occurrence, row in enumerate(checkpoint_rows)
        ],
    )

    client = _FakeClient()
    with pytest.raises(ValueError, match="does not match the checkpoint file"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_ignores_pending_rows_when_validating_output_rows(
    tmp_path: Path,
) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0],
                occurrence=0,
                index=0,
                status="success",
            ),
            checkpoint_item_for_record(
                rows[1],
                occurrence=0,
                index=1,
                status="pending",
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:b"


def test_resume_tracks_duplicate_parse_errors_by_occurrence(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    bad_line = "not valid json"
    input_path.write_text(
        json.dumps({"prompt": "good"}) + "\n" + bad_line + "\n" + bad_line + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": None, "error": "cached-parse-error"})
        + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_parse_error(
                bad_line,
                occurrence=0,
                index=0,
                status="error",
                error="cached-parse-error",
            ),
            checkpoint_item_for_record(
                {"prompt": "good"},
                occurrence=0,
                index=1,
                status="pending",
            ),
            checkpoint_item_for_parse_error(
                bad_line,
                occurrence=1,
                index=2,
                status="pending",
            ),
        ],
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.inputs == ["good"]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:good"
    assert rows_by_index[2]["output_text"] is None
    assert rows_by_index[2]["error"] is not None


def test_stdout_path_creates_no_checkpoint_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{"prompt": "hello"}]
    input_path = _write_input(tmp_path, rows)
    client = _FakeClient()
    monkeypatch.setattr(ResumePlanner, "_temp_dir", lambda: tmp_path)

    _run(client, input_path=input_path, output_path=None)

    assert list(tmp_path.glob("*.checkpoint.sqlite")) == []
    assert not list(tmp_path.glob(".infermesh-resume-plan.*.sqlite"))

    out = capsys.readouterr().out
    out_rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(out_rows) == 1
    assert out_rows[0]["error"] is None


def test_invalid_metadata_becomes_error_row_without_aborting_siblings(
    tmp_path: Path,
) -> None:
    fake_mod = types.ModuleType("_test_wf_metadata_mod")

    def my_mapper(record: dict[str, Any]) -> dict[str, Any]:
        if record["prompt"] == "bad":
            return {
                "input": record["prompt"],
                "metadata": {"created_at": datetime.now()},
            }
        return {
            "input": record["prompt"],
            "metadata": {"kind": "ok"},
        }

    fake_mod.my_mapper = my_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_metadata_mod"] = fake_mod
    try:
        rows = [{"prompt": "bad"}, {"prompt": "good"}]
        input_path = _write_input(tmp_path, rows)
        output_path = tmp_path / "out.jsonl"
        client = _FakeClient()

        _run(
            client,
            input_path=input_path,
            output_path=output_path,
            mapper_spec="_test_wf_metadata_mod:my_mapper",
        )

        assert client.inputs == ["good"]
        out_rows = _read_output(output_path)
        assert len(out_rows) == 2
        rows_by_index = {row["_index"]: row for row in out_rows}
        assert "json serializable" in rows_by_index[0]["error"].lower()
        assert rows_by_index[1]["metadata"] == {"kind": "ok"}
        assert rows_by_index[1]["error"] is None
    finally:
        del sys.modules["_test_wf_metadata_mod"]


def test_file_backed_generate_materialises_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When output_jsonl is set but no input_jsonl/prompt, stdin is spooled to a temp file."""
    stdin_data = json.dumps({"prompt": "from-stdin"}) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_data))
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=None, output_path=output_path)

    out_rows = _read_output(output_path)
    assert len(out_rows) == 1
    assert out_rows[0]["output_text"] == "out:from-stdin"
    assert out_rows[0]["error"] is None
    # No stray temp artefacts should remain in the working directory
    assert not list(tmp_path.glob("*.tmp"))


def test_persistence_sink_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = checkpoint_module._mark_checkpoint_item_settled
    call_count = 0

    def failing_mark(connection, checkpoint_key, *, status, error):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected sink failure")
        original(connection, checkpoint_key, status=status, error=error)

    monkeypatch.setattr(
        checkpoint_module, "_mark_checkpoint_item_settled", failing_mark
    )

    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    with pytest.raises(RuntimeError, match="injected sink failure"):
        _run(client, input_path=input_path, output_path=output_path)


def test_resume_planner_temp_db_is_removed_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    write_checkpoint_db(
        output_path,
        [
            checkpoint_item_for_record(
                rows[0], occurrence=0, index=0, status="success"
            ),
            checkpoint_item_for_record(
                rows[1], occurrence=0, index=1, status="pending"
            ),
        ],
    )

    def failing_mark(connection, checkpoint_key, *, status, error):
        raise RuntimeError("injected sink failure")

    monkeypatch.setattr(
        checkpoint_module, "_mark_checkpoint_item_settled", failing_mark
    )
    monkeypatch.setattr(ResumePlanner, "_temp_dir", lambda: tmp_path)

    client = _FakeClient()
    with pytest.raises(RuntimeError, match="injected sink failure"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert not list(tmp_path.glob(".infermesh-resume-plan.*.sqlite"))
