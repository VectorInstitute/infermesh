"""Unit tests for the workflow engine (infermesh._workflow)."""

from __future__ import annotations

import io
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from infermesh._workflow import (
    _compute_parse_error_fingerprint,
    _compute_record_fingerprint,
    _load_resume_state,
    _state_path_for,
    run_generate_workflow,
)
from tests.fakes import state_row_for_record

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
    """Synchronous fake that records generate_batch calls and fires on_result."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def generate_batch(self, input_batch: list[Any], **kwargs: Any) -> None:
        self.batches.append(list(input_batch))
        on_result = kwargs.get("on_result")
        if on_result is not None:
            for idx, item in enumerate(input_batch):
                on_result(idx, _FakeResult(output_text=f"out:{item}"), None)

    def close(self) -> None:
        pass


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
    prompt: str | None = None,
    mapper_spec: str | None = None,
    resume: bool = False,
    window_size: int = 128,
    parse_json: bool = False,
) -> None:
    run_generate_workflow(
        client,  # type: ignore[arg-type]
        prompt=prompt,
        input_jsonl=str(input_path) if input_path is not None else None,
        output_jsonl=str(output_path) if output_path is not None else None,
        mapper_spec=mapper_spec,
        resume=resume,
        endpoint="chat_completion",
        window_size=window_size,
        parse_json=parse_json,
    )


def _state_row_for_parse_error(
    raw_line: str,
    *,
    occurrence: int,
    index: int,
    status: str,
    error: str | None = None,
    mapping_fingerprint: str | None = None,
) -> dict[str, Any]:
    row = {
        "record_fingerprint": _compute_parse_error_fingerprint(raw_line),
        "occurrence": occurrence,
        "_index": index,
        "status": status,
        "error": error,
    }
    if mapping_fingerprint is not None:
        row["mapping_fingerprint"] = mapping_fingerprint
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_file_backed_generate_streams_input(tmp_path: Path) -> None:
    rows = [{"prompt": f"p{i}"} for i in range(7)]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path, window_size=3)

    # 7 items, window_size=3 → batches of [3, 3, 1]
    assert len(client.batches) == 3
    assert all(len(b) <= 3 for b in client.batches)
    assert sum(len(b) for b in client.batches) == 7

    out_rows = _read_output(output_path)
    assert len(out_rows) == 7


def test_builtin_row_conventions(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"prompt": "hello"},
        {"messages": [{"role": "user", "content": "hi"}]},
        {"responses_input": [{"role": "user", "content": "hey"}]},
    ]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=output_path)

    assert len(client.batches) == 1
    assert len(client.batches[0]) == 3
    out_rows = _read_output(output_path)
    assert len(out_rows) == 3
    assert all(r["error"] is None for r in out_rows)


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

        assert client.batches == [["HELLO"]]
        out_rows = _read_output(output_path)
        assert out_rows[0]["metadata"] == {"original": "hello"}
        assert out_rows[0]["error"] is None
    finally:
        del sys.modules["_test_wf_mapper_mod"]


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

        # Both rows become error rows; generate_batch is never called
        assert client.batches == []
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

    assert client.batches == []
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


def test_fresh_run_bootstrap_failure_preserves_existing_artifacts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    old_output = json.dumps({"_index": 99, "output_text": "keep-me"}) + "\n"
    old_state = (
        json.dumps(
            state_row_for_record(
                {"prompt": "old"},
                occurrence=0,
                index=99,
                status="success",
            )
        )
        + "\n"
    )
    output_path.write_text(old_output, encoding="utf-8")
    state_path.write_text(old_state, encoding="utf-8")
    client = _FakeClient()

    with pytest.raises(FileNotFoundError):
        _run(client, input_path=input_path, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == old_output
    assert state_path.read_text(encoding="utf-8") == old_state
    assert client.batches == []
def test_resume_skips_items_from_state_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
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

    # Pre-populate state file for rows 0 (a) and 2 (c)
    with state_path.open("w", encoding="utf-8") as fh:
        for source_index, row, status in [
            (0, rows[0], "success"),
            (1, rows[1], "pending"),
            (2, rows[2], "success"),
        ]:
            fh.write(
                json.dumps(
                    state_row_for_record(
                        row,
                        occurrence=0,
                        index=source_index,
                        status=status,
                    )
                )
                + "\n"
            )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    all_inputs = [item for batch in client.batches for item in batch]
    assert all_inputs == ["b"]


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
    state_path = _state_path_for(str(output_path))
    state_path.write_text(
        json.dumps(
            state_row_for_record(rows[0], occurrence=0, index=0, status="success")
        )
        + "\n",
        encoding="utf-8",
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="requires output file"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_rejects_truncated_output_file(tmp_path: Path) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            json.dumps(
                state_row_for_record(
                    row,
                    occurrence=0,
                    index=index,
                    status="success",
                )
            )
            for index, row in enumerate(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    client = _FakeClient()

    with pytest.raises(ValueError, match="missing settled checkpoint rows"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_keeps_latest_state_per_occurrence(tmp_path: Path) -> None:
    record = {"prompt": "hello"}
    checkpoint_key = (_compute_record_fingerprint(record), 0)
    state_path = tmp_path / "out.state.jsonl"

    with state_path.open("w", encoding="utf-8") as fh:
        # First entry: error; second entry: success (last-write-wins)
        fh.write(
            json.dumps(
                state_row_for_record(
                    record,
                    occurrence=0,
                    index=0,
                    status="error",
                    error="oops",
                )
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                state_row_for_record(
                    record,
                    occurrence=0,
                    index=0,
                    status="success",
                )
            )
            + "\n"
        )

    state = _load_resume_state(state_path)
    assert state[checkpoint_key]["status"] == "success"


def test_resume_tracks_duplicate_records_independently(tmp_path: Path) -> None:
    rows = [{"prompt": "dup"}, {"prompt": "dup"}, {"prompt": "tail"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-dup"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    state_row_for_record(
                        rows[0],
                        occurrence=0,
                        index=0,
                        status="success",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        rows[1],
                        occurrence=1,
                        index=1,
                        status="pending",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        rows[2],
                        occurrence=0,
                        index=2,
                        status="pending",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    all_inputs = [item for batch in client.batches for item in batch]
    assert all_inputs == ["dup", "tail"]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:dup"
    assert rows_by_index[2]["output_text"] == "out:tail"


def test_resume_legacy_manifest_without_mapping_fingerprint_still_works(
    tmp_path: Path,
) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    state_row_for_record(
                        rows[0],
                        occurrence=0,
                        index=0,
                        status="success",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        rows[1],
                        occurrence=0,
                        index=1,
                        status="pending",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.batches == [["b"]]


def test_resume_rejects_legacy_manifest_with_custom_mapper(tmp_path: Path) -> None:
    rows = [{"prompt": "first"}, {"prompt": "second"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-first"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    state_row_for_record(
                        rows[0],
                        occurrence=0,
                        index=0,
                        status="success",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        rows[1],
                        occurrence=0,
                        index=1,
                        status="pending",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_mod = types.ModuleType("_test_wf_legacy_mapper_mod")

    def my_mapper(record: dict[str, Any]) -> dict[str, Any]:
        return {"input": record["prompt"].upper()}

    my_mapper.__module__ = "_test_wf_legacy_mapper_mod"
    fake_mod.my_mapper = my_mapper  # type: ignore[attr-defined]
    sys.modules["_test_wf_legacy_mapper_mod"] = fake_mod
    try:
        client = _FakeClient()
        with pytest.raises(ValueError, match="predates mapper fingerprints"):
            _run(
                client,
                input_path=input_path,
                output_path=output_path,
                mapper_spec="_test_wf_legacy_mapper_mod:my_mapper",
                resume=True,
            )
    finally:
        del sys.modules["_test_wf_legacy_mapper_mod"]


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

        state_path = _state_path_for(str(output_path))
        state_rows = [
            json.loads(line)
            for line in state_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert all("mapping_fingerprint" in row for row in state_rows)

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

        state_path = _state_path_for(str(output_path))
        assert output_path.read_text(encoding="utf-8") == ""
        assert state_path.read_text(encoding="utf-8") == ""
        assert resume_client.batches == []
    finally:
        del sys.modules["_test_wf_empty_mapper_mod"]


def test_resume_preserves_original_indexes_after_reorder(tmp_path: Path) -> None:
    original_rows = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    reordered_rows = [original_rows[2], original_rows[0], original_rows[1]]
    input_path = _write_input(tmp_path, reordered_rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    state_row_for_record(
                        original_rows[0],
                        occurrence=0,
                        index=0,
                        status="success",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        original_rows[1],
                        occurrence=0,
                        index=1,
                        status="pending",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        original_rows[2],
                        occurrence=0,
                        index=2,
                        status="pending",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    all_inputs = [item for batch in client.batches for item in batch]
    assert all_inputs == ["c", "b"]
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
    state_path = _state_path_for(str(output_path))
    output_path.write_text("", encoding="utf-8")
    state_path.write_text(
        "\n".join(
            json.dumps(
                state_row_for_record(
                    row,
                    occurrence=occurrence,
                    index=occurrence,
                    status="pending",
                )
            )
            for occurrence, row in enumerate(checkpoint_rows)
        )
        + "\n",
        encoding="utf-8",
    )

    client = _FakeClient()
    with pytest.raises(ValueError, match="does not match the checkpoint manifest"):
        _run(client, input_path=input_path, output_path=output_path, resume=True)


def test_resume_ignores_pending_rows_when_validating_output_rows(
    tmp_path: Path,
) -> None:
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "out.jsonl"
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    state_row_for_record(
                        rows[0], occurrence=0, index=0, status="success"
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        rows[1], occurrence=0, index=1, status="pending"
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
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
    state_path = _state_path_for(str(output_path))
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": None, "error": "cached-parse-error"})
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        "\n".join(
            [
                json.dumps(
                    _state_row_for_parse_error(
                        bad_line,
                        occurrence=0,
                        index=0,
                        status="error",
                        error="cached-parse-error",
                    )
                ),
                json.dumps(
                    state_row_for_record(
                        {"prompt": "good"},
                        occurrence=0,
                        index=1,
                        status="pending",
                    )
                ),
                json.dumps(
                    _state_row_for_parse_error(
                        bad_line,
                        occurrence=1,
                        index=2,
                        status="pending",
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    client = _FakeClient()
    _run(client, input_path=input_path, output_path=output_path, resume=True)

    assert client.batches == [["good"]]
    rows_by_index = {row["_index"]: row for row in _read_output(output_path)}
    assert rows_by_index[1]["output_text"] == "out:good"
    assert rows_by_index[2]["output_text"] is None
    assert rows_by_index[2]["error"] is not None


def test_stdout_path_creates_no_state_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{"prompt": "hello"}]
    input_path = _write_input(tmp_path, rows)
    client = _FakeClient()

    _run(client, input_path=input_path, output_path=None)

    assert list(tmp_path.glob("*.state.jsonl")) == []

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

        assert client.batches == [["good"]]
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
