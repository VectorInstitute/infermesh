from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from infermesh import cli
from infermesh._cli_support import _client_config_from_args
from infermesh._workflow.checkpoint import _checkpoint_path_for
from tests.fakes import FakeCLIClient, checkpoint_item_for_record, write_checkpoint_db

_BASE_ARGS = [
    "generate",
    "--model",
    "openai/test",
    "--api-base",
    "http://localhost:8000/v1",
]


@pytest.mark.usefixtures("fake_client_builder")
def test_embed_with_text(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "embed",
            "--model",
            "embed-model",
            "--api-base",
            "http://localhost:8000/v1",
            "--text",
            "hello",
            "--no-vectors",
        ]
    )
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out.strip())
    assert output["dimensions"] == 2
    assert output["embedding"] is None


def test_repeated_api_base_builds_router_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client_ctor(**kwargs: Any) -> FakeCLIClient:
        captured.update(kwargs)
        return FakeCLIClient(**kwargs)

    monkeypatch.setattr(cli, "LMClient", fake_client_ctor)
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--model",
            "gpt-4",
            "--api-base",
            "http://a",
            "--api-base",
            "http://b",
            "--prompt",
            "hello",
        ]
    )
    cli._build_client(_client_config_from_args(args))
    assert captured["deployments"] is not None
    assert len(captured["deployments"]) == 2


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_rejects_malformed_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps({"text": "wrong"}) + "\n", encoding="utf-8")
    exit_code = cli.main(
        [
            "generate",
            "--model",
            "openai/test",
            "--api-base",
            "http://localhost:8000/v1",
            "--input-jsonl",
            str(input_path),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["_index"] == 0
    assert rows[0]["output_text"] is None
    assert "Generation rows require" in rows[0]["error"]


def test_generate_surfaces_provider_failure_as_error_row(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingClient(FakeCLIClient):
        async def agenerate(self, input_data: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_build_client", lambda *a, **k: FailingClient())
    exit_code = cli.main([*_BASE_ARGS, "--prompt", "hello"])

    assert exit_code == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["error"] == "boom"
    assert row["_index"] == 0


def test_generate_surfaces_workflow_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_build_client", lambda *a, **k: FakeCLIClient())

    def blow_up(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(cli, "run_generate_workflow", blow_up)

    exit_code = cli.main([*_BASE_ARGS, "--prompt", "hello"])

    assert exit_code == 1
    assert "error: disk full" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_index_field_always_present(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
        ]
    )
    assert exit_code == 0
    rows = sorted(
        (
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ),
        key=lambda row: row["_index"],
    )
    assert rows[0]["_index"] == 0
    assert rows[1]["_index"] == 1


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_no_resume_overwrites_existing_file(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    output_path.write_text(
        json.dumps({"_index": 99, "output_text": "stale"}) + "\n",
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
        ]
    )
    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["_index"] == 0
    assert rows[0]["output_text"] == "generated:hello"


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_uses_checkpoint_dir_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    checkpoint_dir = tmp_path / "env-checkpoints"
    input_path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("INFERMESH_CHECKPOINT_DIR", str(checkpoint_dir))

    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
        ]
    )

    checkpoint_path = _checkpoint_path_for(
        str(output_path), checkpoint_dir=str(checkpoint_dir)
    )
    assert exit_code == 0
    assert checkpoint_path.exists()
    assert not _checkpoint_path_for(str(output_path)).exists()


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_checkpoint_dir_flag_overrides_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    env_checkpoint_dir = tmp_path / "env-checkpoints"
    arg_checkpoint_dir = tmp_path / "arg-checkpoints"
    input_path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("INFERMESH_CHECKPOINT_DIR", str(env_checkpoint_dir))

    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--checkpoint-dir",
            str(arg_checkpoint_dir),
        ]
    )

    env_checkpoint_path = _checkpoint_path_for(
        str(output_path), checkpoint_dir=str(env_checkpoint_dir)
    )
    arg_checkpoint_path = _checkpoint_path_for(
        str(output_path), checkpoint_dir=str(arg_checkpoint_dir)
    )
    assert exit_code == 0
    assert arg_checkpoint_path.exists()
    assert not env_checkpoint_path.exists()


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_rejects_identical_input_and_output_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    same_path = tmp_path / "same.jsonl"
    same_path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")

    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(same_path),
            "--output-jsonl",
            str(same_path),
        ]
    )

    assert exit_code == 1
    assert "must be different files" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_resume_nothing_to_do(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    row = {"prompt": "a"}
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "done"}) + "\n", encoding="utf-8"
    )
    checkpoint_path = write_checkpoint_db(
        output_path,
        [checkpoint_item_for_record(row, occurrence=0, index=0, status="success")],
    )
    assert checkpoint_path == _checkpoint_path_for(str(output_path))
    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--resume",
        ]
    )

    assert exit_code == 0
    assert "Nothing to do" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_client_builder")
def test_generate_resume_requires_state_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should surface strict resume validation failures cleanly."""
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        json.dumps({"prompt": "a"}) + "\n" + json.dumps({"prompt": "b"}) + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"}) + "\n",
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            *_BASE_ARGS,
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--resume",
        ]
    )
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "checkpoint file" in stderr.lower()


def test_generate_resume_requires_output_jsonl() -> None:
    exit_code = cli.main([*_BASE_ARGS, "--prompt", "hello", "--resume"])
    assert exit_code == 1


@pytest.mark.usefixtures("fake_client_builder")
def test_env_file_loads_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-secret-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cli.main(
        [
            "generate",
            "--model",
            "openai/test",
            "--api-base",
            "http://localhost:8000/v1",
            "--env-file",
            str(env_file),
            "--prompt",
            "hello",
        ]
    )
    assert os.environ.get("OPENAI_API_KEY") == "test-secret-key"


def test_deployments_toml_rejects_top_level_api_key(tmp_path: Path) -> None:
    deployments_toml = tmp_path / "deployments.toml"
    deployments_toml.write_text(
        (
            "[deployments.replica]\n"
            'model = "openai/gpt-4o"\n'
            'api_base = "https://api.openai.com/v1"\n'
            'api_key = "plaintext-secret"\n'
        ),
        encoding="utf-8",
    )
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--model",
            "gpt-4o",
            "--deployments-toml",
            str(deployments_toml),
            "--prompt",
            "hello",
        ]
    )
    with pytest.raises(ValueError, match="plaintext secret") as excinfo:
        cli._build_client(_client_config_from_args(args))
    assert "deployments.replica.api_key" in str(excinfo.value)
    assert "--env-file" in str(excinfo.value)


def test_deployments_toml_rejects_nested_extra_kwargs_api_key(tmp_path: Path) -> None:
    deployments_toml = tmp_path / "deployments.toml"
    deployments_toml.write_text(
        (
            "[deployments.replica]\n"
            'model = "openai/gpt-4o"\n'
            'api_base = "https://api.openai.com/v1"\n'
            "\n"
            "[deployments.replica.extra_kwargs]\n"
            'api_key = "plaintext-secret"\n'
        ),
        encoding="utf-8",
    )
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--model",
            "gpt-4o",
            "--deployments-toml",
            str(deployments_toml),
            "--prompt",
            "hello",
        ]
    )
    with pytest.raises(ValueError, match="plaintext secret") as excinfo:
        cli._build_client(_client_config_from_args(args))
    assert "deployments.replica.extra_kwargs.api_key" in str(excinfo.value)
    assert "--env-file" in str(excinfo.value)


def test_deployments_toml_rejects_missing_deployments_table(tmp_path: Path) -> None:
    deployments_toml = tmp_path / "deployments.toml"
    deployments_toml.write_text('title = "no deployments table"\n', encoding="utf-8")
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--model",
            "gpt-4o",
            "--deployments-toml",
            str(deployments_toml),
            "--prompt",
            "hello",
        ]
    )
    with pytest.raises(ValueError, match="missing a \\[deployments\\] table"):
        cli._build_client(_client_config_from_args(args))


def test_handle_generate_surfaces_build_client_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_build_client ValueError must produce a clean error message, not a traceback."""
    deployments_toml = tmp_path / "deployments.toml"
    deployments_toml.write_text(
        (
            "[deployments.replica]\n"
            'model = "openai/gpt-4o"\n'
            'api_base = "https://api.openai.com/v1"\n'
            'api_key = "plaintext-secret"\n'
        ),
        encoding="utf-8",
    )
    exit_code = cli.main(
        [
            "generate",
            "--model",
            "gpt-4o",
            "--deployments-toml",
            str(deployments_toml),
            "--prompt",
            "hello",
        ]
    )
    assert exit_code == 1
    assert "plaintext secret" in capsys.readouterr().err
