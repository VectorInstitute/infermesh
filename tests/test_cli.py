from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from lm_client import cli
from tests.fakes import FakeCLIClient

_BASE_ARGS = [
    "generate",
    "--model",
    "openai/test",
    "--api-base",
    "http://localhost:8000/v1",
]


def test_embed_with_text(
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
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
    cli._build_client(cli._client_config_from_args(args))
    assert captured["deployments"] is not None
    assert len(captured["deployments"]) == 2


def test_generate_rejects_malformed_rows(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps({"text": "wrong"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Generation rows require"):
        cli.main(
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


def test_generate_propagates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient(FakeCLIClient):
        def generate_batch(self, input_batch: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_build_client", lambda *a, **k: FailingClient())
    with pytest.raises(RuntimeError, match="boom"):
        cli.main([*_BASE_ARGS, "--prompt", "hello"])


def test_generate_index_field_always_present(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
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
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["_index"] == 0
    assert rows[1]["_index"] == 1


def test_generate_resume_skips_completed_rows(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ["a", "b", "c"]) + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "cached-a"})
        + "\n"
        + json.dumps({"_index": 2, "output_text": "cached-c"})
        + "\n",
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
    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    new_rows = [row for row in rows if row.get("_index") == 1]
    assert len(new_rows) == 1
    assert new_rows[0]["output_text"] == "generated:b"


def test_generate_resume_appends_to_existing_file(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
) -> None:
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
    assert exit_code == 0
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["_index"] for row in rows} == {0, 1}
    assert any(row["_index"] == 0 and row["output_text"] == "cached-a" for row in rows)


def test_generate_no_resume_overwrites_existing_file(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
) -> None:
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


def test_generate_resume_nothing_to_do(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    input_path.write_text(json.dumps({"prompt": "a"}) + "\n", encoding="utf-8")
    output_path.write_text(
        json.dumps({"_index": 0, "output_text": "done"}) + "\n", encoding="utf-8"
    )

    called: list[bool] = []

    class TrackingClient(FakeCLIClient):
        def generate_batch(self, input_batch: Any, **kwargs: Any) -> Any:
            called.append(True)
            return super().generate_batch(input_batch, **kwargs)

    def patched_build(*a: Any, **kw: Any) -> TrackingClient:
        return TrackingClient()

    monkeypatch.setattr(cli, "_build_client", patched_build)
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
    assert not called
    assert "Nothing to do" in capsys.readouterr().err


def test_generate_resume_requires_output_jsonl(
    fake_client_builder: list[FakeCLIClient],
) -> None:
    exit_code = cli.main([*_BASE_ARGS, "--prompt", "hello", "--resume"])
    assert exit_code == 1


def test_env_file_loads_secrets(
    tmp_path: Path,
    fake_client_builder: list[FakeCLIClient],
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
        cli._build_client(cli._client_config_from_args(args))
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
        cli._build_client(cli._client_config_from_args(args))
    assert "deployments.replica.extra_kwargs.api_key" in str(excinfo.value)
    assert "--env-file" in str(excinfo.value)
