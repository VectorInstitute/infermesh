from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from infermesh import cli
from infermesh._cli_bench import _compute_generate_recommendation
from infermesh.types import BatchResult, EmbeddingResult, RequestMetrics, TokenUsage
from tests.fakes import FakeCLIClient


def test_bench_generate_default_sweep(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
    output_json = tmp_path / "bench.json"
    exit_code = cli.main(
        [
            "bench",
            "generate",
            "--model",
            "openai/test",
            "--api-base",
            "http://localhost:8000/v1",
            "--prompt",
            "hello",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--output-json",
            str(output_json),
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "recommended_max_parallel_requests" in stdout
    assert "peak" in stdout

    data = json.loads(output_json.read_text(encoding="utf-8"))
    level = data["sweep_results"][0]
    assert "recommended_max_parallel_requests" in data
    assert "succeeded" in level
    assert "completed" not in level
    assert "p99_latency_s" in level
    assert "p95_service_time_s" in level
    assert "output_tokens_per_second" in level
    assert "prompt_tokens_per_second" in level


def test_bench_generate_fixed_concurrency(
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
    exit_code = cli.main(
        [
            "bench",
            "generate",
            "--model",
            "openai/test",
            "--api-base",
            "http://localhost:8000/v1",
            "--prompt",
            "hello",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--concurrency",
            "4",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "c=4  " in stdout
    assert "recommended_max_parallel_requests" not in stdout


def test_bench_generate_max_concurrency(
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
    exit_code = cli.main(
        [
            "bench",
            "generate",
            "--model",
            "openai/test",
            "--api-base",
            "http://localhost:8000/v1",
            "--prompt",
            "hello",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--max-concurrency",
            "4",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "c=1  " in stdout
    assert "c=4  " in stdout
    assert "c=8  " not in stdout
    assert "recommended_max_parallel_requests" in stdout


def test_bench_generate_plateau_warning() -> None:
    sweep_results = [
        {
            "concurrency": concurrency,
            "failures": 0,
            "output_tokens_per_second": float(concurrency * 100),
            "requests_per_second": float(concurrency * 5),
        }
        for concurrency in [1, 2, 4, 8]
    ]
    recommended, metric_label, plateau_reached = _compute_generate_recommendation(
        sweep_results
    )
    assert recommended == 8
    assert plateau_reached is True
    assert "tok" in metric_label


def test_bench_embed_batch_size_sweep(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
    output_json = tmp_path / "embed_bench.json"
    exit_code = cli.main(
        [
            "bench",
            "embed",
            "--model",
            "embed-model",
            "--api-base",
            "http://localhost:8000/v1",
            "--text",
            "hello",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--output-json",
            str(output_json),
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "batch=" in stdout
    assert "vec/s=" in stdout
    assert "recommended_batch_size" in stdout
    assert "recommended_max_parallel_requests" not in stdout

    data = json.loads(output_json.read_text(encoding="utf-8"))
    level = data["sweep_results"][0]
    assert "recommended_batch_size" in data
    assert "batch_size" in level
    assert "vectors_per_second" in level
    assert "p99_latency_s" in level
    assert "p95_service_time_s" in level


def test_bench_embed_fixed_batch_size(
    capsys: pytest.CaptureFixture[str],
    fake_client_builder: list[FakeCLIClient],
) -> None:
    exit_code = cli.main(
        [
            "bench",
            "embed",
            "--model",
            "embed-model",
            "--api-base",
            "http://localhost:8000/v1",
            "--text",
            "hello world",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--batch-size",
            "8",
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "batch=8   " in stdout
    assert "recommended_batch_size" not in stdout


def test_bench_embed_fixed_batch_size_cycles_inputs_to_fill_batch(
    fake_client_builder: list[FakeCLIClient],
) -> None:
    exit_code = cli.main(
        [
            "bench",
            "embed",
            "--model",
            "embed-model",
            "--api-base",
            "http://localhost:8000/v1",
            "--text",
            "hello world",
            "--warmup",
            "1",
            "--requests",
            "2",
            "--batch-size",
            "8",
        ]
    )
    assert exit_code == 0
    recorded_sizes = [
        size for client in fake_client_builder for size in client.embed_batch_sizes
    ]
    recorded_micro_batch_sizes = [
        size
        for client in fake_client_builder
        for size in client.embed_micro_batch_sizes
    ]
    assert recorded_sizes == [8, 8, 8]
    assert recorded_micro_batch_sizes == [8, 8, 8]


def test_bench_embed_counts_partial_failures_per_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartialFailureCLIClient(FakeCLIClient):
        def embed_batch(
            self,
            input_batch: list[str],
            **kwargs: Any,
        ) -> BatchResult[EmbeddingResult]:
            self.embed_batch_sizes.append(len(input_batch))
            self.embed_micro_batch_sizes.append(kwargs.get("micro_batch_size"))
            success = EmbeddingResult(
                model_id="embed-model",
                embedding=[0.1, 0.2],
                request_id="embed-shared",
                token_usage=TokenUsage(
                    prompt_tokens=8,
                    completion_tokens=0,
                    total_tokens=8,
                ),
                metrics=RequestMetrics(
                    queue_wait_s=0.01,
                    service_time_s=0.02,
                    end_to_end_s=0.03,
                    deployment="replica-1",
                ),
            )
            return BatchResult(
                results=[success, None],
                errors=[None, RuntimeError("boom")],
            )

    def build_client(*args: Any, **kwargs: Any) -> PartialFailureCLIClient:
        return PartialFailureCLIClient(**kwargs)

    monkeypatch.setattr(cli, "_build_client", build_client)
    output_json = tmp_path / "embed_partial.json"

    exit_code = cli.main(
        [
            "bench",
            "embed",
            "--model",
            "embed-model",
            "--api-base",
            "http://localhost:8000/v1",
            "--text",
            "hello",
            "--text",
            "world",
            "--warmup",
            "0",
            "--requests",
            "1",
            "--batch-size",
            "2",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    data = json.loads(output_json.read_text(encoding="utf-8"))
    level = data["sweep_results"][0]
    assert level["total_submitted"] == 2
    assert level["succeeded"] == 1
    assert level["failures"] == 1
