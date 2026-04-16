"""Private benchmark helpers for ``lm_client.cli``."""

from __future__ import annotations

import itertools
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lm_client._cli_support import ClientConfig
from lm_client._utils import batched_cycle, percentile
from lm_client.types import BatchResult, EmbeddingResult

DEFAULT_SWEEP = [1, 2, 4, 8, 16, 32]
DEFAULT_EMBED_BATCH_SIZES = [1, 8, 32, 128, 512]


def _run_benchmark(
    *,
    task_name: str,
    warmup: int,
    requests: int,
    duration_s: float | None,
    concurrency_sweep: list[int],
    recommend: bool,
    workload_factory: Any,
    runner: Any,
    workload: list[Any],
) -> dict[str, Any]:
    """Run a client-side generation benchmark."""

    warmup_items = workload[:warmup]
    measured_items = workload[:requests]
    sweep_results: list[dict[str, Any]] = []

    if warmup_items:
        warmup_client = workload_factory(concurrency_sweep[0])
        try:
            runner(warmup_client, warmup_items)
        finally:
            warmup_client.close()

    for concurrency in concurrency_sweep:
        client = workload_factory(concurrency)
        try:
            if duration_s is not None:
                batch_result, elapsed = _run_duration_loop(
                    client=client,
                    runner=runner,
                    items=measured_items,
                    concurrency=concurrency,
                    duration_s=duration_s,
                )
            else:
                n = len(measured_items)
                with tqdm(
                    total=n,
                    desc=f"c={concurrency:2d}",
                    unit="req",
                    file=sys.stderr,
                ) as pbar:
                    started_at = time.perf_counter()
                    batch_result = runner(
                        client,
                        measured_items,
                        on_progress=lambda _d, _t: pbar.update(1),
                    )
                    elapsed = time.perf_counter() - started_at
        finally:
            client.close()

        sweep_result = _summarize_batch(
            task_name=task_name,
            concurrency=concurrency,
            batch_result=batch_result,
            elapsed_s=elapsed,
        )
        sweep_results.append(sweep_result)
        _print_generate_level(sweep_result)

    return {
        "task": task_name,
        "sweep_results": sweep_results,
        "recommend": recommend,
    }


def _run_duration_loop(
    *,
    client: Any,
    runner: Any,
    items: list[Any],
    concurrency: int,
    duration_s: float,
) -> tuple[BatchResult[Any], float]:
    """Run concurrency-sized request chunks until ``duration_s`` has elapsed."""

    all_results: list[Any] = []
    all_errors: list[Any] = []
    has_errors = False
    items_cycle = itertools.cycle(items)
    started_at = time.perf_counter()

    with tqdm(
        total=None,
        desc=f"c={concurrency:2d}",
        unit="req",
        file=sys.stderr,
    ) as pbar:
        while time.perf_counter() - started_at < duration_s:
            chunk = list(itertools.islice(items_cycle, concurrency))
            sub = runner(client, chunk)
            all_results.extend(sub.results)
            if sub.errors is not None:
                has_errors = True
                all_errors.extend(sub.errors)
            else:
                all_errors.extend([None] * len(sub.results))
            pbar.update(len(sub.results))

    elapsed = time.perf_counter() - started_at
    return (
        BatchResult(results=all_results, errors=all_errors if has_errors else None),
        elapsed,
    )


def _print_generate_level(result: dict[str, Any]) -> None:
    """Print a single generation sweep-level result line."""

    out_tok_s = result.get("output_tokens_per_second", 0.0)
    tok_part = f"  out_tok/s={out_tok_s:.0f}" if out_tok_s else ""
    print(
        f"c={result['concurrency']:<3d}"
        f"{tok_part}"
        f"  rps={result['requests_per_second']:.2f}"
        f"  p50={result['p50_latency_s']:.3f}s"
        f"  p95={result['p95_latency_s']:.3f}s"
        f"  p99={result['p99_latency_s']:.3f}s"
        f"  svc_p95={result['p95_service_time_s']:.3f}s"
        f"  q_p95={result['p95_queue_wait_s']:.3f}s"
        f"  err={result['total_submitted'] - result['succeeded']}/{result['total_submitted']}"
        f"  elapsed={result['elapsed_s']:.1f}s"
    )


def _percentile_stats(
    latencies: list[float],
    service_times: list[float],
    queue_waits: list[float] | None = None,
) -> dict[str, float]:
    """Compute p50/p95/p99 stats for benchmark latency lists."""

    stats: dict[str, float] = {
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "p50_service_time_s": percentile(service_times, 50),
        "p95_service_time_s": percentile(service_times, 95),
        "p99_service_time_s": percentile(service_times, 99),
    }
    if queue_waits is not None:
        stats["p50_queue_wait_s"] = percentile(queue_waits, 50)
        stats["p95_queue_wait_s"] = percentile(queue_waits, 95)
        stats["p99_queue_wait_s"] = percentile(queue_waits, 99)
    return stats


def _summarize_batch(
    *,
    task_name: str,
    concurrency: int,
    batch_result: BatchResult[Any],
    elapsed_s: float,
) -> dict[str, Any]:
    """Summarize a completed benchmark batch."""

    total_submitted = len(batch_result)
    failures = 0
    queue_waits: list[float] = []
    latencies: list[float] = []
    service_times: list[float] = []
    deployments: Counter[str] = Counter()
    total_tokens = 0
    output_tokens = 0
    prompt_tokens = 0
    vector_count = 0

    for item in batch_result:
        if item is None:
            failures += 1
            continue
        metrics = getattr(item, "metrics", None)
        if metrics is not None:
            queue_waits.append(metrics.queue_wait_s)
            latencies.append(metrics.end_to_end_s)
            service_times.append(metrics.service_time_s)
            if metrics.deployment:
                deployments[metrics.deployment] += 1
        usage = getattr(item, "token_usage", None)
        if usage is not None:
            total_tokens += usage.total_tokens
            output_tokens += usage.output_tokens
            prompt_tokens += usage.prompt_tokens
        if isinstance(item, EmbeddingResult):
            vector_count += 1

    succeeded = total_submitted - failures
    summary: dict[str, Any] = {
        "task": task_name,
        "concurrency": concurrency,
        "total_submitted": total_submitted,
        "succeeded": succeeded,
        "failures": failures,
        "elapsed_s": elapsed_s,
        "requests_per_second": succeeded / elapsed_s if elapsed_s > 0 else 0.0,
        **_percentile_stats(latencies, service_times, queue_waits),
        "deployment_distribution": dict(deployments),
    }
    if task_name == "generate":
        summary["output_tokens_per_second"] = (
            output_tokens / elapsed_s if elapsed_s > 0 else 0.0
        )
        summary["prompt_tokens_per_second"] = (
            prompt_tokens / elapsed_s if elapsed_s > 0 else 0.0
        )
        summary["total_tokens_per_second"] = (
            total_tokens / elapsed_s if elapsed_s > 0 else 0.0
        )
    if task_name == "embed":
        summary["vectors_per_second"] = (
            vector_count / elapsed_s if elapsed_s > 0 else 0.0
        )
        if total_tokens:
            summary["total_tokens_per_second"] = (
                total_tokens / elapsed_s if elapsed_s > 0 else 0.0
            )
    return summary


def _compute_generate_recommendation(
    sweep_results: list[dict[str, Any]],
) -> tuple[int | None, str, bool]:
    """Choose the first level within 95% of peak zero-failure throughput."""

    use_tok = any(r.get("output_tokens_per_second", 0) > 0 for r in sweep_results)
    metric_key = "output_tokens_per_second" if use_tok else "requests_per_second"
    metric_label = "output_tok/s" if use_tok else "rps"

    eligible = [r for r in sweep_results if r["failures"] == 0]
    if not eligible:
        return None, metric_label, False

    best = max(r[metric_key] for r in eligible)
    if best == 0:
        return None, metric_label, False

    recommended = next(
        (
            result["concurrency"]
            for result in eligible
            if result[metric_key] >= best * 0.95
        ),
        None,
    )
    plateau_reached = (
        recommended is not None and recommended == sweep_results[-1]["concurrency"]
    )
    return recommended, metric_label, plateau_reached


def _write_generate_summary(summary: dict[str, Any], output_path: str | None) -> None:
    """Write a generation benchmark summary to stdout and optionally a JSON file."""

    recommend = summary.pop("recommend", True)
    sweep_results = summary["sweep_results"]

    if recommend:
        recommended, metric_label, plateau_reached = _compute_generate_recommendation(
            sweep_results
        )
        summary["recommended_max_parallel_requests"] = recommended
        print(
            f"recommended_max_parallel_requests={recommended}"
            f"  (lowest concurrency within 95% of peak {metric_label}, 0 failures)"
        )
        if plateau_reached and recommended is not None:
            max_c = sweep_results[-1]["concurrency"]
            print(
                "note: throughput was still increasing at the highest sweep level "
                f"({max_c}). Re-run with --max-concurrency {max_c * 2} to find the "
                "true optimum."
            )
    else:
        summary["recommended_max_parallel_requests"] = None

    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_embed_batch_benchmark(
    *,
    texts: list[str],
    config: ClientConfig,
    warmup: int,
    requests: int,
    batch_size_sweep: list[int],
    recommend: bool,
    build_client: Callable[..., Any],
) -> dict[str, Any]:
    """Sweep embed batch sizes and measure throughput vs. latency."""

    if warmup > 0:
        warmup_client = build_client(config)
        try:
            warmup_batch = batched_cycle(texts, batch_size_sweep[0])
            for _ in range(warmup):
                warmup_client.embed_batch(warmup_batch)
        finally:
            warmup_client.close()

    sweep_results: list[dict[str, Any]] = []
    for batch_size in batch_size_sweep:
        sample = batched_cycle(texts, batch_size)
        client = build_client(config)
        try:
            latencies: list[float] = []
            service_times: list[float] = []
            total_tokens = 0
            failures = 0

            started_at = time.perf_counter()
            with tqdm(
                total=requests,
                desc=f"batch={batch_size:<4d}",
                unit="call",
                file=sys.stderr,
            ) as pbar:
                for _ in range(requests):
                    result = client.embed_batch(sample)
                    first = result.results[0] if result.results else None
                    if first is None:
                        failures += 1
                    else:
                        if first.metrics is not None:
                            latencies.append(first.metrics.end_to_end_s)
                            service_times.append(first.metrics.service_time_s)
                        if first.token_usage is not None:
                            total_tokens += first.token_usage.total_tokens
                    pbar.update(1)
            elapsed_s = time.perf_counter() - started_at
        finally:
            client.close()

        succeeded = requests - failures
        level: dict[str, Any] = {
            "batch_size": batch_size,
            "total_submitted": requests,
            "succeeded": succeeded,
            "failures": failures,
            "elapsed_s": elapsed_s,
            "vectors_per_second": succeeded * len(sample) / elapsed_s
            if elapsed_s > 0
            else 0.0,
            **_percentile_stats(latencies, service_times),
        }
        if total_tokens:
            level["total_tokens_per_second"] = (
                total_tokens / elapsed_s if elapsed_s > 0 else 0.0
            )
        sweep_results.append(level)
        _print_embed_level(level)

    return {
        "task": "embed",
        "sweep_results": sweep_results,
        "recommend": recommend,
    }


def _print_embed_level(result: dict[str, Any]) -> None:
    """Print a single embed sweep-level result line."""

    print(
        f"batch={result['batch_size']:<4d}"
        f"  vec/s={result['vectors_per_second']:.0f}"
        f"  p50={result['p50_latency_s']:.3f}s"
        f"  p95={result['p95_latency_s']:.3f}s"
        f"  p99={result['p99_latency_s']:.3f}s"
        f"  svc_p95={result['p95_service_time_s']:.3f}s"
        f"  err={result['failures']}/{result['total_submitted']}"
        f"  elapsed={result['elapsed_s']:.1f}s"
    )


def _write_embed_summary(summary: dict[str, Any], output_path: str | None) -> None:
    """Write an embedding benchmark summary to stdout and optionally a JSON file."""

    recommend = summary.pop("recommend", True)
    sweep_results = summary["sweep_results"]

    if recommend:
        eligible = [r for r in sweep_results if r["failures"] == 0]
        recommended_batch_size = (
            max(eligible, key=lambda r: r["vectors_per_second"])["batch_size"]
            if eligible
            else None
        )
        summary["recommended_batch_size"] = recommended_batch_size
        print(
            f"recommended_batch_size={recommended_batch_size}  (highest vec/s, 0 failures)"
        )

        plateau_reached = (
            recommended_batch_size is not None
            and recommended_batch_size == sweep_results[-1]["batch_size"]
            and len(sweep_results) > 1
        )
        if plateau_reached:
            max_b = sweep_results[-1]["batch_size"]
            print(
                "note: throughput was still increasing at the largest batch size "
                f"({max_b}). Re-run with --max-batch-size {max_b * 4} to find the "
                "true optimum."
            )
    else:
        summary["recommended_batch_size"] = None

    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _resolve_sweep_levels(
    *,
    single: int | None,
    maximum: int | None,
    default: list[int],
    sweep_fn: Callable[[int], list[int]],
) -> tuple[list[int], bool]:
    """Return ``(sweep_levels, recommend)`` from mutually exclusive CLI args."""

    if single is not None:
        return [single], False
    if maximum is not None:
        return sweep_fn(maximum), True
    return default, True


def _build_concurrency_sweep(max_concurrency: int) -> list[int]:
    """Return ``[1, 2, 4, 8, …, max_concurrency]``."""

    sweep: list[int] = []
    concurrency = 1
    while concurrency <= max_concurrency:
        sweep.append(concurrency)
        concurrency *= 2
    if not sweep or sweep[-1] != max_concurrency:
        sweep.append(max_concurrency)
    return sweep


def _build_embed_batch_sweep(max_batch_size: int) -> list[int]:
    """Return embed batch sizes up to ``max_batch_size`` from the default set."""

    sweep = [size for size in DEFAULT_EMBED_BATCH_SIZES if size <= max_batch_size]
    if not sweep or sweep[-1] != max_batch_size:
        sweep.append(max_batch_size)
    return sweep
