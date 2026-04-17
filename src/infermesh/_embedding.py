"""Private helpers for embeddings."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from infermesh._batch_utils import cancel_tasks
from infermesh._utils import (
    build_embedding_results,
    estimate_token_count,
    normalize_embedding_input,
)
from infermesh.types import (
    BatchResult,
    EmbeddingBatchResult,
    EmbeddingResult,
    OnEmbeddingResult,
)

if TYPE_CHECKING:
    from infermesh.client import LMClient


type _EmbeddingSettlement = tuple[int, EmbeddingResult | None, BaseException | None]


def _validate_micro_batch_size(micro_batch_size: int) -> None:
    """Validate embedding micro-batch sizing."""

    if micro_batch_size < 1:
        raise ValueError("``micro_batch_size`` must be a positive integer.")


def _should_isolate_embedding_failure(self: LMClient, exc: Exception) -> bool:
    """Return whether a failed micro-batch should be recursively isolated."""

    return not isinstance(exc, self._retryable_exceptions)


async def _aembed_one(
    self: LMClient,
    input_data: str,
    *,
    request_kwargs: dict[str, Any],
    queue_started_at: float | None = None,
) -> EmbeddingResult:
    """Create one embedding result."""

    results = await _aembed_chunk(
        self,
        [input_data],
        request_kwargs=request_kwargs,
        queue_started_at=queue_started_at,
    )
    return results[0]


async def _aembed_chunk(
    self: LMClient,
    input_data: list[str],
    *,
    request_kwargs: dict[str, Any],
    queue_started_at: float | None = None,
) -> list[EmbeddingResult]:
    """Run one embedding request for a micro-batch of texts."""

    estimated_tokens = estimate_token_count(
        self._litellm,
        self.model,
        input_data,
        endpoint="chat_completion",
        max_tokens=0,
    )
    response, metrics = await self._with_retry(
        lambda: self._dispatch_with_controls(
            estimated_tokens=estimated_tokens,
            request_callable=self._call_embedding,
            request_args=(input_data,),
            request_kwargs={"request_kwargs": dict(request_kwargs)},
            queue_started_at=queue_started_at,
        )
    )
    results = build_embedding_results(response, metrics=metrics)
    if len(results) != len(input_data):
        raise ValueError(
            "Embedding response did not contain one vector per input item."
        )
    return results


async def _resolve_embedding_chunk_capture(
    self: LMClient,
    input_data: list[str],
    start_index: int,
    *,
    request_kwargs: dict[str, Any],
    queue_started_at: float,
) -> list[_EmbeddingSettlement]:
    """Resolve one embedding chunk, recursively isolating failures."""

    try:
        results = await _aembed_chunk(
            self,
            input_data,
            request_kwargs=request_kwargs,
            queue_started_at=queue_started_at,
        )
    except Exception as exc:
        if len(input_data) == 1:
            return [(start_index, None, exc)]
        if not _should_isolate_embedding_failure(self, exc):
            return [
                (start_index + offset, None, exc) for offset in range(len(input_data))
            ]
        midpoint = len(input_data) // 2
        left = await _resolve_embedding_chunk_capture(
            self,
            input_data[:midpoint],
            start_index,
            request_kwargs=request_kwargs,
            queue_started_at=queue_started_at,
        )
        right = await _resolve_embedding_chunk_capture(
            self,
            input_data[midpoint:],
            start_index + midpoint,
            request_kwargs=request_kwargs,
            queue_started_at=queue_started_at,
        )
        return left + right
    return [
        (start_index + offset, result, None) for offset, result in enumerate(results)
    ]


def _apply_embedding_settlements(
    settlements: list[_EmbeddingSettlement],
    *,
    results: list[EmbeddingResult | None],
    errors: list[BaseException | None] | None,
    on_result: OnEmbeddingResult,
    on_progress: Callable[[int, int], None] | None,
    completed: int,
    total: int,
) -> int:
    """Apply chunk settlements to the batch result buffers and fire callbacks."""

    for index, result, error in settlements:
        if error is not None:
            assert errors is not None
            errors[index] = error
            if on_result is not None:
                on_result(index, None, error)
        else:
            assert result is not None
            results[index] = result
            if on_result is not None:
                on_result(index, result, None)
        completed += 1
        if on_progress is not None:
            on_progress(completed, total)
    return completed


async def _aembed_batch(
    self: LMClient,
    input_batch: Sequence[str],
    *,
    micro_batch_size: int = 32,
    return_exceptions: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: OnEmbeddingResult = None,
    **kwargs: Any,
) -> EmbeddingBatchResult:
    """Implement ``LMClient.aembed_batch`` with resilient micro-batching."""

    _validate_micro_batch_size(micro_batch_size)
    inputs = normalize_embedding_input(input_batch)
    total = len(inputs)
    errors: list[BaseException | None] | None = (
        [None] * total if return_exceptions else None
    )
    results: list[EmbeddingResult | None] = [None] * total
    if total == 0:
        return BatchResult(results=results, errors=errors)

    request_kwargs = self._merge_request_kwargs(kwargs)
    admission_started = time.perf_counter()
    active_tasks: set[asyncio.Task[list[_EmbeddingSettlement]]] = set()
    completed = 0
    next_index = 0

    async def run_chunk(
        start_index: int,
        chunk: list[str],
    ) -> list[_EmbeddingSettlement]:
        if return_exceptions:
            return await _resolve_embedding_chunk_capture(
                self,
                chunk,
                start_index,
                request_kwargs=request_kwargs,
                queue_started_at=admission_started,
            )
        chunk_results = await _aembed_chunk(
            self,
            chunk,
            request_kwargs=request_kwargs,
            queue_started_at=admission_started,
        )
        return [
            (start_index + offset, result, None)
            for offset, result in enumerate(chunk_results)
        ]

    def admit_chunks() -> int:
        nonlocal next_index
        while next_index < total and (
            self.max_parallel_requests is None
            or len(active_tasks) < self.max_parallel_requests
        ):
            start_index = next_index
            chunk = inputs[start_index : start_index + micro_batch_size]
            active_tasks.add(asyncio.create_task(run_chunk(start_index, chunk)))
            next_index += len(chunk)
        return next_index

    admit_chunks()
    try:
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                active_tasks.remove(task)
                try:
                    settlements = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pending = list(active_tasks)
                    active_tasks.clear()
                    await cancel_tasks(pending)
                    raise
                completed = _apply_embedding_settlements(
                    settlements,
                    results=results,
                    errors=errors,
                    on_result=on_result,
                    on_progress=on_progress,
                    completed=completed,
                    total=total,
                )
            admit_chunks()
    except BaseException:
        pending = list(active_tasks)
        active_tasks.clear()
        await cancel_tasks(pending)
        raise
    return BatchResult(results=results, errors=errors)
