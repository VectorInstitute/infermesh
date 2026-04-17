"""Private helpers for audio transcription."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from infermesh._batch_utils import cancel_tasks
from infermesh._utils import build_transcription_result, normalize_transcription_input
from infermesh.types import (
    BatchResult,
    OnTranscriptionResult,
    TranscriptionBatchResult,
    TranscriptionInput,
    TranscriptionResult,
)

if TYPE_CHECKING:
    from infermesh.client import LMClient


async def _atranscribe_one(
    self: LMClient,
    input_data: TranscriptionInput,
    *,
    max_transcription_bytes: int | None,
    request_kwargs: dict[str, Any],
    queue_started_at: float | None = None,
) -> TranscriptionResult:
    """Transcribe one audio input with optional batch admission timing."""

    normalized = normalize_transcription_input(
        input_data,
        max_bytes=max_transcription_bytes,
    )
    response, metrics = await self._with_retry(
        lambda: self._dispatch_with_controls(
            estimated_tokens=0,
            request_callable=self._call_transcription,
            request_args=(normalized,),
            request_kwargs={"request_kwargs": dict(request_kwargs)},
            queue_started_at=queue_started_at,
        )
    )
    return build_transcription_result(response, metrics=metrics)


async def _atranscribe_batch(
    self: LMClient,
    input_batch: Sequence[TranscriptionInput],
    *,
    max_transcription_bytes: int | None,
    return_exceptions: bool,
    on_progress: Callable[[int, int], None] | None,
    on_result: OnTranscriptionResult,
    **kwargs: Any,
) -> TranscriptionBatchResult:
    """Implement ``LMClient.atranscribe_batch`` with optional bounded admission."""

    total = len(input_batch)
    results: list[TranscriptionResult | None] = [None] * total
    errors: list[BaseException | None] | None = (
        [None] * total if return_exceptions else None
    )
    if total == 0:
        return BatchResult(results=results, errors=errors)

    request_kwargs = self._merge_request_kwargs(kwargs)
    admission_started = time.perf_counter()
    active_tasks: dict[asyncio.Task[TranscriptionResult], int] = {}
    completed = 0
    next_index = 0

    def admit_inputs() -> int:
        nonlocal next_index
        while next_index < total and (
            self.max_parallel_requests is None
            or len(active_tasks) < self.max_parallel_requests
        ):
            active_tasks[
                asyncio.create_task(
                    _atranscribe_one(
                        self,
                        input_batch[next_index],
                        max_transcription_bytes=max_transcription_bytes,
                        request_kwargs=request_kwargs,
                        queue_started_at=admission_started,
                    )
                )
            ] = next_index
            next_index += 1
        return next_index

    admit_inputs()
    try:
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                index = active_tasks.pop(task)
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if not return_exceptions:
                        pending = list(active_tasks)
                        active_tasks.clear()
                        await cancel_tasks(pending)
                        raise exc
                    assert errors is not None
                    errors[index] = exc
                    if on_result is not None:
                        on_result(index, None, exc)
                else:
                    results[index] = result
                    if on_result is not None:
                        on_result(index, result, None)
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
            admit_inputs()
    except BaseException:
        pending = list(active_tasks)
        active_tasks.clear()
        await cancel_tasks(pending)
        raise
    return BatchResult(results=results, errors=errors)
