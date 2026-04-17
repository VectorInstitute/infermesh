"""Private helpers for text generation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from infermesh._utils import (
    build_generation_result,
    estimate_token_count,
    normalize_batch_input,
    normalize_generate_input,
)
from infermesh.types import (
    BatchResult,
    EndpointType,
    GenerateInput,
    GenerationBatchResult,
    GenerationResult,
    OnGenerationResult,
)

if TYPE_CHECKING:
    from infermesh.client import LMClient


async def _agenerate_one(
    self: LMClient,
    input_data: GenerateInput,
    *,
    endpoint: EndpointType,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    request_kwargs: dict[str, Any],
    queue_started_at: float | None = None,
) -> GenerationResult:
    """Run one generation request with optional batch admission timing."""

    normalized_input = normalize_generate_input(input_data, endpoint)
    request_kwargs = dict(request_kwargs)
    per_request_max = self._get_generation_output_token_limit(
        request_kwargs,
        endpoint,
    )
    estimated_tokens = estimate_token_count(
        self._litellm,
        self.model,
        normalized_input,
        endpoint=endpoint,
        max_tokens=per_request_max or self.default_output_tokens,
    )
    response, metrics = await self._with_retry(
        lambda: self._dispatch_with_controls(
            estimated_tokens=estimated_tokens,
            request_callable=self._call_generation,
            request_args=(normalized_input,),
            request_kwargs={
                "endpoint": endpoint,
                "request_kwargs": request_kwargs,
            },
            queue_started_at=queue_started_at,
        )
    )
    return build_generation_result(
        response,
        endpoint=endpoint,
        response_format=response_format,
        parse_output=parse_output or response_format is not None,
        metrics=metrics,
    )


async def _gather_with_progress(
    coros: list[Coroutine[Any, Any, GenerationResult]],
    on_progress: Callable[[int, int], None] | None,
    on_result: OnGenerationResult = None,
) -> GenerationBatchResult:
    """Gather coroutines with per-item callbacks."""

    async def indexed(
        index: int,
        coro: Coroutine[Any, Any, GenerationResult],
    ) -> tuple[int, GenerationResult | BaseException]:
        try:
            return index, await coro
        except BaseException as exc:  # noqa: BLE001
            return index, exc

    total = len(coros)
    results: list[GenerationResult | None] = [None] * total
    errors: list[BaseException | None] = [None] * total
    tasks = [
        asyncio.create_task(indexed(index, coro)) for index, coro in enumerate(coros)
    ]
    try:
        for completed, future in enumerate(asyncio.as_completed(tasks), start=1):
            index, outcome = await future
            if isinstance(outcome, BaseException):
                errors[index] = outcome
                if on_result is not None:
                    on_result(index, None, outcome)
            else:
                results[index] = outcome
                if on_result is not None:
                    on_result(index, outcome, None)
            if on_progress is not None:
                on_progress(completed, total)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return BatchResult(results=results, errors=errors)


async def _cancel_tasks(tasks: Sequence[asyncio.Task[Any]]) -> None:
    """Cancel unfinished tasks and await their cleanup."""

    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _fill_bounded_generation_window(
    self: LMClient,
    active_tasks: dict[asyncio.Task[GenerationResult], int],
    *,
    inputs: Sequence[GenerateInput],
    next_index: int,
    endpoint: EndpointType,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    request_kwargs: dict[str, Any],
    queue_started_at: float,
) -> int:
    """Admit generation tasks until the bounded window is full."""

    assert self.max_parallel_requests is not None
    while next_index < len(inputs) and len(active_tasks) < self.max_parallel_requests:
        active_tasks[
            asyncio.create_task(
                _agenerate_one(
                    self,
                    inputs[next_index],
                    endpoint=endpoint,
                    response_format=response_format,
                    parse_output=parse_output,
                    request_kwargs=request_kwargs,
                    queue_started_at=queue_started_at,
                )
            )
        ] = next_index
        next_index += 1
    return next_index


def _settle_bounded_generation_task(
    task: asyncio.Task[GenerationResult],
    *,
    index: int,
    results: list[GenerationResult | None],
    errors: list[BaseException | None] | None,
    return_exceptions: bool,
    on_progress: Callable[[int, int], None] | None,
    on_result: OnGenerationResult,
    completed: int,
    total: int,
) -> tuple[int, BaseException | None]:
    """Store one completed bounded-batch task and fire callbacks."""

    try:
        result = task.result()
    except BaseException as exc:  # noqa: BLE001
        if not return_exceptions:
            return completed, exc
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
    return completed, None


async def _run_bounded_generation_batch(
    self: LMClient,
    inputs: Sequence[GenerateInput],
    *,
    endpoint: EndpointType,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    return_exceptions: bool,
    on_progress: Callable[[int, int], None] | None,
    on_result: OnGenerationResult,
    request_kwargs: dict[str, Any],
) -> GenerationBatchResult:
    """Run generation batch work through a bounded in-flight window."""

    total = len(inputs)
    results: list[GenerationResult | None] = [None] * total
    errors: list[BaseException | None] | None = (
        [None] * total if return_exceptions else None
    )
    if total == 0:
        return BatchResult(results=results, errors=errors)

    assert self.max_parallel_requests is not None
    admission_started = time.perf_counter()
    active_tasks: dict[asyncio.Task[GenerationResult], int] = {}
    completed = 0
    next_index = _fill_bounded_generation_window(
        self,
        active_tasks,
        inputs=inputs,
        next_index=0,
        endpoint=endpoint,
        response_format=response_format,
        parse_output=parse_output,
        request_kwargs=request_kwargs,
        queue_started_at=admission_started,
    )

    try:
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            first_error: BaseException | None = None
            for task in done:
                index = active_tasks.pop(task)
                completed, task_error = _settle_bounded_generation_task(
                    task,
                    index=index,
                    results=results,
                    errors=errors,
                    return_exceptions=return_exceptions,
                    on_progress=on_progress,
                    on_result=on_result,
                    completed=completed,
                    total=total,
                )
                if first_error is None:
                    first_error = task_error

            if first_error is not None:
                pending = list(active_tasks)
                active_tasks.clear()
                await _cancel_tasks(pending)
                raise first_error

            next_index = _fill_bounded_generation_window(
                self,
                active_tasks,
                inputs=inputs,
                next_index=next_index,
                endpoint=endpoint,
                response_format=response_format,
                parse_output=parse_output,
                request_kwargs=request_kwargs,
                queue_started_at=admission_started,
            )
    except BaseException:
        pending = list(active_tasks)
        active_tasks.clear()
        await _cancel_tasks(pending)
        raise
    return BatchResult(results=results, errors=errors)


async def _agenerate_batch(
    self: LMClient,
    input_batch: Sequence[GenerateInput],
    *,
    endpoint: EndpointType | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    parse_output: bool = False,
    return_exceptions: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: OnGenerationResult = None,
    **kwargs: Any,
) -> GenerationBatchResult:
    """Implement ``LMClient.agenerate_batch`` with optional bounded admission."""

    inputs = normalize_batch_input(input_batch)
    request_endpoint = endpoint or self.endpoint
    request_kwargs = self._merge_request_kwargs(kwargs)

    if self.max_parallel_requests is not None and self.max_parallel_requests > 0:
        return await _run_bounded_generation_batch(
            self,
            inputs,
            endpoint=request_endpoint,
            response_format=response_format,
            parse_output=parse_output,
            return_exceptions=return_exceptions,
            on_progress=on_progress,
            on_result=on_result,
            request_kwargs=request_kwargs,
        )

    coros = [
        _agenerate_one(
            self,
            item,
            endpoint=request_endpoint,
            response_format=response_format,
            parse_output=parse_output,
            request_kwargs=request_kwargs,
        )
        for item in inputs
    ]

    if return_exceptions:
        if on_progress is None and on_result is None:
            raw = await asyncio.gather(*coros, return_exceptions=True)
            results: list[GenerationResult | None] = []
            errors: list[BaseException | None] = []
            for item in raw:
                if isinstance(item, BaseException):
                    results.append(None)
                    errors.append(item)
                else:
                    results.append(item)
                    errors.append(None)
            return BatchResult(results=results, errors=errors)
        return await _gather_with_progress(coros, on_progress, on_result)

    completed = [0]

    async def progress_wrapper(
        index: int, coro: Coroutine[Any, Any, GenerationResult]
    ) -> GenerationResult:
        """Wrap a coroutine with progress and result callbacks."""

        result = await coro
        completed[0] += 1
        if on_result is not None:
            on_result(index, result, None)
        if on_progress is not None:
            on_progress(completed[0], len(coros))
        return result

    strict_tasks: list[asyncio.Task[GenerationResult]] = []
    try:
        async with asyncio.TaskGroup() as task_group:
            strict_tasks = [
                task_group.create_task(progress_wrapper(index, coro))
                for index, coro in enumerate(coros)
            ]
    except BaseException as exc:
        if isinstance(exc, BaseExceptionGroup):
            raise exc.exceptions[0] from exc.exceptions[0].__cause__
        raise
    return BatchResult(
        results=cast(
            list[GenerationResult | None], [task.result() for task in strict_tasks]
        )
    )
