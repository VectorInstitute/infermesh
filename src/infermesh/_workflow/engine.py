"""Generate workflow engine orchestration."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from infermesh._batch_utils import cancel_tasks
from infermesh._cli_support import _build_generation_record, _write_jsonl

from .checkpoint import _ERROR_STATUS, _SUCCESS_STATUS, _FileBackedPersistenceSink
from .models import CheckpointKey, _PreparedWorkItem, _SourceExhausted, _WorkItem
from .prepare import Preparer
from .runtime import _prepare_generate_run_resources

if TYPE_CHECKING:
    from infermesh.client import LMClient
    from infermesh.types import EndpointType


def _write_item_result(
    persistence_sink: _FileBackedPersistenceSink | None,
    output_index: int,
    checkpoint_key: CheckpointKey,
    result: Any,
    error: BaseException | None,
    *,
    metadata: dict[str, Any] | None,
    parse_json: bool,
) -> dict[str, Any]:
    """Build one settled item and optionally persist it through the sink."""

    record = _build_generation_record(
        output_index,
        result,
        error,
        parse_json=parse_json,
    )
    if metadata is not None:
        record["metadata"] = metadata
    if persistence_sink is not None:
        persistence_sink.write_record(
            record,
            checkpoint_key,
            status=_ERROR_STATUS if error else _SUCCESS_STATUS,
            error=str(error) if error else None,
        )
    return record


async def _agenerate_work_item(
    client: LMClient,
    item: _WorkItem,
    *,
    endpoint: EndpointType,
) -> tuple[Any, Exception | None]:
    """Run one workflow item and return its settled outcome."""

    try:
        result = await client.agenerate(item.mapped_input, endpoint=endpoint)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return None, exc
    return result, None


def _emit_settled_work_item(
    persistence_sink: _FileBackedPersistenceSink | None,
    item: _WorkItem,
    result: Any,
    error: BaseException | None,
    *,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Persist or print one settled workflow item, then tick progress."""

    record = _write_item_result(
        persistence_sink,
        item.output_index,
        item.checkpoint_key,
        result,
        error,
        metadata=item.metadata,
        parse_json=parse_json,
    )
    if persistence_sink is None:
        _write_jsonl([record], None)
    if on_progress is not None:
        on_progress()


def _emit_immediate_error(
    persistence_sink: _FileBackedPersistenceSink | None,
    prepared: _PreparedWorkItem,
    *,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Write one per-item error row without aborting the run, then invoke progress."""

    record = _write_item_result(
        persistence_sink,
        prepared.output_index,
        prepared.checkpoint_key,
        None,
        prepared.immediate_error,
        metadata=None,
        parse_json=parse_json,
    )
    if persistence_sink is None:
        _write_jsonl([record], None)
    if on_progress is not None:
        on_progress()


async def _arun_generate_source_rows(
    client: LMClient,
    *,
    preparer: Preparer,
    preparer_executor: ThreadPoolExecutor,
    resume: bool,
    persistence_sink: _FileBackedPersistenceSink | None,
    window_size: int,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Stream mapped rows through a rolling in-flight generation window."""

    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    loop = asyncio.get_running_loop()
    active_tasks: dict[asyncio.Task[tuple[Any, Exception | None]], _WorkItem] = {}
    source_exhausted = False
    any_work = False

    async def fill_window() -> None:
        nonlocal source_exhausted
        nonlocal any_work

        while len(active_tasks) < window_size and not source_exhausted:
            prepared = await loop.run_in_executor(
                preparer_executor, preparer.next_prepared
            )
            if isinstance(prepared, _SourceExhausted):
                source_exhausted = True
                return
            any_work = True
            if prepared.immediate_error is not None:
                _emit_immediate_error(
                    persistence_sink,
                    prepared,
                    parse_json=parse_json,
                    on_progress=on_progress,
                )
                continue

            work_item = prepared.work_item
            assert work_item is not None
            task = asyncio.create_task(
                _agenerate_work_item(client, work_item, endpoint=endpoint)
            )
            active_tasks[task] = work_item

    try:
        await fill_window()
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                item = active_tasks.pop(task)
                result, error = task.result()
                _emit_settled_work_item(
                    persistence_sink,
                    item,
                    result,
                    error,
                    parse_json=parse_json,
                    on_progress=on_progress,
                )
            await fill_window()
    except BaseException:
        await cancel_tasks(list(active_tasks))
        raise
    finally:
        await loop.run_in_executor(preparer_executor, preparer.close)

    if resume and not any_work:
        sys.stderr.write("Nothing to do — all rows already completed.\n")


def _run_generate_source_rows(
    client: LMClient,
    *,
    preparer: Preparer,
    resume: bool,
    persistence_sink: _FileBackedPersistenceSink | None,
    window_size: int,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Run the rolling generate scheduler on the client's background loop."""

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="infermesh-generate-prep",
    ) as preparer_executor:
        client._run_sync(
            _arun_generate_source_rows(
                client,
                preparer=preparer,
                preparer_executor=preparer_executor,
                resume=resume,
                persistence_sink=persistence_sink,
                window_size=window_size,
                endpoint=endpoint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
        )


def run_generate_workflow(
    client: LMClient,
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_jsonl: str | None,
    checkpoint_dir: str | None,
    mapper_spec: str | None,
    resume: bool,
    endpoint: EndpointType,
    window_size: int,
    parse_json: bool,
    on_progress: Callable[[], Any] | None = None,
    on_status: Callable[[str], Any] | None = None,
) -> None:
    """Run the generate workflow engine."""

    run = _prepare_generate_run_resources(
        prompt=prompt,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        checkpoint_dir=checkpoint_dir,
        mapper_spec=mapper_spec,
        resume=resume,
        on_status=on_status,
    )
    try:
        _run_generate_source_rows(
            client,
            preparer=run.preparer,
            resume=resume,
            persistence_sink=run.persistence_sink,
            window_size=window_size,
            endpoint=endpoint,
            parse_json=parse_json,
            on_progress=on_progress,
        )
    finally:
        run.close()
