"""Generate workflow engine orchestration."""

import asyncio
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from infermesh._batch_utils import cancel_tasks

from .models import (
    BatchWorkflowSummary,
    Operation,
    RecordBuilder,
    RunStore,
    SettledRecord,
    SourceItem,
    SourceStream,
    WorkStream,
    _SourceExhausted,
)


@dataclass(slots=True)
class _BatchWorkflowCounters:
    completed: int = 0
    errored: int = 0

    def add(self, *, completed: int, errored: int) -> None:
        self.completed += completed
        self.errored += errored


async def _run_source_item(
    item: SourceItem, operation: Operation
) -> tuple[Any, BaseException | None]:
    """Run one endpoint-neutral workflow item."""

    try:
        return await operation(item)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        return None, exc


async def _settle_batch_workflow_item(
    store: RunStore,
    item: SourceItem,
    result: Any,
    error: BaseException | None,
    *,
    record_builder: RecordBuilder,
    on_progress: Callable[[], Any] | None,
) -> tuple[int, int]:
    """Build, settle, and count one workflow item outcome."""

    record = record_builder(item, result, error)
    await store.settle(
        SettledRecord(
            output_index=item.output_index,
            checkpoint_key=item.checkpoint_key,
            record=record,
            error=error,
        )
    )
    if on_progress is not None:
        on_progress()
    if error is None:
        return 1, 0
    return 0, 1


async def _fill_work_window(
    *,
    work_stream: WorkStream,
    executor: ThreadPoolExecutor,
    active_tasks: dict[asyncio.Task[tuple[Any, BaseException | None]], SourceItem],
    window_size: int,
    operation: Operation,
    store: RunStore,
    record_builder: RecordBuilder,
    counters: _BatchWorkflowCounters,
    on_progress: Callable[[], Any] | None,
) -> bool:
    """Fill active tasks until the window is full or source is exhausted."""

    loop = asyncio.get_running_loop()
    while len(active_tasks) < window_size:
        # Source iteration may block on file reads or SQLite lookups; keep it off
        # the event loop so in-flight provider calls can continue making progress.
        prepared = await loop.run_in_executor(executor, work_stream.next_prepared)
        if isinstance(prepared, _SourceExhausted):
            return True
        if prepared.immediate_error is not None:
            item = SourceItem(
                output_index=prepared.output_index,
                checkpoint_key=prepared.checkpoint_key,
                mapped_input=None,
                metadata=None,
            )
            completed, errored = await _settle_batch_workflow_item(
                store,
                item,
                None,
                prepared.immediate_error,
                record_builder=record_builder,
                on_progress=on_progress,
            )
            counters.add(completed=completed, errored=errored)
            continue
        scheduled_item = prepared.item
        assert scheduled_item is not None
        active_tasks[
            asyncio.create_task(_run_source_item(scheduled_item, operation))
        ] = scheduled_item
    return False


async def _settle_completed_tasks(
    *,
    done_tasks: set[asyncio.Task[tuple[Any, BaseException | None]]],
    active_tasks: dict[asyncio.Task[tuple[Any, BaseException | None]], SourceItem],
    store: RunStore,
    record_builder: RecordBuilder,
    counters: _BatchWorkflowCounters,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Settle all tasks that completed in the active window."""

    for task in done_tasks:
        item = active_tasks.pop(task)
        result, error = task.result()
        completed, errored = await _settle_batch_workflow_item(
            store,
            item,
            result,
            error,
            record_builder=record_builder,
            on_progress=on_progress,
        )
        counters.add(completed=completed, errored=errored)


async def _close_batch_workflow_resources(
    *,
    work_stream: WorkStream | None,
    executor: ThreadPoolExecutor,
    source: SourceStream,
    store: RunStore,
) -> None:
    """Close resources and re-raise the first cleanup error."""

    loop = asyncio.get_running_loop()
    cleanup_error: BaseException | None = None
    if work_stream is not None:
        try:
            await loop.run_in_executor(executor, work_stream.close)
        except BaseException as exc:  # noqa: BLE001
            cleanup_error = exc
    for close in (source.close, store.close):
        try:
            close()
        except BaseException as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


async def run_batch_workflow(
    *,
    source: SourceStream,
    store: RunStore,
    operation: Operation,
    record_builder: RecordBuilder,
    resume: bool,
    mapping_fingerprint: str,
    window_size: int,
    on_progress: Callable[[], Any] | None = None,
) -> BatchWorkflowSummary:
    """Run source items through a bounded endpoint operation.

    Parameters
    ----------
    source
        Source stream that can produce all logical workflow rows.
    store
        Run store that owns persistence, resume selection, and settlement.
    operation
        Async endpoint operation to run for each schedulable source item.
    record_builder
        Function that turns an item result or error into the output record.
    resume
        Whether the run should skip rows already settled by the store.
    mapping_fingerprint
        Stable fingerprint for the row mapping strategy. The store uses it to
        reject incompatible resumed runs.
    window_size
        Maximum number of endpoint operations in flight.
    on_progress
        Optional callback invoked after each row is settled.

    Returns
    -------
    BatchWorkflowSummary
        Counts for rows completed, errored, skipped, and the output path.

    Raises
    ------
    ValueError
        If ``window_size`` is less than one.
    BaseException
        Propagates operation setup, settlement, cancellation, and cleanup
        failures after cancelling active work.
    """

    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    counters = _BatchWorkflowCounters()
    active_tasks: dict[asyncio.Task[tuple[Any, BaseException | None]], SourceItem] = {}
    source_exhausted = False
    work_stream: WorkStream | None = None

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="infermesh-workflow-prep",
    ) as preparer_executor:
        try:
            # The store returns a work stream instead of a list so large resumes
            # can stay streaming and avoid loading all pending rows into memory.
            work_stream = store.open(
                source,
                resume=resume,
                mapping_fingerprint=mapping_fingerprint,
            )

            source_exhausted = await _fill_work_window(
                work_stream=work_stream,
                executor=preparer_executor,
                active_tasks=active_tasks,
                window_size=window_size,
                operation=operation,
                store=store,
                record_builder=record_builder,
                counters=counters,
                on_progress=on_progress,
            )
            while active_tasks:
                done_tasks, _ = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await _settle_completed_tasks(
                    done_tasks=done_tasks,
                    active_tasks=active_tasks,
                    store=store,
                    record_builder=record_builder,
                    counters=counters,
                    on_progress=on_progress,
                )
                if not source_exhausted:
                    source_exhausted = await _fill_work_window(
                        work_stream=work_stream,
                        executor=preparer_executor,
                        active_tasks=active_tasks,
                        window_size=window_size,
                        operation=operation,
                        store=store,
                        record_builder=record_builder,
                        counters=counters,
                        on_progress=on_progress,
                    )
        except BaseException:
            await cancel_tasks(list(active_tasks))
            raise
        finally:
            await _close_batch_workflow_resources(
                work_stream=work_stream,
                executor=preparer_executor,
                source=source,
                store=store,
            )

    if resume and counters.completed == 0 and counters.errored == 0:
        sys.stderr.write("Nothing to do — all rows already completed.\n")
    skipped = store.skipped
    return BatchWorkflowSummary(
        total=counters.completed + counters.errored + skipped,
        completed=counters.completed,
        errored=counters.errored,
        skipped=skipped,
        output_path=store.output_path,
    )
