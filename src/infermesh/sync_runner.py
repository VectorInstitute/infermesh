"""Notebook-safe synchronous wrapper for async coroutines.

`SyncRunner` runs coroutines on a dedicated background event-loop
thread, which makes the synchronous wrappers in [LMClient][infermesh.LMClient]
safe to call from notebook environments that already have a running event loop
(e.g. Jupyter, IPython).  Using `asyncio.run_until_complete` on the caller's
thread would raise a `RuntimeError` in those contexts; submitting work to a
separate thread avoids the conflict entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")


class SyncRunner:
    """Run async coroutines synchronously on a dedicated background event loop.

    A single daemon thread owns the event loop for the lifetime of the runner.
    Coroutines submitted via `run` are scheduled onto that loop with
    `asyncio.run_coroutine_threadsafe` and the calling thread blocks
    until the result (or exception) is available.

    Notes
    -----
    This design is intentionally simple: one loop, one thread.  It is not
    intended for high-throughput parallel synchronous calls from multiple OS
    threads; use the async API directly for that.

    The background thread is started in `__init__` and the loop is kept
    alive until `close` is called (or the process exits, since the
    thread is a daemon).

    Examples
    --------
    Typical usage (via [LMClient][infermesh.LMClient]; you do not need to
    instantiate `SyncRunner` directly):

    >>> runner = SyncRunner()
    >>> result = runner.run(some_async_function())
    >>> runner.close()

    Safe inside a running event loop (e.g. a Jupyter cell):

    >>> # The background loop is separate from the notebook loop, so this works:
    >>> result = runner.run(some_async_function())
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="infermesh-sync-runner",
            daemon=True,
        )
        self._started = threading.Event()
        self._thread.start()
        self._started.wait()

    def run(self, coroutine: Coroutine[object, object, T]) -> T:
        """Run a coroutine to completion and return its result.

        Submits ``coroutine`` to the background event loop and blocks the
        calling thread until the coroutine finishes.  Any exception raised
        inside the coroutine is re-raised on the calling thread.

        Parameters
        ----------
        coroutine : Coroutine
            An unawaited coroutine object (the result of calling an ``async
            def`` function without ``await``).

        Returns
        -------
        T
            Whatever the coroutine returns.

        Raises
        ------
        Exception
            Re-raises any exception the coroutine raised, on the calling
            thread.

        Examples
        --------
        >>> async def add(a: int, b: int) -> int:
        ...     return a + b
        >>> runner = SyncRunner()
        >>> runner.run(add(1, 2))
        3
        """
        future: Future[T] = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self) -> None:
        """Stop the background event loop and join the worker thread.

        After `close()` is called, any subsequent call to `run` will
        raise because the underlying loop is no longer running.  Calling
        ``close()`` more than once is safe and has no effect.

        Notes
        -----
        Before stopping the loop, all pending tasks are cancelled and awaited.
        This suppresses "Task was destroyed but it is pending!" warnings from
        third-party libraries (e.g. LiteLLM's internal ``LoggingWorker``) that
        leave background tasks on the event loop.

        The worker thread is given a 1-second join timeout.  If the loop does
        not shut down within that window (e.g. a coroutine is stuck), the
        thread may still be alive but the loop will be closed on return.
        """
        if self._loop.is_closed():
            return
        if self._loop.is_running():
            # Cancel any background tasks left by third-party libraries before
            # stopping, so they don't trigger "Task was destroyed" warnings.
            cancel_future: Future[None] = asyncio.run_coroutine_threadsafe(
                self._cancel_pending_tasks(), self._loop
            )
            with contextlib.suppress(Exception):
                cancel_future.result(timeout=2.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if not self._loop.is_closed():
            self._loop.close()

    @staticmethod
    async def _cancel_pending_tasks() -> None:
        """Cancel and await all tasks on the current loop except this one."""
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    def _run_loop(self) -> None:
        """Run the event loop on the dedicated background thread."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()
