from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any

import pytest

from infermesh.sync_runner import SyncRunner


def test_sync_runner_waits_for_cancel_cleanup_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SyncRunner()
    started = threading.Event()
    cleanup_started = threading.Event()
    allow_cleanup_finish = threading.Event()
    cleaned = threading.Event()
    returned = threading.Event()
    result: dict[str, BaseException | None] = {"error": None}

    async def blocked() -> None:
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        finally:
            cleanup_started.set()
            await asyncio.to_thread(allow_cleanup_finish.wait)
            cleaned.set()

    original_wait_for_future = runner._wait_for_future
    blocking_waits = 0

    def interrupting_wait_for_future(
        future: Future[Any], timeout: float | None = None
    ) -> Any:
        nonlocal blocking_waits
        if timeout is None:
            blocking_waits += 1
        if blocking_waits == 2 and timeout is None:
            started.wait(timeout=1.0)
            blocking_waits += 1
            raise KeyboardInterrupt
        return original_wait_for_future(future, timeout)

    monkeypatch.setattr(runner, "_wait_for_future", interrupting_wait_for_future)

    def run_and_capture() -> None:
        try:
            runner.run(blocked())
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            returned.set()

    try:
        worker = threading.Thread(target=run_and_capture, daemon=True)
        worker.start()
        assert cleanup_started.wait(timeout=1.0)
        assert not returned.wait(timeout=0.05)
        allow_cleanup_finish.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert isinstance(result["error"], KeyboardInterrupt)
        assert cleaned.is_set()
    finally:
        allow_cleanup_finish.set()
        runner.close()


def test_sync_runner_waits_for_cancel_cleanup_when_interrupt_during_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SyncRunner()
    started = threading.Event()
    cleaned = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        finally:
            cleaned.set()

    original_wait_for_future = runner._wait_for_future
    blocking_waits = 0

    def interrupting_wait_for_future(
        future: Future[Any], timeout: float | None = None
    ) -> Any:
        nonlocal blocking_waits
        if timeout is None:
            blocking_waits += 1
        if blocking_waits == 1 and timeout is None:
            started.wait(timeout=1.0)
            blocking_waits += 1
            raise KeyboardInterrupt
        return original_wait_for_future(future, timeout)

    monkeypatch.setattr(runner, "_wait_for_future", interrupting_wait_for_future)

    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run(blocked())
        assert cleaned.is_set()
    finally:
        runner.close()
