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
    cleaned = threading.Event()

    async def blocked() -> None:
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        finally:
            # Cleanup can take longer than the main result wait; run() should
            # still wait for it before returning control to the caller.
            await asyncio.sleep(2.1)
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

    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run(blocked())
        assert cleaned.is_set()
    finally:
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
