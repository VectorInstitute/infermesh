import asyncio
import heapq
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class MockTimer:
    fire_at: float
    callback: Callable[..., Any]
    args: tuple[Any, ...]
    handle: "MockTimerHandle | None" = None

    def __lt__(self, other: "MockTimer") -> bool:
        if not isinstance(other, MockTimer):
            return NotImplemented
        return self.fire_at < other.fire_at


@dataclass
class MockTimerHandle:
    timer_ref: MockTimer
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class MockLoop:
    def __init__(self, time_provider: "MockTime") -> None:
        self._time_provider = time_provider
        self._tasks: list[asyncio.Task[Any]] = []

    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: tuple[Any, ...],
    ) -> MockTimerHandle:
        return self._time_provider.schedule_timer(delay, callback, *args)

    def create_future(self) -> asyncio.Future[bool]:
        return asyncio.Future()

    def call_soon(
        self,
        callback: Callable[..., Any],
        *args: tuple[Any, ...],
    ) -> None:
        callback(*args)

    def time(self) -> float:
        return self._time_provider.monotonic()

    def create_task(self, coroutine: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        self._tasks.append(task)
        return task


class MockTime:
    def __init__(self) -> None:
        self.mock_loop = MockLoop(self)
        self._current_time = 0.0
        self._timers: list[MockTimer] = []

    def monotonic(self) -> float:
        return self._current_time

    def get_running_loop(self) -> MockLoop:
        return self.mock_loop

    def schedule_timer(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: tuple[Any, ...],
    ) -> MockTimerHandle:
        fire_at = self._current_time + delay
        timer = MockTimer(fire_at=fire_at, callback=callback, args=args)
        handle = MockTimerHandle(timer_ref=timer)
        timer.handle = handle
        heapq.heappush(self._timers, timer)
        return handle

    async def advance_time(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("Cannot advance time backwards.")
        target_time = self._current_time + delta
        processed_callbacks = False
        while self._timers and self._timers[0].fire_at <= target_time:
            next_timer = heapq.heappop(self._timers)
            self._current_time = next_timer.fire_at
            if next_timer.handle is not None and next_timer.handle.cancelled:
                continue
            next_timer.callback(*next_timer.args)
            processed_callbacks = True
            await asyncio.sleep(0)
        self._current_time = target_time
        if processed_callbacks:
            await asyncio.sleep(0)
