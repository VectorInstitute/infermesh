"""Shared helpers for internal batch runners."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any


async def cancel_tasks(tasks: Sequence[asyncio.Task[Any]]) -> None:
    """Cancel unfinished tasks and await their cleanup."""

    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
