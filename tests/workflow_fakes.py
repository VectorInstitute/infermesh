"""In-memory fakes for workflow boundary tests."""

from __future__ import annotations

from pathlib import Path

from infermesh._workflow.models import (
    _SOURCE_EXHAUSTED,
    PreparedItem,
    SettledRecord,
    SourceStream,
    WorkStream,
    _SourceExhausted,
)


class InMemoryWorkStream:
    """Blocking work stream backed by a Python list."""

    def __init__(self, items: list[PreparedItem]) -> None:
        self._items = list(items)
        self._index = 0
        self.closed = False

    def next_prepared(self) -> PreparedItem | _SourceExhausted:
        if self._index >= len(self._items):
            return _SOURCE_EXHAUSTED
        item = self._items[self._index]
        self._index += 1
        return item

    def close(self) -> None:
        self.closed = True


class InMemorySourceStream:
    """Source stream backed by prepared items."""

    def __init__(self, items: list[PreparedItem]) -> None:
        self._items = list(items)
        self.closed = False

    def iter_all(self) -> WorkStream:
        return InMemoryWorkStream(self._items)

    def close(self) -> None:
        self.closed = True


class InMemoryRunStore:
    """Run store that records settled rows in memory."""

    def __init__(
        self,
        *,
        skipped: int = 0,
        output_path: Path | None = None,
    ) -> None:
        self._skipped = skipped
        self._output_path = output_path
        self.records: list[SettledRecord] = []
        self.closed = False

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def open(
        self,
        source: SourceStream,
        *,
        resume: bool,
        mapping_fingerprint: str,
    ) -> WorkStream:
        del resume, mapping_fingerprint
        return source.iter_all()

    async def settle(self, settled: SettledRecord) -> None:
        self.records.append(settled)

    def close(self) -> None:
        self.closed = True


__all__ = [
    "InMemoryRunStore",
    "InMemorySourceStream",
    "InMemoryWorkStream",
]
