from __future__ import annotations

import asyncio
from typing import Any

import pytest

from infermesh.client import LMClient
from tests.fakes import FakeLiteLLM


class FailingFakeLiteLLM(FakeLiteLLM):
    """Raises ``RuntimeError`` for configured inputs."""

    def __init__(self, fail_on: set[str]) -> None:
        super().__init__()
        self.fail_on = fail_on

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        content = kwargs["messages"][0]["content"]
        if content in self.fail_on:
            raise RuntimeError(f"Simulated failure for: {content}")
        return await super().acompletion(**kwargs)

    async def aembedding(self, **kwargs: Any) -> dict[str, Any]:
        for input_text in kwargs["input"]:
            if input_text in self.fail_on:
                raise RuntimeError(f"Simulated failure for: {input_text}")
        return await super().aembedding(**kwargs)


@pytest.fixture
def failing_fake_client(monkeypatch: pytest.MonkeyPatch) -> LMClient:
    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: FailingFakeLiteLLM(fail_on={"bad"}),
    )
    return LMClient(model="openai/test", api_base="http://localhost:8000/v1")


@pytest.mark.asyncio
async def test_sync_batch_inside_running_loop(fake_client: LMClient) -> None:
    result = fake_client.generate_batch(["alpha", "beta"])
    assert len(result) == 2


def test_generate_batch_errors_populated_on_all_success(fake_client: LMClient) -> None:
    result = fake_client.generate_batch(["hello", "world"])
    assert result.errors is not None
    assert all(error is None for error in result.errors)


@pytest.mark.asyncio
async def test_agenerate_batch_default_captures_partial_failure(
    failing_fake_client: LMClient,
) -> None:
    result = await failing_fake_client.agenerate_batch(["good", "bad", "good"])
    assert len(result) == 3
    assert result.errors is not None
    assert result.results[0] is not None
    assert result.results[1] is None
    assert isinstance(result.errors[1], RuntimeError)


def test_generate_batch_sync_captures_partial_failure(
    failing_fake_client: LMClient,
) -> None:
    result = failing_fake_client.generate_batch(["good", "bad"])
    assert result.errors is not None
    assert result.results[0] is not None
    assert result.results[1] is None
    assert isinstance(result.errors[1], RuntimeError)


@pytest.mark.asyncio
async def test_agenerate_batch_all_fail(failing_fake_client: LMClient) -> None:
    result = await failing_fake_client.agenerate_batch(["bad"])
    assert result.results == [None]
    assert result.errors is not None
    assert isinstance(result.errors[0], RuntimeError)


@pytest.mark.asyncio
async def test_agenerate_batch_return_exceptions_false_raises(
    failing_fake_client: LMClient,
) -> None:
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await failing_fake_client.agenerate_batch(
            ["good", "bad"],
            return_exceptions=False,
        )


@pytest.mark.asyncio
async def test_agenerate_batch_return_exceptions_false_cancels_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    completed: list[str] = []
    cancelled: list[str] = []

    class GatedFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> Any:
            content = kwargs["messages"][0]["content"]
            if content == "bad":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await gate.wait()
                result = await super().acompletion(**kwargs)
                completed.append(content)
                return result
            except asyncio.CancelledError:
                cancelled.append(content)
                raise

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: GatedFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    with pytest.raises(RuntimeError, match="boom"):
        await client.agenerate_batch(["slow", "bad"], return_exceptions=False)

    assert completed == []
    assert cancelled == ["slow"]
    client.close()


@pytest.mark.asyncio
async def test_agenerate_batch_bounded_window_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0
    completion_order: list[str] = []
    delays = {"a": 0.05, "b": 0.01, "c": 0.01, "d": 0.01}

    class WindowedFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            nonlocal active, peak
            content = kwargs["messages"][0]["content"]
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(delays[content])
                completion_order.append(content)
                return await super().acompletion(**kwargs)
            finally:
                active -= 1

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: WindowedFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )
    batch = await client.agenerate_batch(["a", "b", "c", "d"])

    assert peak == 2
    assert completion_order[0] == "b"
    assert [result.output_text if result is not None else None for result in batch] == [
        "a",
        "b",
        "c",
        "d",
    ]
    client.close()


@pytest.mark.asyncio
async def test_agenerate_batch_bounded_strict_failure_does_not_start_queued_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    seen: list[str] = []
    cancelled: list[str] = []

    class BoundedStrictFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            content = kwargs["messages"][0]["content"]
            seen.append(content)
            if content == "bad":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.append(content)
                raise
            return await super().acompletion(**kwargs)

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: BoundedStrictFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.agenerate_batch(
            ["slow", "bad", "queued"],
            return_exceptions=False,
        )

    assert "queued" not in seen
    assert sorted(seen) == ["bad", "slow"]
    assert cancelled == ["slow"]
    client.close()


@pytest.mark.asyncio
async def test_batch_result_length_parity(failing_fake_client: LMClient) -> None:
    inputs = ["good", "bad", "good", "bad", "good"]
    result = await failing_fake_client.agenerate_batch(inputs)
    assert len(result.results) == len(inputs)
    assert result.errors is not None
    assert len(result.errors) == len(inputs)


@pytest.mark.asyncio
async def test_aembed_batch_recursively_isolates_bad_items(
    failing_fake_client: LMClient,
) -> None:
    result = await failing_fake_client.aembed_batch(["good-a", "bad", "good-b"])
    assert result.errors is not None
    assert result.results[0] is not None
    assert result.results[1] is None
    assert result.results[2] is not None
    assert isinstance(result.errors[1], RuntimeError)
    assert result.errors[0] is None
    assert result.errors[2] is None


@pytest.mark.asyncio
async def test_aembed_batch_return_exceptions_false_raises(
    failing_fake_client: LMClient,
) -> None:
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await failing_fake_client.aembed_batch(["good", "bad"], return_exceptions=False)


@pytest.mark.asyncio
async def test_aembed_batch_callbacks_are_per_item(
    fake_client: LMClient,
) -> None:
    progress_calls: list[tuple[int, int]] = []
    result_calls: list[tuple[int, Any, Any]] = []

    batch = await fake_client.aembed_batch(
        ["a", "b", "c"],
        micro_batch_size=2,
        on_progress=lambda done, total: progress_calls.append((done, total)),
        on_result=lambda index, result, error: result_calls.append(
            (index, result, error)
        ),
    )

    assert len(batch.results) == 3
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)
    assert len(result_calls) == 3
    assert sorted(index for index, _, _ in result_calls) == [0, 1, 2]
    assert all(error is None for _, _, error in result_calls)


@pytest.mark.asyncio
async def test_aembed_batch_strict_mode_cancels_siblings_and_skips_queued_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    seen: list[str] = []
    cancelled: list[str] = []

    class StrictEmbeddingFakeLiteLLM(FakeLiteLLM):
        async def aembedding(self, **kwargs: Any) -> dict[str, Any]:
            content = kwargs["input"][0]
            seen.append(content)
            if content == "bad":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.append(content)
                raise
            return await super().aembedding(**kwargs)

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: StrictEmbeddingFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.aembed_batch(
            ["slow", "bad", "queued"],
            micro_batch_size=1,
            return_exceptions=False,
        )

    assert "queued" not in seen
    assert sorted(seen) == ["bad", "slow"]
    assert cancelled == ["slow"]
    client.close()


def test_embed_batch_rejects_invalid_micro_batch_size(fake_client: LMClient) -> None:
    with pytest.raises(ValueError, match="micro_batch_size"):
        fake_client.embed_batch(["a"], micro_batch_size=0)
    fake_client.close()


@pytest.mark.asyncio
async def test_atranscribe_batch_captures_per_item_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTranscriptionFakeLiteLLM(FakeLiteLLM):
        async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["file"] == b"bad":
                raise RuntimeError("transcription boom")
            return await super().atranscription(**kwargs)

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: FailingTranscriptionFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    batch = await client.atranscribe_batch([b"good", b"bad"])

    assert batch.errors is not None
    assert batch.results[0] is not None
    assert batch.results[1] is None
    assert isinstance(batch.errors[1], RuntimeError)
    client.close()


@pytest.mark.asyncio
async def test_atranscribe_batch_return_exceptions_false_cancels_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    seen: list[bytes] = []
    cancelled: list[bytes] = []

    class StrictTranscriptionFakeLiteLLM(FakeLiteLLM):
        async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["file"]
            assert isinstance(payload, bytes)
            seen.append(payload)
            if payload == b"bad":
                await asyncio.sleep(0)
                raise RuntimeError("boom")
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.append(payload)
                raise
            return await super().atranscription(**kwargs)

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: StrictTranscriptionFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await client.atranscribe_batch(
            [b"slow", b"bad", b"queued"],
            return_exceptions=False,
        )

    assert b"queued" not in seen
    assert sorted(seen) == [b"bad", b"slow"]
    assert cancelled == [b"slow"]
    client.close()


@pytest.mark.asyncio
async def test_atranscribe_batch_normalizes_inputs_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    def fake_normalize(
        input_data: Any,
        *,
        max_bytes: int | None = None,
    ) -> tuple[str, bytes]:
        assert max_bytes is None
        assert isinstance(input_data, str)
        normalized.append(input_data)
        return input_data, b"audio"

    class LazyTranscriptionFakeLiteLLM(FakeLiteLLM):
        async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["file"]
            assert isinstance(payload, tuple)
            if payload[0] == "first":
                started.set()
                await release.wait()
            return await super().atranscription(**kwargs)

    monkeypatch.setattr(
        "infermesh._transcription.normalize_transcription_input",
        fake_normalize,
    )
    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: LazyTranscriptionFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=1,
    )

    batch_task = asyncio.create_task(
        client.atranscribe_batch(
            ["first", "second"],
            max_transcription_bytes=None,
            return_exceptions=False,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0.02)
    assert normalized == ["first"]
    release.set()
    await batch_task
    assert normalized == ["first", "second"]
    client.close()


@pytest.mark.asyncio
async def test_atranscribe_batch_on_progress_callback_error_cancels_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[bytes] = []

    class ProgressAbortTranscriptionFakeLiteLLM(FakeLiteLLM):
        async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["file"]
            assert isinstance(payload, bytes)
            if payload == b"fast":
                return await super().atranscription(**kwargs)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(payload)
                raise
            raise AssertionError("request should have been cancelled")

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: ProgressAbortTranscriptionFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    with pytest.raises(RuntimeError, match="progress callback failed"):
        await client.atranscribe_batch(
            [b"fast", b"slow-a", b"slow-b"],
            on_progress=lambda *_: (_ for _ in ()).throw(
                RuntimeError("progress callback failed")
            ),
        )
    assert sorted(cancelled) == [b"slow-a", b"slow-b"]
    client.close()


@pytest.mark.asyncio
async def test_atranscribe_batch_cancellation_cancels_active_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled: list[bytes] = []
    seen: list[bytes] = []

    class GatedTranscriptionFakeLiteLLM(FakeLiteLLM):
        async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs["file"]
            assert isinstance(payload, bytes)
            seen.append(payload)
            if len(seen) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(payload)
                raise
            raise AssertionError("request should have been cancelled")

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: GatedTranscriptionFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )
    batch_task = asyncio.create_task(client.atranscribe_batch([b"a", b"b"]))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    batch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch_task
    assert sorted(cancelled) == [b"a", b"b"]
    client.close()


def test_generate_batch_on_progress_calls_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    client = LMClient(model="openai/test", api_base="http://localhost")
    calls: list[tuple[int, int]] = []
    client.generate_batch(
        ["a", "b", "c"],
        on_progress=lambda done, total: calls.append((done, total)),
    )
    assert len(calls) == 3
    assert calls[-1] == (3, 3)
    assert all(total == 3 for _, total in calls)
    client.close()


@pytest.mark.asyncio
async def test_agenerate_batch_on_progress_cancellation_cancels_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled: list[str] = []
    seen: list[str] = []

    class GatedFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            content = kwargs["messages"][0]["content"]
            seen.append(content)
            if len(seen) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(content)
                raise
            raise AssertionError("request should have been cancelled")

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: GatedFakeLiteLLM(),
    )
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        max_parallel_requests=2,
    )
    batch_task = asyncio.create_task(
        client.agenerate_batch(["a", "b"], on_progress=lambda *_: None)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    batch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch_task
    assert sorted(cancelled) == ["a", "b"]
    client.close()


@pytest.mark.asyncio
async def test_agenerate_batch_on_progress_callback_error_cancels_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled: list[str] = []

    class ProgressAbortFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            content = kwargs["messages"][0]["content"]
            if content == "fast":
                return await super().acompletion(**kwargs)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(content)
                raise
            raise AssertionError("request should have been cancelled")

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: ProgressAbortFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    with pytest.raises(RuntimeError, match="progress callback failed"):
        await client.agenerate_batch(
            ["fast", "slow-a", "slow-b"],
            on_progress=lambda *_: (_ for _ in ()).throw(
                RuntimeError("progress callback failed")
            ),
        )
    assert sorted(cancelled) == ["slow-a", "slow-b"]
    client.close()


def test_on_result_fires_with_correct_index_and_result(
    fake_client: LMClient,
) -> None:
    calls: list[tuple[int, Any, Any]] = []
    fake_client.generate_batch(
        ["a", "b", "c"],
        on_result=lambda idx, result, error: calls.append((idx, result, error)),
    )
    assert len(calls) == 3
    assert sorted(idx for idx, _, _ in calls) == [0, 1, 2]
    for idx, result, error in calls:
        assert error is None
        assert result is not None
        assert result.output_text == ["a", "b", "c"][idx]
    fake_client.close()


def test_on_result_captures_error_on_failure(
    failing_fake_client: LMClient,
) -> None:
    calls: list[tuple[int, Any, Any]] = []
    failing_fake_client.generate_batch(
        ["good", "bad", "good"],
        on_result=lambda idx, result, error: calls.append((idx, result, error)),
    )
    error_calls = [
        (idx, result, error) for idx, result, error in calls if error is not None
    ]
    assert len(calls) == 3
    assert len(error_calls) == 1
    _, result, error = error_calls[0]
    assert result is None
    assert isinstance(error, RuntimeError)
    failing_fake_client.close()
