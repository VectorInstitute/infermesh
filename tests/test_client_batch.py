from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lm_client.client import LMClient
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
async def test_batch_result_length_parity(failing_fake_client: LMClient) -> None:
    inputs = ["good", "bad", "good", "bad", "good"]
    result = await failing_fake_client.agenerate_batch(inputs)
    assert len(result.results) == len(inputs)
    assert result.errors is not None
    assert len(result.errors) == len(inputs)


@pytest.mark.asyncio
async def test_aembed_batch_on_failure_all_items_get_same_error(
    failing_fake_client: LMClient,
) -> None:
    result = await failing_fake_client.aembed_batch(["good", "bad"])
    assert result.errors is not None
    assert all(item is None for item in result.results)
    assert all(isinstance(error, RuntimeError) for error in result.errors)
    assert result.errors[0] is result.errors[1]


@pytest.mark.asyncio
async def test_aembed_batch_return_exceptions_false_raises(
    failing_fake_client: LMClient,
) -> None:
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await failing_fake_client.aembed_batch(["good", "bad"], return_exceptions=False)


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
    client = LMClient(model="openai/test", api_base="http://localhost")
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
