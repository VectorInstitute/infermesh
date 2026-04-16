from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from lm_client._utils import parse_model_output_with_format
from lm_client.client import LMClient
from lm_client.types import DeploymentConfig, image_block
from tests.fakes import (
    FakeLiteLLM,
    FakeRouter,
    ParsedResponse,
    ToolCallingFakeLiteLLM,
)


@pytest.mark.asyncio
async def test_generate_responses_path(fake_client: LMClient) -> None:
    result = await fake_client.agenerate(
        {"input": "hello"},
        endpoint="responses",
        response_format=ParsedResponse,
        parse_output=True,
    )
    assert result.output_parsed is not None
    assert result.output_parsed.answer == "ok"


@pytest.mark.asyncio
async def test_response_format_implies_parse_output(fake_client: LMClient) -> None:
    result = await fake_client.agenerate(
        {"input": "hello"},
        endpoint="responses",
        response_format=ParsedResponse,
    )
    assert result.output_parsed is not None
    assert result.output_parsed.answer == "ok"


def test_router_mode_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    with pytest.raises(ValueError, match="model.*required"):
        LMClient(
            deployments={
                "replica-1": DeploymentConfig(
                    model="openai/gpt-4",
                    api_base="http://a",
                ),
                "replica-2": DeploymentConfig(
                    model="openai/gpt-4",
                    api_base="http://b",
                ),
            }
        )


@pytest.mark.asyncio
async def test_router_mode_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr(LMClient, "_build_router", lambda self: FakeRouter())
    client = LMClient(
        model="gpt-4",
        deployments={
            "replica-1": DeploymentConfig(model="openai/gpt-4", api_base="http://a"),
            "replica-2": DeploymentConfig(model="openai/gpt-4", api_base="http://b"),
        },
    )
    result = await client.agenerate("hello")
    assert result.output_text == "router-chat-output"
    assert result.metrics is not None
    assert result.metrics.deployment == "replica-1"
    client.close()


@pytest.mark.asyncio
async def test_text_completion_endpoint_uses_text_completion_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    client = LMClient(model="openai/test", api_base="http://localhost:8000/v1")
    result = await client.agenerate("hello world", endpoint="text_completion")
    assert result.output_text == "text::hello world"
    client.close()


@pytest.mark.asyncio
async def test_text_completion_router_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr(LMClient, "_build_router", lambda self: FakeRouter())
    client = LMClient(
        model="gpt-4",
        deployments={
            "replica-1": DeploymentConfig(model="openai/gpt-4", api_base="http://a"),
            "replica-2": DeploymentConfig(model="openai/gpt-4", api_base="http://b"),
        },
    )
    result = await client.agenerate("hello world", endpoint="text_completion")
    assert result.output_text == "router-text::hello world"
    assert result.metrics is not None
    assert result.metrics.deployment == "replica-1"
    client.close()


def test_text_completion_does_not_accept_chat_input(fake_client: LMClient) -> None:
    with pytest.raises(ValueError, match="Text completion expects a string input"):
        fake_client.generate(
            [{"role": "user", "content": "hello"}],
            endpoint="text_completion",
        )


@pytest.mark.asyncio
async def test_chat_completion_tool_calls_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: ToolCallingFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    result = await client.agenerate("hello")
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "lookup_weather"
    assert result.tool_calls[0].arguments == '{"city":"Paris"}'
    client.close()


def test_dict_schema_accepts_conformant_json() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    assert parse_model_output_with_format('{"answer": "ok"}', schema) == {
        "answer": "ok"
    }


def test_dict_schema_rejects_nonconformant_json() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    assert parse_model_output_with_format('{"answer": 42}', schema) is None


@pytest.mark.asyncio
async def test_inline_thinking_tokens_stripped_from_output_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ThinkingFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "id": "think-1",
                "model": kwargs["model"],
                "choices": [
                    {
                        "message": {
                            "content": '<think>\nsome reasoning\n</think>\n\n{"answer": "ok"}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 10,
                    "total_tokens": 13,
                },
            }

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: ThinkingFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    result = await client.agenerate(
        'say {"answer": "ok"}',
        response_format=ParsedResponse,
    )
    assert result.output_text == '{"answer": "ok"}'
    assert result.reasoning == "some reasoning"
    assert result.output_parsed is not None
    client.close()


@pytest.mark.asyncio
async def test_reasoning_content_field_populates_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReasoningContentFakeLiteLLM(FakeLiteLLM):
        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "id": "rc-1",
                "model": kwargs["model"],
                "choices": [
                    {
                        "message": {
                            "content": '{"answer": "ok"}',
                            "reasoning_content": "structured reasoning here",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 8,
                    "total_tokens": 11,
                },
            }

    monkeypatch.setattr(
        LMClient,
        "_create_litellm_module",
        lambda self: ReasoningContentFakeLiteLLM(),
    )
    client = LMClient(model="openai/test", api_base="http://localhost")
    result = await client.agenerate(
        'say {"answer": "ok"}',
        response_format=ParsedResponse,
    )
    assert result.output_text == '{"answer": "ok"}'
    assert result.reasoning == "structured reasoning here"
    assert result.output_parsed is not None
    client.close()


def test_image_block_from_bytes_requires_mime_type() -> None:
    with pytest.raises(ValueError, match="mime_type"):
        image_block(b"\xff\xd8\xff")


def test_image_block_from_bytes_roundtrips() -> None:
    raw = b"\x89PNG\r\n"
    url = image_block(raw, mime_type="image/png")["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_image_block_from_path_infers_mime_and_encodes(tmp_path: Path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n")
    url = image_block(image)["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG\r\n"


class FlakyFakeLiteLLM(FakeLiteLLM):
    def __init__(self, fail_count: int, error: Exception) -> None:
        super().__init__()
        self._fail_count = fail_count
        self._error = error
        self._calls = 0

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise self._error
        return await super().acompletion(**kwargs)


def test_generate_retries_on_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = FakeLiteLLM.RateLimitError("rate limited")
    flaky = FlakyFakeLiteLLM(fail_count=2, error=error)
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: flaky)
    client = LMClient(model="openai/test", api_base="http://localhost", max_retries=3)
    with patch("lm_client.client.asyncio.sleep", new_callable=AsyncMock):
        result = client.generate("hi")
    assert result.metrics is not None
    assert result.metrics.retries == 2
    assert flaky._calls == 3
    client.close()


def test_generate_raises_after_max_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = FakeLiteLLM.ServiceUnavailableError("down")
    flaky = FlakyFakeLiteLLM(fail_count=99, error=error)
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: flaky)
    client = LMClient(model="openai/test", api_base="http://localhost", max_retries=2)
    with (
        patch("lm_client.client.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(FakeLiteLLM.ServiceUnavailableError),
    ):
        client.generate("hi")
    assert flaky._calls == 3
    client.close()


def test_generate_does_not_retry_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonRetryableError(Exception):
        pass

    class AlwaysFailFakeLiteLLM(FakeLiteLLM):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
            self._calls += 1
            raise NonRetryableError("bad input")

    fake = AlwaysFailFakeLiteLLM()
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: fake)
    client = LMClient(model="openai/test", api_base="http://localhost", max_retries=3)
    with pytest.raises(NonRetryableError):
        client.generate("hi")
    assert fake._calls == 1
    client.close()


def test_generate_max_retries_zero_disables_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = FakeLiteLLM.RateLimitError("rate limited")
    flaky = FlakyFakeLiteLLM(fail_count=1, error=error)
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: flaky)
    client = LMClient(model="openai/test", api_base="http://localhost", max_retries=0)
    with pytest.raises(FakeLiteLLM.RateLimitError):
        client.generate("hi")
    assert flaky._calls == 1
    client.close()


def test_retry_after_header_sets_sleep_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MockResponse:
        headers = {"Retry-After": "7"}

    exc = FakeLiteLLM.RateLimitError("rate limited")
    exc.response = MockResponse()  # type: ignore[attr-defined]
    flaky = FlakyFakeLiteLLM(fail_count=1, error=exc)
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: flaky)
    client = LMClient(model="openai/test", api_base="http://localhost", max_retries=1)
    sleep_args: list[float] = []

    async def capture_sleep(seconds: float) -> None:
        sleep_args.append(seconds)

    with patch("lm_client.client.asyncio.sleep", capture_sleep):
        client.generate("hi")
    assert sleep_args == [7.0]
    client.close()
