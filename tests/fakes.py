from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from infermesh import cli
from infermesh.client import LMClient
from infermesh.types import (
    BatchResult,
    EmbeddingResult,
    GenerationResult,
    RequestMetrics,
    TokenUsage,
    TranscriptionResult,
)


class FakeLiteLLM:
    class RateLimitError(Exception):
        pass

    class ServiceUnavailableError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    class Timeout(Exception):  # noqa: N818
        pass

    class InternalServerError(Exception):
        pass

    def __init__(self) -> None:
        self.last_calls: list[tuple[str, dict[str, Any]]] = []

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("acompletion", kwargs))
        return {
            "id": "chat-1",
            "model": kwargs["model"],
            "choices": [
                {
                    "message": {"content": kwargs["messages"][0]["content"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "_response_headers": {"x-ratelimit-limit-requests": "100"},
        }

    async def aresponses(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("aresponses", kwargs))
        return {
            "id": "resp-1",
            "model": kwargs["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"answer":"ok"}'}],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 6,
                "total_tokens": 11,
            },
            "_response_headers": {"x-ratelimit-limit-requests": "100"},
        }

    async def atext_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("atext_completion", kwargs))
        return {
            "id": "textcomp-1",
            "model": kwargs["model"],
            "choices": [{"text": f"text::{kwargs['prompt']}", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "_response_headers": {"x-ratelimit-limit-requests": "100"},
        }

    async def aembedding(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("aembedding", kwargs))
        return {
            "id": "embed-1",
            "model": kwargs["model"],
            "data": [{"embedding": [0.1, 0.2, 0.3]} for _ in kwargs["input"]],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
        }

    async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("atranscription", kwargs))
        return {
            "id": "tx-1",
            "model": kwargs["model"],
            "text": "hello transcript",
            "duration": 1.2,
            "language": "en",
        }

    def token_counter(self, **kwargs: Any) -> int:
        return len(str(kwargs["messages"])) // 4 + 1


class FakeRouter:
    def __init__(self) -> None:
        self.last_calls: list[tuple[str, dict[str, Any]]] = []

    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("acompletion", kwargs))
        return {
            "id": "router-chat-1",
            "model": kwargs["model"],
            "choices": [
                {
                    "message": {"content": "router-chat-output"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "_hidden_params": {"deployment": "replica-1"},
        }

    async def atext_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("atext_completion", kwargs))
        return {
            "id": "router-textcomp-1",
            "model": kwargs["model"],
            "choices": [
                {"text": f"router-text::{kwargs['prompt']}", "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "_hidden_params": {"deployment": "replica-1"},
        }

    async def aembedding(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("aembedding", kwargs))
        return {
            "id": "router-embed-1",
            "model": kwargs["model"],
            "data": [{"embedding": [0.4, 0.5]} for _ in kwargs["input"]],
        }

    async def atranscription(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("atranscription", kwargs))
        return {
            "id": "router-tx-1",
            "model": kwargs["model"],
            "text": "router transcript",
        }


class ParsedResponse(BaseModel):
    answer: str


class ToolCallingFakeLiteLLM(FakeLiteLLM):
    async def acompletion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_calls.append(("acompletion", kwargs))
        return {
            "id": "chat-tool-1",
            "model": kwargs["model"],
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }


class FakeCLIClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.embed_batch_sizes: list[int] = []

    def close(self) -> None:
        self.closed = True

    def generate(self, input_data: Any, **kwargs: Any) -> GenerationResult:
        return GenerationResult(
            model_id="test-model",
            output_text=f"generated:{input_data}",
            request_id="req-1",
        )

    def generate_batch(
        self,
        input_batch: list[Any],
        **kwargs: Any,
    ) -> BatchResult[GenerationResult]:
        on_result = kwargs.get("on_result")
        results = [
            GenerationResult(
                model_id="test-model",
                output_text=f"generated:{item}",
                request_id=f"req-{index}",
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
                metrics=RequestMetrics(
                    queue_wait_s=0.01,
                    service_time_s=0.02,
                    end_to_end_s=0.03,
                    deployment="replica-1",
                ),
            )
            for index, item in enumerate(input_batch, start=1)
        ]
        if on_result is not None:
            for index, result in enumerate(results):
                on_result(index, result, None)
        return BatchResult(results=cast(list[GenerationResult | None], results))

    def embed_batch(
        self,
        input_batch: list[str],
        **kwargs: Any,
    ) -> BatchResult[EmbeddingResult]:
        self.embed_batch_sizes.append(len(input_batch))
        results = [
            EmbeddingResult(
                model_id="embed-model",
                embedding=[0.1, 0.2],
                request_id=f"embed-{index}",
                token_usage=TokenUsage(
                    prompt_tokens=8,
                    completion_tokens=0,
                    total_tokens=8,
                ),
                metrics=RequestMetrics(
                    queue_wait_s=0.01,
                    service_time_s=0.02,
                    end_to_end_s=0.03,
                    deployment="replica-1",
                ),
            )
            for index, _ in enumerate(input_batch, start=1)
        ]
        return BatchResult(results=cast(list[EmbeddingResult | None], results))

    def transcribe(self, path: str, **kwargs: Any) -> TranscriptionResult:
        return TranscriptionResult(
            model_id="tx-model",
            text=f"transcribed:{Path(path).name}",
            request_id="tx-1",
        )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> LMClient:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    return LMClient(model="openai/test", api_base="http://localhost:8000/v1")


@pytest.fixture
def fake_client_builder(monkeypatch: pytest.MonkeyPatch) -> list[FakeCLIClient]:
    created: list[FakeCLIClient] = []

    def build_client(*args: Any, **kwargs: Any) -> FakeCLIClient:
        client = FakeCLIClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(cli, "_build_client", build_client)
    return created
