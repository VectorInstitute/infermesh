from __future__ import annotations

import asyncio
import io
import logging
import time as _time
from pathlib import Path
from typing import Any

import pytest

from lm_client._utils import normalize_transcription_input
from lm_client.client import DEFAULT_MAX_TRANSCRIPTION_BYTES, LMClient
from lm_client.rate_limiter import RateLimiterAcquisitionHandle
from tests.fakes import FakeLiteLLM


@pytest.mark.asyncio
async def test_rate_limiter_adjust_receives_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, dict[str, Any] | None]] = []

    async def fake_adjust(
        self: Any,
        handle: Any,
        actual_tokens: int,
        response_headers: dict[str, Any] | None = None,
    ) -> None:
        calls.append((actual_tokens, response_headers))

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.adjust", fake_adjust)
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        rpm=10,
        tpm=1000,
    )
    await client.agenerate("hello")
    assert calls[-1][1] == {"x-ratelimit-limit-requests": "100"}
    client.close()


@pytest.mark.asyncio
async def test_sync_and_async_draw_from_shared_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    client = LMClient(model="openai/test", api_base="http://localhost", rpm=100)
    assert client._rate_limiter is not None

    await client.agenerate("hello")
    client.generate("world")

    rpm_bucket = client._rate_limiter._request_buckets["rpm"]
    assert rpm_bucket.get_bucket_level(_time.monotonic()) == pytest.approx(98, abs=0.1)
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_kwargs", "bucket_group", "bucket_name", "initial_level"),
    [
        ({"rpd": 5}, "_request_buckets", "rpd", 5),
        ({"tpd": 1_000}, "_token_buckets", "tpd", 1_000),
    ],
)
async def test_daily_only_limits_create_and_use_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
    client_kwargs: dict[str, Any],
    bucket_group: str,
    bucket_name: str,
    initial_level: int,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    client = LMClient(model="openai/test", api_base="http://localhost", **client_kwargs)
    assert client._rate_limiter is not None
    bucket = getattr(client._rate_limiter, bucket_group)[bucket_name]
    assert bucket.get_bucket_level(_time.monotonic()) == initial_level
    await client.agenerate("hello")
    assert bucket.get_bucket_level(_time.monotonic()) < initial_level
    client.close()


@pytest.mark.asyncio
async def test_cancelled_rate_limiter_wait_does_not_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_acquire(self: Any, estimated_tokens: int) -> None:
        return None

    async def fake_call_generation(
        self: LMClient,
        input_data: Any,
        *,
        endpoint: Any,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str | None]:
        nonlocal called
        called = True
        return {}, None

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.acquire", fake_acquire)
    monkeypatch.setattr(LMClient, "_call_generation", fake_call_generation)
    client = LMClient(model="openai/test", api_base="http://localhost", rpm=10)
    with pytest.raises(asyncio.CancelledError):
        await client.agenerate("hello")
    assert not called
    client.close()


@pytest.mark.asyncio
async def test_default_output_tokens_used_when_no_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[int] = []

    async def fake_acquire(self: Any, tokens: int) -> Any:
        acquired.append(tokens)
        return RateLimiterAcquisitionHandle(tokens)

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.acquire", fake_acquire)
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        rpm=10,
        tpm=1000,
        default_output_tokens=200,
    )
    await client.agenerate("hello")
    assert acquired[0] >= 200
    client.close()


@pytest.mark.asyncio
async def test_per_request_max_tokens_overrides_default_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[int] = []

    async def fake_acquire(self: Any, tokens: int) -> Any:
        acquired.append(tokens)
        return RateLimiterAcquisitionHandle(tokens)

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.acquire", fake_acquire)
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        rpm=10,
        tpm=10000,
        default_output_tokens=500,
    )
    await client.agenerate("hello", max_tokens=50)
    assert acquired[0] < 500
    client.close()


@pytest.mark.asyncio
async def test_default_request_kwargs_max_tokens_overrides_default_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[int] = []

    async def fake_acquire(self: Any, tokens: int) -> Any:
        acquired.append(tokens)
        return RateLimiterAcquisitionHandle(tokens)

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.acquire", fake_acquire)
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        rpm=10,
        tpm=10000,
        default_output_tokens=500,
        default_request_kwargs={"max_tokens": 75},
    )
    await client.agenerate("hello")
    assert acquired[0] < 500
    client.close()


@pytest.mark.asyncio
async def test_responses_max_output_tokens_overrides_default_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[int] = []

    async def fake_acquire(self: Any, tokens: int) -> Any:
        acquired.append(tokens)
        return RateLimiterAcquisitionHandle(tokens)

    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    monkeypatch.setattr("lm_client.client.RateLimiter.acquire", fake_acquire)
    client = LMClient(
        model="openai/test",
        api_base="http://localhost",
        rpm=10,
        tpm=10000,
        default_output_tokens=0,
    )
    await client.agenerate(
        {"input": "hello"},
        endpoint="responses",
        max_output_tokens=50,
    )
    assert acquired[0] >= 50
    client.close()


def test_transcribe_rejects_nonregular_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        normalize_transcription_input(tmp_path)


def test_transcribe_enforces_max_bytes_on_raw_bytes() -> None:
    with pytest.raises(ValueError, match="10 bytes exceeds limit 5 bytes"):
        normalize_transcription_input(b"x" * 10, max_bytes=5)


def test_transcribe_enforces_max_bytes_on_path(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x" * 10)
    with pytest.raises(ValueError, match="exceeds limit"):
        normalize_transcription_input(audio, max_bytes=5)


def test_transcribe_enforces_max_bytes_on_filelike() -> None:
    buf = io.BytesIO(b"x" * 10)
    with pytest.raises(ValueError, match="exceeds limit"):
        normalize_transcription_input(buf, max_bytes=5)


def test_transcribe_default_limit_applies_to_raw_bytes(fake_client: LMClient) -> None:
    oversized_audio = b"x" * (DEFAULT_MAX_TRANSCRIPTION_BYTES + 1)
    with pytest.raises(ValueError, match=f"{DEFAULT_MAX_TRANSCRIPTION_BYTES} bytes"):
        fake_client.transcribe(oversized_audio)


def test_transcribe_none_disables_default_limit(fake_client: LMClient) -> None:
    oversized_audio = b"x" * (DEFAULT_MAX_TRANSCRIPTION_BYTES + 1)
    result = fake_client.transcribe(oversized_audio, max_transcription_bytes=None)
    assert result.text == "hello transcript"


def test_http_api_base_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    with caplog.at_level(logging.WARNING, logger="lm_client.client"):
        client = LMClient(model="openai/test", api_base="http://remote-server:8000/v1")
    assert "unencrypted" in caplog.text
    client.close()


def test_http_localhost_no_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(LMClient, "_create_litellm_module", lambda self: FakeLiteLLM())
    for api_base in (
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="lm_client.client"):
            client = LMClient(model="openai/test", api_base=api_base)
        assert "unencrypted" not in caplog.text
        client.close()
