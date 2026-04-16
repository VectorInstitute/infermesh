import asyncio
import contextlib
import time
from collections.abc import Generator
from typing import Any
from unittest import mock

import pytest

from lm_client.rate_limiter import (
    RateLimiter,
    RateLimiterAcquisitionHandle,
    _parse_rate_limit_info_from_response_headers,
    _parse_reset_time,
)

from .time_mock import MockTime

RPM_LIMIT = 120
TPM_LIMIT = 2_000
RPD_LIMIT = 1_000
TPD_LIMIT = 10_000
MAX_REQ_BURST = 60
MAX_TOKEN_BURST = 1_000


@pytest.fixture
def mock_time() -> Generator[MockTime, None, None]:
    mock_time_instance = MockTime()
    module_path = "lm_client.rate_limiter"
    bucket_module_path = "lm_client._bucket"
    patches: list[mock._patch[Any]] = [
        mock.patch(f"{module_path}.time.monotonic", new=mock_time_instance.monotonic),
        mock.patch(
            f"{module_path}.asyncio.get_running_loop",
            new=mock_time_instance.get_running_loop,
        ),
        mock.patch(
            f"{bucket_module_path}.time.monotonic",
            new=mock_time_instance.monotonic,
        ),
    ]
    try:
        for patcher in patches:
            patcher.start()
        yield mock_time_instance
    finally:
        for patcher in patches:
            patcher.stop()


@pytest.fixture
def simple_rate_limiter() -> RateLimiter:
    return RateLimiter(requests_per_minute=RPM_LIMIT)


@pytest.fixture
def multi_bucket_rate_limiter(mock_time: MockTime) -> RateLimiter:  # noqa: ARG001
    return RateLimiter(
        requests_per_minute=RPM_LIMIT,
        requests_per_day=RPD_LIMIT,
        tokens_per_minute=TPM_LIMIT,
        tokens_per_day=TPD_LIMIT,
        max_request_burst=MAX_REQ_BURST,
        max_token_burst=MAX_TOKEN_BURST,
    )


@pytest.mark.parametrize(
    ("rpm", "tpm", "rpd", "tpd", "max_req_burst", "max_token_burst"),
    [
        (0, 10, 100, 1000, 10, 100),
        (10, 0, 100, 1000, 10, 100),
        (10, 10, 0, 1000, 10, 100),
        (10, 10, 100, 0, 10, 100),
        (10, 10, 100, 1000, 0, 100),
        (10, 10, 100, 1000, 10, 0),
        (10, 10, 100, 1000, -1, 100),
        (10, 10, 100, 1000, 10, -1),
    ],
)
def test_failed_initialization(
    rpm: int,
    tpm: int,
    rpd: int,
    tpd: int,
    max_req_burst: int,
    max_token_burst: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RateLimiter(
            requests_per_minute=rpm,
            tokens_per_minute=tpm,
            requests_per_day=rpd,
            tokens_per_day=tpd,
            max_request_burst=max_req_burst,
            max_token_burst=max_token_burst,
        )


@pytest.mark.asyncio
async def test_acquire_immediate(simple_rate_limiter: RateLimiter) -> None:
    current_time = time.monotonic()
    assert (
        simple_rate_limiter._request_buckets["rpm"].get_bucket_level(current_time)
        == RPM_LIMIT
    )
    handle = await simple_rate_limiter.acquire(256)
    assert isinstance(handle, RateLimiterAcquisitionHandle)
    assert handle.estimated_tokens == 256
    assert (
        simple_rate_limiter._request_buckets["rpm"].get_bucket_level(current_time)
        == RPM_LIMIT - 1
    )


@pytest.mark.asyncio
async def test_acquire_blocks_on_requests(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    rpm_bucket_ref = multi_bucket_rate_limiter._request_buckets["rpm"]
    tasks = [
        asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)
    assert rpm_bucket_ref.get_bucket_level(mock_time.monotonic()) == pytest.approx(0.0)
    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
    await asyncio.sleep(0)
    assert not blocked_request.done()
    await mock_time.advance_time((60 / RPM_LIMIT) + 0.05)
    assert blocked_request.done()


@pytest.mark.asyncio
async def test_multiple_waiters(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    tasks = [
        asyncio.create_task(multi_bucket_rate_limiter.acquire(1))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)
    medium_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(50))
    await asyncio.sleep(0)
    small_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
    await asyncio.sleep(0)
    large_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(150))
    await asyncio.sleep(0)
    assert len(multi_bucket_rate_limiter._waiters) == 3
    assert multi_bucket_rate_limiter._waiters[0][0] == 10
    await mock_time.advance_time((60 / RPM_LIMIT) + 0.1)
    assert small_request.done()
    assert not medium_request.done()
    assert not large_request.done()


@pytest.mark.asyncio
async def test_adjust_failed_request(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    handles = [await multi_bucket_rate_limiter.acquire(1) for _ in range(MAX_REQ_BURST)]
    blocked_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(1))
    await asyncio.sleep(0)
    assert not blocked_request.done()
    assert handles[0] is not None
    await multi_bucket_rate_limiter.adjust(handles[0], actual_tokens=0)
    await asyncio.sleep(0)
    assert blocked_request.done()


@pytest.mark.asyncio
async def test_adjust_with_response_headers(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST)
    assert handle is not None
    headers = {
        "x-ratelimit-limit-requests": str(RPD_LIMIT),
        "x-ratelimit-remaining-requests": str(100),
        "x-ratelimit-reset-requests": "5000s",
        "x-ratelimit-limit-tokens": str(TPD_LIMIT + 5000),
        "x-ratelimit-remaining-tokens": str(TPD_LIMIT - MAX_TOKEN_BURST + 1000),
        "x-ratelimit-reset-tokens": str(mock_time.monotonic() + 6000.0),
    }
    await multi_bucket_rate_limiter.adjust(
        handle,
        actual_tokens=MAX_TOKEN_BURST,
        response_headers=headers,
    )
    assert multi_bucket_rate_limiter._request_buckets["rpd"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(100)


@pytest.mark.asyncio
async def test_cancel_waiters(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    tasks = [
        asyncio.create_task(multi_bucket_rate_limiter.acquire(10))
        for _ in range(MAX_REQ_BURST)
    ]
    await asyncio.gather(*tasks)
    large_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(100))
    await asyncio.sleep(0)
    medium_request = asyncio.create_task(multi_bucket_rate_limiter.acquire(50))
    await asyncio.sleep(0)
    medium_request.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await medium_request
    await mock_time.advance_time(0.1)
    assert len(multi_bucket_rate_limiter._waiters) == 1
    await mock_time.advance_time(60 / RPM_LIMIT)
    assert large_request.done()


@pytest.mark.parametrize(
    ("value", "expected_delta"),
    [
        ("10", 10.0),
        ("15.5", 15.5),
        ("8.64s", 8.64),
        ("1m", 60.0),
        ("1.5m", 90.0),
        ("1h", 3600.0),
        ("100ms", 0.1),
        ("1m30s", 90.0),
    ],
)
def test_parse_reset_time_valid(value: str, expected_delta: float) -> None:
    current_time = time.monotonic()
    parsed_time = _parse_reset_time(value, current_time)
    assert parsed_time == pytest.approx(current_time + expected_delta)


@pytest.mark.parametrize(
    "value",
    ["abc", "10 seconds", "1 h", "1m 1d", "1m 30s", "", "1.2.3"],
)
def test_parse_reset_time_invalid(value: str) -> None:
    assert _parse_reset_time(value, time.monotonic()) is None


@pytest.mark.asyncio
async def test_adjust_headers_route_to_minute_bucket(
    multi_bucket_rate_limiter: RateLimiter,
    mock_time: MockTime,
) -> None:
    handle = await multi_bucket_rate_limiter.acquire(MAX_TOKEN_BURST)
    assert handle is not None
    headers = {
        "x-ratelimit-limit-requests": str(RPM_LIMIT),
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-reset-requests": "30s",
        "x-ratelimit-limit-tokens": str(TPM_LIMIT),
        "x-ratelimit-remaining-tokens": str(TPM_LIMIT - MAX_TOKEN_BURST + 200),
        "x-ratelimit-reset-tokens": "45s",
    }
    await multi_bucket_rate_limiter.adjust(
        handle, actual_tokens=MAX_TOKEN_BURST, response_headers=headers
    )
    assert multi_bucket_rate_limiter._request_buckets["rpm"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(50)


@pytest.mark.asyncio
async def test_adjust_headers_explicit_scope_minute(
    mock_time: MockTime,  # noqa: ARG001
) -> None:
    limiter = RateLimiter(
        requests_per_minute=RPM_LIMIT,
        tokens_per_minute=TPM_LIMIT,
        requests_per_day=RPD_LIMIT,
        tokens_per_day=TPD_LIMIT,
        max_request_burst=MAX_REQ_BURST,
        max_token_burst=MAX_TOKEN_BURST,
        header_bucket_scope="minute",
    )
    handle = await limiter.acquire(10)
    assert handle is not None
    headers = {
        "x-ratelimit-limit-requests": str(RPM_LIMIT),
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-reset-requests": "5000s",
        "x-ratelimit-limit-tokens": str(TPM_LIMIT),
        "x-ratelimit-remaining-tokens": str(TPM_LIMIT - 10 + 100),
        "x-ratelimit-reset-tokens": "5000s",
    }
    await limiter.adjust(handle, actual_tokens=10, response_headers=headers)
    assert limiter._request_buckets["rpm"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(50)


@pytest.mark.asyncio
async def test_adjust_headers_explicit_scope_day(
    mock_time: MockTime,  # noqa: ARG001
) -> None:
    limiter = RateLimiter(
        requests_per_minute=RPM_LIMIT,
        tokens_per_minute=TPM_LIMIT,
        requests_per_day=RPD_LIMIT,
        tokens_per_day=TPD_LIMIT,
        max_request_burst=MAX_REQ_BURST,
        max_token_burst=MAX_TOKEN_BURST,
        header_bucket_scope="day",
    )
    handle = await limiter.acquire(10)
    assert handle is not None
    headers = {
        "x-ratelimit-limit-requests": str(RPD_LIMIT),
        "x-ratelimit-remaining-requests": "200",
        "x-ratelimit-reset-requests": "30s",
        "x-ratelimit-limit-tokens": str(TPD_LIMIT),
        "x-ratelimit-remaining-tokens": str(TPD_LIMIT - 10 + 500),
        "x-ratelimit-reset-tokens": "30s",
    }
    await limiter.adjust(handle, actual_tokens=10, response_headers=headers)
    assert limiter._request_buckets["rpd"].get_bucket_level(
        mock_time.monotonic()
    ) == pytest.approx(200)


def test_parse_rate_limit_headers_complete() -> None:
    current_time = time.monotonic()
    headers = {
        "X-RateLimit-Limit-Requests": "100",
        "X-RateLimit-Limit-Tokens": "10000",
        "x-ratelimit-remaining-requests": "50",
        "x-ratelimit-remaining-tokens": "5000",
        "X-RateLimit-Reset-Requests": "60",
        "X-RATELIMIT-RESET-TOKENS": "120.5s",
    }
    parsed = _parse_rate_limit_info_from_response_headers(headers, current_time)
    assert parsed is not None
    assert parsed["requests_limit"] == 100
    assert parsed["tokens_limit"] == 10000
    assert parsed["requests_remaining"] == 50
    assert parsed["tokens_remaining"] == 5000
