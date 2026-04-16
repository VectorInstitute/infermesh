"""Async token-bucket rate limiter.

[RateLimiter][lm_client.RateLimiter] enforces per-minute and per-day request
and token limits across both synchronous (via `SyncRunner`) and async call
paths.  Unlike asyncio-native alternatives, its counters are protected by a
`threading.Lock`, so a single instance can be shared safely between the
caller's event loop and the `SyncRunner`'s background loop.

The limiter exposes two operations:

- [acquire][lm_client.RateLimiter.acquire] — called before a request is
  dispatched.  Blocks asynchronously until all configured buckets have
  capacity, then atomically deducts the estimated token cost.
- [adjust][lm_client.RateLimiter.adjust] — called after the response arrives.
  Corrects for estimation error, refunds tokens on failure, and optionally
  syncs the buckets from `x-ratelimit-*` response headers.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from lm_client._bucket import Bucket

logger = logging.getLogger(__name__)

_RESET_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)


@dataclass(slots=True)
class RateLimiterAcquisitionHandle:
    """Opaque handle returned by [acquire][lm_client.RateLimiter.acquire].

    Pass this handle to [adjust][lm_client.RateLimiter.adjust] after the
    request completes so the limiter can reconcile the actual token usage
    against the pre-dispatch estimate and release or reclaim the difference.

    Parameters
    ----------
    estimated_tokens : int
        The number of tokens that were reserved when the handle was created.
        This field is used internally by [adjust][lm_client.RateLimiter.adjust];
        external callers generally do not need to inspect it.

    Notes
    -----
    Do not create [RateLimiterAcquisitionHandle][lm_client.RateLimiterAcquisitionHandle]
    instances directly.  They are produced exclusively by
    [acquire][lm_client.RateLimiter.acquire] and should be treated as opaque
    tokens.
    """

    estimated_tokens: int


class RateLimiter:
    """Thread-safe async rate limiter backed by token buckets.

    Enforces up to four independent limits simultaneously:

    - **RPM** (requests per minute) — always active; the only required limit.
    - **TPM** (tokens per minute) — optional; activated by passing ``tokens_per_minute``.
    - **RPD** (requests per day) — optional; activated by passing ``requests_per_day``.
    - **TPD** (tokens per day) — optional; activated by passing ``tokens_per_day``.

    A request acquires capacity from *all* active buckets before it proceeds.
    Waiters are queued in a min-heap ordered by token cost (smallest-first),
    which prevents large requests from being starved by a constant stream of
    small ones.

    After a provider response arrives, [adjust][lm_client.RateLimiter.adjust]
    refines the token accounting with the actual usage and can also sync the
    bucket state from the `x-ratelimit-*` headers returned by OpenAI-compatible
    APIs.

    Parameters
    ----------
    requests_per_minute : int
        Maximum number of requests allowed per 60-second window.  This is the
        only required parameter.
    tokens_per_minute : int or None, optional
        Maximum number of tokens allowed per 60-second window.  When supplied,
        a TPM bucket is created and all requests must also fit within this limit.
    requests_per_day : int or None, optional
        Maximum requests per 24-hour window.  Creates an RPD bucket in addition
        to the RPM bucket.
    tokens_per_day : int or None, optional
        Maximum tokens per 24-hour window.  Creates a TPD bucket.
    max_request_burst : int or None, optional
        Burst capacity for the RPM bucket.  When ``None`` (default), the burst
        equals ``requests_per_minute`` (no burst above the base rate).  Set
        higher to allow short spikes above the steady-state RPM.
    max_token_burst : int or None, optional
        Burst capacity for the TPM bucket.  Analogous to ``max_request_burst``.
    header_bucket_scope : {"auto", "minute", "day"}, optional
        Controls which bucket receives updates from ``x-ratelimit-*`` response
        headers.

        - ``"auto"`` (default): resets arriving within 120 s → per-minute
          buckets (RPM/TPM); later resets → per-day buckets (RPD/TPD).
        - ``"minute"``: always route header updates to RPM/TPM.
        - ``"day"``: always route header updates to RPD/TPD.

        Override ``"auto"`` when your provider uses non-standard header
        conventions.

    Raises
    ------
    ValueError
        If ``header_bucket_scope`` is not one of the accepted string literals.

    Examples
    --------
    A simple 100 RPM limiter:

    >>> from lm_client import RateLimiter
    >>> limiter = RateLimiter(requests_per_minute=100)

    A combined 500 RPM / 100 000 TPM limiter appropriate for OpenAI Tier-2:

    >>> limiter = RateLimiter(
    ...     requests_per_minute=500,
    ...     tokens_per_minute=100_000,
    ... )

    Using the limiter outside of ``LMClient`` (advanced):

    >>> import asyncio
    >>> async def limited_call(limiter: RateLimiter, tokens: int) -> None:
    ...     handle = await limiter.acquire(tokens)
    ...     try:
    ...         response = await some_api_call()
    ...         actual = response.usage.total_tokens
    ...     except Exception:
    ...         await limiter.adjust(handle, actual_tokens=0)
    ...         raise
    ...     await limiter.adjust(handle, actual_tokens=actual)
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int | None = None,
        requests_per_day: int | None = None,
        tokens_per_day: int | None = None,
        max_request_burst: int | None = None,
        max_token_burst: int | None = None,
        header_bucket_scope: Literal["minute", "day", "auto"] = "auto",
    ) -> None:
        if header_bucket_scope not in {"minute", "day", "auto"}:
            raise ValueError(
                "Expected ``header_bucket_scope`` to be one of "
                "['auto', 'day', 'minute'], "
                f"but got {header_bucket_scope!r}."
            )
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_day = requests_per_day
        self.tokens_per_day = tokens_per_day
        self.max_request_burst = max_request_burst
        self.max_token_burst = max_token_burst
        self.header_bucket_scope = header_bucket_scope
        self._request_buckets: dict[str, Bucket] = {
            "rpm": Bucket(
                requests_per_minute,
                capacity=max_request_burst,
                time_period=60,
            )
        }
        if requests_per_day is not None:
            self._request_buckets["rpd"] = Bucket(
                requests_per_day,
                time_period=24 * 60 * 60,
            )

        self._token_buckets: dict[str, Bucket] = {}
        if tokens_per_minute is not None:
            self._token_buckets["tpm"] = Bucket(
                tokens_per_minute,
                capacity=max_token_burst,
                time_period=60,
            )
        if tokens_per_day is not None:
            self._token_buckets["tpd"] = Bucket(
                tokens_per_day,
                time_period=24 * 60 * 60,
            )

        self._lock = threading.Lock()
        self._waiters: list[tuple[int, float, asyncio.Future[None]]] = []
        self._timer_handle: asyncio.TimerHandle | None = None

    async def acquire(
        self, estimated_tokens: int
    ) -> RateLimiterAcquisitionHandle | None:
        """Reserve capacity for one request and return an acquisition handle.

        Blocks asynchronously until *all* active buckets (RPM, TPM, RPD, TPD)
        have enough capacity for the request.  Once capacity is available,
        the tokens are atomically deducted and a
        [RateLimiterAcquisitionHandle][lm_client.RateLimiterAcquisitionHandle]
        is returned.

        Call [adjust][lm_client.RateLimiter.adjust] with the returned handle
        after the request completes to reconcile actual token usage.

        Parameters
        ----------
        estimated_tokens : int
            Pre-dispatch estimate of the total tokens this request will
            consume (prompt + expected output).  Must be non-negative.  A
            value of ``0`` means "only reserve one request slot".

        Returns
        -------
        RateLimiterAcquisitionHandle or None
            A handle encapsulating the reservation.  Returns ``None`` if the
            wait future was cancelled (i.e. the calling task was cancelled
            while waiting in the queue); the request was *not* dispatched in
            that case.

        Raises
        ------
        ValueError
            If ``estimated_tokens`` exceeds the capacity of any token bucket.
            This prevents a request that can *never* fit from blocking the
            queue indefinitely.  Either reduce the estimate or increase
            ``max_token_burst``.

        Notes
        -----
        Waiters are ordered by token cost (smallest first) in a min-heap.
        This avoids starvation of small requests by large ones but means that
        a very large request may wait until the bucket refills enough to serve
        it.

        Examples
        --------
        >>> handle = await limiter.acquire(estimated_tokens=512)
        >>> if handle is None:
        ...     return  # task was cancelled; do not dispatch
        >>> try:
        ...     response = await api_call()
        ... except Exception:
        ...     await limiter.adjust(handle, actual_tokens=0)
        ...     raise
        >>> await limiter.adjust(handle, actual_tokens=response.usage.total_tokens)
        """
        if any(
            estimated_tokens > bucket.capacity
            for bucket in self._token_buckets.values()
        ):
            raise ValueError(
                f"Requested number of tokens {estimated_tokens} exceeds the capacity "
                "of at least one token bucket. Please reduce the estimate or "
                "increase ``max_token_burst``."
            )

        wait_future: asyncio.Future[None] | None = None
        while True:
            with self._lock:
                current_time = time.monotonic()
                self._notify_waiters_locked()
                bucket_token_pairs = self._get_bucket_token_pairs(estimated_tokens)
                if self._has_capacity(bucket_token_pairs, current_time):
                    self._consume_all(bucket_token_pairs, current_time)
                    self._notify_waiters_locked()
                    return RateLimiterAcquisitionHandle(estimated_tokens)

                if wait_future is None:
                    loop = asyncio.get_running_loop()
                    wait_future = loop.create_future()
                    heapq.heappush(
                        self._waiters,
                        (estimated_tokens, current_time, wait_future),
                    )
                    self._notify_waiters_locked()

            if wait_future is not None:
                try:
                    await wait_future
                    return RateLimiterAcquisitionHandle(estimated_tokens)
                except asyncio.CancelledError:
                    with self._lock:
                        self._notify_waiters_locked()
                    return None

    async def adjust(
        self,
        handle: RateLimiterAcquisitionHandle,
        actual_tokens: int,
        response_headers: dict[str, Any] | None = None,
    ) -> None:
        """Reconcile a reservation with the actual outcome of the request.

        This method must be called after every [acquire][lm_client.RateLimiter.acquire]
        call, regardless of whether the request succeeded or failed.  It
        performs three tasks:

        1. **Failure refund** — when ``actual_tokens == 0`` (signalling a
           failed request), the one request slot that was deducted by
           [acquire][lm_client.RateLimiter.acquire] is returned to the RPM
           bucket so the failed request does not count against the rate.
        2. **Token correction** — the difference between the pre-dispatch
           estimate (``handle.estimated_tokens``) and the actual usage is
           added back to all token buckets (a negative delta removes tokens;
           a positive delta returns over-reserved tokens).
        3. **Header sync** — when ``response_headers`` is provided, any
           ``x-ratelimit-*`` headers are parsed and used to authoritative-
           ly synchronise the bucket state, overriding the local estimate.

        Parameters
        ----------
        handle : RateLimiterAcquisitionHandle
            The handle returned by the preceding [acquire][lm_client.RateLimiter.acquire]
            call.
        actual_tokens : int
            Total tokens actually consumed, as reported in the response's
            ``usage.total_tokens`` field.  Pass ``0`` to indicate a failed
            request (no tokens billed) and trigger the failure refund.
        response_headers : dict or None, optional
            The provider's response headers.  When present, any
            ``x-ratelimit-limit-*``, ``x-ratelimit-remaining-*``, and
            ``x-ratelimit-reset-*`` headers are parsed and used to sync the
            corresponding buckets.  Pass ``None`` (default) to skip header
            syncing (appropriate for local vLLM servers that do not send
            rate-limit headers).

        Raises
        ------
        ValueError
            If ``actual_tokens`` or ``handle.estimated_tokens`` is negative.

        Notes
        -----
        After the adjustments are applied, any queued waiters that can now
        proceed are notified.

        Examples
        --------
        Successful request:

        >>> handle = await limiter.acquire(256)
        >>> response = await api_call()
        >>> await limiter.adjust(
        ...     handle,
        ...     actual_tokens=response.usage.total_tokens,
        ...     response_headers=dict(response.headers),
        ... )

        Failed request (refund the slot):

        >>> handle = await limiter.acquire(256)
        >>> try:
        ...     await api_call()
        ... except Exception:
        ...     await limiter.adjust(handle, actual_tokens=0)
        ...     raise
        """
        if actual_tokens < 0:
            raise ValueError("Actual tokens consumed must be non-negative.")
        if handle.estimated_tokens < 0:
            raise ValueError("Estimated tokens consumed must be non-negative.")

        estimated_tokens = handle.estimated_tokens
        delta = estimated_tokens - actual_tokens
        is_failure_release = actual_tokens == 0

        with self._lock:
            current_time = time.monotonic()
            needs_notify = False
            if is_failure_release:
                for bucket in self._request_buckets.values():
                    bucket.adjust_bucket_level(1, current_time)
                needs_notify = True

            if delta != 0:
                for bucket in self._token_buckets.values():
                    bucket.adjust_bucket_level(delta, current_time)
                if delta > 0:
                    needs_notify = True

            response_info = _parse_rate_limit_info_from_response_headers(
                response_headers,
                current_time,
            )
            if response_info:

                def _sync_bucket(
                    bucket: Bucket | None,
                    limit: int | None,
                    remaining: int | None,
                    reset_time: float | None,
                ) -> bool:
                    if bucket is None or remaining is None or reset_time is None:
                        return False
                    bucket.sync_from_response_header(
                        server_token_limit=limit,
                        server_tokens_remaining=remaining,
                        server_reset_time=reset_time,
                        current_time=current_time,
                    )
                    return True

                req_scope = self._header_scope(
                    _maybe_float(response_info["requests_reset"]), current_time
                )
                tok_scope = self._header_scope(
                    _maybe_float(response_info["tokens_reset"]), current_time
                )
                needs_notify = (
                    _sync_bucket(
                        self._request_buckets.get(
                            "rpm" if req_scope == "minute" else "rpd"
                        ),
                        _maybe_int(response_info["requests_limit"]),
                        _maybe_int(response_info["requests_remaining"]),
                        _maybe_float(response_info["requests_reset"]),
                    )
                    or needs_notify
                )
                needs_notify = (
                    _sync_bucket(
                        self._token_buckets.get(
                            "tpm" if tok_scope == "minute" else "tpd"
                        ),
                        _maybe_int(response_info["tokens_limit"]),
                        _maybe_int(response_info["tokens_remaining"]),
                        _maybe_float(response_info["tokens_reset"]),
                    )
                    or needs_notify
                )

            if needs_notify:
                self._notify_waiters_locked()

    def _header_scope(
        self, reset_time: float | None, current_time: float
    ) -> Literal["minute", "day"]:
        """Resolve whether a header's reset window is per-minute or per-day."""
        if self.header_bucket_scope == "minute":
            return "minute"
        if self.header_bucket_scope == "day":
            return "day"
        # auto: resets arriving in < 120 s belong to the per-minute window
        if reset_time is not None and (reset_time - current_time) < 120.0:
            return "minute"
        return "day"

    def _get_bucket_token_pairs(
        self, estimated_tokens: int
    ) -> list[tuple[Bucket, int]]:
        """Return the buckets touched by a request."""
        bucket_token_pairs: list[tuple[Bucket, int]] = []
        for bucket in self._token_buckets.values():
            bucket_token_pairs.append((bucket, estimated_tokens))
        for bucket in self._request_buckets.values():
            bucket_token_pairs.append((bucket, 1))
        return bucket_token_pairs

    def _has_capacity(
        self,
        bucket_token_pairs: list[tuple[Bucket, int]],
        current_time: float,
    ) -> bool:
        """Return whether all buckets have enough capacity."""
        return all(
            bucket.get_bucket_level(current_time) >= tokens
            for bucket, tokens in bucket_token_pairs
        )

    def _consume_all(
        self,
        bucket_token_pairs: list[tuple[Bucket, int]],
        current_time: float,
    ) -> None:
        """Consume from all buckets."""
        for bucket, tokens in bucket_token_pairs:
            bucket.consume_tokens(tokens, current_time)

    def _notify_waiters_locked(self) -> None:
        """Wake any waiter that can now proceed (called while holding ``_lock``)."""
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

        current_time = time.monotonic()
        while self._waiters:
            tokens_needed, _, future = self._waiters[0]
            if future.done():
                heapq.heappop(self._waiters)
                continue

            bucket_token_pairs = self._get_bucket_token_pairs(tokens_needed)
            if self._has_capacity(bucket_token_pairs, current_time):
                heapq.heappop(self._waiters)
                self._consume_all(bucket_token_pairs, current_time)
                future_loop = future.get_loop()
                if getattr(future_loop, "_thread_id", None) == threading.get_ident():
                    future.set_result(None)
                else:
                    future_loop.call_soon_threadsafe(future.set_result, None)
                continue

            delay = 0.0
            for bucket, tokens in bucket_token_pairs:
                delay = max(
                    delay,
                    bucket.estimate_next_refill_time(tokens, current_time),
                )
            self._schedule_timer_locked(delay)
            break

    def _schedule_timer_locked(self, delay: float) -> None:
        """Schedule the next waiter wake-up (called while holding ``_lock``)."""
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

        if delay <= 0 or delay == float("inf"):
            return

        loop = asyncio.get_running_loop()
        self._timer_handle = loop.call_later(delay, self._wake_next_timer)

    def _wake_next_timer(self) -> None:
        """Wake waiters from the timer callback."""

        async def do_notify() -> None:
            with self._lock:
                self._notify_waiters_locked()

        asyncio.create_task(do_notify())


def _parse_rate_limit_info_from_response_headers(
    response_headers: dict[str, Any] | None,
    current_time: float,
) -> dict[str, int | float | None] | None:
    """Parse rate-limit headers into normalized values."""
    if response_headers is None:
        return None

    requests_limit: int | None = None
    tokens_limit: int | None = None
    requests_remaining: int | None = None
    tokens_remaining: int | None = None
    requests_reset: float | None = None
    tokens_reset: float | None = None

    for header, value in response_headers.items():
        key = header.lower()
        try:
            if "x-ratelimit-limit-requests" in key:
                requests_limit = int(value)
            elif "x-ratelimit-limit-tokens" in key:
                tokens_limit = int(value)
            elif "x-ratelimit-remaining-requests" in key:
                requests_remaining = int(value)
            elif "x-ratelimit-remaining-tokens" in key:
                tokens_remaining = int(value)
            elif "x-ratelimit-reset-requests" in key:
                requests_reset = _parse_reset_time(str(value), current_time)
            elif "x-ratelimit-reset-tokens" in key:
                tokens_reset = _parse_reset_time(str(value), current_time)
        except ValueError:
            logger.error("Failed to parse rate-limit header: %s=%s", header, value)

    return {
        "requests_limit": requests_limit,
        "tokens_limit": tokens_limit,
        "requests_remaining": requests_remaining,
        "tokens_remaining": tokens_remaining,
        "requests_reset": requests_reset,
        "tokens_reset": tokens_reset,
    }


def _parse_reset_time(header_value: str, current_time: float) -> float | None:
    """Parse a reset header into a future monotonic timestamp."""
    header_value = header_value.strip()
    if not header_value:
        return None

    try:
        value = float(header_value)
        if value > time.time() - (5 * 365 * 86400):
            delay = max(0.0, value - time.time())
            return current_time + delay
        return current_time + max(0.0, value)
    except ValueError:
        pass

    total_seconds = 0.0
    last_match_end = 0
    found_match = False

    for match in _RESET_TIME_RE.finditer(header_value):
        found_match = True
        start, end = match.span()
        if start != last_match_end:
            return None
        value_str, unit = match.groups()
        parsed_value = float(value_str)
        unit_lower = unit.lower()
        if unit_lower == "h":
            total_seconds += parsed_value * 3600
        elif unit_lower == "m":
            total_seconds += parsed_value * 60
        elif unit_lower == "s":
            total_seconds += parsed_value
        elif unit_lower == "ms":
            total_seconds += parsed_value / 1000.0
        last_match_end = end

    if not found_match or last_match_end != len(header_value):
        return None
    return current_time + total_seconds


def _maybe_int(value: int | float | None) -> int | None:
    """Coerce a parsed header value to ``int`` when possible."""

    if value is None:
        return None
    return int(value)


def _maybe_float(value: int | float | None) -> float | None:
    """Coerce a parsed header value to ``float`` when possible."""

    if value is None:
        return None
    return float(value)
