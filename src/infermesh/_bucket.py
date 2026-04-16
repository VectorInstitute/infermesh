"""Token-bucket implementation for rate limiting.

A `Bucket` is a fixed-capacity counter that refills at a constant rate.
Tokens are consumed when a request is dispatched and added back over time as
the rate window progresses.  The bucket also supports synchronisation from
server-side `x-ratelimit-*` response headers so that the local estimate stays
aligned with the provider's authoritative view.

This module is internal to `infermesh`; external callers should interact with
[RateLimiter][infermesh.RateLimiter] instead.
"""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)


class Bucket:
    """A fixed-capacity token bucket that refills at a constant rate.

    The bucket starts full (at `capacity` tokens) and is drained by
    `consume_tokens`.  Tokens are added continuously at a rate of
    `tokens_per_period / time_period` tokens per second.  The level is
    capped at `capacity`.

    After a call to `sync_from_response_header`, the bucket adopts a
    *server-derived effective rate* computed from the remaining tokens and
    reset timestamp reported in `x-ratelimit-*` headers.  This adjusted rate
    is used for up to `SYNC_RATE_VALIDITY_DURATION` seconds before
    falling back to the configured rate.

    Parameters
    ----------
    tokens_per_period : int
        The number of tokens the bucket receives every ``time_period`` seconds.
        Must be a positive integer.
    capacity : int or None, optional
        Maximum number of tokens the bucket can hold.  When ``None`` (default),
        the capacity equals ``tokens_per_period``, meaning no burst above the
        base rate is permitted.  Set this higher to allow short bursts.
    time_period : int, optional
        The refill window in seconds.  Default is ``60`` (per-minute buckets).
        Use ``86400`` for per-day buckets.

    Raises
    ------
    ValueError
        If ``tokens_per_period``, ``capacity``, or ``time_period`` are not
        positive integers.

    Examples
    --------
    Create a 100 RPM bucket with no burst allowance:

    >>> bucket = Bucket(tokens_per_period=100, time_period=60)
    >>> bucket.capacity
    100
    >>> bucket.consume_tokens(1)  # one request
    True

    Create a 1000 TPM bucket with a 200-token burst capacity:

    >>> tpm_bucket = Bucket(tokens_per_period=1000, capacity=200, time_period=60)
    """

    SYNC_RATE_VALIDITY_DURATION = 60.0
    """Seconds for which a server-synced rate stays valid.

    After this window, the bucket reverts to its configured
    ``tokens_per_period / time_period`` rate.
    """

    def __init__(
        self,
        tokens_per_period: int,
        capacity: int | None = None,
        time_period: int = 60,
    ) -> None:
        if not isinstance(tokens_per_period, int) or tokens_per_period <= 0:
            raise ValueError(
                "Expected ``tokens_per_period`` to be a positive integer, "
                f"but got {tokens_per_period}"
            )
        if capacity is not None and (not isinstance(capacity, int) or capacity <= 0):
            raise ValueError(
                f"Expected ``capacity`` to be a positive integer, but got {capacity}"
            )
        if not isinstance(time_period, int) or time_period <= 0:
            raise ValueError(
                "Expected ``time_period`` to be a positive integer, "
                f"but got {time_period}"
            )

        self._tokens_per_period = tokens_per_period
        self._capacity = capacity or tokens_per_period
        self._time_period = time_period
        self._rate_per_second = tokens_per_period / time_period
        self._last_refill_time = time.monotonic()
        self._tokens = float(self.capacity)
        self._last_sync_time: float = 0.0
        self._server_reset_time: float | None = None
        self._last_sync_rate: float | None = None

    @property
    def capacity(self) -> int:
        """Maximum number of tokens the bucket can hold.

        Returns
        -------
        int
            The bucket capacity.  Equal to ``tokens_per_period`` when no
            explicit capacity was provided.
        """
        return self._capacity

    @property
    def tokens_per_period(self) -> int:
        """Configured token replenishment per ``time_period``.

        Returns
        -------
        int
            The ``tokens_per_period`` value supplied at construction.
        """
        return self._tokens_per_period

    @property
    def time_period(self) -> int:
        """Refill window duration in seconds.

        Returns
        -------
        int
            The ``time_period`` value supplied at construction (e.g. ``60``
            for per-minute buckets, ``86400`` for per-day buckets).
        """
        return self._time_period

    def get_bucket_level(self, current_time: float | None = None) -> int:
        """Return the current number of available tokens (floor).

        Applies any accumulated refill since the last call before returning the
        level.

        Parameters
        ----------
        current_time : float or None, optional
            Monotonic timestamp (from `time.monotonic`).  When ``None``,
            the current time is obtained automatically.

        Returns
        -------
        int
            Available tokens, floored to the nearest integer.
        """
        current_time = current_time or time.monotonic()
        self._refill(current_time)
        return math.floor(self._tokens)

    def consume_tokens(
        self,
        num_tokens_needed: int,
        current_time: float | None = None,
    ) -> bool:
        """Consume tokens from the bucket if sufficient capacity exists.

        Parameters
        ----------
        num_tokens_needed : int
            Number of tokens to consume.  If ``<= 0``, the call is a no-op
            and returns ``True``.
        current_time : float or None, optional
            Monotonic timestamp.  Obtained automatically when ``None``.

        Returns
        -------
        bool
            ``True`` if the tokens were consumed, ``False`` if the bucket did
            not have enough tokens (no tokens are consumed in this case).
        """
        if num_tokens_needed <= 0:
            return True

        current_time = current_time or time.monotonic()
        if self.get_bucket_level(current_time) >= num_tokens_needed:
            self._tokens -= float(num_tokens_needed)
            self._last_sync_rate = None
            self._server_reset_time = None
            return True
        return False

    def adjust_bucket_level(
        self,
        delta: int,
        current_time: float | None = None,
    ) -> None:
        """Add or remove tokens from the bucket by a signed delta.

        Used to correct the bucket after a request fails (refund) or to
        reconcile over- / under-estimation of token usage.  The resulting level
        is clamped to ``[0, capacity]``.

        Parameters
        ----------
        delta : int
            Signed token adjustment.  Positive values add tokens back
            (e.g. refund after a failed request); negative values remove them.
            A delta of ``0`` is a no-op.
        current_time : float or None, optional
            Monotonic timestamp.  Obtained automatically when ``None``.
        """
        if delta == 0:
            return

        current_time = current_time or time.monotonic()
        self._refill(current_time)
        self._tokens = max(0.0, min(self.capacity, self._tokens + delta))
        self._last_sync_rate = None
        self._server_reset_time = None

    def estimate_next_refill_time(
        self,
        num_tokens_needed: int,
        current_time: float | None = None,
    ) -> float:
        """Estimate the seconds until ``num_tokens_needed`` tokens are available.

        Used by the rate limiter to schedule wake-up timers for waiting
        requests.

        Parameters
        ----------
        num_tokens_needed : int
            The number of tokens a waiting request requires.
        current_time : float or None, optional
            Monotonic timestamp.  Obtained automatically when ``None``.

        Returns
        -------
        float
            Estimated seconds until the bucket will have enough tokens.
            Returns ``0.0`` if the bucket already has sufficient tokens.
            Returns ``float('inf')`` if the effective refill rate is zero and
            no future server reset time is known (a warning is logged).

        Notes
        -----
        A tiny epsilon (``1e-9``) is added to the estimate to avoid waking up
        fractionally too early and immediately re-sleeping.
        """
        current_time = current_time or time.monotonic()
        available_tokens = self.get_bucket_level(current_time)
        shortfall = num_tokens_needed - available_tokens
        if shortfall <= 0:
            return 0.0

        effective_rate = self._get_effective_rate(current_time)
        if effective_rate <= 0:
            if (
                self._server_reset_time is not None
                and self._server_reset_time > current_time
            ):
                return max(0.0, self._server_reset_time - current_time) + 1e-9

            logger.warning(
                "Bucket refill rate is zero and no future reset time is known."
            )
            return float("inf")

        return (shortfall / effective_rate) + 1e-9

    def sync_from_response_header(
        self,
        server_token_limit: int | None,
        server_tokens_remaining: int,
        server_reset_time: float,
        current_time: float | None = None,
    ) -> None:
        """Synchronise the bucket with the server's authoritative rate-limit state.

        After calling this method, the bucket's level is set to
        `server_tokens_remaining` (capped at `capacity`) and its
        effective refill rate is recalculated from the remaining tokens and the
        time until the reset window ends.  The server-derived rate stays active
        for up to `SYNC_RATE_VALIDITY_DURATION` seconds.

        This method is called automatically by
        [adjust][infermesh.RateLimiter.adjust] when `x-ratelimit-*` response
        headers are present.

        Parameters
        ----------
        server_token_limit : int or None
            The provider's stated limit for this window (from the
            ``x-ratelimit-limit-*`` header).  ``None`` if not available; the
            bucket's own capacity is used as the refill target.
        server_tokens_remaining : int
            Tokens remaining in the current window, from the
            ``x-ratelimit-remaining-*`` header.  Must be non-negative.
        server_reset_time : float
            Monotonic timestamp at which the provider will reset the window,
            derived from the ``x-ratelimit-reset-*`` header.  Must be in the
            future relative to ``current_time``.
        current_time : float or None, optional
            Monotonic timestamp.  Obtained automatically when ``None``.

        Raises
        ------
        ValueError
            If ``server_token_limit`` is not positive, ``server_tokens_remaining``
            is negative, or ``server_reset_time`` is not in the future.
        """
        current_time = current_time or time.monotonic()
        self._refill(current_time)

        if server_token_limit is not None and server_token_limit <= 0:
            raise ValueError("Expected ``server_token_limit`` to be positive.")
        if server_tokens_remaining < 0:
            raise ValueError("Expected ``server_tokens_remaining`` to be non-negative.")
        if server_reset_time <= current_time:
            raise ValueError("Expected ``server_reset_time`` to be in the future.")

        self._tokens = float(min(self.capacity, server_tokens_remaining))
        self._last_refill_time = current_time
        self._server_reset_time = server_reset_time
        if server_token_limit is None:
            refill_target = self.capacity
        else:
            refill_target = min(self.capacity, server_token_limit)
        delay = server_reset_time - current_time
        self._last_sync_rate = max(0.0, (refill_target - self._tokens) / delay)
        self._last_sync_time = current_time

    def _get_effective_rate(self, current_time: float) -> float:
        """Return the active refill rate (tokens per second).

        Returns the server-synced rate when it is still within its validity
        window, otherwise falls back to the configured rate.
        """
        if (
            self._last_sync_rate is not None
            and (current_time - self._last_sync_time)
            <= self.SYNC_RATE_VALIDITY_DURATION
        ):
            return self._last_sync_rate
        return self._rate_per_second

    def _refill(self, current_time: float) -> None:
        """Add tokens proportional to elapsed time since the last refill."""
        elapsed = max(0.0, current_time - self._last_refill_time)
        if elapsed <= 0:
            return
        refill_amount = elapsed * self._get_effective_rate(current_time)
        self._tokens = min(self.capacity, self._tokens + refill_amount)
        self._last_refill_time = current_time
