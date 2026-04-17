"""Private runtime helpers for ``LMClient``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse

from infermesh._utils import (
    extract_deployment_label,
    extract_response_headers,
    validate_endpoint,
)
from infermesh.rate_limiter import RateLimiter, RateLimiterAcquisitionHandle
from infermesh.sync_runner import SyncRunner
from infermesh.types import (
    DeploymentConfig,
    EndpointType,
    RequestMetrics,
    ResponsesInput,
)

logger = logging.getLogger("infermesh.client")

_RoutingStrategy = Literal[
    "simple-shuffle",
    "least-busy",
    "usage-based-routing",
    "latency-based-routing",
    "cost-based-routing",
    "usage-based-routing-v2",
]


@dataclass(slots=True)
class _LoopState:
    """Async state local to an event loop."""

    semaphore: asyncio.Semaphore | None
    router: Any | None


@contextlib.asynccontextmanager
async def _null_async_context() -> Any:
    """Return a no-op async context manager."""

    yield


class _ClientRuntimeMixin:
    """Private request-dispatch helpers shared by ``LMClient`` methods."""

    model: str
    api_base: str | None
    api_key: str | None
    endpoint: EndpointType
    max_parallel_requests: int | None
    rpm: int | None
    tpm: int | None
    rpd: int | None
    tpd: int | None
    max_request_burst: int | None
    max_token_burst: int | None
    header_bucket_scope: Literal["minute", "day", "auto"]
    default_output_tokens: int
    timeout: float | None
    max_retries: int
    default_request_kwargs: dict[str, Any]
    routing_strategy: str
    router_kwargs: dict[str, Any]
    _rate_limiter: RateLimiter | None
    _litellm: Any
    _retryable_exceptions: tuple[type[BaseException], ...]
    _sync_runner: SyncRunner
    _loop_states: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState]
    _loop_states_lock: threading.Lock
    _routing_lock: threading.Lock
    _round_robin_index: int
    _deployments: dict[str, DeploymentConfig] | None
    _closed: bool

    @staticmethod
    def _validate_init_args(
        *,
        model: str | None,
        api_base: str | None,
        api_key: str | None,
        deployments: dict[str, DeploymentConfig | dict[str, Any]] | None,
        endpoint: EndpointType,
        max_parallel_requests: int | None,
    ) -> None:
        """Validate top-level constructor arguments."""

        validate_endpoint(endpoint)
        if model is None:
            raise ValueError("``model`` is required.")
        if max_parallel_requests is not None and max_parallel_requests < 1:
            raise ValueError(
                "``max_parallel_requests`` must be ``None`` or a positive integer."
            )
        if deployments is not None and (api_base is not None or api_key is not None):
            raise ValueError(
                "``api_base`` and ``api_key`` cannot be set when ``deployments`` "
                "is provided."
            )

    @staticmethod
    def _warn_on_insecure_api_base(api_base: str | None) -> None:
        """Warn when a remote ``api_base`` uses plain HTTP."""

        if api_base is None or not api_base.startswith("http://"):
            return
        host = urlparse(api_base).hostname or ""
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return
        logger.warning(
            "api_base %r uses an unencrypted HTTP connection. "
            "Credentials and data may be exposed in transit. "
            "Use HTTPS for production deployments.",
            api_base,
        )

    @staticmethod
    def _build_default_request_kwargs(
        default_request_kwargs: dict[str, Any] | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Merge constructor-level default request kwargs."""

        request_kwargs = dict(default_request_kwargs or {})
        if timeout is not None:
            request_kwargs.setdefault("timeout", timeout)
        return request_kwargs

    @staticmethod
    def _build_rate_limiter(
        *,
        rpm: int | None,
        tpm: int | None,
        rpd: int | None,
        tpd: int | None,
        max_request_burst: int | None,
        max_token_burst: int | None,
        header_bucket_scope: Literal["minute", "day", "auto"],
    ) -> RateLimiter | None:
        """Create a client-side rate limiter when limits are configured."""

        if not any(limit is not None for limit in (rpm, tpm, rpd, tpd)):
            return None

        requests_per_minute = rpm
        if requests_per_minute is None:
            if tpm is not None:
                requests_per_minute = max(1, tpm // 6000)
            elif rpd is not None:
                requests_per_minute = max(1, rpd)
            else:
                requests_per_minute = max(1, tpd or 1)

        return RateLimiter(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tpm,
            requests_per_day=rpd,
            tokens_per_day=tpd,
            max_request_burst=max_request_burst,
            max_token_burst=max_token_burst,
            header_bucket_scope=header_bucket_scope,
        )

    def _build_retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return the provider exceptions retried by ``_with_retry``."""

        return (
            self._litellm.RateLimitError,
            self._litellm.ServiceUnavailableError,
            self._litellm.APIConnectionError,
            self._litellm.Timeout,
            self._litellm.InternalServerError,
        )

    def _initialize_runtime_state(
        self, deployments: dict[str, DeploymentConfig | dict[str, Any]] | None
    ) -> None:
        """Initialize runtime-only state that is not user configuration."""

        self._sync_runner = SyncRunner()
        self._loop_states = weakref.WeakKeyDictionary()
        self._loop_states_lock = threading.Lock()
        self._routing_lock = threading.Lock()
        self._round_robin_index = 0
        self._deployments = self._coerce_deployments(deployments)
        self._closed = False

    async def _dispatch_with_controls(
        self,
        *,
        estimated_tokens: int,
        request_callable: Any,
        request_args: tuple[Any, ...],
        request_kwargs: dict[str, Any],
        queue_started_at: float | None = None,
    ) -> tuple[Any, RequestMetrics]:
        """Run a request with concurrency and rate-limiting controls."""

        state = self._get_loop_state()
        queue_started = (
            queue_started_at if queue_started_at is not None else time.perf_counter()
        )
        handle: RateLimiterAcquisitionHandle | None = None
        semaphore_context = (
            state.semaphore if state.semaphore is not None else _null_async_context()
        )
        selected_deployment: str | None = None
        async with semaphore_context:
            if self._rate_limiter is not None:
                handle = await self._rate_limiter.acquire(estimated_tokens)
                if handle is None:
                    raise asyncio.CancelledError
            dispatch_started = time.perf_counter()
            try:
                response, selected_deployment = await request_callable(
                    *request_args,
                    **request_kwargs,
                )
            except BaseException:
                if handle is not None and self._rate_limiter is not None:
                    await self._rate_limiter.adjust(handle, actual_tokens=0)
                raise
            completed = time.perf_counter()

        metrics = RequestMetrics(
            queue_wait_s=dispatch_started - queue_started,
            service_time_s=completed - dispatch_started,
            end_to_end_s=completed - queue_started,
            deployment=selected_deployment or extract_deployment_label(response),
        )
        if handle is not None and self._rate_limiter is not None:
            usage = self._extract_usage_total(response)
            await self._rate_limiter.adjust(
                handle,
                actual_tokens=usage if usage is not None else handle.estimated_tokens,
                response_headers=extract_response_headers(response),
            )
        return response, metrics

    async def _call_generation(
        self,
        input_data: str | list[dict[str, Any]] | ResponsesInput,
        *,
        endpoint: EndpointType,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str | None]:
        """Call the provider for generation."""

        deployment_name = None
        if endpoint == "responses":
            if not isinstance(input_data, dict):
                raise ValueError("Responses endpoint expects a mapping input.")
            if self._deployments is not None:
                deployment_name, deployment_config = self._select_deployment()
                response = await self._litellm.aresponses(
                    model=deployment_config.model,
                    input=input_data["input"],
                    instructions=input_data.get("instructions"),
                    api_base=deployment_config.api_base,
                    api_key=deployment_config.api_key,
                    **(deployment_config.extra_kwargs or {}),
                    **request_kwargs,
                )
                return response, deployment_name
            response = await self._litellm.aresponses(
                model=self.model,
                input=input_data["input"],
                instructions=input_data.get("instructions"),
                api_base=self.api_base,
                api_key=self.api_key,
                **request_kwargs,
            )
            return response, None

        if self._deployments is not None:
            state = self._get_loop_state()
            if state.router is None:
                raise RuntimeError("Expected a router in deployment mode.")
            if endpoint == "text_completion":
                response = await state.router.atext_completion(
                    model=self.model,
                    prompt=input_data,
                    **request_kwargs,
                )
            else:
                response = await state.router.acompletion(
                    model=self.model,
                    messages=input_data,
                    **request_kwargs,
                )
            return response, extract_deployment_label(response)

        if endpoint == "text_completion":
            response = await self._litellm.atext_completion(
                model=self.model,
                prompt=input_data,
                api_base=self.api_base,
                api_key=self.api_key,
                **request_kwargs,
            )
        else:
            response = await self._litellm.acompletion(
                model=self.model,
                messages=input_data,
                api_base=self.api_base,
                api_key=self.api_key,
                **request_kwargs,
            )
        return response, None

    async def _call_embedding(
        self,
        input_data: list[str],
        *,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str | None]:
        """Call the provider for embeddings."""

        if self._deployments is not None:
            state = self._get_loop_state()
            if state.router is None:
                raise RuntimeError("Expected a router in deployment mode.")
            response = await state.router.aembedding(
                model=self.model,
                input=input_data,
                **request_kwargs,
            )
            return response, extract_deployment_label(response)

        response = await self._litellm.aembedding(
            model=self.model,
            input=input_data,
            api_base=self.api_base,
            api_key=self.api_key,
            **request_kwargs,
        )
        return response, None

    async def _call_transcription(
        self,
        input_data: bytes | tuple[str, bytes],
        *,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str | None]:
        """Call the provider for transcription."""

        if self._deployments is not None:
            state = self._get_loop_state()
            if state.router is not None and hasattr(state.router, "atranscription"):
                response = await state.router.atranscription(
                    model=self.model,
                    file=input_data,
                    **request_kwargs,
                )
                return response, extract_deployment_label(response)

            deployment_name, deployment_config = self._select_deployment()
            response = await self._litellm.atranscription(
                model=deployment_config.model,
                file=input_data,
                api_base=deployment_config.api_base,
                api_key=deployment_config.api_key,
                **(deployment_config.extra_kwargs or {}),
                **request_kwargs,
            )
            return response, deployment_name

        response = await self._litellm.atranscription(
            model=self.model,
            file=input_data,
            api_base=self.api_base,
            api_key=self.api_key,
            **request_kwargs,
        )
        return response, None

    def _get_loop_state(self) -> _LoopState:
        """Get per-event-loop state."""

        loop = asyncio.get_running_loop()
        state = self._loop_states.get(loop)
        if state is not None:
            return state
        with self._loop_states_lock:
            state = self._loop_states.get(loop)
            if state is not None:
                return state
            semaphore = (
                asyncio.Semaphore(self.max_parallel_requests)
                if self.max_parallel_requests is not None
                else None
            )
            router = self._build_router() if self._deployments is not None else None
            state = _LoopState(semaphore=semaphore, router=router)
            self._loop_states[loop] = state
            return state

    def _build_router(self) -> Any:
        """Build a LiteLLM Router client."""

        from litellm.router import Router

        model_list: list[dict[str, Any]] = []
        assert self._deployments is not None
        for deployment in self._deployments.values():
            model_list.append(
                {
                    "model_name": self.model,
                    "litellm_params": {
                        "model": deployment.model,
                        "api_base": deployment.api_base,
                        **(
                            {"api_key": deployment.api_key}
                            if deployment.api_key
                            else {}
                        ),
                        **(deployment.extra_kwargs or {}),
                    },
                }
            )
        return Router(
            model_list=model_list,
            routing_strategy=cast(_RoutingStrategy, self.routing_strategy),
            **self.router_kwargs,
        )

    def _select_deployment(self) -> tuple[str, DeploymentConfig]:
        """Select one configured deployment label for a manual per-replica call."""

        assert self._deployments is not None
        items = list(self._deployments.items())
        with self._routing_lock:
            index = self._round_robin_index % len(items)
            self._round_robin_index += 1
            return items[index]

    def _merge_request_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge request kwargs with client defaults."""

        merged = dict(self.default_request_kwargs)
        merged.update(overrides)
        if "num_retries" in merged and self.max_retries > 0:
            merged.pop("num_retries")
            logger.warning(
                "num_retries kwarg ignored; use LMClient(max_retries=N) instead."
            )
        return merged

    def _get_generation_output_token_limit(
        self,
        request_kwargs: dict[str, Any],
        endpoint: EndpointType,
    ) -> int:
        """Return the per-request output-token limit used for reservation."""

        keys = (
            ("max_output_tokens", "max_tokens")
            if endpoint == "responses"
            else ("max_tokens", "max_output_tokens")
        )
        for key in keys:
            value = request_kwargs.get(key)
            if value is not None:
                return int(value or 0)
        return 0

    def _extract_usage_total(self, response: Any) -> int | None:
        """Extract total token usage from a provider response."""

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return None
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
        else:
            total = getattr(usage, "total_tokens", None)
        return int(total) if total is not None else None

    def _coerce_deployments(
        self,
        deployments: dict[str, DeploymentConfig | dict[str, Any]] | None,
    ) -> dict[str, DeploymentConfig] | None:
        """Normalize deployment configuration values."""

        if deployments is None:
            return None
        normalized: dict[str, DeploymentConfig] = {}
        for name, value in deployments.items():
            if isinstance(value, DeploymentConfig):
                normalized[name] = value
                continue
            normalized[name] = DeploymentConfig(
                model=str(value["model"]),
                api_base=str(value["api_base"]),
                api_key=value.get("api_key"),
                extra_kwargs=dict(value.get("extra_kwargs", {}) or {}),
            )
        return normalized

    def _create_litellm_module(self) -> Any:
        """Import and configure LiteLLM."""

        import litellm
        import litellm._logging

        litellm._logging.verbose_logger.setLevel(logging.WARNING)
        if hasattr(litellm._logging, "verbose_router_logger"):
            litellm._logging.verbose_router_logger.setLevel(logging.WARNING)
        return litellm
