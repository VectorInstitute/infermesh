"""Public ``LMClient`` interface built on top of LiteLLM.

The single public class in this module, [LMClient][infermesh.LMClient], wraps
LiteLLM to provide typed generation, embedding, and transcription calls for
batch-heavy workflows. Every public method has both a synchronous form and an
``a``-prefixed async counterpart that can be awaited directly. The synchronous
methods are notebook-safe: they run coroutine work on a managed background loop
so callers can use batching helpers without manually creating or coordinating an
event loop.

Examples
--------
>>> from infermesh import LMClient
>>> with LMClient(
...     model="openai/gpt-4o-mini",
...     api_base="https://api.openai.com/v1",
... ) as client:
...     result = client.generate("Say hello in five languages.")
>>> result.output_text

See Also
--------
``docs/guide.md``
    Fuller routing, batching, and notebook examples.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import random
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel

from infermesh._client_runtime import _ClientRuntimeMixin
from infermesh._generation import _agenerate_batch, _agenerate_one
from infermesh._utils import (
    build_embedding_results,
    build_transcription_result,
    estimate_token_count,
    normalize_embedding_input,
    normalize_transcription_input,
)
from infermesh.rate_limiter import RateLimiter as _RateLimiter
from infermesh.types import (
    BatchResult,
    DeploymentConfig,
    EmbeddingBatchResult,
    EmbeddingResult,
    EndpointType,
    GenerateInput,
    GenerationBatchResult,
    GenerationResult,
    OnGenerationResult,
    RequestMetrics,
    TranscriptionInput,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)
DEFAULT_MAX_TRANSCRIPTION_BYTES = 25 * 1024 * 1024
# Backwards-compatible module alias for existing patch/import targets.
RateLimiter = _RateLimiter


class LMClient(_ClientRuntimeMixin):
    """Batch-friendly language-model interface built on LiteLLM.

    [LMClient][infermesh.LMClient] supports two operating modes selected at construction
    time:

    **Single-endpoint mode** — one model, one server.  Provide ``model`` and
    ``api_base``:

    ```python
    client = LMClient(
        model="openai/gpt-4.1-mini",
        api_base="https://api.openai.com/v1",
    )
    result = client.generate("What is the capital of France?")
    print(result.output_text)  # "Paris"
    client.close()
    ```

    **Router mode** — multiple replicas, load-balanced by a LiteLLM Router.
    Provide ``model`` (the logical model name) and ``deployments`` (a dict of
    free-form label → [DeploymentConfig][infermesh.DeploymentConfig]):

    ```python
    client = LMClient(
        model="llama-3",
        deployments={
            "gpu-0": DeploymentConfig(
                model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
                api_base="http://gpu0:8000/v1",
            ),
            "gpu-1": DeploymentConfig(
                model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
                api_base="http://gpu1:8000/v1",
            ),
        },
    )
    ```

    If you only need a few single requests, plain LiteLLM or the provider SDK
    is usually simpler. [LMClient][infermesh.LMClient] becomes useful when you
    need concurrent batches, per-item failure handling, client-side rate
    limiting, or routing across several replicas of the same logical model.

    The client can be used as a context manager (sync or async) to ensure
    [close][infermesh.LMClient.close] is always called:

    ```python
    with LMClient(
        model="openai/gpt-4o", api_base="https://api.openai.com/v1"
    ) as client:
        batch = client.generate_batch(prompts)
    ```

    The sync methods delegate to their async counterparts through a managed
    background loop, which keeps notebook and REPL usage simple while preserving
    retries, throttling, and batching behavior.

    Notes
    -----
    Always call [close][infermesh.LMClient.close] (or use the context-manager form)
    when the client is no longer needed. [close][infermesh.LMClient.close] stops
    the background `SyncRunner` thread; failing to call it leaves a daemon thread
    running until process exit.

    A single [RateLimiter][infermesh.RateLimiter] instance is shared between
    the caller's event loop and the `SyncRunner` background loop, so sync and
    async calls are accounted together and do not double the effective rate.

    See Also
    --------
    [DeploymentConfig][infermesh.DeploymentConfig] : Per-replica configuration
    for router mode.

    [RateLimiter][infermesh.RateLimiter] : The rate-limiter used internally;
    can also be used standalone.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        deployments: dict[str, DeploymentConfig | dict[str, Any]] | None = None,
        endpoint: EndpointType = "chat_completion",
        max_parallel_requests: int | None = None,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        tpd: int | None = None,
        max_request_burst: int | None = None,
        max_token_burst: int | None = None,
        header_bucket_scope: Literal["minute", "day", "auto"] = "auto",
        default_output_tokens: int = 0,
        timeout: float | None = None,
        max_retries: int = 3,
        default_request_kwargs: dict[str, Any] | None = None,
        routing_strategy: str = "simple-shuffle",
        router_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Create an ``LMClient`` instance.

        Parameters
        ----------
        model : str
            Provider model name used for all requests. This is required in both
            single-endpoint and deployment-router modes.
        api_base, api_key : str | None, optional
            Connection details for direct, single-endpoint usage. Leave these
            unset when routing through ``deployments``.
        deployments : dict[str, DeploymentConfig | dict[str, Any]] | None, optional
            Named deployment definitions used for router mode. Each deployment
            can override model, API base, API key, and provider-specific kwargs.
        endpoint : EndpointType, default="chat_completion"
            Default generation endpoint used by ``generate`` and
            ``generate_batch`` unless a per-call override is supplied.
        max_parallel_requests : int | None, optional
            Per-event-loop cap on concurrent in-flight requests. When set,
            ``generate_batch`` and ``agenerate_batch`` also admit generation
            work through a bounded in-flight window instead of creating one
            task per item up front. Must be ``None`` or a positive integer.
        rpm, tpm, rpd, tpd : int | None, optional
            Client-side rate-limit settings. When any limit is set, the client
            creates a shared limiter used by sync and async methods.
        max_request_burst, max_token_burst : int | None, optional
            Burst allowances applied by the client-side rate limiter.
        header_bucket_scope : {"minute", "day", "auto"}, default="auto"
            How provider rate-limit headers should be interpreted when updating
            limiter state after a request.
        default_output_tokens : int, default=0
            Default completion-token budget used when estimating token
            reservations for rate limiting.
        timeout : float | None, optional
            Default request timeout forwarded to LiteLLM unless a per-call
            timeout is supplied in ``kwargs``.
        max_retries : int, default=3
            Number of retry attempts for transient provider failures such as
            rate limits, timeouts, and internal server errors.
        default_request_kwargs : dict[str, Any] | None, optional
            Request kwargs merged into every provider call.
        routing_strategy : str, default="simple-shuffle"
            Router selection strategy used when ``deployments`` are configured.
        router_kwargs : dict[str, Any] | None, optional
            Extra kwargs forwarded when building the LiteLLM router.

        Warns
        -----
        UserWarning
            A logger warning is emitted when ``api_base`` uses insecure HTTP for
            a non-local host.

        Raises
        ------
        ValueError
            If ``model`` is missing, if ``endpoint`` is invalid, or if
            deployment mode is mixed with direct ``api_base``/``api_key``
            settings, or if ``max_parallel_requests`` is not positive.

        Examples
        --------
        >>> client = LMClient(
        ...     model="openai/gpt-4o-mini",
        ...     api_base="https://api.openai.com/v1",
        ...     timeout=30,
        ... )
        >>> client.close()
        """

        self._validate_init_args(
            model=model,
            api_base=api_base,
            api_key=api_key,
            deployments=deployments,
            endpoint=endpoint,
            max_parallel_requests=max_parallel_requests,
        )
        assert model is not None
        self._warn_on_insecure_api_base(api_base)
        self._store_configuration(
            model=model,
            api_base=api_base,
            api_key=api_key,
            endpoint=endpoint,
            max_parallel_requests=max_parallel_requests,
            rpm=rpm,
            tpm=tpm,
            rpd=rpd,
            tpd=tpd,
            max_request_burst=max_request_burst,
            max_token_burst=max_token_burst,
            header_bucket_scope=header_bucket_scope,
            default_output_tokens=default_output_tokens,
            timeout=timeout,
            max_retries=max_retries,
            default_request_kwargs=default_request_kwargs,
            routing_strategy=routing_strategy,
            router_kwargs=router_kwargs,
        )
        self._rate_limiter = self._build_rate_limiter(
            rpm=rpm,
            tpm=tpm,
            rpd=rpd,
            tpd=tpd,
            max_request_burst=max_request_burst,
            max_token_burst=max_token_burst,
            header_bucket_scope=header_bucket_scope,
        )
        self._litellm = self._create_litellm_module()
        self._retryable_exceptions = self._build_retryable_exceptions()
        self._initialize_runtime_state(deployments)

    def _store_configuration(
        self,
        *,
        model: str,
        api_base: str | None,
        api_key: str | None,
        endpoint: EndpointType,
        max_parallel_requests: int | None,
        rpm: int | None,
        tpm: int | None,
        rpd: int | None,
        tpd: int | None,
        max_request_burst: int | None,
        max_token_burst: int | None,
        header_bucket_scope: Literal["minute", "day", "auto"],
        default_output_tokens: int,
        timeout: float | None,
        max_retries: int,
        default_request_kwargs: dict[str, Any] | None,
        routing_strategy: str,
        router_kwargs: dict[str, Any] | None,
    ) -> None:
        """Store validated constructor configuration on the instance."""

        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_parallel_requests = max_parallel_requests
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.tpd = tpd
        self.max_request_burst = max_request_burst
        self.max_token_burst = max_token_burst
        self.header_bucket_scope = header_bucket_scope
        self.default_output_tokens = default_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_request_kwargs = self._build_default_request_kwargs(
            default_request_kwargs,
            timeout,
        )
        self.routing_strategy = routing_strategy
        self.router_kwargs = dict(router_kwargs or {})

    def close(self) -> None:
        """Release background resources used by the synchronous API.

        ``generate``, ``embed``, and ``transcribe`` run on a managed background
        event loop. Call [close][infermesh.LMClient.close] when you are finished
        with the client, or prefer ``with`` / ``async with`` so cleanup happens
        automatically.
        """

        if self._closed:
            return
        self._sync_runner.close()
        self._closed = True

    def generate(
        self,
        input_data: GenerateInput,
        *,
        endpoint: EndpointType | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        parse_output: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate one response on the notebook-safe background loop.

        Parameters
        ----------
        input_data : GenerateInput
            Prompt text, a chat-style message list, or a ``responses`` payload.
            The accepted shape depends on the selected endpoint.
        endpoint : EndpointType | None, optional
            Per-call override for the generation endpoint. Defaults to the
            client-wide ``endpoint`` set at construction time.
        response_format : type[BaseModel] | dict[str, Any] | None, optional
            Structured output target. Pass a Pydantic model class or provider
            schema mapping to parse the response into ``output_parsed``.
        parse_output : bool, default=False
            When ``True``, attempt to parse structured output even without an
            explicit ``response_format``.
        **kwargs : Any
            Additional LiteLLM request kwargs such as temperature or max tokens.

        Returns
        -------
        GenerationResult
            The generated text, optional parsed output, token usage, request
            id, finish reason, and timing metrics.

        Raises
        ------
        ValueError
            If ``input_data`` does not match the selected endpoint contract.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = client.generate("Summarize the French Revolution.")
        >>> result.output_text
        """

        return self._sync_runner.run(
            self.agenerate(
                input_data,
                endpoint=endpoint,
                response_format=response_format,
                parse_output=parse_output,
                **kwargs,
            )
        )

    def generate_batch(
        self,
        input_batch: Sequence[GenerateInput],
        *,
        endpoint: EndpointType | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        parse_output: bool = False,
        return_exceptions: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: OnGenerationResult = None,
        **kwargs: Any,
    ) -> GenerationBatchResult:
        """Generate a batch of responses synchronously.

        Parameters
        ----------
        input_batch : Sequence[GenerateInput]
            Ordered inputs to run. Each item may be a prompt string, chat
            message list, or ``responses`` payload.
        endpoint, response_format, parse_output, **kwargs
            Behave the same as in ``generate`` and are applied to every item in
            the batch.
        return_exceptions : bool, default=True
            When ``True``, item failures are captured in ``errors`` and the
            rest of the batch keeps running. When ``False``, the first failure
            cancels remaining work and is raised.
        on_progress : callable | None, optional
            Callback invoked as ``on_progress(completed, total)`` each time an
            item finishes.
        on_result : callable | None, optional
            Callback invoked as ``on_result(index, result, error)`` for each
            completed item.

        Returns
        -------
        GenerationBatchResult
            A batch result with one slot per input item. Successful items appear
            in ``results`` and failures, when captured, appear in ``errors``.

        Notes
        -----
        For large or memory-sensitive Python batch runs, set
        ``max_parallel_requests`` on the client. When it is unset,
        ``generate_batch`` may start work for the full batch up front.

        Raises
        ------
        ValueError
            If any batch item has an invalid input shape.
        Exception
            The first provider error when ``return_exceptions`` is ``False``.

        Examples
        --------
        >>> batch = client.generate_batch(
        ...     ["ELI5: Artificial Intelligence", "ELI5: Quantum Computing"]
        ... )
        >>> [item.output_text if item else None for item in batch.results]
        """

        return self._sync_runner.run(
            self.agenerate_batch(
                input_batch,
                endpoint=endpoint,
                response_format=response_format,
                parse_output=parse_output,
                return_exceptions=return_exceptions,
                on_progress=on_progress,
                on_result=on_result,
                **kwargs,
            )
        )

    async def agenerate(
        self,
        input_data: GenerateInput,
        *,
        endpoint: EndpointType | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        parse_output: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate one response asynchronously.

        Parameters
        ----------
        input_data, endpoint, response_format, parse_output, **kwargs
            Follow the same contract as ``generate``.

        Returns
        -------
        GenerationResult
            The generated text, structured output if requested, and request
            metadata for a single input.

        Raises
        ------
        ValueError
            If ``input_data`` does not match the selected endpoint contract.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = await client.agenerate("Summarize the French Revolution.")
        >>> result.output_text
        """

        request_endpoint = endpoint or self.endpoint
        return await _agenerate_one(
            self,
            input_data,
            endpoint=request_endpoint,
            response_format=response_format,
            parse_output=parse_output,
            request_kwargs=self._merge_request_kwargs(kwargs),
        )

    async def agenerate_batch(
        self,
        input_batch: Sequence[GenerateInput],
        *,
        endpoint: EndpointType | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        parse_output: bool = False,
        return_exceptions: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        on_result: OnGenerationResult = None,
        **kwargs: Any,
    ) -> GenerationBatchResult:
        """Generate a batch of responses asynchronously.

        Parameters
        ----------
        input_batch, endpoint, response_format, parse_output, **kwargs
            Follow the same contract as ``generate_batch``.
        return_exceptions : bool, default=True
            Capture per-item failures in ``errors`` when ``True``. When
            ``False``, the first failure cancels the remaining tasks and is
            raised.
        on_progress, on_result : callable | None, optional
            Optional callbacks invoked as items finish.

        Returns
        -------
        GenerationBatchResult
            Batch-sized ``results`` and ``errors`` collections aligned to the
            original input order.

        Notes
        -----
        For large or memory-sensitive Python batch runs, set
        ``max_parallel_requests`` on the client. When it is unset,
        ``agenerate_batch`` may start work for the full batch up front.

        Raises
        ------
        ValueError
            If any input item is invalid for the selected endpoint.
        Exception
            The first provider error when ``return_exceptions`` is ``False``.

        Examples
        --------
        >>> batch = await client.agenerate_batch(
        ...     ["ELI5: Artificial Intelligence", "ELI5: Quantum Computing"]
        ... )
        >>> len(batch.results)
        """

        return await _agenerate_batch(
            self,
            input_batch,
            endpoint=endpoint,
            response_format=response_format,
            parse_output=parse_output,
            return_exceptions=return_exceptions,
            on_progress=on_progress,
            on_result=on_result,
            **kwargs,
        )

    def embed(self, input_data: str, **kwargs: Any) -> EmbeddingResult:
        """Create an embedding for a single text string synchronously.

        Parameters
        ----------
        input_data : str
            Text to embed.
        **kwargs : Any
            Additional LiteLLM embedding kwargs such as dimension hints.

        Returns
        -------
        EmbeddingResult
            One embedding vector plus request metadata and token-usage details.

        Raises
        ------
        ValueError
            If the provider response does not contain an embedding vector.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = client.embed("The quick brown fox")
        >>> len(result.embedding)
        """

        return self._sync_runner.run(self.aembed(input_data, **kwargs))

    def embed_batch(
        self,
        input_batch: Sequence[str],
        *,
        return_exceptions: bool = True,
        **kwargs: Any,
    ) -> EmbeddingBatchResult:
        """Create embeddings for a batch of text strings synchronously.

        Parameters
        ----------
        input_batch : Sequence[str]
            Text values to embed together.
        return_exceptions : bool, default=True
            When ``True``, a provider failure is captured once per item in the
            ``errors`` list. When ``False``, the provider error is raised.
        **kwargs : Any
            Additional LiteLLM embedding kwargs applied to the batch request.

        Returns
        -------
        EmbeddingBatchResult
            A result slot for each input string, aligned to the original order.

        Raises
        ------
        Exception
            The provider error when ``return_exceptions`` is ``False``.

        Examples
        --------
        >>> batch = client.embed_batch(["Queen", "King", "Card", "Ace"])
        >>> [len(item.embedding) if item else None for item in batch.results]
        """

        return self._sync_runner.run(
            self.aembed_batch(
                input_batch,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        )

    async def aembed(self, input_data: str, **kwargs: Any) -> EmbeddingResult:
        """Create an embedding for a single text string asynchronously.

        Parameters
        ----------
        input_data, **kwargs
            Follow the same contract as ``embed``.

        Returns
        -------
        EmbeddingResult
            The embedding vector and request metadata for one text value.

        Raises
        ------
        ValueError
            If the provider response does not contain an embedding vector.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = await client.aembed("The quick brown fox")
        >>> len(result.embedding)
        """

        response, metrics = await self._with_retry(
            lambda: self._dispatch_with_controls(
                estimated_tokens=estimate_token_count(
                    self._litellm,
                    self.model,
                    input_data,
                    endpoint="chat_completion",
                    max_tokens=0,
                ),
                request_callable=self._call_embedding,
                request_args=([input_data],),
                request_kwargs={"request_kwargs": self._merge_request_kwargs(kwargs)},
            )
        )
        results = build_embedding_results(response, metrics=metrics)
        if not results:
            raise ValueError("Embedding response did not contain any vectors.")
        return results[0]

    async def aembed_batch(
        self,
        input_batch: Sequence[str],
        *,
        return_exceptions: bool = True,
        **kwargs: Any,
    ) -> EmbeddingBatchResult:
        """Create embeddings for a batch of text strings asynchronously.

        Parameters
        ----------
        input_batch, return_exceptions, **kwargs
            Follow the same contract as ``embed_batch``.

        Returns
        -------
        EmbeddingBatchResult
            One result slot per input string, aligned to the original order.

        Raises
        ------
        Exception
            The provider error when ``return_exceptions`` is ``False``.

        Examples
        --------
        >>> batch = await client.aembed_batch(["Queen", "King", "Card", "Ace"])
        >>> len(batch.results)
        """

        normalized = normalize_embedding_input(input_batch)
        if not normalized:
            return BatchResult(results=[])
        estimated_tokens = estimate_token_count(
            self._litellm,
            self.model,
            normalized,
            endpoint="chat_completion",
            max_tokens=0,
        )
        try:
            response, metrics = await self._with_retry(
                lambda: self._dispatch_with_controls(
                    estimated_tokens=estimated_tokens,
                    request_callable=self._call_embedding,
                    request_args=(normalized,),
                    request_kwargs={
                        "request_kwargs": self._merge_request_kwargs(kwargs)
                    },
                )
            )
        except Exception as exc:
            if not return_exceptions:
                raise
            return BatchResult(
                results=[None] * len(normalized), errors=[exc] * len(normalized)
            )
        results = cast(
            list[EmbeddingResult | None],
            list(build_embedding_results(response, metrics=metrics)),
        )
        return BatchResult(results=results, errors=[None] * len(results))

    def transcribe(
        self,
        input_data: TranscriptionInput,
        *,
        max_transcription_bytes: int | None = DEFAULT_MAX_TRANSCRIPTION_BYTES,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe one audio input synchronously.

        Parameters
        ----------
        input_data : TranscriptionInput
            Local path, raw bytes, or a binary file-like object containing
            audio data.
        max_transcription_bytes : int | None, optional
            Defensive size limit applied before the request is sent. Pass
            ``None`` to disable the check.
        **kwargs : Any
            Additional LiteLLM transcription kwargs such as language hints.

        Returns
        -------
        TranscriptionResult
            The transcript text plus request metadata, language, and duration
            when the provider returns them.

        Raises
        ------
        ValueError
            If the input cannot be normalized or exceeds
            ``max_transcription_bytes``.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = client.transcribe("sample.wav")
        >>> result.text
        """

        return self._sync_runner.run(
            self.atranscribe(
                input_data,
                max_transcription_bytes=max_transcription_bytes,
                **kwargs,
            )
        )

    async def atranscribe(
        self,
        input_data: TranscriptionInput,
        *,
        max_transcription_bytes: int | None = DEFAULT_MAX_TRANSCRIPTION_BYTES,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe one audio input asynchronously.

        Parameters
        ----------
        input_data, max_transcription_bytes, **kwargs
            Follow the same contract as ``transcribe``.

        Returns
        -------
        TranscriptionResult
            The transcript text and request metadata for one audio input.

        Raises
        ------
        ValueError
            If the input cannot be normalized or exceeds the configured size
            limit.
        Exception
            Re-raises provider errors after retries are exhausted.

        Examples
        --------
        >>> result = await client.atranscribe("sample.wav")
        >>> result.text
        """

        normalized = normalize_transcription_input(
            input_data,
            max_bytes=max_transcription_bytes,
        )
        response, metrics = await self._with_retry(
            lambda: self._dispatch_with_controls(
                estimated_tokens=0,
                request_callable=self._call_transcription,
                request_args=(normalized,),
                request_kwargs={"request_kwargs": self._merge_request_kwargs(kwargs)},
            )
        )
        return build_transcription_result(response, metrics=metrics)

    async def _with_retry(
        self,
        coro_fn: Callable[[], Coroutine[Any, Any, tuple[Any, RequestMetrics]]],
    ) -> tuple[Any, RequestMetrics]:
        """Retry transient provider failures with exponential backoff."""

        for attempt in range(self.max_retries + 1):
            try:
                response, metrics = await coro_fn()
                if attempt:
                    metrics = dataclasses.replace(metrics, retries=attempt)
                return response, metrics
            except self._retryable_exceptions as exc:
                if attempt == self.max_retries:
                    raise
                wait = self._compute_retry_wait(exc, attempt)
                logger.warning(
                    "Retrying request (attempt %d/%d) after %.1fs: %s: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    @staticmethod
    def _compute_retry_wait(exc: BaseException, attempt: int) -> float:
        """Return seconds to wait before the next retry attempt."""

        retry_after = LMClient._extract_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, 60.0)
        return min(math.pow(2, attempt), 60.0) + random.random() * 0.5

    @staticmethod
    def _extract_retry_after(exc: BaseException) -> float | None:
        """Extract ``Retry-After`` seconds from a provider exception."""

        response = getattr(exc, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", {})
        if not hasattr(headers, "get"):
            return None
        for key in ("Retry-After", "retry-after"):
            value = headers.get(key)
            if value is not None:
                with contextlib.suppress(ValueError):
                    return float(value)
        return None

    def __enter__(self) -> LMClient:
        """Enter a synchronous context-manager scope."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Exit a synchronous context-manager scope."""

        self.close()

    async def __aenter__(self) -> LMClient:
        """Enter an asynchronous context-manager scope."""

        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Exit an asynchronous context-manager scope."""

        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup for interpreter shutdown."""

        with contextlib.suppress(Exception):
            self.close()
