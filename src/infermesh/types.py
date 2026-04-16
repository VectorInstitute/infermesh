"""Shared public types for ``infermesh``.

This module defines the typed public contract used by ``infermesh`` batch
workflows. Every result returned by [LMClient][infermesh.LMClient] is an
instance of one of the classes defined here.
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Generic, Literal, TypeAlias, TypeVar

ChatMessage: TypeAlias = dict[str, Any]
"""A single chat message dict.

Must contain at least a ``"role"`` key and a ``"content"`` key, e.g.
``{"role": "user", "content": "Hello!"}``.

For multimodal (VLM) inputs the ``"content"`` value may be a list of content
blocks instead of a plain string. Text blocks have the form
``{"type": "text", "text": "..."}``; image blocks use
``{"type": "image_url", "image_url": {"url": "https://..."}}``.

Use [image_block][infermesh.image_block] to build image blocks from local files
or raw bytes.
"""

ChatInput: TypeAlias = list[ChatMessage]
"""A full chat conversation: an ordered list of `ChatMessage` dicts.

Supports both plain-text and multimodal (VLM) messages; see `ChatMessage` and
[image_block][infermesh.image_block].
"""

ResponsesInput: TypeAlias = dict[str, Any]
"""Input for the ``"responses"`` endpoint.

Contains an ``"input"`` key (required) and an optional ``"instructions"``
key for a system prompt.
"""

GenerateInput: TypeAlias = str | ChatInput | ResponsesInput
"""Union of the three accepted generation input formats.

- ``str``: plain text; converted to a single user message internally.
- `ChatInput`: a pre-built list of role/content dicts. Supports multimodal
  messages; see `ChatMessage` and [image_block][infermesh.image_block].
- `ResponsesInput`: a dict suitable for the ``responses`` endpoint.
"""


def image_block(
    source: str | Path | bytes,
    *,
    detail: Literal["auto", "low", "high"] | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Build an image content block for a multimodal chat message.

    For URL-based images no helper is needed — pass the dict directly::

        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}

    Use this function when the image is a local file or raw bytes that must be
    base64-encoded before sending to the provider.  The LLM servers cannot read
    the caller's filesystem; both require either a publicly reachable URL or a
    base64 data URL embedded in the request body.

    Parameters
    ----------
    source : str or Path or bytes
        The image source:

        - A URL string (``"https://..."`` or ``"http://..."``) — returned
          as-is inside an ``image_url`` block.
        - A `pathlib.Path` — the file is read and base64-encoded
          automatically.  MIME type is inferred from the file extension via
          `mimetypes`; supply ``mime_type`` to override.  Raises
          ``ValueError`` when the MIME type cannot be inferred and
          ``mime_type`` is not provided.
        - ``bytes`` — raw image bytes, base64-encoded.  ``mime_type`` is
          **required**.
    detail : {"auto", "low", "high"} or None, optional
        OpenAI vision detail level controlling how many image tokens are
        consumed.  ``None`` (default) omits the field and lets the provider
        choose.
    mime_type : str or None, optional
        MIME type string (e.g. ``"image/png"``).  Required when ``source``
        is ``bytes``; optional override when ``source`` is a `pathlib.Path`.

    Returns
    -------
    dict
        ``{"type": "image_url", "image_url": {"url": ...}}`` ready for use
        as an element in a `ChatMessage` ``"content"`` list.

    Raises
    ------
    ValueError
        If ``source`` is ``bytes`` and ``mime_type`` is not provided, or if
        ``source`` is a `pathlib.Path` and the MIME type cannot be inferred
        from the file extension and ``mime_type`` is not provided.  Also
        raised when a plain string is passed that is not an ``http://`` or
        ``https://`` URL — use a `pathlib.Path` for local files.
    FileNotFoundError
        If ``source`` is a `pathlib.Path` that does not exist.

    Examples
    --------
    URL (plain string is fine for URLs):

    >>> block = image_block("https://example.com/cat.jpg")

    Local file — pass a ``Path``, not a plain string:

    >>> msg = {
    ...     "role": "user",
    ...     "content": [
    ...         {"type": "text", "text": "What's in this image?"},
    ...         image_block(Path("photo.jpg")),
    ...         image_block(Path("diagram.png"), detail="high"),
    ...     ],
    ... }
    >>> result = client.generate([msg])

    Raw bytes:

    >>> with open("photo.jpg", "rb") as f:
    ...     data = f.read()
    >>> block = image_block(data, mime_type="image/jpeg")
    """
    if isinstance(source, bytes):
        if mime_type is None:
            raise ValueError(
                "``mime_type`` is required when ``source`` is raw bytes "
                "(e.g. mime_type='image/jpeg')."
            )
        encoded = base64.b64encode(source).decode("ascii")
        url = f"data:{mime_type};base64,{encoded}"
    elif isinstance(source, str):
        if not source.startswith(("http://", "https://")):
            raise ValueError(
                f"str source must be an HTTP/HTTPS URL. "
                f"For local files pass a Path: image_block(Path({source!r}))"
            )
        url = source
    elif isinstance(source, Path):
        raw = source.read_bytes()  # raises FileNotFoundError if absent
        effective_mime = mime_type or mimetypes.guess_type(str(source))[0]
        if effective_mime is None:
            raise ValueError(
                f"Cannot infer MIME type from {source!r}. "
                "Specify mime_type explicitly, e.g. mime_type='image/jpeg'."
            )
        encoded = base64.b64encode(raw).decode("ascii")
        url = f"data:{effective_mime};base64,{encoded}"
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__!r}")

    payload: dict[str, Any] = {"url": url}
    if detail is not None:
        payload["detail"] = detail
    return {"type": "image_url", "image_url": payload}


EmbeddingInput: TypeAlias = str | list[str]
"""Accepted embedding input: a single string or a list of strings."""

TranscriptionInput: TypeAlias = str | Path | bytes | BinaryIO
"""Accepted transcription input.

- ``str`` / `pathlib.Path`: path to an audio file on disk; opened and read
  automatically.
- ``bytes``: raw audio bytes.
- ``BinaryIO``: any file-like object with a ``.read()`` method.
"""

EndpointType: TypeAlias = Literal["text_completion", "chat_completion", "responses"]
"""The three supported generation endpoint identifiers.

- ``"chat_completion"`` (default): standard chat API
  (``/v1/chat/completions``).
- ``"text_completion"``: legacy completions API (``/v1/completions``). Input
  must be a plain string; LiteLLM's ``atext_completion`` is called.
- ``"responses"``: OpenAI Responses API (``/v1/responses``).
"""


@dataclass(slots=True)
class DeploymentConfig:
    """Configuration for a single deployment replica used in router mode.

    In router mode [LMClient][infermesh.LMClient] accepts a mapping of free-form
    labels (for example ``"gpu-0"`` or ``"us-east-1"``) to
    [DeploymentConfig][infermesh.DeploymentConfig] instances. The client builds
    a LiteLLM Router from these configs and load-balances requests across the
    replicas.

    Parameters
    ----------
    model : str
        Full LiteLLM model identifier understood by the provider, e.g.
        ``"hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"`` for a vLLM server
        or ``"anthropic/claude-3-5-sonnet-20241022"`` for Anthropic.
    api_base : str
        Base URL of the server, e.g. ``"http://gpu0:8000/v1"``.
    api_key : str or None, optional
        API key for this replica.  Pass ``None`` (default) when the server
        does not require authentication, which is typical for local vLLM
        deployments.
    extra_kwargs : dict or None, optional
        Additional LiteLLM keyword arguments applied only to this deployment.
        Useful for provider-specific settings such as custom request timeouts
        or Azure deployment names.

    Examples
    --------
    Create a deployment for a local vLLM replica:

    >>> from infermesh import DeploymentConfig
    >>> cfg = DeploymentConfig(
    ...     model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
    ...     api_base="http://gpu0:8000/v1",
    ... )

    Create a deployment with an environment-sourced API key and custom timeout:

    >>> import os
    >>> cfg = DeploymentConfig(
    ...     model="openai/gpt-4o",
    ...     api_base="https://api.openai.com/v1",
    ...     api_key=os.environ["OPENAI_API_KEY"],
    ...     extra_kwargs={"timeout": 30},
    ... )
    """

    model: str
    api_base: str
    api_key: str | None = None
    extra_kwargs: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolCall:
    """A tool call emitted by a model during a generation request.

    Appears in `tool_calls` when the model decides to
    invoke a function.  Use `id` to correlate the tool result back to
    the original call when continuing a multi-turn conversation.

    Parameters
    ----------
    id : str
        Unique identifier assigned by the provider for this specific tool call.
    name : str
        The function name the model wants to invoke.
    arguments : str or None, optional
        JSON-encoded string containing the arguments the model supplied.
        Parse with ``json.loads(tool_call.arguments)`` to obtain a ``dict``.
        ``None`` if the model emitted a tool call with no arguments.

    Examples
    --------
    >>> import json
    >>> result = client.generate("What is the weather in Paris?", ...)
    >>> if result.tool_calls:
    ...     for tc in result.tool_calls:
    ...         args = json.loads(tc.arguments or "{}")
    ...         print(f"Call {tc.id}: {tc.name}({args})")
    """

    id: str
    name: str
    arguments: str | None = None


@dataclass(slots=True)
class TokenUsage:
    """Token-count information returned by a provider for a single request.

    Parameters
    ----------
    prompt_tokens : int
        Number of tokens in the input (prompt / context window content).
    completion_tokens : int
        Number of tokens in the generated output.
    total_tokens : int
        Combined token count as reported by the provider.  May differ from
        ``prompt_tokens + completion_tokens`` for providers that count
        internal reasoning tokens separately.
    reasoning_tokens : int or None, optional
        Tokens consumed by chain-of-thought reasoning, when disclosed by the
        provider (e.g. OpenAI ``o1`` / ``o3`` families).  ``None`` when not
        reported.

    Attributes
    ----------
    output_tokens : int
        Provider-neutral alias for `completion_tokens`.

    Notes
    -----
    Use ``output_tokens`` (alias for ``completion_tokens``) when writing
    code that should work with multiple providers, as some SDKs use the term
    "output tokens" rather than "completion tokens".

    Examples
    --------
    >>> result = client.generate("Explain backpropagation briefly.")
    >>> if result.token_usage:
    ...     u = result.token_usage
    ...     print(
    ...         f"Prompt: {u.prompt_tokens}, Output: {u.output_tokens}, "
    ...         f"Total: {u.total_tokens}"
    ...     )
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None

    @property
    def output_tokens(self) -> int:
        """Return completion tokens under a provider-neutral alias.

        Returns
        -------
        int
            The value of `completion_tokens`.
        """
        return self.completion_tokens


@dataclass(slots=True)
class RequestMetrics:
    """Per-request timing and routing metadata.

    Attached to every [GenerationResult][infermesh.GenerationResult], [EmbeddingResult][infermesh.EmbeddingResult],
    and [TranscriptionResult][infermesh.TranscriptionResult] produced by [LMClient][infermesh.LMClient].

    Parameters
    ----------
    queue_wait_s : float
        Seconds spent waiting in the concurrency semaphore and/or rate-limiter
        queue before the request was dispatched to the provider.  A
        persistently high value indicates the client is regularly hitting its
        configured RPM / TPM limits or its ``max_parallel_requests`` cap.
    service_time_s : float
        Seconds from request dispatch to response receipt — essentially
        network round-trip time plus provider inference latency.
    end_to_end_s : float
        Total wall-clock seconds from when the call entered the client to when
        the response was received.  Always equal to
        ``queue_wait_s + service_time_s``.
    deployment : str or None, optional
        The deployment label selected for this request in router mode
        (e.g. ``"replica-1"``), extracted from LiteLLM's ``_hidden_params``
        or ``x-litellm-deployment`` header.  ``None`` in single-endpoint mode.
    retries : int, optional
        Number of retry attempts made before this response was received.
        ``0`` means the first attempt succeeded.

    Examples
    --------
    >>> result = client.generate("Hello")
    >>> m = result.metrics
    >>> if m:
    ...     print(
    ...         f"Queue wait: {m.queue_wait_s:.3f}s, "
    ...         f"Service: {m.service_time_s:.3f}s, "
    ...         f"Deployment: {m.deployment}, "
    ...         f"Retries: {m.retries}"
    ...     )
    """

    queue_wait_s: float
    service_time_s: float
    end_to_end_s: float
    deployment: str | None = None
    retries: int = 0


@dataclass(slots=True)
class GenerationResult:
    """The typed result of a text-generation request.

    Returned by [generate][infermesh.LMClient.generate],
    [agenerate][infermesh.LMClient.agenerate], and contained in
    [BatchResult][infermesh.BatchResult] for ``*_batch`` methods.

    Parameters
    ----------
    model_id : str
        The provider-reported model identifier (e.g. ``"gpt-4o-mini"``).
    output_text : str
        The generated text.  For ``responses``-endpoint calls this is the
        concatenation of all ``output_text`` content blocks in the response.
    output_parsed : object or None, optional
        The structured result when ``response_format`` was supplied, or when
        ``parse_output=True`` was used with a Pydantic model or JSON-schema
        dict. The type
        matches the supplied ``response_format``.  When ``response_format`` is
        a Pydantic model class the output is validated via
        ``model_validate_json``; when it is a ``dict`` the parsed JSON is
        validated against the provided JSON Schema before being returned — a
        schema violation is treated as a parse failure.  ``None`` when parsing
        was not requested or failed (a warning is logged on parse failure).
    reasoning : str or None, optional
        Extended chain-of-thought reasoning text, when disclosed by the
        provider (e.g. certain Anthropic or OpenAI reasoning models).
    token_usage : TokenUsage or None, optional
        Token-count breakdown.  ``None`` if the provider did not include usage
        information in the response.
    finish_reason : str or None, optional
        The stop condition reported by the provider.  Common values are
        ``"stop"`` (normal completion), ``"length"`` (hit ``max_tokens``),
        and ``"tool_calls"`` (model requested a tool).
    tool_calls : list[ToolCall] or None, optional
        Structured tool calls emitted by the model.  ``None`` when the model
        completed without requesting any tool invocation.
    raw_response : object or None, optional
        The unmodified provider response object.  Useful for accessing
        provider-specific fields not surfaced by this dataclass.
    request_id : str or None, optional
        The provider-assigned request identifier (e.g. the ``id`` field from
        an OpenAI response).
    cost : float or None, optional
        Estimated cost in USD, when reported by LiteLLM's cost tracking.
    metrics : RequestMetrics or None, optional
        Queue-wait and service-time metadata for this request.

    Notes
    -----
    ``str(result)`` returns `output_text`, so a
    [GenerationResult][infermesh.GenerationResult] can be used directly wherever a string is
    expected.

    Examples
    --------
    Basic generation:

    >>> result = client.generate("Summarize backpropagation in one sentence.")
    >>> print(result.output_text)
    >>> print(f"Cost: ${result.cost:.6f}" if result.cost else "no cost info")

    Structured output with a Pydantic model:

    >>> from pydantic import BaseModel
    >>> class Summary(BaseModel):
    ...     headline: str
    ...     body: str
    >>> result = client.generate(
    ...     "Summarize the French Revolution.",
    ...     response_format=Summary,
    ... )
    >>> summary: Summary = result.output_parsed  # type: ignore[assignment]
    >>> print(summary.headline)
    """

    model_id: str
    output_text: str
    output_parsed: Any | None = None
    reasoning: str | None = None
    token_usage: TokenUsage | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw_response: Any | None = None
    request_id: str | None = None
    cost: float | None = None
    metrics: RequestMetrics | None = None

    def __str__(self) -> str:
        """Return the generated text.

        Returns
        -------
        str
            The value of `output_text`.
        """
        return self.output_text


@dataclass(slots=True)
class EmbeddingResult:
    """The typed result of an embedding request.

    Returned by [embed][infermesh.LMClient.embed] for single-string input and
    contained in [BatchResult][infermesh.BatchResult] for [embed_batch][infermesh.LMClient.embed_batch]
    calls.

    Parameters
    ----------
    model_id : str
        The provider-reported model identifier.
    embedding : list[float]
        The dense embedding vector.  Its length equals the model's output
        dimension (e.g. 1536 for ``text-embedding-3-small``).
    token_usage : TokenUsage or None, optional
        Token-count breakdown.  ``None`` if the provider did not report usage.
    raw_response : object or None, optional
        The unmodified provider response for advanced use cases.
    request_id : str or None, optional
        The provider-assigned request identifier.
    metrics : RequestMetrics or None, optional
        Queue-wait and service-time metadata for this request.

    Examples
    --------
    >>> import numpy as np
    >>> result = client.embed("The quick brown fox jumps over the lazy dog.")
    >>> vec = np.array(result.embedding)
    >>> print(f"Dim: {vec.shape[0]}, Norm: {np.linalg.norm(vec):.4f}")
    """

    model_id: str
    embedding: list[float]
    token_usage: TokenUsage | None = None
    raw_response: Any | None = None
    request_id: str | None = None
    metrics: RequestMetrics | None = None


@dataclass(slots=True)
class TranscriptionResult:
    """The typed result of an audio-transcription request.

    Returned by [transcribe][infermesh.LMClient.transcribe] and
    [atranscribe][infermesh.LMClient.atranscribe].

    Parameters
    ----------
    model_id : str
        The provider-reported model identifier (e.g. ``"whisper-1"``).
    text : str
        The transcribed text.
    duration_s : float or None, optional
        Duration of the audio clip in seconds, when reported by the provider.
    language : str or None, optional
        Detected or explicitly requested language code (e.g. ``"en"``), when
        reported by the provider.
    raw_response : object or None, optional
        The unmodified provider response for advanced use cases.
    request_id : str or None, optional
        The provider-assigned request identifier.
    metrics : RequestMetrics or None, optional
        Queue-wait and service-time metadata for this request.

    Examples
    --------
    >>> result = client.transcribe("interview.mp3")
    >>> print(result.text)
    >>> if result.language:
    ...     print(f"Detected language: {result.language}")
    """

    model_id: str
    text: str
    duration_s: float | None = None
    language: str | None = None
    raw_response: Any | None = None
    request_id: str | None = None
    metrics: RequestMetrics | None = None


T = TypeVar("T")


@dataclass(slots=True)
class BatchResult(Generic[T]):
    """A typed container for the results of a batch request.

    Returned by [generate_batch][infermesh.LMClient.generate_batch],
    [agenerate_batch][infermesh.LMClient.agenerate_batch],
    [embed_batch][infermesh.LMClient.embed_batch], and
    [aembed_batch][infermesh.LMClient.aembed_batch].

    When ``return_exceptions=True`` (the default), a failed item does **not**
    raise and discard the whole batch.  Instead, `results` contains
    ``None`` at that position and `errors` holds the exception.  Both
    lists are always the same length as the input, enabling index-based
    correlation.

    Parameters
    ----------
    results : list[T or None]
        One entry per input item.  Successful items have type ``T``; items
        where an exception occurred are ``None``
        (only when ``return_exceptions=True``).
    errors : list[BaseException or None] or None, optional
        One entry per input item when ``return_exceptions=True`` was used.
        ``None`` at positions where the request succeeded; the exception at
        positions where it failed.  This attribute is ``None`` itself when
        ``return_exceptions=False``.

    Notes
    -----
    - ``len(batch)`` always equals the number of input items.
    - Iterating over ``batch`` yields from `results` (may include
      ``None`` values on failure).
    - Index access (``batch[i]``) returns ``results[i]``.
    - To split successes from failures::

        successes = [r for r, e in zip(batch.results, batch.errors or []) if e is None]
        failures = [(i, e) for i, e in enumerate(batch.errors or []) if e is not None]

    Examples
    --------
    Process a batch tolerating partial failures (default behaviour):

    >>> prompts = ["Translate 'cat' to French", "bad-prompt", "What is 42?"]
    >>> batch = client.generate_batch(prompts)
    >>> for i, (result, error) in enumerate(zip(batch.results, batch.errors or [])):
    ...     if error:
    ...         print(f"[{i}] ERROR: {error}")
    ...     else:
    ...         print(f"[{i}] {result.output_text}")

    Opt in to raise-on-first-failure (legacy behaviour):

    >>> batch = client.generate_batch(prompts, return_exceptions=False)
    """

    results: list[T | None]
    errors: list[BaseException | None] | None = None

    def __iter__(self) -> Iterator[T | None]:
        """Iterate over batch items in input order.

        Yields
        ------
        T or None
            Each item from `results`.  ``None`` at positions where the
            corresponding request failed (when ``return_exceptions=True``).
        """
        return iter(self.results)

    def __getitem__(self, index: int) -> T | None:
        """Return the result at ``index``.

        Parameters
        ----------
        index : int
            Zero-based position in the batch.

        Returns
        -------
        T or None
            ``results[index]``.  ``None`` if the request at that position
            failed (when ``return_exceptions=True``).
        """
        return self.results[index]

    def __len__(self) -> int:
        """Return the number of items in the batch.

        Returns
        -------
        int
            Always equal to the number of input items, regardless of how many
            requests succeeded or failed.
        """
        return len(self.results)


GenerationBatchResult: TypeAlias = BatchResult[GenerationResult]
"""Type alias for a batch of generation results."""

EmbeddingBatchResult: TypeAlias = BatchResult[EmbeddingResult]
"""Type alias for a batch of embedding results."""

OnGenerationResult: TypeAlias = (
    Callable[[int, GenerationResult | None, BaseException | None], None] | None
)
"""Callback type for per-result notifications in batch generation.

Called as ``on_result(index, result, error)`` each time a single request
settles.  ``index`` is the position in ``input_batch``; exactly one of
``result`` or ``error`` is ``None``.
"""
