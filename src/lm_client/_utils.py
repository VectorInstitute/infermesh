"""Internal utility helpers."""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import jsonschema
from pydantic import BaseModel

from lm_client.types import (
    ChatInput,
    ChatMessage,
    EmbeddingResult,
    GenerateInput,
    GenerationResult,
    RequestMetrics,
    ResponsesInput,
    TokenUsage,
    ToolCall,
    TranscriptionInput,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Used only when vLLM embeds thinking tokens inline in `content` rather than
# exposing them via the structured `reasoning_content` message field.
_LEADING_THINK_RE = re.compile(r"<think>(.*?)</think>[ \t\r\n]*", re.DOTALL)


def validate_endpoint(endpoint: str) -> None:
    """Validate a generation endpoint."""

    if endpoint not in {"text_completion", "chat_completion", "responses"}:
        raise ValueError(
            "Expected ``endpoint`` to be one of "
            "['text_completion', 'chat_completion', 'responses']"
        )


def parse_model_output_with_format(
    output_text: str,
    response_format: type[BaseModel] | dict[str, Any] | None,
) -> Any | None:
    """Parse model output using a Pydantic model or JSON schema.

    When ``response_format`` is a Pydantic model class, the output is validated
    via ``model_validate_json``.  When it is a ``dict``, the output is parsed as
    JSON and then validated against the provided JSON Schema; a schema violation
    is treated as a parse failure (returns ``None`` and logs a warning).
    """

    if response_format is None:
        return None
    try:
        if hasattr(response_format, "model_validate_json"):
            return response_format.model_validate_json(output_text)
        parsed = json.loads(output_text)
        jsonschema.validate(parsed, response_format)
        return parsed
    except Exception as exc:
        logger.warning("Failed to parse output with the requested schema: %s", exc)
        return None


def extract_token_usage(usage_obj: Any) -> TokenUsage | None:
    """Extract provider token-usage information."""

    if usage_obj is None:
        return None
    if hasattr(usage_obj, "input_tokens") and hasattr(usage_obj, "output_tokens"):
        return TokenUsage(
            prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
            reasoning_tokens=_extract_reasoning_tokens(
                getattr(usage_obj, "output_tokens_details", None)
            ),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )
    if hasattr(usage_obj, "prompt_tokens") or hasattr(usage_obj, "completion_tokens"):
        return TokenUsage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            reasoning_tokens=_extract_reasoning_tokens(
                getattr(usage_obj, "completion_tokens_details", None)
            ),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )
    if isinstance(usage_obj, dict):
        if "input_tokens" in usage_obj or "output_tokens" in usage_obj:
            return TokenUsage(
                prompt_tokens=int(usage_obj.get("input_tokens", 0) or 0),
                completion_tokens=int(usage_obj.get("output_tokens", 0) or 0),
                reasoning_tokens=_extract_reasoning_tokens(
                    usage_obj.get("output_tokens_details")
                ),
                total_tokens=int(usage_obj.get("total_tokens", 0) or 0),
            )
        return TokenUsage(
            prompt_tokens=int(usage_obj.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_obj.get("completion_tokens", 0) or 0),
            reasoning_tokens=_extract_reasoning_tokens(
                usage_obj.get("completion_tokens_details")
            ),
            total_tokens=int(usage_obj.get("total_tokens", 0) or 0),
        )
    return None


def estimate_token_count(
    litellm_module: Any,
    model: str,
    input_data: GenerateInput | list[str],
    *,
    endpoint: str,
    max_tokens: int = 0,
) -> int:
    """Estimate tokens for rate limiting.

    Notes
    -----
    When LiteLLM token counting cannot handle the input shape, this helper falls
    back to a conservative response-only estimate.
    """

    try:
        if endpoint == "responses":
            payload = (
                input_data if isinstance(input_data, dict) else {"input": input_data}
            )
            messages = []
            instructions = payload.get("instructions")
            if instructions:
                messages.append({"role": "system", "content": instructions})
            messages.append({"role": "user", "content": payload.get("input", "")})
        elif isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif _is_chat_input(input_data):
            messages = cast(list[ChatMessage], input_data)
        elif isinstance(input_data, list) and all(
            isinstance(item, str) for item in input_data
        ):
            return sum(
                estimate_token_count(
                    litellm_module,
                    model,
                    item,
                    endpoint="chat_completion",
                    max_tokens=max_tokens,
                )
                for item in input_data
            )
        else:
            messages = [{"role": "user", "content": json.dumps(input_data)}]

        input_tokens = int(
            litellm_module.token_counter(
                model=model,
                messages=messages,
                count_response_tokens=False,
                default_token_count=0,
            )
            or 0
        )
        return input_tokens + max(0, max_tokens)
    except Exception:
        return max(0, max_tokens)


def normalize_generate_input(
    input_data: GenerateInput, endpoint: str
) -> str | ChatInput | ResponsesInput:
    """Normalize a generation input for a single request."""

    validate_endpoint(endpoint)
    if endpoint == "responses":
        if isinstance(input_data, str):
            return {"input": input_data}
        if isinstance(input_data, dict) and "input" in input_data:
            return input_data
        raise ValueError(
            "Responses requests require a string or a dictionary with an 'input' key."
        )

    if endpoint == "text_completion":
        if isinstance(input_data, str):
            return input_data
        raise ValueError("Text completion expects a string input.")

    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if _is_chat_input(input_data):
        return input_data
    raise ValueError(
        "Chat completion expects either a string or a list of message dictionaries."
    )


def normalize_batch_input(input_batch: Sequence[GenerateInput]) -> list[GenerateInput]:
    """Normalize batch input to a concrete list."""

    return list(input_batch)


def normalize_embedding_input(input_data: str | Sequence[str]) -> list[str]:
    """Normalize embedding inputs."""

    if isinstance(input_data, str):
        return [input_data]
    return [str(item) for item in input_data]


def normalize_transcription_input(
    input_data: TranscriptionInput, *, max_bytes: int | None = None
) -> bytes | tuple[str, bytes]:
    """Normalize transcription inputs to bytes.

    Parameters
    ----------
    input_data : str or bytes or file-like
        Audio to normalize.
    max_bytes : int or None, optional
        Maximum allowed size in bytes.  Raises ``ValueError`` when exceeded.
        Also raises ``ValueError`` if ``input_data`` is a path that does not
        point to a regular file.
    """

    if isinstance(input_data, bytes):
        if max_bytes is not None and len(input_data) > max_bytes:
            raise ValueError(
                f"Audio input size {len(input_data)} bytes exceeds limit "
                f"{max_bytes} bytes."
            )
        return input_data
    if isinstance(input_data, str):
        input_data = Path(input_data)
    if isinstance(input_data, Path):
        try:
            with open(input_data, "rb") as file_handle:
                if max_bytes is not None:
                    size = os.fstat(file_handle.fileno()).st_size
                    if size > max_bytes:
                        raise ValueError(
                            f"Audio input size {size} bytes exceeds limit {max_bytes} bytes."
                        )
                return str(input_data), file_handle.read()
        except OSError as exc:
            raise ValueError(
                f"Transcription input must be a regular file: {input_data!r}"
            ) from exc
    name = getattr(input_data, "name", "audio")
    if max_bytes is not None:
        data = input_data.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(
                f"Audio input size is at least {len(data)} bytes and exceeds "
                f"limit {max_bytes} bytes."
            )
        return str(name), data
    return str(name), input_data.read()


def build_generation_result(
    response: Any,
    *,
    endpoint: str,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    metrics: RequestMetrics | None,
) -> GenerationResult:
    """Build a public generation result from a provider response."""

    if endpoint == "responses":
        return _parse_responses_result(
            response,
            response_format=response_format,
            parse_output=parse_output,
            metrics=metrics,
        )
    return _parse_completion_result(
        response,
        response_format=response_format,
        parse_output=parse_output,
        metrics=metrics,
    )


def build_embedding_results(
    response: Any, *, metrics: RequestMetrics | None
) -> list[EmbeddingResult]:
    """Build embedding results from a provider response."""

    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data", [])
    usage = extract_token_usage(
        getattr(response, "usage", None) or response.get("usage")
    )
    request_id = _get_value(response, "id")
    model_id = _get_value(response, "model") or ""
    results: list[EmbeddingResult] = []
    for item in data or []:
        embedding = _get_value(item, "embedding") or []
        results.append(
            EmbeddingResult(
                model_id=model_id,
                embedding=list(embedding),
                token_usage=usage,
                raw_response=response,
                request_id=request_id,
                metrics=metrics,
            )
        )
    return results


def build_transcription_result(
    response: Any, *, metrics: RequestMetrics | None
) -> TranscriptionResult:
    """Build a transcription result from a provider response."""

    request_id = _get_value(response, "id")
    model_id = _get_value(response, "model") or ""
    text = _get_value(response, "text") or ""
    duration = _get_value(response, "duration")
    language = _get_value(response, "language")
    return TranscriptionResult(
        model_id=model_id,
        text=text,
        duration_s=float(duration) if duration is not None else None,
        language=language,
        raw_response=response,
        request_id=request_id,
        metrics=metrics,
    )


def extract_response_headers(response: Any) -> dict[str, Any] | None:
    """Extract provider response headers when present."""

    for candidate in (
        getattr(response, "_response_headers", None),
        getattr(response, "headers", None),
    ):
        if candidate:
            return dict(candidate)
    if isinstance(response, dict):
        headers = response.get("_response_headers") or response.get("headers")
        if headers:
            return dict(headers)
    return None


def extract_deployment_label(response: Any) -> str | None:
    """Extract the chosen deployment label if LiteLLM exposes one."""

    hidden = getattr(response, "_hidden_params", None)
    if hidden is None and isinstance(response, dict):
        hidden = response.get("_hidden_params")
    if isinstance(hidden, dict):
        for key in ("deployment", "api_base", "model_id", "model"):
            value = hidden.get(key)
            if value:
                return str(value)
    return None


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a percentile from ``values``."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile_value / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def batched_cycle[T](items: Sequence[T], count: int) -> list[T]:
    """Repeat ``items`` until ``count`` elements are available."""

    if not items:
        return []
    return list(itertools.islice(itertools.cycle(items), count))


def _extract_reasoning_tokens(token_details: Any) -> int | None:
    """Extract reasoning token information."""

    if token_details is None:
        return None
    if isinstance(token_details, dict):
        value = token_details.get("reasoning_tokens")
    else:
        value = getattr(token_details, "reasoning_tokens", None)
    if value is None:
        return None
    return int(value)


def _extract_choice_content(choice: Any) -> tuple[Any, str | None]:
    """Return ``(content, reasoning_content)`` from a single completion choice."""
    msg = _get_value(choice, "message")
    if msg is not None:
        # or None coerces "" to None so callers can use `is not None` guards.
        return _get_value(msg, "content"), _get_value(msg, "reasoning_content") or None
    return _get_value(choice, "text"), None


def _parse_completion_result(
    response: Any,
    *,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    metrics: RequestMetrics | None,
) -> GenerationResult:
    """Parse a chat or text completion response."""

    choices = _get_value(response, "choices") or []
    content_texts: list[str] = []
    reasoning_texts: list[str] = []

    for choice in choices:
        content, rc = _extract_choice_content(choice)
        if isinstance(content, list):
            content_texts.append(
                "".join(
                    str(
                        part.get("text")
                        if isinstance(part, dict)
                        else getattr(part, "text", "")
                    )
                    for part in content
                )
            )
        elif content is not None:
            content_texts.append(str(content))
        if rc is not None:
            reasoning_texts.append(rc)

    output_text = _join_texts(content_texts)

    if reasoning_texts:
        joined = "\n\n".join(r.strip() for r in reasoning_texts if r.strip())
        reasoning = joined if joined else None
    else:
        output_text, reasoning = _extract_thinking_tokens(output_text)

    output_parsed = (
        parse_model_output_with_format(output_text, response_format)
        if parse_output
        else None
    )
    usage = extract_token_usage(_get_value(response, "usage"))
    finish_reason = None
    if choices:
        finish_reason = _get_value(choices[-1], "finish_reason")
    tool_calls = _extract_completion_tool_calls(choices)
    return GenerationResult(
        model_id=str(_get_value(response, "model") or ""),
        output_text=output_text,
        output_parsed=output_parsed,
        reasoning=reasoning,
        token_usage=usage,
        finish_reason=finish_reason,
        tool_calls=tool_calls or None,
        raw_response=response,
        request_id=_get_value(response, "id"),
        metrics=metrics,
    )


def _parse_responses_result(
    response: Any,
    *,
    response_format: type[BaseModel] | dict[str, Any] | None,
    parse_output: bool,
    metrics: RequestMetrics | None,
) -> GenerationResult:
    """Parse an OpenAI responses-style result."""

    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    outputs = _get_value(response, "output") or []
    for output in outputs:
        output_type = _get_value(output, "type")
        if output_type == "message":
            contents = _get_value(output, "content") or []
            for content in contents:
                if _get_value(content, "type") == "output_text":
                    text = _get_value(content, "text")
                    if text:
                        texts.append(str(text))
        elif output_type == "function_call":
            tool_calls.append(
                ToolCall(
                    id=str(_get_value(output, "id") or ""),
                    name=str(_get_value(output, "name") or ""),
                    arguments=_get_value(output, "arguments"),
                )
            )
    output_text = _join_texts(texts)
    output_parsed = (
        parse_model_output_with_format(output_text, response_format)
        if parse_output
        else None
    )
    usage = extract_token_usage(_get_value(response, "usage"))
    return GenerationResult(
        model_id=str(_get_value(response, "model") or ""),
        output_text=output_text,
        output_parsed=output_parsed,
        tool_calls=tool_calls or None,
        token_usage=usage,
        raw_response=response,
        request_id=_get_value(response, "id"),
        metrics=metrics,
    )


def _extract_completion_tool_calls(choices: list[Any]) -> list[ToolCall]:
    """Extract tool calls from chat-completion choices."""

    tool_calls: list[ToolCall] = []
    for choice in choices:
        message = _get_value(choice, "message")
        for tool_call in _get_value(message, "tool_calls") or []:
            function = _get_value(tool_call, "function")
            tool_calls.append(
                ToolCall(
                    id=str(_get_value(tool_call, "id") or ""),
                    name=str(
                        _get_value(function, "name")
                        or _get_value(tool_call, "name")
                        or ""
                    ),
                    arguments=(
                        _get_value(function, "arguments")
                        if function is not None
                        else _get_value(tool_call, "arguments")
                    ),
                )
            )
    return tool_calls


def _join_texts(texts: list[str]) -> str:
    """Return a single string from a list of text fragments."""
    return texts[0] if len(texts) == 1 else "".join(texts)


def _extract_thinking_tokens(text: str) -> tuple[str, str | None]:
    """Strip leading ``<think>…</think>`` blocks from model output.

    Only removes blocks that appear at the very start of the string (after any
    leading whitespace).  A ``<think>`` tag that appears mid-response is left
    in place and treated as literal content, avoiding false-positive stripping
    when a model legitimately outputs XML-like tags.

    Parameters
    ----------
    text : str
        Raw model output, possibly containing leading thinking tokens.

    Returns
    -------
    tuple[str, str | None]
        ``(clean_text, reasoning)`` where *reasoning* is the concatenated
        content of all removed leading blocks, or ``None`` when none were found.
    """
    if "<think>" not in text:
        return text, None
    remaining = text.lstrip()
    reasoning_parts: list[str] = []
    while remaining.startswith("<think>"):
        m = _LEADING_THINK_RE.match(remaining)
        if m is None:
            break
        reasoning_parts.append(m.group(1).strip())
        remaining = remaining[m.end() :]

    if not reasoning_parts:
        return text, None
    return remaining, "\n\n".join(reasoning_parts)


def _is_chat_input(input_data: Any) -> bool:
    """Return whether ``input_data`` is a chat message list."""

    return isinstance(input_data, list) and all(
        isinstance(item, dict) and "role" in item for item in input_data
    )


def _get_value(obj: Any, key: str) -> Any:
    """Get a field from either an object or a dict."""

    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
