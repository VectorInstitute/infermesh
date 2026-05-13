"""Generate workflow veneer over the endpoint-neutral batch runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .engine import run_batch_workflow
from .mapping import _compute_mapping_fingerprint, _load_mapper
from .models import BatchWorkflowSummary, SourceItem
from .source import FileSourceStream
from .store import SqliteFileRunStore, StdoutRunStore

if TYPE_CHECKING:
    from infermesh.client import LMClient
    from infermesh.types import EndpointType


def _token_usage_to_dict(result: Any) -> dict[str, Any] | None:
    """Convert token usage to a JSON-serializable dict."""

    usage = getattr(result, "token_usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }


def _maybe_parse_json(value: str) -> Any:
    """Try to parse a JSON string."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _build_generation_record(
    orig_idx: int,
    result: Any,
    error: BaseException | None,
    *,
    parse_json: bool,
) -> dict[str, Any]:
    """Convert one generation result into its JSONL output shape."""

    if result is None:
        return {
            "_index": orig_idx,
            "output_text": None,
            "output_parsed": None,
            "token_usage": None,
            "request_id": None,
            "finish_reason": None,
            "error": str(error) if error else "unknown error",
        }
    return {
        "_index": orig_idx,
        "output_text": result.output_text,
        "output_parsed": _maybe_parse_json(result.output_text) if parse_json else None,
        "token_usage": _token_usage_to_dict(result),
        "request_id": result.request_id,
        "finish_reason": result.finish_reason,
        "error": None,
    }


def _build_generate_record_builder(
    *, parse_json: bool
) -> Callable[[SourceItem, Any, BaseException | None], dict[str, Any]]:
    def build_record(
        item: SourceItem,
        result: Any,
        error: BaseException | None,
    ) -> dict[str, Any]:
        record = _build_generation_record(
            item.output_index,
            result,
            error,
            parse_json=parse_json,
        )
        if item.metadata is not None:
            record["metadata"] = item.metadata
        return record

    return build_record


def run_generate_from_files(
    client: LMClient,
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_jsonl: str | None,
    checkpoint_dir: str | None,
    mapper_spec: str | None,
    resume: bool,
    endpoint: EndpointType,
    window_size: int,
    parse_json: bool,
    on_progress: Callable[[], Any] | None = None,
    on_status: Callable[[str], Any] | None = None,
) -> BatchWorkflowSummary:
    """Run the generate workflow from prompt, JSONL, or stdin inputs.

    Parameters
    ----------
    client
        Language-model client used for generation requests.
    prompt
        Optional single prompt. Mutually exclusive with ``input_jsonl`` in
        normal CLI use.
    input_jsonl
        Optional JSONL input path. If omitted, stdin is used.
    output_jsonl
        Optional JSONL output path. When omitted, rows are written to stdout and
        resume is not available.
    checkpoint_dir
        Optional directory for checkpoint files.
    mapper_spec
        Optional ``"module:function"`` mapper spec for JSONL source records.
    resume
        Whether to resume a previous file-backed run.
    endpoint
        Generation endpoint type to pass to ``LMClient.agenerate``.
    window_size
        Maximum number of generation requests in flight.
    parse_json
        Whether to populate ``output_parsed`` by parsing model output text.
    on_progress
        Optional callback invoked after each row is settled.
    on_status
        Optional callback for long-running setup and resume status messages.

    Returns
    -------
    BatchWorkflowSummary
        Counts for rows completed, errored, skipped, and the output path.
    """

    mapper = _load_mapper(mapper_spec) if mapper_spec else None
    mapping_fingerprint = _compute_mapping_fingerprint(
        mapper_spec=mapper_spec,
        mapper=mapper,
    )
    source = FileSourceStream.from_options(
        prompt=prompt,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        mapper=mapper,
    )
    # Store choice is the persistence boundary: stdout has no durable progress,
    # while file-backed output gets checkpoint and resume behavior.
    store = (
        StdoutRunStore()
        if output_jsonl is None
        else SqliteFileRunStore(
            output_jsonl=output_jsonl,
            checkpoint_dir=checkpoint_dir,
            on_status=on_status,
        )
    )

    async def operation(item: SourceItem) -> tuple[Any, BaseException | None]:
        result = await client.agenerate(item.mapped_input, endpoint=endpoint)
        return result, None

    return client._run_sync(
        run_batch_workflow(
            source=source,
            store=store,
            operation=operation,
            record_builder=_build_generate_record_builder(parse_json=parse_json),
            resume=resume,
            mapping_fingerprint=mapping_fingerprint,
            window_size=window_size,
            on_progress=on_progress,
        )
    )


__all__ = ["run_generate_from_files"]
