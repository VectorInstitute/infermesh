"""Private support helpers for ``infermesh.cli``."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, overload

from infermesh.client import LMClient
from infermesh.types import DeploymentConfig, EndpointType


@dataclass
class ClientConfig:
    """Typed configuration for building an ``LMClient`` from CLI arguments."""

    model: str | None
    api_bases: list[str] = field(default_factory=list)
    deployments_toml: str | None = None
    routing_strategy: str = "simple-shuffle"
    endpoint: EndpointType = "chat_completion"
    max_parallel_requests: int | None = None
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None
    max_request_burst: int | None = None
    max_token_burst: int | None = None
    max_retries: int = 3


def _client_config_from_args(args: argparse.Namespace) -> ClientConfig:
    """Convert parsed CLI arguments to a typed ``ClientConfig``."""

    return ClientConfig(
        model=args.model,
        api_bases=args.api_bases or [],
        deployments_toml=getattr(args, "deployments_toml", None),
        routing_strategy=getattr(args, "routing_strategy", "simple-shuffle"),
        endpoint=cast(EndpointType, getattr(args, "endpoint", "chat_completion")),
        max_parallel_requests=getattr(args, "max_parallel_requests", None),
        rpm=getattr(args, "rpm", None),
        tpm=getattr(args, "tpm", None),
        rpd=getattr(args, "rpd", None),
        tpd=getattr(args, "tpd", None),
        max_request_burst=getattr(args, "max_request_burst", None),
        max_token_burst=getattr(args, "max_token_burst", None),
        max_retries=getattr(args, "max_retries", 3),
    )


@overload
def _build_client(
    config: ClientConfig,
    *,
    max_parallel_requests: int | None = None,
) -> LMClient: ...


@overload
def _build_client[TClient](
    config: ClientConfig,
    *,
    max_parallel_requests: int | None = None,
    client_cls: Callable[..., TClient],
) -> TClient: ...


def _build_client(
    config: ClientConfig,
    *,
    max_parallel_requests: int | None = None,
    client_cls: Callable[..., object] | None = None,
) -> object:
    """Build an ``LMClient`` from a ``ClientConfig``."""

    builder: Callable[..., object] = LMClient if client_cls is None else client_cls
    deployments: dict[str, DeploymentConfig | dict[str, Any]] | None = None
    if config.deployments_toml:
        if not config.model:
            raise ValueError("--model is required when --deployments-toml is used.")
        with open(config.deployments_toml, "rb") as file_handle:
            loaded = tomllib.load(file_handle)
        _validate_cli_deployments_toml(
            loaded=loaded,
            deployments_toml_path=config.deployments_toml,
        )
        deployments = {
            name: DeploymentConfig(
                model=str(deployment_cfg["model"]),
                api_base=str(deployment_cfg["api_base"]),
                extra_kwargs=dict(deployment_cfg.get("extra_kwargs", {}) or {}),
            )
            for name, deployment_cfg in loaded["deployments"].items()
        }
    elif len(config.api_bases) > 1:
        if not config.model:
            raise ValueError("Repeated --api-base routing requires --model.")
        deployments = {
            f"replica-{index}": DeploymentConfig(
                model=config.model,
                api_base=api_base,
            )
            for index, api_base in enumerate(config.api_bases, start=1)
        }

    api_base = config.api_bases[0] if len(config.api_bases) == 1 else None
    return builder(
        model=config.model,
        api_base=api_base,
        deployments=deployments,
        endpoint=config.endpoint,
        max_parallel_requests=(
            max_parallel_requests
            if max_parallel_requests is not None
            else config.max_parallel_requests
        ),
        rpm=config.rpm,
        tpm=config.tpm,
        rpd=config.rpd,
        tpd=config.tpd,
        max_request_burst=config.max_request_burst,
        max_token_burst=config.max_token_burst,
        max_retries=config.max_retries,
        routing_strategy=config.routing_strategy,
        router_kwargs={},
    )


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add common connection arguments."""

    parser.add_argument(
        "--model",
        required=False,
        help=(
            "LiteLLM model string, for example openai/gpt-4.1-mini, "
            "anthropic/claude-3-5-sonnet-20241022, or hosted_vllm/..."
        ),
    )
    parser.add_argument(
        "--api-base",
        dest="api_bases",
        action="append",
        help=(
            "Base URL for single-endpoint mode. Repeat this flag to spread "
            "requests across multiple replicas of the same model."
        ),
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        help=(
            "Path to a .env file containing provider secrets "
            "(e.g. OPENAI_API_KEY=sk-...). "
            "Env vars already set in the shell take precedence."
        ),
    )
    parser.add_argument(
        "--deployments-toml",
        help=(
            "Path to a TOML file describing router deployments. "
            "Keep secrets out of this file; export provider env vars or "
            "load them with --env-file instead."
        ),
    )
    parser.add_argument(
        "--routing-strategy",
        default="simple-shuffle",
        help="LiteLLM router strategy for multi-replica mode. Default: simple-shuffle.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        help="Client-side cap on in-flight requests.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        help="Client-side requests-per-minute limit.",
    )
    parser.add_argument(
        "--tpm",
        type=int,
        help="Client-side tokens-per-minute limit.",
    )
    parser.add_argument(
        "--rpd",
        type=int,
        help="Client-side requests-per-day limit.",
    )
    parser.add_argument(
        "--tpd",
        type=int,
        help="Client-side tokens-per-day limit.",
    )
    parser.add_argument(
        "--max-request-burst",
        type=int,
        help="Burst capacity for the requests-per-minute bucket.",
    )
    parser.add_argument(
        "--max-token-burst",
        type=int,
        help="Burst capacity for the tokens-per-minute bucket.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help=(
            "Maximum number of automatic retries for transient provider errors "
            "(429, 503, 500, network failures, timeouts). Default: 3. Set to 0 "
            "to disable retries."
        ),
    )


def _validate_cli_deployments_toml(
    *,
    loaded: dict[str, Any],
    deployments_toml_path: str,
) -> None:
    """Reject plaintext secrets in CLI-loaded router deployment config."""

    for name, config in loaded["deployments"].items():
        forbidden_path = _find_forbidden_secret_path(
            config,
            path=f"deployments.{name}",
        )
        if forbidden_path is None:
            continue
        raise ValueError(
            f"Deployment {name!r} in {deployments_toml_path!r} contains a "
            f"plaintext secret at {forbidden_path!r}. CLI deployment configs "
            "must not contain 'api_key'. Export provider environment variables "
            "or load them with --env-file instead."
        )


def _find_forbidden_secret_path(
    value: Any,
    *,
    path: str,
) -> str | None:
    """Return the first nested ``api_key`` path found in ``value``."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "api_key":
                return child_path
            nested = _find_forbidden_secret_path(child, path=child_path)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested = _find_forbidden_secret_path(item, path=f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _load_generation_rows(
    *,
    prompt: str | None,
    input_jsonl: str | None,
) -> list[dict[str, Any]]:
    """Load generation inputs from CLI arguments."""

    if prompt is not None:
        return [{"prompt": prompt}]
    return _load_jsonl_rows(input_jsonl)


def _load_embed_texts(*, text: str | None, input_jsonl: str | None) -> list[str]:
    """Load embedding input texts."""

    if text is not None:
        return [text]
    rows = _load_jsonl_rows(input_jsonl)
    texts = [str(row["text"]) for row in rows if "text" in row]
    if not texts:
        raise ValueError("Embedding input rows require a 'text' field.")
    return texts


def _load_transcription_paths(
    *,
    path: str | None,
    input_jsonl: str | None,
) -> list[Path]:
    """Load transcription paths."""

    if path is not None:
        return [Path(path)]
    rows = _load_jsonl_rows(input_jsonl)
    paths = [Path(row["path"]) for row in rows if "path" in row]
    if not paths:
        raise ValueError("Transcription input rows require a 'path' field.")
    return paths


def _load_jsonl_rows(path: str | None) -> list[dict[str, Any]]:
    """Load JSONL rows from a file or stdin."""

    if path is None:
        return [json.loads(line) for line in sys.stdin if line.strip()]
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(rows: list[dict[str, Any]], output_path: str | None) -> None:
    """Write JSONL rows to a file or stdout."""

    payload = "\n".join(json.dumps(row) for row in rows)
    if output_path is None:
        if payload:
            sys.stdout.write(payload + "\n")
        return
    Path(output_path).write_text(payload + ("\n" if payload else ""), encoding="utf-8")


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
