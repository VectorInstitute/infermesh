"""Command-line interface for ``lm_client``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from lm_client._cli_bench import (
    DEFAULT_EMBED_BATCH_SIZES,
    DEFAULT_SWEEP,
    _build_concurrency_sweep,
    _build_embed_batch_sweep,
    _resolve_sweep_levels,
    _run_benchmark,
    _run_embed_batch_benchmark,
    _write_embed_summary,
    _write_generate_summary,
)
from lm_client._cli_support import (
    ClientConfig,
    _add_connection_args,
    _load_embed_texts,
    _load_generation_rows,
    _load_transcription_paths,
    _maybe_parse_json,
    _token_usage_to_dict,
    _write_jsonl,
)
from lm_client._cli_support import (
    _build_client as _support_build_client,
)
from lm_client._cli_support import (
    _client_config_from_args as _support_client_config_from_args,
)
from lm_client._utils import batched_cycle
from lm_client.client import LMClient
from lm_client.types import EndpointType

_client_config_from_args = _support_client_config_from_args


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    env_file = getattr(args, "env_file", None)
    if env_file is not None:
        load_dotenv(env_file, override=False)
    return int(args.handler(args))


def _build_client(
    config: ClientConfig,
    *,
    max_parallel_requests: int | None = None,
) -> LMClient:
    """Build an ``LMClient`` instance for CLI commands."""

    return _support_build_client(
        config,
        max_parallel_requests=max_parallel_requests,
        client_cls=LMClient,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="lm-client",
        description=(
            "Run LLM generation, embeddings, transcription, and client-side "
            "benchmarks from scripts, notebooks, and JSONL batches."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", title="commands")
    _add_generate_parser(subparsers)
    _add_embed_parser(subparsers)
    _add_transcribe_parser(subparsers)
    _add_bench_parser(subparsers)
    return parser


def _add_generate_parser(subparsers: Any) -> None:
    """Register the ``generate`` subcommand."""

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text from one prompt or a JSONL batch.",
        description=(
            "Generate text from a single prompt or a JSONL batch. Input rows "
            "may contain 'prompt', 'messages', or 'responses_input'."
        ),
    )
    _add_connection_args(generate_parser)
    generate_parser.add_argument("--prompt", help="Single prompt to run.")
    generate_parser.add_argument(
        "--input-jsonl",
        help=(
            "Path to a JSONL file. Each line must contain 'prompt', "
            "'messages', or 'responses_input'."
        ),
    )
    generate_parser.add_argument(
        "--output-jsonl",
        help="Write one result object per input row.",
    )
    generate_parser.add_argument(
        "--endpoint",
        default="chat_completion",
        help="Generation endpoint: chat_completion, text_completion, or responses.",
    )
    generate_parser.add_argument(
        "--parse-json",
        action="store_true",
        help="Try to JSON-decode each output_text into output_parsed.",
    )
    generate_parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run by reading completed _index values from "
            "--output-jsonl and appending the remaining rows."
        ),
    )
    generate_parser.set_defaults(handler=_handle_generate)


def _add_embed_parser(subparsers: Any) -> None:
    """Register the ``embed`` subcommand."""

    embed_parser = subparsers.add_parser(
        "embed",
        help="Create embeddings for one text or a JSONL batch.",
        description=(
            "Create embeddings for a single text or a JSONL file whose rows "
            "contain a 'text' field."
        ),
    )
    _add_connection_args(embed_parser)
    embed_parser.add_argument("--text", help="Single text to embed.")
    embed_parser.add_argument(
        "--input-jsonl",
        help="Path to a JSONL file. Each line must contain a 'text' field.",
    )
    embed_parser.add_argument("--output-jsonl", help="Write one result per input row.")
    embed_parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Omit raw embedding arrays from stdout or JSONL output.",
    )
    embed_parser.set_defaults(handler=_handle_embed)


def _add_transcribe_parser(subparsers: Any) -> None:
    """Register the ``transcribe`` subcommand."""

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe one audio file or a JSONL batch of paths.",
        description=(
            "Transcribe one audio file or a JSONL file whose rows contain a "
            "'path' field."
        ),
    )
    _add_connection_args(transcribe_parser)
    transcribe_parser.add_argument(
        "path",
        nargs="?",
        help="Path to a single local audio file to transcribe.",
    )
    transcribe_parser.add_argument(
        "--input-jsonl",
        help="Path to a JSONL file. Each line must contain a 'path' field.",
    )
    transcribe_parser.add_argument(
        "--output-jsonl",
        help="Write one result object per input file.",
    )
    transcribe_parser.set_defaults(handler=_handle_transcribe)


def _add_bench_parser(subparsers: Any) -> None:
    """Register the ``bench`` command tree."""

    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark client-side throughput for generation or embeddings.",
        description=(
            "Measure client-side throughput across concurrency or batch-size "
            "sweeps. This is a client benchmark, not a server capacity test."
        ),
    )
    bench_subparsers = bench_parser.add_subparsers(
        dest="bench_command",
        title="bench commands",
    )
    _add_bench_generate_parser(bench_subparsers)
    _add_bench_embed_parser(bench_subparsers)


def _add_bench_generate_parser(bench_subparsers: Any) -> None:
    """Register ``bench generate``."""

    bench_generate_parser = bench_subparsers.add_parser(
        "generate",
        help="Benchmark generation throughput across concurrency levels.",
        description=(
            "Benchmark generation throughput for one prompt or a JSONL batch "
            "while sweeping client-side concurrency."
        ),
    )
    _add_connection_args(bench_generate_parser)
    bench_generate_parser.add_argument("--prompt", help="Single prompt to benchmark.")
    bench_generate_parser.add_argument(
        "--input-jsonl",
        help=(
            "Path to a JSONL file. Each line must contain 'prompt', "
            "'messages', or 'responses_input'."
        ),
    )
    bench_generate_parser.add_argument(
        "--endpoint",
        default="chat_completion",
        help="Generation endpoint: chat_completion, text_completion, or responses.",
    )
    bench_generate_parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests before measured runs begin.",
    )
    bench_generate_parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Requests per concurrency level when --duration is not used.",
    )
    bench_generate_parser.add_argument(
        "--output-json",
        help="Optional path for a JSON benchmark summary.",
    )
    concurrency_group = bench_generate_parser.add_mutually_exclusive_group()
    concurrency_group.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help="Run at a single concurrency level instead of sweeping.",
    )
    concurrency_group.add_argument(
        "--max-concurrency",
        type=int,
        metavar="N",
        help="Sweep concurrency levels [1, 2, 4, …, N] instead of the default sweep.",
    )
    bench_generate_parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Run for this many seconds per level instead of a fixed request count.",
    )
    bench_generate_parser.set_defaults(handler=_handle_bench_generate)


def _add_bench_embed_parser(bench_subparsers: Any) -> None:
    """Register ``bench embed``."""

    bench_embed_parser = bench_subparsers.add_parser(
        "embed",
        help="Benchmark embedding throughput across batch sizes.",
        description=(
            "Benchmark embed_batch throughput for one text or a JSONL input "
            "while sweeping request batch size."
        ),
    )
    _add_connection_args(bench_embed_parser)
    bench_embed_parser.add_argument(
        "--text",
        help="Single text that will be cycled to fill benchmark batches.",
    )
    bench_embed_parser.add_argument(
        "--input-jsonl",
        help="Path to a JSONL file. Each line must contain a 'text' field.",
    )
    bench_embed_parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup embed_batch calls before measured runs begin.",
    )
    bench_embed_parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of embed_batch API calls per sweep level.",
    )
    bench_embed_parser.add_argument(
        "--output-json",
        help="Optional path for a JSON benchmark summary.",
    )
    batch_group = bench_embed_parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="Run at a single batch size instead of sweeping.",
    )
    batch_group.add_argument(
        "--max-batch-size",
        type=int,
        metavar="N",
        help="Sweep batch sizes up to N instead of the default sweep.",
    )
    bench_embed_parser.set_defaults(handler=_handle_bench_embed)


def _load_completed_generation_indices(output_jsonl: str) -> set[int]:
    """Read completed ``_index`` values from an existing output JSONL file."""

    completed: set[int] = set()
    if not Path(output_jsonl).exists():
        return completed

    with open(output_jsonl, encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            try:
                index = json.loads(stripped_line).get("_index")
            except json.JSONDecodeError:
                continue
            if isinstance(index, int):
                completed.add(index)
    return completed


def _build_pending_generation_inputs(
    all_rows: list[dict[str, Any]],
    done: set[int],
) -> list[tuple[int, Any]]:
    """Return the generation inputs that still need to run."""

    pending: list[tuple[int, Any]] = []
    for orig_idx, row in enumerate(all_rows):
        if orig_idx in done:
            continue
        input_data = (
            row.get("responses_input") or row.get("messages") or row.get("prompt")
        )
        if input_data is None:
            raise ValueError(
                "Generation rows require 'prompt', 'messages', or 'responses_input'."
            )
        pending.append((orig_idx, input_data))
    return pending


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


def _write_generation_batch_to_file(
    client: LMClient,
    *,
    inputs: list[Any],
    pending: list[tuple[int, Any]],
    output_jsonl: str,
    resume: bool,
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Any,
) -> None:
    """Stream generation results to disk as each request completes."""

    file_mode = "a" if resume else "w"
    with open(output_jsonl, file_mode, encoding="utf-8") as out_file:

        def on_result(
            batch_idx: int,
            result: Any,
            error: BaseException | None,
        ) -> None:
            orig_idx = pending[batch_idx][0]
            record = _build_generation_record(
                orig_idx,
                result,
                error,
                parse_json=parse_json,
            )
            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

        client.generate_batch(
            inputs,
            endpoint=endpoint,
            on_progress=on_progress,
            on_result=on_result,
        )


def _write_generation_batch_to_stdout(
    client: LMClient,
    *,
    inputs: list[Any],
    pending: list[tuple[int, Any]],
    endpoint: EndpointType,
    parse_json: bool,
    on_progress: Any,
) -> None:
    """Collect generation results and write them to stdout together."""

    batch = client.generate_batch(
        inputs,
        endpoint=endpoint,
        on_progress=on_progress,
    )
    records = []
    for batch_idx, result in enumerate(batch):
        error = batch.errors[batch_idx] if batch.errors else None
        records.append(
            _build_generation_record(
                pending[batch_idx][0],
                result,
                error,
                parse_json=parse_json,
            )
        )
    _write_jsonl(records, None)


def _run_generate_command(
    client: LMClient,
    args: argparse.Namespace,
    *,
    endpoint: EndpointType,
) -> int:
    """Execute the core generation workflow after client construction."""

    resume = bool(getattr(args, "resume", False))
    all_rows = _load_generation_rows(prompt=args.prompt, input_jsonl=args.input_jsonl)
    done = (
        _load_completed_generation_indices(args.output_jsonl)
        if resume and args.output_jsonl
        else set()
    )
    if done:
        sys.stderr.write(f"Resuming: skipping {len(done)} already-completed row(s).\n")

    pending = _build_pending_generation_inputs(all_rows, done)
    if not pending:
        sys.stderr.write("Nothing to do — all rows already completed.\n")
        return 0

    inputs = [input_data for _, input_data in pending]
    parse_json = bool(getattr(args, "parse_json", False))
    with tqdm(
        total=len(pending),
        desc="Generating",
        unit="req",
        disable=(len(pending) <= 1),
        file=sys.stderr,
    ) as progress_bar:

        def on_progress(_done: int, _total: int) -> None:
            progress_bar.update(1)

        if args.output_jsonl:
            _write_generation_batch_to_file(
                client,
                inputs=inputs,
                pending=pending,
                output_jsonl=args.output_jsonl,
                resume=resume,
                endpoint=endpoint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
        else:
            _write_generation_batch_to_stdout(
                client,
                inputs=inputs,
                pending=pending,
                endpoint=endpoint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
    return 0


def _handle_generate(args: argparse.Namespace) -> int:
    """Handle the ``generate`` subcommand."""

    resume = getattr(args, "resume", False)
    if resume and not args.output_jsonl:
        sys.stderr.write("error: --resume requires --output-jsonl\n")
        return 1

    config = _client_config_from_args(args)
    client = _build_client(config)
    try:
        return _run_generate_command(client, args, endpoint=config.endpoint)
    finally:
        client.close()


def _handle_embed(args: argparse.Namespace) -> int:
    """Handle the ``embed`` subcommand."""

    client = _build_client(_client_config_from_args(args))
    try:
        texts = _load_embed_texts(text=args.text, input_jsonl=args.input_jsonl)
        batch = client.embed_batch(texts)
        rows: list[dict[str, Any]] = []
        for index, result in enumerate(batch):
            if result is None:
                error = batch.errors[index] if batch.errors else None
                rows.append(
                    {
                        "embedding": None,
                        "dimensions": None,
                        "request_id": None,
                        "token_usage": None,
                        "error": str(error) if error else "unknown error",
                    }
                )
            else:
                rows.append(
                    {
                        "embedding": None if args.no_vectors else result.embedding,
                        "dimensions": len(result.embedding),
                        "request_id": result.request_id,
                        "token_usage": _token_usage_to_dict(result),
                        "error": None,
                    }
                )
        _write_jsonl(rows, args.output_jsonl)
    finally:
        client.close()
    return 0


def _handle_transcribe(args: argparse.Namespace) -> int:
    """Handle the ``transcribe`` subcommand."""

    client = _build_client(_client_config_from_args(args))
    try:
        paths = _load_transcription_paths(path=args.path, input_jsonl=args.input_jsonl)
        rows: list[dict[str, Any]] = []
        for path in tqdm(
            paths,
            desc="Transcribing",
            unit="file",
            disable=(len(paths) <= 1),
            file=sys.stderr,
        ):
            result = client.transcribe(path)
            rows.append(
                {
                    "text": result.text,
                    "duration_s": result.duration_s,
                    "language": result.language,
                    "request_id": result.request_id,
                    "error": None,
                }
            )
        _write_jsonl(rows, args.output_jsonl)
    finally:
        client.close()
    return 0


def _handle_bench_generate(args: argparse.Namespace) -> int:
    """Handle ``bench generate``."""

    config = _client_config_from_args(args)
    rows = _load_generation_rows(prompt=args.prompt, input_jsonl=args.input_jsonl)
    input_items = [
        row.get("responses_input") or row.get("messages") or row.get("prompt")
        for row in rows
    ]
    if not input_items:
        raise ValueError("Generation benchmark requires input rows or --prompt.")

    concurrency_levels, recommend = _resolve_sweep_levels(
        single=getattr(args, "concurrency", None),
        maximum=getattr(args, "max_concurrency", None),
        default=DEFAULT_SWEEP,
        sweep_fn=_build_concurrency_sweep,
    )
    summary = _run_benchmark(
        task_name="generate",
        warmup=args.warmup,
        requests=args.requests,
        duration_s=getattr(args, "duration", None),
        concurrency_sweep=concurrency_levels,
        recommend=recommend,
        workload_factory=lambda concurrency: _build_client(
            config,
            max_parallel_requests=concurrency,
        ),
        runner=lambda client, batch, *, on_progress=None: client.generate_batch(
            batch,
            endpoint=config.endpoint,
            on_progress=on_progress,
        ),
        workload=batched_cycle(input_items, max(args.requests, args.warmup)),
    )
    _write_generate_summary(summary, args.output_json)
    return 0


def _handle_bench_embed(args: argparse.Namespace) -> int:
    """Handle ``bench embed``."""

    config = _client_config_from_args(args)
    texts = _load_embed_texts(text=args.text, input_jsonl=args.input_jsonl)
    if not texts:
        raise ValueError("Embedding benchmark requires input rows or --text.")

    batch_sizes, recommend = _resolve_sweep_levels(
        single=getattr(args, "batch_size", None),
        maximum=getattr(args, "max_batch_size", None),
        default=DEFAULT_EMBED_BATCH_SIZES,
        sweep_fn=_build_embed_batch_sweep,
    )
    summary = _run_embed_batch_benchmark(
        texts=texts,
        config=config,
        warmup=args.warmup,
        requests=args.requests,
        batch_size_sweep=batch_sizes,
        recommend=recommend,
        build_client=_build_client,
    )
    _write_embed_summary(summary, args.output_json)
    return 0
