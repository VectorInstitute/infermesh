"""Command-line interface for ``infermesh``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from infermesh._cli_bench import (
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
from infermesh._cli_support import (
    ClientConfig,
    _add_connection_args,
    _client_config_from_args,
    _load_embed_texts,
    _load_generation_rows,
    _load_transcription_paths,
    _token_usage_to_dict,
    _write_jsonl,
)
from infermesh._cli_support import _build_client as _support_build_client
from infermesh._utils import batched_cycle
from infermesh._workflow import run_generate_workflow
from infermesh.client import LMClient


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
    config: ClientConfig, *, max_parallel_requests: int | None = None
) -> LMClient:
    """Build an ``LMClient`` instance for CLI commands."""

    return _support_build_client(
        config, max_parallel_requests=max_parallel_requests, client_cls=LMClient
    )


@contextmanager
def _managed_client(
    config: ClientConfig, *, max_parallel_requests: int | None = None
) -> Iterator[LMClient]:
    """Build a client and guarantee ``close()`` on exit regardless of errors."""
    client = _build_client(config, max_parallel_requests=max_parallel_requests)
    try:
        yield client
    finally:
        client.close()


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="infermesh",
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
        "--output-jsonl", help="Write one result object per input row."
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
            "Resume a previous run by reading the checkpoint file "
            "*.checkpoint.sqlite and appending the unsettled rows."
        ),
    )
    generate_parser.add_argument(
        "--mapper",
        metavar="MODULE:FUNC",
        help=(
            "Import path of a mapper function: 'package.module:function'. "
            "The function receives a raw source record (dict) and must return "
            "a dict with at least an 'input' key."
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


def _handle_generate(args: argparse.Namespace) -> int:
    """Handle the ``generate`` subcommand."""

    resume = bool(getattr(args, "resume", False))
    if resume and not args.output_jsonl:
        sys.stderr.write("error: --resume requires --output-jsonl\n")
        return 1

    config = _client_config_from_args(args)
    window_size = config.max_parallel_requests or 128

    # Use an open-ended progress bar for file-backed runs (total unknown up
    # front).  For a single --prompt to stdout, suppress it entirely.
    disable_bar = args.output_jsonl is None and args.prompt is not None
    try:
        with (
            _managed_client(config) as client,
            tqdm(
                desc="Generating", unit="req", disable=disable_bar, file=sys.stderr
            ) as bar,
        ):

            def report_status(message: str) -> None:
                if disable_bar:
                    sys.stderr.write(message + "\n")
                else:
                    bar.write(message)

            run_generate_workflow(
                client,
                prompt=args.prompt,
                input_jsonl=args.input_jsonl,
                output_jsonl=args.output_jsonl,
                mapper_spec=getattr(args, "mapper", None),
                resume=resume,
                endpoint=config.endpoint,
                window_size=window_size,
                parse_json=bool(getattr(args, "parse_json", False)),
                on_progress=lambda: bar.update(),  # noqa: PLW0108
                on_status=report_status,
            )
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _handle_embed(args: argparse.Namespace) -> int:
    """Handle the ``embed`` subcommand."""

    try:
        with _managed_client(_client_config_from_args(args)) as client:
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
    except (ImportError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _handle_transcribe(args: argparse.Namespace) -> int:
    """Handle the ``transcribe`` subcommand."""

    try:
        with _managed_client(_client_config_from_args(args)) as client:
            paths = _load_transcription_paths(
                path=args.path, input_jsonl=args.input_jsonl
            )
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
    except (ImportError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _handle_bench_generate(args: argparse.Namespace) -> int:
    """Handle ``bench generate``."""

    try:
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
    except (ImportError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


def _handle_bench_embed(args: argparse.Namespace) -> int:
    """Handle ``bench embed``."""

    try:
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
    except (ImportError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0
