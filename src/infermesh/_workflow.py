"""Internal workflow engine for the CLI."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from infermesh._cli_support import _build_generation_record, _write_jsonl

if TYPE_CHECKING:
    from infermesh.client import LMClient
    from infermesh.types import EndpointType


@dataclass
class _WorkItem:
    """One unit of work in the generate workflow."""

    output_index: int
    checkpoint_key: tuple[str, int]
    mapped_input: Any
    metadata: dict[str, Any] | None


@dataclass
class _SourceRow:
    """One source row and its parse outcome."""

    source_index: int
    raw_line: str
    raw_record: dict[str, Any] | None
    error: Exception | None


_PENDING_STATUS = "pending"
_SUCCESS_STATUS = "success"
_ERROR_STATUS = "error"
_SETTLED_STATUSES = frozenset({_SUCCESS_STATUS, _ERROR_STATUS})
_VALID_STATUSES = frozenset({_PENDING_STATUS, *_SETTLED_STATUSES})
_BUILTIN_MAPPING_FINGERPRINT = hashlib.sha256(
    b"infermesh.generate.builtin_mapping.v1"
).hexdigest()


def _compute_record_fingerprint(raw_record: dict[str, Any]) -> str:
    """Return a stable SHA-256 of the canonical JSON representation of ``raw_record``."""

    canonical = json.dumps(
        raw_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_parse_error_fingerprint(raw_line: str) -> str:
    """Return a stable fingerprint for one malformed JSONL source line."""

    return hashlib.sha256(f"__parse_error__{raw_line}".encode()).hexdigest()


def _state_path_for(output_jsonl: str) -> Path:
    """Derive the checkpoint file path from the output JSONL path.

    ``results.jsonl``  ->  ``results.state.jsonl``
    ``results``        ->  ``results.state.jsonl``
    """
    p = Path(output_jsonl)
    if p.suffix == ".jsonl":
        return p.with_name(p.stem + ".state.jsonl")
    return p.with_name(p.name + ".state.jsonl")


def _load_resume_state(state_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load the latest checkpoint row per ``(record_fingerprint, occurrence)``."""

    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not state_path.exists():
        return completed
    with state_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            fingerprint = row.get("record_fingerprint")
            occurrence = row.get("occurrence")
            output_index = row.get("_index")
            status = row.get("status")
            mapping_fingerprint = row.get("mapping_fingerprint")
            if (
                isinstance(fingerprint, str)
                and isinstance(occurrence, int)
                and isinstance(output_index, int)
                and isinstance(status, str)
                and status in _VALID_STATUSES
                and (
                    mapping_fingerprint is None or isinstance(mapping_fingerprint, str)
                )
            ):
                completed[(fingerprint, occurrence)] = row
    return completed


def _paths_reference_same_file(input_path: Path, output_path: Path) -> bool:
    """Return whether two paths resolve to the same file target."""

    if input_path.exists() and output_path.exists():
        try:
            if input_path.samefile(output_path):
                return True
        except OSError:
            pass
    return input_path.resolve(strict=False) == output_path.resolve(strict=False)


def _validate_distinct_input_output_paths(
    *, input_jsonl: str | None, output_jsonl: str | None
) -> None:
    """Reject file-backed runs that reuse the same path for input and output."""

    if input_jsonl is None or output_jsonl is None:
        return
    if _paths_reference_same_file(Path(input_jsonl), Path(output_jsonl)):
        raise ValueError("--input-jsonl and --output-jsonl must be different files.")


def _iter_source_rows(
    *, prompt: str | None, input_jsonl: str | None
) -> Iterator[_SourceRow]:
    """Yield source rows one line at a time.

    - For ``--prompt``: yields one synthetic record and stops.
    - For ``--input-jsonl``: streams the file line by line.
    - For neither (stdin): streams ``sys.stdin`` line by line.

    Malformed JSON lines yield a source row with ``error`` set so the caller
    can write a per-item error row and continue.
    """
    if prompt is not None:
        yield _SourceRow(
            source_index=0,
            raw_line=prompt,
            raw_record={"prompt": prompt},
            error=None,
        )
        return

    source: IO[str]
    if input_jsonl is not None:
        source = open(input_jsonl, encoding="utf-8")  # noqa: SIM115
        close_after = True
    else:
        source = sys.stdin
        close_after = False

    try:
        index = 0
        for raw_line in source:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield _SourceRow(
                    source_index=index,
                    raw_line=stripped,
                    raw_record=None,
                    error=exc,
                )
            else:
                yield _SourceRow(
                    source_index=index,
                    raw_line=stripped,
                    raw_record=record,
                    error=None,
                )
            index += 1
    finally:
        if close_after:
            source.close()


def _iter_source_rows_with_keys(
    *, prompt: str | None, input_jsonl: str | None
) -> Iterator[tuple[_SourceRow, tuple[str, int]]]:
    """Yield ``(source_row, checkpoint_key)`` pairs with occurrence-aware keys."""
    fingerprint_counts: dict[str, int] = {}
    for source_row in _iter_source_rows(prompt=prompt, input_jsonl=input_jsonl):
        yield source_row, _resume_key_for_source_row(source_row, fingerprint_counts)


def _materialize_stdin_source() -> Path:
    """Copy stdin to a temporary JSONL file so file-backed runs can replay it."""

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".jsonl",
        delete=False,
    ) as file_handle:
        for raw_line in sys.stdin:
            file_handle.write(raw_line)
        return Path(file_handle.name)


def _resume_key_for_source_row(
    source_row: _SourceRow, fingerprint_counts: dict[str, int]
) -> tuple[str, int]:
    """Return the occurrence-aware resume key for one source row."""

    if source_row.raw_record is not None:
        fingerprint = _compute_record_fingerprint(source_row.raw_record)
    else:
        fingerprint = _compute_parse_error_fingerprint(source_row.raw_line)
    occurrence = fingerprint_counts.get(fingerprint, 0)
    fingerprint_counts[fingerprint] = occurrence + 1
    return fingerprint, occurrence


def _compute_mapper_implementation_fingerprint(
    mapper: Callable[[dict[str, Any]], Any],
) -> str:
    """Return a stable fingerprint for the mapper implementation.

    Prefer the full mapper module source when available so helper functions and
    module-level constants also participate in resume compatibility. Fall back
    to the mapper function source, then the code object, for dynamic modules
    where module source cannot be recovered.
    """
    module = inspect.getmodule(mapper)
    if module is not None:
        try:
            module_source = inspect.getsource(module)
        except (OSError, TypeError):
            pass
        else:
            return hashlib.sha256(module_source.encode("utf-8")).hexdigest()

    try:
        return hashlib.sha256(inspect.getsource(mapper).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        pass

    code = getattr(mapper, "__code__", None)
    fallback_payload = repr(
        {
            "co_code": getattr(code, "co_code", None),
            "co_consts": getattr(code, "co_consts", None),
            "co_names": getattr(code, "co_names", None),
            "defaults": getattr(mapper, "__defaults__", None),
            "kwdefaults": getattr(mapper, "__kwdefaults__", None),
        }
    )
    return hashlib.sha256(fallback_payload.encode("utf-8")).hexdigest()


def _compute_mapping_fingerprint(
    *,
    mapper_spec: str | None,
    mapper: Callable[[dict[str, Any]], Any] | None,
) -> str:
    """Return the fingerprint that ties a run to its mapping strategy."""

    if mapper is None:
        return _BUILTIN_MAPPING_FINGERPRINT

    module_name = getattr(mapper, "__module__", type(mapper).__module__)
    qualname = getattr(mapper, "__qualname__", type(mapper).__qualname__)
    payload = json.dumps(
        {
            "mapper_spec": mapper_spec,
            "module_name": module_name,
            "qualname": qualname,
            "implementation_fingerprint": _compute_mapper_implementation_fingerprint(
                mapper
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_state_row(
    checkpoint_key: tuple[str, int],
    output_index: int,
    *,
    mapping_fingerprint: str,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one checkpoint row."""

    return {
        "record_fingerprint": checkpoint_key[0],
        "occurrence": checkpoint_key[1],
        "_index": output_index,
        "mapping_fingerprint": mapping_fingerprint,
        "status": status,
        "error": error,
    }


def _bootstrap_checkpoint(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    state_path: Path,
    mapping_fingerprint: str,
) -> None:
    """Write one pending checkpoint row per source item before execution starts."""

    with state_path.open("w", encoding="utf-8") as state_file:
        for source_row, checkpoint_key in _iter_source_rows_with_keys(
            prompt=prompt, input_jsonl=input_jsonl
        ):
            state_row = _build_state_row(
                checkpoint_key,
                source_row.source_index,
                mapping_fingerprint=mapping_fingerprint,
                status=_PENDING_STATUS,
                error=None,
            )
            state_file.write(json.dumps(state_row) + "\n")


def _stage_fresh_workflow_files(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_path: Path,
    state_path: Path,
    mapping_fingerprint: str,
) -> None:
    """Stage fresh workflow artifacts and replace existing ones after bootstrap."""

    staged_output_path: Path | None = None
    staged_state_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            staged_state_path = Path(file_handle.name)
        _bootstrap_checkpoint(
            prompt=prompt,
            input_jsonl=input_jsonl,
            state_path=staged_state_path,
            mapping_fingerprint=mapping_fingerprint,
        )

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            staged_output_path = Path(file_handle.name)

        staged_state_path.replace(state_path)
        staged_output_path.replace(output_path)
    finally:
        if staged_state_path is not None and staged_state_path.exists():
            staged_state_path.unlink()
        if staged_output_path is not None and staged_output_path.exists():
            staged_output_path.unlink()


def _validate_resume_source(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    checkpoint_state: dict[tuple[str, int], dict[str, Any]],
) -> None:
    """Validate that the current source matches the checkpoint occurrence set."""

    seen_keys: set[tuple[str, int]] = set()
    for _, checkpoint_key in _iter_source_rows_with_keys(
        prompt=prompt, input_jsonl=input_jsonl
    ):
        if checkpoint_key not in checkpoint_state:
            raise ValueError(
                "Resume source does not match the checkpoint manifest. Added, "
                "removed, or modified row occurrences are not supported."
            )
        seen_keys.add(checkpoint_key)

    if seen_keys != set(checkpoint_state):
        raise ValueError(
            "Resume source does not match the checkpoint manifest. Added, "
            "removed, or modified row occurrences are not supported."
        )


def _validate_resume_mapping_fingerprint(
    checkpoint_state: dict[tuple[str, int], dict[str, Any]],
    *,
    mapping_fingerprint: str,
    using_custom_mapper: bool,
) -> None:
    """Validate mapping consistency between the checkpoint and current run."""

    if not checkpoint_state:
        return

    observed_fingerprints = {
        row["mapping_fingerprint"]
        for row in checkpoint_state.values()
        if isinstance(row.get("mapping_fingerprint"), str)
    }
    missing_fingerprints = any(
        row.get("mapping_fingerprint") is None for row in checkpoint_state.values()
    )

    if len(observed_fingerprints) > 1 or (
        missing_fingerprints and observed_fingerprints
    ):
        raise ValueError(
            "Checkpoint manifest is inconsistent: mixed mapping_fingerprint "
            "values were found."
        )

    if not observed_fingerprints:
        if using_custom_mapper:
            raise ValueError(
                "Checkpoint manifest predates mapper fingerprints and cannot "
                "be resumed with --mapper. Restart the run without --resume."
            )
        return

    checkpoint_mapping_fingerprint = next(iter(observed_fingerprints))
    if checkpoint_mapping_fingerprint != mapping_fingerprint:
        raise ValueError(
            "Resume mapping does not match the checkpoint manifest. Use the "
            "original mapper implementation or restart without --resume."
        )


def _load_mapper(mapper_spec: str) -> Callable[[dict[str, Any]], Any]:
    """Load a mapper function from a ``"package.module:function"`` spec.

    Raises ``ValueError`` if the spec is malformed or the resolved attribute is
    not callable.
    """
    module_path, sep, func_name = mapper_spec.rpartition(":")
    if not sep or not module_path or not func_name:
        raise ValueError(
            f"--mapper must be 'package.module:function', got {mapper_spec!r}"
        )
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None:
        raise ValueError(f"--mapper: {module_path!r} has no attribute {func_name!r}")
    if not callable(func):
        raise ValueError(
            f"--mapper: {mapper_spec!r} resolved to a non-callable {type(func).__name__!r}"
        )
    return cast(Callable[[dict[str, Any]], Any], func)


def _apply_mapper_or_builtin(
    raw_record: dict[str, Any], mapper: Callable[[dict[str, Any]], Any] | None
) -> tuple[Any, dict[str, Any] | None] | Exception:
    """Apply the mapper (or built-in field extraction) to ``raw_record``.

    Returns ``(mapped_input, metadata_or_None)`` on success, or an ``Exception``
    on failure. Mapper failures become per-item error rows, not run aborts.

    Extra keys beyond ``"input"`` and ``"metadata"`` in the mapper return value
    are silently ignored.
    """
    if mapper is not None:
        try:
            result = mapper(raw_record)
        except Exception as exc:  # noqa: BLE001
            return exc
        if not isinstance(result, dict):
            return ValueError(
                f"Mapper must return a dict, got {type(result).__name__!r}"
            )
        if "input" not in result:
            return KeyError("Mapper return value is missing required key 'input'")
        return result["input"], result.get("metadata")

    # Built-in mapping: extract the first recognised input field.
    input_data = (
        raw_record.get("responses_input")
        or raw_record.get("messages")
        or raw_record.get("prompt")
    )
    if input_data is None:
        return ValueError(
            "Generation rows require 'prompt', 'messages', or 'responses_input'."
        )
    return input_data, None


def _validate_metadata(metadata: Any) -> dict[str, Any] | None | Exception:
    """Validate mapper metadata before it reaches the sink."""

    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return TypeError("Mapper 'metadata' must be a dict when provided.")
    try:
        json.dumps(metadata)
    except TypeError as exc:
        return TypeError(f"Mapper 'metadata' must be JSON serializable: {exc}")
    return metadata


def _write_item_result(
    out_file: IO[str] | None,
    state_file: IO[str] | None,
    output_index: int,
    checkpoint_key: tuple[str, int],
    result: Any,
    error: BaseException | None,
    *,
    mapping_fingerprint: str,
    metadata: dict[str, Any] | None,
    parse_json: bool,
) -> None:
    """Write one settled item to the output file and the state file.

    Output row is written first so a crash between the two writes never loses
    a completed result — the worst case is one duplicate output row on resume.
    """
    record = _build_generation_record(
        output_index, result, error, parse_json=parse_json
    )
    if metadata is not None:
        record["metadata"] = metadata

    if out_file is not None:
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

    if state_file is not None:
        state_row = _build_state_row(
            checkpoint_key,
            output_index,
            mapping_fingerprint=mapping_fingerprint,
            status=_ERROR_STATUS if error else _SUCCESS_STATUS,
            error=str(error) if error else None,
        )
        state_file.write(json.dumps(state_row) + "\n")
        state_file.flush()


def _write_immediate_error(
    out_file: IO[str] | None,
    state_file: IO[str] | None,
    output_index: int,
    checkpoint_key: tuple[str, int],
    error: BaseException,
    *,
    mapping_fingerprint: str,
    metadata: dict[str, Any] | None,
    parse_json: bool,
) -> None:
    """Write one per-item workflow error without aborting the run."""

    if out_file is not None:
        _write_item_result(
            out_file,
            state_file,
            output_index,
            checkpoint_key,
            None,
            error,
            mapping_fingerprint=mapping_fingerprint,
            metadata=metadata,
            parse_json=parse_json,
        )
        return

    row = _build_generation_record(output_index, None, error, parse_json=parse_json)
    if metadata is not None:
        row["metadata"] = metadata
    _write_jsonl([row], None)


def _flush_window(
    client: LMClient,
    window: list[_WorkItem],
    out_file: IO[str] | None,
    state_file: IO[str] | None,
    endpoint: EndpointType,
    parse_json: bool,
    mapping_fingerprint: str,
    on_progress: Callable[[], Any] | None,
) -> list[dict[str, Any]]:
    """Execute one window of work through ``generate_batch`` and write results.

    Returns a list of output records for stdout-mode callers (empty for
    file-backed runs where rows are written inside the ``on_result`` callback).
    """
    inputs = [item.mapped_input for item in window]
    stdout_rows: list[dict[str, Any]] = []

    def on_result(batch_idx: int, result: Any, error: BaseException | None) -> None:
        item = window[batch_idx]
        if out_file is not None:
            _write_item_result(
                out_file,
                state_file,
                item.output_index,
                item.checkpoint_key,
                result,
                error,
                mapping_fingerprint=mapping_fingerprint,
                metadata=item.metadata,
                parse_json=parse_json,
            )
        else:
            record = _build_generation_record(
                item.output_index, result, error, parse_json=parse_json
            )
            if item.metadata is not None:
                record["metadata"] = item.metadata
            stdout_rows.append(record)
        if on_progress is not None:
            on_progress()

    client.generate_batch(
        inputs,
        endpoint=endpoint,
        return_exceptions=True,
        on_result=on_result,
    )
    return stdout_rows


def _load_resume_checkpoint(
    output_path: Path,
    state_path: Path,
    *,
    mapping_fingerprint: str,
    using_custom_mapper: bool,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load and validate the resume checkpoint for a file-backed workflow."""

    if not state_path.exists():
        raise ValueError(
            f"--resume requires checkpoint file {state_path}. "
            "Start a fresh file-backed run first."
        )
    if not output_path.exists():
        raise ValueError(
            f"--resume requires output file {output_path} because checkpoint "
            f"{state_path} already exists."
        )

    checkpoint_state = _load_resume_state(state_path)
    _validate_resume_mapping_fingerprint(
        checkpoint_state,
        mapping_fingerprint=mapping_fingerprint,
        using_custom_mapper=using_custom_mapper,
    )
    output_indices = _load_output_indices(output_path)
    settled_indices = {
        int(row["_index"])
        for row in checkpoint_state.values()
        if row["status"] in _SETTLED_STATUSES
    }
    missing_indices = sorted(settled_indices - output_indices)
    if missing_indices:
        missing_text = ", ".join(str(index) for index in missing_indices[:10])
        if len(missing_indices) > 10:
            missing_text += ", ..."
        raise ValueError(
            "Output file is missing settled checkpoint rows for _index values "
            f"{missing_text}. Restore the output artifact or restart the run "
            "without --resume."
        )
    return checkpoint_state


def _load_output_indices(output_path: Path) -> set[int]:
    """Return the set of integer ``_index`` values present in the output file."""

    observed_indices: set[int] = set()
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            output_index = row.get("_index")
            if isinstance(output_index, int):
                observed_indices.add(output_index)
    return observed_indices


def _generate_output_index_for_resume_row(
    *,
    resume: bool,
    checkpoint_state: dict[tuple[str, int], dict[str, Any]],
    checkpoint_key: tuple[str, int],
    source_index: int,
) -> int | None:
    """Return the output index for this row, or ``None`` if the row is already settled."""

    if not resume:
        return source_index
    manifest_row = checkpoint_state.get(checkpoint_key)
    if manifest_row is None:
        raise ValueError(
            "Resume source does not match the checkpoint manifest. "
            "Added, removed, or modified row occurrences are not "
            "supported."
        )
    if manifest_row["status"] in _SETTLED_STATUSES:
        return None
    return int(manifest_row["_index"])


def _emit_generate_immediate_error(
    out_file: IO[str] | None,
    state_file: IO[str] | None,
    output_index: int,
    checkpoint_key: tuple[str, int],
    error: BaseException,
    *,
    mapping_fingerprint: str,
    parse_json: bool,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Write one per-item error row and optionally invoke progress."""

    _write_immediate_error(
        out_file,
        state_file,
        output_index,
        checkpoint_key,
        error,
        mapping_fingerprint=mapping_fingerprint,
        metadata=None,
        parse_json=parse_json,
    )
    if on_progress is not None:
        on_progress()


def _run_generate_source_rows(
    client: LMClient,
    *,
    prompt: str | None,
    effective_input_jsonl: str | None,
    resume: bool,
    checkpoint_state: dict[tuple[str, int], dict[str, Any]],
    mapper: Callable[[dict[str, Any]], Any] | None,
    window_size: int,
    out_file: IO[str] | None,
    state_file: IO[str] | None,
    endpoint: EndpointType,
    parse_json: bool,
    mapping_fingerprint: str,
    on_progress: Callable[[], Any] | None,
) -> None:
    """Stream mapped rows into ``generate_batch`` windows."""

    window: list[_WorkItem] = []
    any_work = False

    for source_row, checkpoint_key in _iter_source_rows_with_keys(
        prompt=prompt, input_jsonl=effective_input_jsonl
    ):
        output_index = _generate_output_index_for_resume_row(
            resume=resume,
            checkpoint_state=checkpoint_state,
            checkpoint_key=checkpoint_key,
            source_index=source_row.source_index,
        )
        if output_index is None:
            continue

        if source_row.error is not None:
            _emit_generate_immediate_error(
                out_file,
                state_file,
                output_index,
                checkpoint_key,
                source_row.error,
                mapping_fingerprint=mapping_fingerprint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            any_work = True
            continue

        raw_record = source_row.raw_record
        if raw_record is None:
            raise RuntimeError(
                "Invariant violated: source_row.raw_record is None after error check."
            )

        mapping_result = _apply_mapper_or_builtin(raw_record, mapper)
        if isinstance(mapping_result, Exception):
            _emit_generate_immediate_error(
                out_file,
                state_file,
                output_index,
                checkpoint_key,
                mapping_result,
                mapping_fingerprint=mapping_fingerprint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            any_work = True
            continue

        mapped_input, metadata = mapping_result
        metadata_result = _validate_metadata(metadata)
        if isinstance(metadata_result, Exception):
            _emit_generate_immediate_error(
                out_file,
                state_file,
                output_index,
                checkpoint_key,
                metadata_result,
                mapping_fingerprint=mapping_fingerprint,
                parse_json=parse_json,
                on_progress=on_progress,
            )
            any_work = True
            continue

        any_work = True
        window.append(
            _WorkItem(
                output_index=output_index,
                checkpoint_key=checkpoint_key,
                mapped_input=mapped_input,
                metadata=metadata_result,
            )
        )

        if len(window) >= window_size:
            stdout_rows = _flush_window(
                client,
                window,
                out_file,
                state_file,
                endpoint,
                parse_json,
                mapping_fingerprint,
                on_progress,
            )
            if stdout_rows:
                _write_jsonl(stdout_rows, None)
            window.clear()

    if window:
        stdout_rows = _flush_window(
            client,
            window,
            out_file,
            state_file,
            endpoint,
            parse_json,
            mapping_fingerprint,
            on_progress,
        )
        if stdout_rows:
            _write_jsonl(stdout_rows, None)

    if resume and not any_work:
        sys.stderr.write("Nothing to do — all rows already completed.\n")


def run_generate_workflow(
    client: LMClient,
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_jsonl: str | None,
    mapper_spec: str | None,
    resume: bool,
    endpoint: EndpointType,
    window_size: int,
    parse_json: bool,
    on_progress: Callable[[], Any] | None = None,
) -> None:
    """Run the generate workflow engine.

    Streams source rows one at a time, maps them, submits them to
    ``generate_batch`` in bounded windows, and writes results to disk (or
    stdout) as each item settles.

    For file-backed runs a checkpoint file ``*.state.jsonl`` is kept as the
    checkpoint; for stdout runs no checkpoint is created.
    """
    mapper = _load_mapper(mapper_spec) if mapper_spec else None
    mapping_fingerprint = _compute_mapping_fingerprint(
        mapper_spec=mapper_spec,
        mapper=mapper,
    )

    effective_input_jsonl = input_jsonl
    staged_stdin_path: Path | None = None
    if output_jsonl and prompt is None and input_jsonl is None:
        staged_stdin_path = _materialize_stdin_source()
        effective_input_jsonl = str(staged_stdin_path)

    output_path = Path(output_jsonl) if output_jsonl else None
    state_path = _state_path_for(output_jsonl) if output_jsonl else None
    checkpoint_state: dict[tuple[str, int], dict[str, Any]] = {}

    out_file: IO[str] | None = None
    state_file: IO[str] | None = None

    try:
        _validate_distinct_input_output_paths(
            input_jsonl=effective_input_jsonl,
            output_jsonl=output_jsonl,
        )

        if output_path is not None and state_path is not None and not resume:
            _stage_fresh_workflow_files(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                output_path=output_path,
                state_path=state_path,
                mapping_fingerprint=mapping_fingerprint,
            )

        if resume and output_jsonl:
            assert state_path is not None
            assert output_path is not None
            checkpoint_state = _load_resume_checkpoint(
                output_path,
                state_path,
                mapping_fingerprint=mapping_fingerprint,
                using_custom_mapper=mapper is not None,
            )
            _validate_resume_source(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                checkpoint_state=checkpoint_state,
            )

        if output_jsonl:
            out_file = open(output_jsonl, "a", encoding="utf-8")  # noqa: SIM115
        if state_path:
            state_file = open(state_path, "a", encoding="utf-8")  # noqa: SIM115

        _run_generate_source_rows(
            client,
            prompt=prompt,
            effective_input_jsonl=effective_input_jsonl,
            resume=resume,
            checkpoint_state=checkpoint_state,
            mapper=mapper,
            window_size=window_size,
            out_file=out_file,
            state_file=state_file,
            endpoint=endpoint,
            parse_json=parse_json,
            mapping_fingerprint=mapping_fingerprint,
            on_progress=on_progress,
        )

    finally:
        if out_file is not None:
            out_file.close()
        if state_file is not None:
            state_file.close()
        if staged_stdin_path is not None:
            staged_stdin_path.unlink(missing_ok=True)
