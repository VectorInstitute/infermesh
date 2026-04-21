"""Run setup and cleanup helpers for the generate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint import (
    _checkpoint_path_for,
    _FileBackedPersistenceSink,
    _stage_fresh_workflow_files,
)
from .mapping import _compute_mapping_fingerprint, _load_mapper
from .models import _ResumePlan
from .prepare import PlannedResumePreparer, Preparer, SequentialPreparer
from .resume import ResumePlanner, validate_resume_checkpoint
from .source import _materialize_stdin_source, _validate_distinct_input_output_paths

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class _GenerateRunResources:
    """Normalized state for one generate-workflow invocation."""

    staged_stdin_path: Path | None
    persistence_sink: _FileBackedPersistenceSink | None
    resume_plan: _ResumePlan | None
    preparer: Preparer

    def close(self) -> None:
        """Release non-scheduler resources and re-raise the first cleanup error."""

        _cleanup_generate_run_resources(
            persistence_sink=self.persistence_sink,
            resume_plan=self.resume_plan,
            staged_stdin_path=self.staged_stdin_path,
        )


def _cleanup_generate_run_resources(
    *,
    persistence_sink: _FileBackedPersistenceSink | None,
    resume_plan: _ResumePlan | None,
    staged_stdin_path: Path | None,
) -> None:
    """Release setup-owned resources and re-raise the first cleanup error."""

    cleanup_error: BaseException | None = None

    try:
        if persistence_sink is not None:
            persistence_sink.close()
    except BaseException as exc:  # noqa: BLE001
        cleanup_error = exc

    try:
        ResumePlanner.cleanup(resume_plan)
    except BaseException as exc:  # noqa: BLE001
        if cleanup_error is None:
            cleanup_error = exc

    try:
        if staged_stdin_path is not None:
            staged_stdin_path.unlink(missing_ok=True)
    except BaseException as exc:  # noqa: BLE001
        if cleanup_error is None:
            cleanup_error = exc

    if cleanup_error is not None:
        raise cleanup_error


def _prepare_generate_run_resources(
    *,
    prompt: str | None,
    input_jsonl: str | None,
    output_jsonl: str | None,
    checkpoint_dir: str | None,
    mapper_spec: str | None,
    resume: bool,
    on_status: Callable[[str], Any] | None = None,
) -> _GenerateRunResources:
    """Build the normalized runtime resources for one generate run."""

    mapper = _load_mapper(mapper_spec) if mapper_spec else None
    mapping_fingerprint = _compute_mapping_fingerprint(
        mapper_spec=mapper_spec,
        mapper=mapper,
    )

    effective_input_jsonl = input_jsonl
    staged_stdin_path: Path | None = None
    output_path = Path(output_jsonl) if output_jsonl else None
    checkpoint_path = (
        _checkpoint_path_for(output_jsonl, checkpoint_dir=checkpoint_dir)
        if output_jsonl
        else None
    )
    persistence_sink: _FileBackedPersistenceSink | None = None
    resume_plan: _ResumePlan | None = None

    try:
        if output_jsonl and prompt is None and input_jsonl is None:
            # File-backed runs need a replayable source so bootstrap/resume can
            # scan it again later; stdout-only runs can keep streaming stdin.
            staged_stdin_path = _materialize_stdin_source()
            effective_input_jsonl = str(staged_stdin_path)

        _validate_distinct_input_output_paths(
            input_jsonl=effective_input_jsonl,
            output_jsonl=output_jsonl,
        )

        if output_path is not None and not resume:
            assert checkpoint_path is not None
            if on_status is not None:
                on_status("Preparing fresh workflow artifacts...")
            # Bootstrap into temp artifacts first so source/bootstrap failures
            # never clobber an existing output/checkpoint pair.
            _stage_fresh_workflow_files(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
            )

        if resume and output_path is not None:
            assert checkpoint_path is not None
            resume_plan = validate_resume_checkpoint(
                output_path,
                checkpoint_path,
                mapping_fingerprint=mapping_fingerprint,
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                on_status=on_status,
            )

        if output_path is not None:
            assert checkpoint_path is not None
            if on_status is not None:
                on_status("Opening output and checkpoint files...")
            persistence_sink = _FileBackedPersistenceSink(
                output_path=output_path,
                checkpoint_path=checkpoint_path,
            )

        if resume_plan is not None:
            assert effective_input_jsonl is not None
            preparer: Preparer = PlannedResumePreparer(
                input_jsonl=effective_input_jsonl,
                resume_plan=resume_plan,
                mapper=mapper,
            )
        else:
            preparer = SequentialPreparer(
                prompt=prompt,
                input_jsonl=effective_input_jsonl,
                resume=resume,
                checkpoint_path=checkpoint_path,
                mapper=mapper,
            )

        return _GenerateRunResources(
            staged_stdin_path=staged_stdin_path,
            persistence_sink=persistence_sink,
            resume_plan=resume_plan,
            preparer=preparer,
        )
    except BaseException:
        # Setup failed before the scheduler took ownership, so reuse the shared
        # cleanup helper here instead of fabricating a dummy resource wrapper.
        _cleanup_generate_run_resources(
            persistence_sink=persistence_sink,
            resume_plan=resume_plan,
            staged_stdin_path=staged_stdin_path,
        )
        raise
