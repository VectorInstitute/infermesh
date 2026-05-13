"""Mapper loading and mapping strategy helpers for the workflow engine."""

import contextlib
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from typing import Any, cast

# Encodes built-in field-extraction semantics. Bump the literal to invalidate
# existing checkpoints whenever the built-in mapping logic changes.
_BUILTIN_MAPPING_FINGERPRINT = hashlib.sha256(
    b"infermesh.generate.builtin_mapping.v1"
).hexdigest()


def _load_mapper(mapper_spec: str) -> Callable[[dict[str, Any]], Any]:
    r"""Load a mapper function from a ``\"package.module:function\"`` spec."""

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


def _compute_mapper_implementation_fingerprint(
    mapper: Callable[[dict[str, Any]], Any],
) -> str:
    """Return a stable fingerprint for the mapper implementation."""

    module = inspect.getmodule(mapper)
    if module is not None:
        with contextlib.suppress(OSError, TypeError):
            return hashlib.sha256(inspect.getsource(module).encode("utf-8")).hexdigest()

    with contextlib.suppress(OSError, TypeError):
        return hashlib.sha256(inspect.getsource(mapper).encode("utf-8")).hexdigest()

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
    *, mapper_spec: str | None, mapper: Callable[[dict[str, Any]], Any] | None
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
