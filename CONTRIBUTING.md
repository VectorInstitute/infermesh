# Contributing

## Development setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd infermesh
uv sync --group dev
uv run pre-commit install
```

## Running the test suite

```bash
uv run pytest                          # all tests
uv run pytest tests/test_client.py    # single file
uv run pytest -k "batch"              # filter by name
```

## Type checking

```bash
uv run mypy src/infermesh tests
```

## Linting and formatting

Pre-commit runs ruff and mypy automatically on every commit.  To run manually:

```bash
uv run pre-commit run --all-files
```

## Dependency security

```bash
uv run pip-audit
```

## Pull requests

- Keep changes focused; one concern per PR.
- Add or update tests for any behaviour change.
- Run `uv run pre-commit run --all-files` and `uv run pytest` before opening a PR.
