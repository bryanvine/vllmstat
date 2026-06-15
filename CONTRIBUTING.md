# Contributing to vllmstat

Thank you for your interest in contributing.

## Design spec

Before diving in, read the design document — it explains the architecture, data-flow, metric names, and panel layout:

```
docs/superpowers/specs/2026-06-14-vllmstat-design.md
```

## Dev setup

Python 3.10 or later is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,proxy]"
```

This installs `vllmstat` in editable mode together with all development dependencies (pytest, pytest-asyncio, ruff, pyright). The optional `proxy` extra adds `aiohttp` for tee/proxy integration tests.

## Run the tests

```bash
pytest -q
```

All **216** tests should pass. The suite covers pure logic (histogram quantiles, EWMA rates, KV compression), fleet polling, tee/log tailing, reverse proxy, GPU vendors (NVIDIA/AMD/Intel via sysfs and fdinfo), Docker discovery, providers (mock transport for the vLLM HTTP client), and the Textual app itself.

## Lint and format

```bash
ruff check .
ruff format .
```

`ruff check` enforces style rules (E, F, I, UP, B, W). `ruff format` auto-formats. Format before committing.

## Type checking

```bash
pyright
```

Type-checking mode is `basic` targeting Python 3.10. Aim for zero errors.

## TDD expectation

This project follows test-driven development. For any new feature or bug fix:

1. Write a failing test that captures the desired behaviour.
2. Run the test to confirm it fails for the right reason.
3. Implement the minimum code to make it pass.
4. Confirm the full suite is still green.
5. Commit.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `ci: ...`, `test: ...`, etc.

## Full verification before a PR

```bash
ruff check . && ruff format --check . && pyright && pytest -q
```

All four must be clean (zero ruff issues, zero format diffs, zero pyright errors, all tests passing) before opening a pull request.

**Note:** GitHub Actions CI (`.github/workflows/ci.yml`) is planned but not yet wired up in this repository. Run the full local gate above before submitting changes.
