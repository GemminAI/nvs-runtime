# Contributing to NVS Runtime

Thank you for contributing. Please also read the detailed
[Contribution Guide](docs/contribution-guide.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/GemminAI/nvs-runtime.git
cd nvs-runtime
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
# nvs-observation is not on PyPI yet, so install it from GitHub first
pip install "git+https://github.com/GemminAI/nvs-observation.git"
pip install -e ".[dev]"
```

## Running tests

```bash
pytest --cov=nvs_runtime --cov-report=term-missing
```

Examples (offline, no API keys):

```bash
python examples/hello_runtime.py
python examples/plan_and_invoke.py
```

## Formatting and lint

```bash
ruff check src tests
ruff format src tests   # optional; keep diffs focused
```

Target: Python 3.12+, typed public APIs, protocols over deep inheritance.

## What belongs here

Changes to the runtime, the providers, the capture path and the readers of stored results belong in
this repository. K measurement, the K checks, experiment drivers and their examples belong to the
Developer Tools, not here. A test that needs a measured K reads the stored data in
`tests/fixtures/measured_k` (see its `manifest.json`); do not edit those files by hand.

## Commit style

- Prefer short, imperative subject lines (≤ ~72 chars)
- Explain *why* in the body when the change is non-obvious
- Keep commits focused; avoid mixing docs-only and large code changes

Examples:

```
Add HttpProvider offline transport contract test
Clarify Runtime API session lifecycle in docs
```

## Pull request workflow

1. Open an issue for non-trivial changes
2. Branch from `main` (or the active release branch)
3. Keep the Runtime API stable unless the issue explicitly changes it
4. Put provider-specific code under `providers/`
5. Add or update offline deterministic tests
6. Update docs / examples when user-facing behavior changes
7. Ensure `pytest --cov=nvs_runtime` and `ruff check` pass
8. Fill out the PR template checklist

## Principles

1. **Runtime first** — do not privilege any provider
2. **Stable API** — `discover` / `plan` / `invoke` / `observe` / `session` / `checkpoint` / `metrics`
3. **Composition** — DI via `build_runtime()`, no module singletons
4. **Deterministic tests** — offline, no secrets
