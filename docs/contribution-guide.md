# Contribution Guide

Detailed contributor documentation. For a short checklist, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Principles

1. **Runtime first** — do not privilege any provider
2. **Stable API** — extend carefully; avoid breaking `discover/plan/invoke/observe/session/checkpoint/metrics`
3. **SOLID + DI** — no globals; compose in `build_runtime()`
4. **Deterministic tests** — offline, no secrets

## Development setup

```bash
git clone https://github.com/GemminAI/nvs-runtime.git
cd nvs-runtime
python -m venv .venv
source .venv/bin/activate
# nvs-observation is not on PyPI yet, so install it from GitHub first
pip install "git+https://github.com/GemminAI/nvs-observation.git"
pip install -e ".[dev]"
```

## Running tests

```bash
pytest --cov=nvs_runtime --cov-report=term-missing
ruff check src tests
python examples/hello_runtime.py
python examples/plan_and_invoke.py
```

## Formatting

```bash
ruff check src tests
ruff format src tests
```

## Commit style

- Imperative subject line
- Explain why in the body when needed
- Keep commits focused

## Pull request workflow

1. Fork and branch from `main`
2. Open an issue for non-trivial work
3. Add or adjust offline tests
4. Update docs / examples when the public surface changes
5. Ensure CI / local `pytest` and `ruff` pass
6. Complete the PR template checklist
7. Follow the [Code of Conduct](../CODE_OF_CONDUCT.md)

## What belongs where

| Change | Location |
|--------|----------|
| Stable API | `core/runtime.py` |
| Planning | `planner/` |
| New provider | `providers/` |
| MCP-only glue | `providers/mcp.py` only |
| Auth | `security/` |
| Quotas / plans | `governance/` |

## License

Contributions are accepted under Apache-2.0.
