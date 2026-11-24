# Developer Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# nvs-observation is not on PyPI yet, so install it from GitHub first
pip install "git+https://github.com/GemminAI/nvs-observation.git"
pip install -e ".[dev]"
```

## Layout

```
src/nvs_runtime/
  core/           NVSRuntime, models, protocols, DI
  planner/        graph-search planner
  graph/          capability DAG
  session/        sessions + checkpoints
  metrics/        counters
  observability/  query store
  providers/      local, http, grpc, mcp, kernel, hekb
  transport/      memory transport
  governance/     plans + quotas
  security/       CBAC + signer
  config/         RuntimeConfig
  events/         EventBus
examples/        hello_runtime.py, plan_and_invoke.py, hidden_trajectory/ (the door scripts)
tests/
docs/
```

K measurement, the K checks and the experiment drivers are not in this repository; see
"What is here, and what is not" in the README.

## Run tests

```bash
pytest --cov=nvs_runtime --cov-report=term-missing
```

Requirements:

- Offline
- Deterministic (`ControllableClock`)
- No API keys
- Coverage ≥ 90%

## Lint

```bash
ruff check src tests
```

## Design checklist

- [ ] New behavior behind an interface
- [ ] Wired only in `build_runtime()`
- [ ] Offline test double available
- [ ] No provider-specific imports in `core/`
- [ ] Public API remains discover / plan / invoke / observe / session / checkpoint / metrics

## Versioning

Package version lives in `src/nvs_runtime/__version__.py` and `pyproject.toml`.  
Runtime API is semver-stable within 1.x for the methods listed above.
