"""Shared fixtures — fresh DI graph per test, fixed clock, no API keys."""

from __future__ import annotations

import pytest

from nvs_runtime import build_runtime
from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.providers.grpc import GrpcProvider

FIXED_NOW = 1_700_000_000_000_000_000

@pytest.fixture
def clock() -> ControllableClock:
    return ControllableClock(FIXED_NOW)


@pytest.fixture
def runtime(clock: ControllableClock):
    rt = build_runtime(
        clock=clock,
        config={"enable_kernel": True, "enable_mcp": True, "enable_hekb": True},
    )
    rt.register_provider(GrpcProvider())
    return rt


@pytest.fixture
def token(runtime):
    return runtime.cbac.issue(
        tenant_id="test-tenant",
        scopes={"scope:invoke", "scope:plan", "scope:session"},
        max_cost_units=1_000,
    )
