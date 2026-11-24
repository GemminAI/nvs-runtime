"""CBAC, signer, governance, quota."""

from __future__ import annotations

import pytest

from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.core.errors import AuthorizationError, GovernanceError
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.models import ExecutionPlan, PlanStep
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.governance.quota import QuotaEngine
from nvs_runtime.security.cbac import CBACEngine
from nvs_runtime.security.signer import HmacSigner


def test_hmac_signer_roundtrip() -> None:
    signer = HmacSigner(b"secret")
    sig = signer.sign(b"payload")
    assert signer.verify(b"payload", sig)
    assert not signer.verify(b"payload", b"bad")


def test_cbac_issue_verify_scopes_budget() -> None:
    clock = ControllableClock()
    cbac = CBACEngine(signer=HmacSigner(), clock=clock)
    token = cbac.issue(
        tenant_id="t1",
        scopes={"scope:invoke"},
        max_cost_units=10,
        ttl_ns=1_000_000_000,
    )
    payload = cbac.verify_token(token)
    assert payload.tenant_id == "t1"
    cbac.require_scopes(token, {"scope:invoke"})
    with pytest.raises(AuthorizationError):
        cbac.require_scopes(token, {"scope:plan"})
    with pytest.raises(AuthorizationError):
        cbac.check_budget(token, 100)
    clock.advance(2_000_000_000)
    with pytest.raises(AuthorizationError):
        cbac.verify_token(token)


def test_governance_admit_cancel() -> None:
    store = MemoryObjectStore()
    gov = GovernanceEngine(store=store, quota=QuotaEngine(max_concurrent=1, max_cost_per_tenant=50))
    plan = ExecutionPlan(
        goal="g",
        executable=True,
        estimated_cost_units=5,
        steps=[
            PlanStep(
                capability_id="c",
                provider_id="local",
                input_type="a",
                output_type="b",
            )
        ],
    )
    oid = gov.store_plan(plan)
    assert oid
    gov.admit(tenant_id="t1", plan=plan)
    with pytest.raises(GovernanceError):
        gov.admit(tenant_id="t1", plan=plan)  # concurrent limit
    gov.release("t1")
    cancelled = gov.cancel_plan(plan.plan_id)
    assert cancelled["cancelled"] is True
    assert gov.is_cancelled(plan.plan_id)
    with pytest.raises(GovernanceError):
        gov.admit(tenant_id="t1", plan=plan)


def test_quota_cost() -> None:
    q = QuotaEngine(max_concurrent=10, max_cost_per_tenant=5)
    q.admit("t", cost=5)
    with pytest.raises(GovernanceError):
        q.admit("t", cost=1)
    snap = q.snapshot()
    assert snap["t"]["spent"] == 5
