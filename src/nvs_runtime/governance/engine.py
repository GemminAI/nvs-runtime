"""Governance — plan admission, cancel, and durable plan store."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import GovernanceError
from nvs_runtime.core.models import ExecutionPlan
from nvs_runtime.core.protocols import ObjectStore
from nvs_runtime.governance.quota import QuotaEngine


class GovernanceEngine:
    """Owns execution plans and admission decisions."""

    def __init__(self, *, store: ObjectStore, quota: QuotaEngine | None = None) -> None:
        self._store = store
        self._quota = quota or QuotaEngine()
        self._plans: dict[str, ExecutionPlan] = {}
        self._cancelled: set[str] = set()

    def store_plan(self, plan: ExecutionPlan) -> str:
        self._plans[plan.plan_id] = plan
        return self._store.put_object(
            plan.to_dict(),
            labels={"kind": "execution_plan", "plan_id": plan.plan_id},
        )

    def get_plan(self, plan_id: str) -> ExecutionPlan:
        try:
            return self._plans[plan_id]
        except KeyError as exc:
            raise GovernanceError(f"unknown plan: {plan_id}", code="PLAN_NOT_FOUND") from exc

    def admit(self, *, tenant_id: str, plan: ExecutionPlan) -> None:
        if not plan.executable:
            raise GovernanceError("plan is not executable", code="PLAN_NOT_EXECUTABLE")
        if plan.plan_id in self._cancelled:
            raise GovernanceError("plan was cancelled", code="PLAN_CANCELLED")
        self._quota.admit(tenant_id, cost=plan.estimated_cost_units)

    def release(self, tenant_id: str) -> None:
        self._quota.release(tenant_id)

    def cancel_plan(self, plan_id: str, *, reason: str = "cancelled") -> dict[str, Any]:
        plan = self.get_plan(plan_id)
        self._cancelled.add(plan_id)
        return {"plan_id": plan.plan_id, "cancelled": True, "reason": reason}

    def is_cancelled(self, plan_id: str) -> bool:
        return plan_id in self._cancelled

    def list_plans(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._plans.values()]

    @property
    def quota(self) -> QuotaEngine:
        return self._quota
