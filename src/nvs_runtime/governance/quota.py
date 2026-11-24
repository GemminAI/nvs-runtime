"""Tenant quota enforcement."""

from __future__ import annotations

from nvs_runtime.core.errors import GovernanceError


class QuotaEngine:
    """Simple in-process concurrency / cost quotas."""

    def __init__(
        self,
        *,
        max_concurrent: int = 32,
        max_cost_per_tenant: int = 10_000,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._max_cost = max_cost_per_tenant
        self._active: dict[str, int] = {}
        self._spent: dict[str, int] = {}

    def admit(self, tenant_id: str, *, cost: int = 1) -> None:
        active = self._active.get(tenant_id, 0)
        if active >= self._max_concurrent:
            raise GovernanceError(
                f"tenant {tenant_id} concurrent limit reached",
                code="QUOTA_CONCURRENT",
            )
        spent = self._spent.get(tenant_id, 0)
        if spent + cost > self._max_cost:
            raise GovernanceError(
                f"tenant {tenant_id} cost quota exceeded",
                code="QUOTA_COST",
            )
        self._active[tenant_id] = active + 1
        self._spent[tenant_id] = spent + cost

    def release(self, tenant_id: str) -> None:
        active = self._active.get(tenant_id, 0)
        if active > 0:
            self._active[tenant_id] = active - 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        tenants = set(self._active) | set(self._spent)
        return {
            t: {"active": self._active.get(t, 0), "spent": self._spent.get(t, 0)}
            for t in sorted(tenants)
        }
