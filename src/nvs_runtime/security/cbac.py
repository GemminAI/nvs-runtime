"""Capability-Based Access Control — lightweight token checks."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

from nvs_runtime.core.errors import AuthorizationError
from nvs_runtime.core.protocols import Signer


class TokenPayload:
    def __init__(
        self,
        *,
        tenant_id: str,
        scopes: set[str],
        max_cost_units: int,
        exp_ns: int,
    ) -> None:
        self.tenant_id = tenant_id
        self.scopes = scopes
        self.max_cost_units = max_cost_units
        self.exp_ns = exp_ns


class CBACEngine:
    """Verify capability tokens and enforce scopes / budgets."""

    def __init__(self, *, signer: Signer, clock: Callable[[], int]) -> None:
        self._signer = signer
        self._clock = clock

    def issue(
        self,
        *,
        tenant_id: str = "anonymous",
        scopes: set[str] | None = None,
        max_cost_units: int = 1_000,
        ttl_ns: int = 3_600_000_000_000,
    ) -> str:
        payload = {
            "tenant_id": tenant_id,
            "scopes": sorted(scopes or {"scope:invoke", "scope:plan", "scope:session"}),
            "max_cost_units": max_cost_units,
            "exp_ns": self._clock() + ttl_ns,
        }
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        sig = self._signer.sign(body)
        return (
            base64.urlsafe_b64encode(body).decode().rstrip("=")
            + "."
            + base64.urlsafe_b64encode(sig).decode().rstrip("=")
        )

    def verify_token(self, token: str) -> TokenPayload:
        try:
            body_b64, sig_b64 = token.split(".", 1)
            body = _b64decode(body_b64)
            sig = _b64decode(sig_b64)
        except (ValueError, IndexError) as exc:
            raise AuthorizationError("malformed token", code="TOKEN_MALFORMED") from exc
        if not self._signer.verify(body, sig):
            raise AuthorizationError("invalid token signature", code="TOKEN_INVALID")
        data = json.loads(body.decode())
        if int(data["exp_ns"]) < self._clock():
            raise AuthorizationError("token expired", code="TOKEN_EXPIRED")
        return TokenPayload(
            tenant_id=str(data["tenant_id"]),
            scopes=set(data.get("scopes") or []),
            max_cost_units=int(data.get("max_cost_units", 0)),
            exp_ns=int(data["exp_ns"]),
        )

    def require_scopes(self, token: str | None, needed: set[str]) -> TokenPayload | None:
        if token is None:
            return None
        payload = self.verify_token(token)
        missing = needed - payload.scopes
        if missing:
            raise AuthorizationError(
                f"missing scopes: {sorted(missing)}",
                code="SCOPE_DENIED",
            )
        return payload

    def check_budget(self, token: str | None, cost: int) -> None:
        if token is None:
            return
        payload = self.verify_token(token)
        if cost > payload.max_cost_units:
            raise AuthorizationError(
                f"cost {cost} exceeds token budget {payload.max_cost_units}",
                code="BUDGET_EXCEEDED",
            )

    def audit_hash(self, token: str) -> str:
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()[:16]


def _b64decode(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)
