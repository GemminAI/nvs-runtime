"""Edge provider — wraps the ``Edge`` of ``nvs_edge`` (one observation per token) through the
provider boundary only (never a local reimplementation of the measurement, matching
``KernelProvider``'s own "DEFERRED, no local geometry reimplementation" rule).

Two ways to reach the kernel: the one at ``base_url`` over HTTP (research, tests), or, with
``in_process=True``, the kernel inside this process (no HTTP server is involved).

Unavailable two ways, both surfaced as ``DEFERRED`` rather than raised:
``nvs_edge`` not installed (dev/editable install missing), or the kernel at
``base_url`` unreachable (``nvs_edge.edge.EdgeTransportError``). Neither case fabricates
a substitute board record or causal log.
"""

from __future__ import annotations

from typing import Any

from nvs_observation.edge_observe import EdgeObserver

from nvs_runtime.core.errors import CapabilityNotFoundError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider

try:
    from nvs_edge.edge import Edge as _Edge
    from nvs_edge.edge import EdgeTransportError as _EdgeTransportError
    from nvs_edge.raw_store import RawCapture as _RawCapture

    _NVS_EDGE_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised by test_edge_provider.py directly
    _Edge = None
    _EdgeTransportError = Exception
    _RawCapture = None
    _NVS_EDGE_AVAILABLE = False


class EdgeProvider(BaseProvider):
    """One ``nvs_edge.Edge`` per ``session_id`` (created lazily on that session's first
    token). ``invoke("edge.observe", {"session_id", "token_index", "hidden_state"})``
    returns ``{"type": "board_record", "status": "OK", "record": BoardRecord.to_wire(),
    "causal_log": <the causal-log entry of that record>}``, or a ``DEFERRED`` result
    (no exception) when the kernel is unavailable.

    Each session's ``Edge`` signs its records with a random key unless one is given with
    ``set_session_key`` before that session's first token. A session that submits to the CDM must
    be given the session key the CDM issued for it: the CDM verifies every record's HMAC with it,
    and a record signed with any other key is rejected."""

    kind = "edge"

    def __init__(
        self,
        provider_id: str = "edge",
        *,
        base_url: str | None = None,
        anchor_policy_sha256: str | None = None,
        experiment_id: str = "EDGE-HEKB-SLICE",
        transport: Any = None,
        wiring_policy: str | None = None,
        capture_raw: bool = False,
        in_process: bool = False,
    ) -> None:
        super().__init__(provider_id)
        self._base_url = base_url
        #: ``True``: no ``base_url``; each Edge measures with the kernel in this process.
        self._in_process = in_process
        self._anchor_policy_sha256 = anchor_policy_sha256
        self._experiment_id = experiment_id
        self._transport = transport
        #: ``None`` keeps nvs_edge's own default policy.
        self._wiring_policy = wiring_policy
        #: ``True``: each session's Edge takes the 4096-D state before P into a ``RawCapture``
        #: (``raw_capture_for``). ``False`` (default) is the behaviour without RAW.
        self._capture_raw = capture_raw
        self._obs = EdgeObserver()
        # The same objects the observation holds, so a subclass that writes ``self._edges``
        # is writing the dict ``observe`` reads.
        self._raw_captures = self._obs.raw_captures
        self._session_keys = self._obs.session_keys
        self._session_tiers = self._obs.session_tiers
        self._session_tier_tokens = self._obs.session_tier_tokens
        self._edges = self._obs.edges
        self._unreachable_sessions = self._obs.unreachable
        self.register_capability(
            Capability(
                id="edge.observe",
                name="Edge: Measure One Token",
                provider_id=self.provider_id,
                input_type="hidden_state",
                output_type="board_record",
                tags=["edge", "kernel"],
                cost_units=5,
                latency_ms=10,
            )
        )

    @property
    def available(self) -> bool:
        return _NVS_EDGE_AVAILABLE and (bool(self._base_url) or self._in_process)

    def set_session_key(self, session_id: str, key: bytes) -> None:
        """Sign ``session_id``'s records with ``key`` (the session key the CDM issued, as
        bytes). Call it before the session's first token: an
        ``Edge`` that already exists keeps the key it was made with, so this raises ``ValueError``
        then. Without a call the session gets a random key, as before."""
        self._obs.set_session_key(session_id, key)

    def set_session_tier(self, session_id: str, tier: str) -> None:
        """Tell ``session_id``'s ``Edge`` which tier it works under (as the CDM
        issued it). Under ``"basic"`` the result's ``causal_log`` is ``{}``; ``"pro"`` returns it.
        Call it before the session's first token (``ValueError`` after). Without a call the session
        is not tiered and returns the log, as before."""
        self._obs.set_session_tier(session_id, tier)

    def set_session_tier_token(self, session_id: str, tier_token: str | None) -> None:
        """Give ``session_id``'s ``Edge`` the signed tier token the CDM issued.

        The Edge verifies the token and takes the tier from it alone (a missing or unverifiable
        token means ``basic``). Call it before the session's first token (``ValueError`` after)."""
        self._obs.set_session_tier_token(session_id, tier_token)

    def session_tier(self, session_id: str) -> str | None:
        """The tier the session's ``Edge`` works under (verified), None before its first token or
        when the Edge is not tiered."""
        return self._obs.session_tier(session_id)

    def _edge_for(self, session_id: str) -> Any:
        return self._obs.edge_for(
            session_id,
            base_url=self._base_url,
            anchor_policy_sha256=self._anchor_policy_sha256,
            experiment_id=self._experiment_id,
            transport=self._transport,
            wiring_policy=self._wiring_policy,
            capture_raw=self._capture_raw,
        )

    def raw_capture_for(self, session_id: str) -> Any:
        """The session's ``nvs_edge.raw_store.RawCapture`` (``states()``, ``finalize()``,
        ``edge_input_cast_exact``), or ``None`` if ``capture_raw`` is off or the session has not
        observed a token yet."""
        return self._obs.raw_capture_for(session_id)

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if capability_id != "edge.observe":
            raise CapabilityNotFoundError(capability_id)
        return self._obs.observe(
            session_id=str(payload.get("session_id") or ""),
            token_index=int(payload.get("token_index", -1)),
            hidden_state=payload.get("hidden_state"),
            available=_NVS_EDGE_AVAILABLE,
            base_url=self._base_url,
            in_process=self._in_process,
            edge_for=self._edge_for,
        )

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        base["nvs_edge_available"] = _NVS_EDGE_AVAILABLE
        base["base_url"] = self._base_url
        base["in_process"] = self._in_process
        return base


__all__ = ["EdgeProvider"]
