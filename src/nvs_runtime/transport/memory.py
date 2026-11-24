"""In-memory transport for deterministic offline tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nvs_runtime.core.errors import TransportError

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


class InMemoryTransport:
    """Routes packets to registered endpoint handlers — no network."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, endpoint: str, handler: Handler) -> None:
        self._handlers[endpoint] = handler

    async def send(
        self,
        endpoint: str,
        packet: dict[str, Any],
        *,
        timeout_ms: int = 5_000,
    ) -> dict[str, Any]:
        del timeout_ms  # offline transport ignores wall-clock timeouts
        handler = self._handlers.get(endpoint)
        if handler is None:
            raise TransportError(f"no handler for endpoint: {endpoint}", code="ENDPOINT_NOT_FOUND")
        result = handler(packet)
        if hasattr(result, "__await__"):
            return await result
        return result
