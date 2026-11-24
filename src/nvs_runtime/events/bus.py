"""In-process event bus with optional transport fan-out."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nvs_runtime.events.types import EventType, RuntimeEvent


class EventBus:
    """Pub/sub bus. Transports attach as subscribers — never embedded."""

    def __init__(self, *, clock: Callable[[], int]) -> None:
        self._clock = clock
        self._subscribers: list[Callable[[RuntimeEvent], None]] = []
        self._history: list[RuntimeEvent] = []
        self._max_history = 1_000

    def subscribe(self, handler: Callable[[RuntimeEvent], None]) -> None:
        self._subscribers.append(handler)

    def publish(
        self,
        event_type: EventType | str,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        et = event_type if isinstance(event_type, EventType) else EventType(event_type)
        event = RuntimeEvent(
            type=et,
            timestamp_ns=self._clock(),
            payload=dict(payload or {}),
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        for handler in list(self._subscribers):
            handler(event)
        return event

    def history(
        self,
        *,
        limit: int = 50,
        event_type: EventType | None = None,
    ) -> list[dict[str, Any]]:
        items = self._history
        if event_type is not None:
            items = [e for e in items if e.type == event_type]
        return [e.to_dict() for e in items[-limit:]]
