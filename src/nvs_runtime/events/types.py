"""Runtime event types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(StrEnum):
    PROVIDER_REGISTERED = "provider.registered"
    PROVIDER_REMOVED = "provider.removed"
    CAPABILITY_DISCOVERED = "capability.discovered"
    PLAN_CREATED = "plan.created"
    INVOKE_STARTED = "invoke.started"
    INVOKE_COMPLETED = "invoke.completed"
    INVOKE_FAILED = "invoke.failed"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_RESTORED = "checkpoint.restored"
    TOPOLOGY_CHANGED = "topology.changed"
    RUNTIME_SHUTDOWN = "runtime.shutdown"


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    type: EventType
    timestamp_ns: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
