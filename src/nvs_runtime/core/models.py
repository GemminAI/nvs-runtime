"""Domain models for the stable Runtime API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Placement(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"


class Capability(BaseModel):
    """A discoverable unit of work exposed by a provider."""

    id: str
    name: str
    provider_id: str
    input_type: str = "any"
    output_type: str = "any"
    tags: list[str] = Field(default_factory=list)
    cost_units: int = 1
    latency_ms: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoverQuery(BaseModel):
    provider_id: str | None = None
    tag: str | None = None
    input_type: str | None = None
    output_type: str | None = None


class DiscoverResult(BaseModel):
    capabilities: list[Capability] = Field(default_factory=list)
    providers: list[dict[str, Any]] = Field(default_factory=list)


class PlanRequest(BaseModel):
    goal: str
    input_type: str = "any"
    target_outputs: list[str] = Field(default_factory=list)
    optimization: str = "LOW_LATENCY"
    max_cost_units: int = 100
    max_latency_ms: int = 200
    allow_remote: bool = True
    constraints: dict[str, Any] = Field(default_factory=dict)
    token: str | None = None


class PlanStep(BaseModel):
    capability_id: str
    provider_id: str
    input_type: str
    output_type: str
    estimated_latency_ms: int = 1
    estimated_cost_units: int = 1


class ExecutionPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")
    goal: str
    executable: bool = True
    steps: list[PlanStep] = Field(default_factory=list)
    placement: Placement = Placement.LOCAL
    estimated_latency_ms: int = 0
    estimated_cost_units: int = 0
    error_code: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class InvokeRequest(BaseModel):
    capability_id: str | None = None
    provider_id: str | None = None
    plan_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    token: str | None = None


class InvokeResult(BaseModel):
    ok: bool = True
    capability_id: str | None = None
    provider_id: str | None = None
    session_id: str | None = None
    plan_id: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    detail: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ObserveQuery(BaseModel):
    kind: str = "timeline"  # timeline | topology | statistics | sessions
    limit: int = 50
    session_id: str | None = None
    plan_id: str | None = None


class ObserveResult(BaseModel):
    kind: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SessionAction(StrEnum):
    CREATE = "create"
    GET = "get"
    LIST = "list"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    COMPLETE = "complete"


class SessionRequest(BaseModel):
    action: SessionAction = SessionAction.CREATE
    session_id: str | None = None
    plan_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class CheckpointAction(StrEnum):
    SAVE = "save"
    RESTORE = "restore"
    LIST = "list"
    GET = "get"


class CheckpointRequest(BaseModel):
    action: CheckpointAction = CheckpointAction.SAVE
    session_id: str | None = None
    checkpoint_id: str | None = None
    label: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
