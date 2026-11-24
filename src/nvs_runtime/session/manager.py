"""Runtime sessions — lifecycle orthogonal to providers."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from nvs_runtime.core.errors import SessionError
from nvs_runtime.core.models import SessionAction, SessionRequest


class SessionState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class RuntimeSession(BaseModel):
    session_id: str
    plan_id: str | None = None
    state: SessionState = SessionState.CREATED
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    checkpoint_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_ns: int = 0
    updated_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SessionManager:
    """In-memory session table — DI-injected, no globals."""

    def __init__(self, *, clock: Callable[[], int]) -> None:
        self._clock = clock
        self._sessions: dict[str, RuntimeSession] = {}

    def handle(self, request: SessionRequest) -> dict[str, Any]:
        action = request.action
        if action == SessionAction.CREATE:
            return self.create(plan_id=request.plan_id, detail=request.detail).to_dict()
        if action == SessionAction.LIST:
            return {"sessions": self.list_sessions()}
        if request.session_id is None:
            raise SessionError("session_id required", code="SESSION_ID_REQUIRED")
        if action == SessionAction.GET:
            return self.get(request.session_id).to_dict()
        if action == SessionAction.START:
            return self.start(request.session_id)
        if action == SessionAction.PAUSE:
            return self.pause(request.session_id)
        if action == SessionAction.RESUME:
            return self.resume(request.session_id)
        if action == SessionAction.CANCEL:
            return self.cancel(request.session_id)
        if action == SessionAction.COMPLETE:
            return self.complete(request.session_id)
        raise SessionError(f"unknown action: {action}", code="SESSION_BAD_ACTION")

    def create(
        self,
        *,
        plan_id: str | None = None,
        total_steps: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> RuntimeSession:
        now = self._clock()
        session = RuntimeSession(
            session_id=f"sess_{uuid4().hex[:12]}",
            plan_id=plan_id,
            state=SessionState.CREATED,
            total_steps=total_steps,
            detail=dict(detail or {}),
            created_ns=now,
            updated_ns=now,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> RuntimeSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionError(f"unknown session: {session_id}", code="SESSION_NOT_FOUND") from exc

    def list_sessions(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def start(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session.state in {SessionState.CANCELLED, SessionState.COMPLETED}:
            raise SessionError(f"cannot start session in state {session.state}")
        session.state = SessionState.RUNNING
        session.updated_ns = self._clock()
        return session.to_dict()

    def pause(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session.state not in {SessionState.RUNNING, SessionState.CREATED}:
            raise SessionError(f"cannot pause session in state {session.state}")
        session.state = SessionState.PAUSED
        session.updated_ns = self._clock()
        return session.to_dict()

    def resume(self, session_id: str, *, checkpoint_id: str | None = None) -> dict[str, Any]:
        session = self.get(session_id)
        if session.state in {SessionState.CANCELLED, SessionState.COMPLETED}:
            raise SessionError(f"cannot resume session in state {session.state}")
        session.state = SessionState.RUNNING
        if checkpoint_id:
            session.checkpoint_id = checkpoint_id
        session.updated_ns = self._clock()
        return session.to_dict()

    def cancel(self, session_id: str, *, reason: str = "CANCELLED") -> dict[str, Any]:
        session = self.get(session_id)
        session.state = SessionState.CANCELLED
        session.detail["cancel_reason"] = reason
        session.updated_ns = self._clock()
        return session.to_dict()

    def complete(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session.state = SessionState.COMPLETED
        session.progress = 1.0
        session.updated_ns = self._clock()
        return session.to_dict()

    def advance(self, session_id: str, *, step: int | None = None) -> RuntimeSession:
        session = self.get(session_id)
        if session.state != SessionState.RUNNING:
            session.state = SessionState.RUNNING
        if step is not None:
            session.current_step = step
        else:
            session.current_step += 1
        if session.total_steps > 0:
            session.progress = min(1.0, session.current_step / session.total_steps)
        session.updated_ns = self._clock()
        return session
