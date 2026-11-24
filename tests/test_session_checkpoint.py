"""Session lifecycle and checkpoints."""

from __future__ import annotations

import pytest

from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.core.errors import CheckpointError, SessionError
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.models import (
    CheckpointAction,
    CheckpointRequest,
    SessionAction,
    SessionRequest,
)
from nvs_runtime.session.checkpoint import CheckpointManager
from nvs_runtime.session.manager import SessionManager, SessionState


def test_session_lifecycle() -> None:
    clock = ControllableClock()
    sm = SessionManager(clock=clock)
    created = sm.handle(SessionRequest(action=SessionAction.CREATE, plan_id="plan_1"))
    sid = created["session_id"]
    assert created["state"] == SessionState.CREATED

    started = sm.handle(SessionRequest(action=SessionAction.START, session_id=sid))
    assert started["state"] == SessionState.RUNNING

    paused = sm.handle(SessionRequest(action=SessionAction.PAUSE, session_id=sid))
    assert paused["state"] == SessionState.PAUSED

    resumed = sm.handle(SessionRequest(action=SessionAction.RESUME, session_id=sid))
    assert resumed["state"] == SessionState.RUNNING

    completed = sm.handle(SessionRequest(action=SessionAction.COMPLETE, session_id=sid))
    assert completed["state"] == SessionState.COMPLETED
    assert completed["progress"] == 1.0

    listed = sm.handle(SessionRequest(action=SessionAction.LIST))
    assert len(listed["sessions"]) == 1


def test_session_cancel_and_errors() -> None:
    clock = ControllableClock()
    sm = SessionManager(clock=clock)
    sid = sm.create().session_id
    cancelled = sm.cancel(sid, reason="user")
    assert cancelled["state"] == SessionState.CANCELLED
    assert cancelled["detail"]["cancel_reason"] == "user"

    with pytest.raises(SessionError):
        sm.start(sid)
    with pytest.raises(SessionError):
        sm.handle(SessionRequest(action=SessionAction.GET))


def test_checkpoint_roundtrip() -> None:
    clock = ControllableClock()
    store = MemoryObjectStore()
    cm = CheckpointManager(store=store, clock=clock)
    saved = cm.handle(
        CheckpointRequest(
            action=CheckpointAction.SAVE,
            session_id="sess_1",
            state={"x": 1},
            label="l1",
        )
    )
    listed = cm.handle(CheckpointRequest(action=CheckpointAction.LIST, session_id="sess_1"))
    assert len(listed["checkpoints"]) == 1
    got = cm.handle(
        CheckpointRequest(action=CheckpointAction.GET, checkpoint_id=saved["checkpoint_id"])
    )
    assert got["state"]["x"] == 1
    restored = cm.handle(
        CheckpointRequest(action=CheckpointAction.RESTORE, checkpoint_id=saved["checkpoint_id"])
    )
    assert restored["state"]["x"] == 1

    with pytest.raises(CheckpointError):
        cm.handle(CheckpointRequest(action=CheckpointAction.RESTORE))
