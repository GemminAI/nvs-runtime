"""Checkpoint manager — persists session state via ObjectStore protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from nvs_runtime.core.errors import CheckpointError
from nvs_runtime.core.models import CheckpointAction, CheckpointRequest
from nvs_runtime.core.protocols import ObjectStore


class CheckpointManager:
    """Save / restore checkpoints without depending on HEKB internals."""

    def __init__(self, *, store: ObjectStore, clock: Callable[[], int]) -> None:
        self._store = store
        self._clock = clock
        self._index: dict[str, dict[str, Any]] = {}

    def handle(self, request: CheckpointRequest) -> dict[str, Any]:
        if request.action == CheckpointAction.SAVE:
            return self.save(
                session_id=request.session_id,
                state=request.state,
                label=request.label,
            )
        if request.action == CheckpointAction.LIST:
            return {"checkpoints": self.list(session_id=request.session_id)}
        if request.action == CheckpointAction.GET:
            if not request.checkpoint_id:
                raise CheckpointError("checkpoint_id required", code="CHECKPOINT_ID_REQUIRED")
            return self.get(request.checkpoint_id)
        if request.action == CheckpointAction.RESTORE:
            if not request.checkpoint_id:
                raise CheckpointError("checkpoint_id required", code="CHECKPOINT_ID_REQUIRED")
            return self.restore(request.checkpoint_id)
        raise CheckpointError(f"unknown action: {request.action}", code="CHECKPOINT_BAD_ACTION")

    def save(
        self,
        *,
        session_id: str | None,
        state: dict[str, Any],
        label: str | None = None,
    ) -> dict[str, Any]:
        checkpoint_id = f"ckpt_{uuid4().hex[:12]}"
        record = {
            "type": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "label": label or "default",
            "state": dict(state),
            "created_ns": self._clock(),
        }
        object_id = self._store.put_object(
            record,
            labels={
                "kind": "checkpoint",
                "checkpoint_id": checkpoint_id,
                "session_id": session_id or "",
            },
        )
        record["object_id"] = object_id
        self._index[checkpoint_id] = record
        return {
            "checkpoint_id": checkpoint_id,
            "object_id": object_id,
            "session_id": session_id,
            "label": record["label"],
        }

    def get(self, checkpoint_id: str) -> dict[str, Any]:
        record = self._index.get(checkpoint_id)
        if record is None:
            raise CheckpointError(
                f"unknown checkpoint: {checkpoint_id}",
                code="CHECKPOINT_NOT_FOUND",
            )
        return dict(record)

    def restore(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.get(checkpoint_id)
        stored = None
        object_id = record.get("object_id")
        if object_id:
            stored = self._store.get_object(str(object_id))
        return {
            "checkpoint_id": checkpoint_id,
            "session_id": record.get("session_id"),
            "state": (stored or record).get("state") or record.get("state") or {},
            "restored": True,
        }

    def list(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        items = list(self._index.values())
        if session_id is not None:
            items = [i for i in items if i.get("session_id") == session_id]
        return [
            {
                "checkpoint_id": i["checkpoint_id"],
                "session_id": i.get("session_id"),
                "label": i.get("label"),
                "created_ns": i.get("created_ns"),
            }
            for i in items
        ]
