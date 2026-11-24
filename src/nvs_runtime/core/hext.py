"""HEXT interface adapters — runtime never imports HEXT internals."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class HextObject(BaseModel):
    """Minimal Semantic Object conforming to the HEXT interface."""

    object_id: str = Field(default_factory=lambda: f"hext_{uuid4().hex[:16]}")
    object_type: str = "object"
    payload: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class MemoryObjectStore:
    """In-process ObjectStore for offline / deterministic operation."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}

    def put_object(self, data: dict[str, Any], *, labels: dict[str, str] | None = None) -> str:
        obj = HextObject(
            object_type=str(data.get("type") or data.get("object_type") or "object"),
            payload=dict(data),
            labels=dict(labels or {}),
        )
        stored = obj.to_dict()
        self._objects[obj.object_id] = stored
        return obj.object_id

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        return self._objects.get(object_id)

    def list_ids(self) -> list[str]:
        return list(self._objects.keys())
