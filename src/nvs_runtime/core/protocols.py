"""Core protocols — interfaces over inheritance."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SemanticObject(Protocol):
    """HEXT Semantic Object Standard — runtime depends only on this interface."""

    @property
    def object_id(self) -> str: ...

    @property
    def object_type(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class ObjectStore(Protocol):
    """Content-addressed object store (HEKB-shaped) — adapter boundary only."""

    def put_object(self, data: dict[str, Any], *, labels: dict[str, str] | None = None) -> str: ...

    def get_object(self, object_id: str) -> dict[str, Any] | None: ...


@runtime_checkable
class Provider(Protocol):
    """Uniform provider contract. No provider is privileged."""

    @property
    def provider_id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def discover(self) -> list[dict[str, Any]]: ...

    def capabilities(self) -> list[dict[str, Any]]: ...

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class Transport(Protocol):
    """Swappable message transport."""

    async def send(
        self,
        endpoint: str,
        packet: dict[str, Any],
        *,
        timeout_ms: int = 5_000,
    ) -> dict[str, Any]: ...


@runtime_checkable
class Clock(Protocol):
    """Injectable clock — nanoseconds since epoch for deterministic tests."""

    def __call__(self) -> int: ...


@runtime_checkable
class Signer(Protocol):
    """Capability token signer — HMAC for tests, Ed25519 for production."""

    def sign(self, payload: bytes) -> bytes: ...

    def verify(self, payload: bytes, signature: bytes) -> bool: ...
