"""Runtime error hierarchy."""

from __future__ import annotations


class RuntimeError(Exception):
    """Base error for NVS Runtime."""

    def __init__(self, message: str, *, code: str = "RUNTIME_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ProviderError(RuntimeError):
    """Provider invocation or lifecycle failure."""

    def __init__(self, message: str, *, code: str = "PROVIDER_ERROR") -> None:
        super().__init__(message, code=code)


class ProviderNotFoundError(ProviderError):
    def __init__(self, provider_id: str) -> None:
        super().__init__(f"unknown provider: {provider_id}", code="PROVIDER_NOT_FOUND")
        self.provider_id = provider_id


class CapabilityNotFoundError(RuntimeError):
    def __init__(self, capability_id: str) -> None:
        super().__init__(
            f"unknown capability: {capability_id}",
            code="CAPABILITY_NOT_FOUND",
        )
        self.capability_id = capability_id


class PlanError(RuntimeError):
    def __init__(self, message: str, *, code: str = "PLAN_ERROR") -> None:
        super().__init__(message, code=code)


class GovernanceError(RuntimeError):
    def __init__(self, message: str, *, code: str = "GOVERNANCE_ERROR") -> None:
        super().__init__(message, code=code)


class SessionError(RuntimeError):
    def __init__(self, message: str, *, code: str = "SESSION_ERROR") -> None:
        super().__init__(message, code=code)


class CheckpointError(RuntimeError):
    def __init__(self, message: str, *, code: str = "CHECKPOINT_ERROR") -> None:
        super().__init__(message, code=code)


class AuthorizationError(RuntimeError):
    def __init__(self, message: str, *, code: str = "UNAUTHORIZED") -> None:
        super().__init__(message, code=code)


class TransportError(RuntimeError):
    def __init__(self, message: str, *, code: str = "TRANSPORT_ERROR") -> None:
        super().__init__(message, code=code)


class ShutdownError(RuntimeError):
    def __init__(self, message: str = "runtime is shut down") -> None:
        super().__init__(message, code="SHUTDOWN")
