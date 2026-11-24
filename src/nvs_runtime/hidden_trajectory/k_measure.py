"""The entrance that measures route K for one position.

Standard library only. The measurement itself is not here: ``default_k_measurer()`` and
``default_key_source()`` (and ``default_k_checks()``, the checks ``rerun`` runs on a stored K) come
from an installed entry point of the group ``nvs_runtime.k_measurer``. An entry point names a module
or an object with the callables ``default_k_measurer() -> KMeasurer``,
``default_key_source() -> Callable[[], bytes]`` and ``default_k_checks() -> KChecks``. More than one
entry point is an error: which one to use would be a guess.

When none is installed there is no measurer: ``default_k_measurer()`` returns one that records every
position as ``not_measured`` with the reason ``edge_unavailable``, and ``default_key_source()``
returns a plain random key maker.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, cast

from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.record_schema import K_KEYS

K_MEASURER_GROUP = "nvs_runtime.k_measurer"


class KMeasurerEntryPointError(RuntimeError):
    """The entry points of ``nvs_runtime.k_measurer`` cannot be used (more than one, or one that
    lacks ``default_k_measurer`` or ``default_key_source``)."""


class KOutcome(Protocol):
    """The three fields ``run_position`` returns and a save keeps.

    The result of a K measurer has them (it may carry more fields, which the save does not read).
    """

    k_path: dict[str, Any]
    board: bytes | None
    causal: bytes | None


class KMeasurer(Protocol):
    """Measures one position. The arguments are ``run_position``'s, keyword only."""

    def run_position(
        self,
        *,
        observation_id: str,
        point: str,
        raw_bytes: bytes,
        trajectory_id: str | None,
        model_build_id: str | None,
        compute_environment: str | None,
        key_source: Callable[[], bytes],
    ) -> KOutcome: ...


def _provider() -> Any | None:
    """The module or object an installed entry point names, or ``None`` when there is none."""
    found = list(metadata.entry_points(group=K_MEASURER_GROUP))
    if not found:
        return None
    if len(found) > 1:
        names = ", ".join(sorted(str(item.name) for item in found))
        raise KMeasurerEntryPointError(
            f"more than one entry point in the group {K_MEASURER_GROUP!r}: {names}"
        )
    return found[0].load()


def _callable_of(provider: Any, name: str) -> Callable[[], Any]:
    found = getattr(provider, name, None)
    if not callable(found):
        raise KMeasurerEntryPointError(
            f"the entry point of {K_MEASURER_GROUP!r} has no callable {name!r}"
        )
    return cast(Callable[[], Any], found)


def _random_key() -> bytes:
    return secrets.token_bytes(32)


def default_key_source() -> Callable[[], bytes]:
    """The key maker of the installed ``nvs_runtime.k_measurer`` entry point. With none installed,
    a plain random 32-byte key maker (nothing is measured then, so no key is used)."""
    provider = _provider()
    if provider is not None:
        return cast(Callable[[], bytes], _callable_of(provider, "default_key_source")())
    return _random_key


@dataclass
class _NotMeasured:
    """The outcome of a position when there is no measurer (``not_measured``)."""

    k_path: dict[str, Any]
    board: bytes | None = None
    causal: bytes | None = None


class _UnavailableKMeasurer:
    """The measurer when no ``nvs_runtime.k_measurer`` entry point is installed."""

    def run_position(
        self,
        *,
        observation_id: str,
        point: str,
        raw_bytes: bytes,
        trajectory_id: str | None,
        model_build_id: str | None,
        compute_environment: str | None,
        key_source: Callable[[], bytes],
    ) -> KOutcome:
        entry: dict[str, Any] = {key: None for key in K_KEYS}
        entry.update(
            status=rec.K_NOT_MEASURED,
            reason=rec.REASON_K_EDGE_UNAVAILABLE,
            trajectory_id=trajectory_id,
            compute_environment=compute_environment,
        )
        return _NotMeasured(entry)


def default_k_measurer() -> KMeasurer:
    """The measurer a save uses when none was passed: that of the installed
    ``nvs_runtime.k_measurer`` entry point, else one that records ``not_measured`` /
    ``edge_unavailable``."""
    provider = _provider()
    if provider is not None:
        return cast(KMeasurer, _callable_of(provider, "default_k_measurer")())
    return _UnavailableKMeasurer()


def default_k_checks() -> Any | None:
    """The ``KChecks`` of the installed ``nvs_runtime.k_measurer`` entry point (what ``rerun`` runs
    on a stored K), or ``None`` when no entry point is installed."""
    provider = _provider()
    if provider is None:
        return None
    return _callable_of(provider, "default_k_checks")()
