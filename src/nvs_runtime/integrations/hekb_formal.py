"""Open a hekb-vnext Formal Store for the save (U6T-2b').

nvs-runtime reaches hekb-vnext the way ``hekb_vnext.py`` does: lazily, from a checkout
(``hekb_vnext_root``), by name. Nothing
here imports at module load, and nothing of sensos is imported (by design, no cross-imports):
the Formal Store is a
hekb-vnext object, and nvs-runtime
only calls ``put(envelope, blobs, *, gate_verdict, checker_bundle)``, ``get`` and ``gate_status`` on
it (``formal_save.py``).
Only ``FormalWriter`` (U6T-2b-b) opens a Formal Store, once, with ``require_formal=True``; the names
here are internal.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class HekbFormalUnavailable(RuntimeError):
    """The hekb-vnext checkout has no ``graph.ic04`` with a Formal Store."""


#: The top-level packages of a hekb-vnext checkout that the Formal Store reads. Two checkouts (or
#: worktrees) have the same names, so
#: one already loaded from another place has to be dropped before the one asked for is imported.
_PACKAGES = ("graph", "vendor")


def _activate(hekb_vnext_root: Path | str) -> str:
    """Put the checkout first on ``sys.path`` and drop ``graph`` / ``vendor`` modules loaded from
    any other place."""
    root = Path(hekb_vnext_root).resolve()
    for name in list(sys.modules):
        if name.split(".")[0] not in _PACKAGES:
            continue
        file = getattr(sys.modules[name], "__file__", None) or ""
        paths = [str(p) for p in getattr(sys.modules[name], "__path__", [])]
        places = [file, *paths]
        if places and not any(Path(p).resolve().is_relative_to(root) for p in places if p):
            del sys.modules[name]
    text = str(root)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    return text


def _has_regular_package(entry: str) -> bool:
    try:
        return any((Path(entry) / name / "__init__.py").is_file() for name in _PACKAGES)
    except (OSError, ValueError):
        return False


def _load_hekb_module(hekb_vnext_root: Path | str, name: str) -> Any:
    """A module of the hekb-vnext checkout by name (``graph.formal``, ``graph.chainlog``...).

    Another checkout whose ``graph`` has an ``__init__.py`` would win over this one's namespace
    ``graph`` while it is on ``sys.path``; so, for the import only, the other entries that hold such
    a package are left out."""
    root = _activate(hekb_vnext_root)
    saved = list(sys.path)
    sys.path[:] = [root, *(p for p in saved if p != root and not _has_regular_package(p))]
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise HekbFormalUnavailable(
            f"cannot import {name} from the hekb-vnext checkout given: {error}"
        ) from error
    finally:
        sys.path[:] = [root, *(p for p in saved if p != root)]


def _store_class(hekb_vnext_root: Path | str) -> Any:
    try:
        module = _load_hekb_module(hekb_vnext_root, "graph.ic04")
        cls = module.IC04Store
        if not hasattr(cls, "create_formal"):
            raise AttributeError("IC04Store.create_formal")
    except (HekbFormalUnavailable, AttributeError) as error:
        raise HekbFormalUnavailable(
            f"no Formal Store in the hekb-vnext checkout given: {error}"
        ) from error
    return cls


def _create_formal_store(
    hekb_vnext_root: Path | str, data_dir: Path | str, declaration: Mapping[str, Any]
) -> Any:
    """``IC04Store.create_formal``: a new Formal Store in an empty folder, opened with
    ``require_formal=True``."""
    return _store_class(hekb_vnext_root).create_formal(data_dir, declaration)


def _open_formal_store(hekb_vnext_root: Path | str, data_dir: Path | str) -> Any:
    """``IC04Store(data_dir, require_formal=True)``: an existing Formal Store (it raises
    ``IC04Error`` if it is not one)."""
    return _store_class(hekb_vnext_root)(data_dir, require_formal=True)


# ------------------------------------------------------------
# One hekb-vnext checkout per process (U6T-2 section 15, condition 2). ``graph`` / ``vendor`` are
# top-level names: two checkouts cannot both be loaded, and swapping one under an open Formal Store
# would change the classes it runs on. ``FormalWriter`` claims the checkout it was given before it
# opens a store and releases it on close.
# ------------------------------------------------------------
class CheckoutConflict(RuntimeError):
    """Another hekb-vnext checkout is in use in this process."""


_WATCHED = ("graph", "vendor", "storage")  # ``storage`` is what the older hekb_vnext.py imports
_claim_lock = threading.Lock()
_claimed_root: str | None = (
    None  # the checkout this module loaded from (kept after the last release)
)
_claims = 0  # open writers using it


def _module_places(module: Any) -> list[str]:
    places = [getattr(module, "__file__", None) or "", *map(str, getattr(module, "__path__", []))]
    return [p for p in places if p]


def _under(path: str, root: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _claim_checkout(hekb_vnext_root: Path | str) -> str:
    """Claim the checkout for one writer; ``CheckoutConflict`` if another one is in use here, either
    by an open writer or because ``graph`` / ``vendor`` / ``storage`` of another checkout is
    already loaded (for example by the older ``hekb_vnext.py``)."""
    global _claimed_root, _claims
    root = os.path.realpath(str(hekb_vnext_root))
    with _claim_lock:
        if _claims and _claimed_root != root:
            raise CheckoutConflict(
                f"a writer already uses the hekb-vnext checkout {_claimed_root}; this one is {root}"
            )
        for name in _WATCHED:
            module = sys.modules.get(name)
            if module is None:
                continue
            places = _module_places(module)
            ours = bool(_claimed_root) and not _claims  # an earlier writer's, all closed: replaced
            # a namespace package can list several checkouts; it is ours if one of them is
            mine = any(_under(p, root) or (ours and _under(p, _claimed_root or "")) for p in places)
            if places and not mine:
                raise CheckoutConflict(
                    f"the module {name} is already loaded from {places[0]}, not from the "
                    f"checkout {root}"
                )
        _claimed_root = root
        _claims += 1
    return root


def _release_checkout() -> None:
    global _claims
    with _claim_lock:
        _claims = max(0, _claims - 1)
