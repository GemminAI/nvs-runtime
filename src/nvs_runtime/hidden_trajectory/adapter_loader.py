"""Load an observation Adapter by module and class name (src never imports ``sensos_inference``).

The catalog gives the strings. Construction follows the real-machine tests' ``_load``: a capture
provider from the snapshot directory, then the Adapter. ``importlib`` is the only way the Adapter
package is reached; a missing ``sensos_inference`` is a clear error, not a traceback.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_MISSING_MESSAGE = (
    "sensos-inference が導入されていません。"
    "PYTHONPATH に sensos-inference の src を足すか、"
    "パッケージ sensos-inference を導入してください。"
)


class AdapterLoadError(RuntimeError):
    """The Adapter package could not be imported, or the snapshot could not be turned into one."""


def _import(name: str, import_module: Callable[[str], Any]) -> Any:
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        missing = getattr(error, "name", None) or str(error)
        if "sensos_inference" in str(missing) or name.startswith("sensos_inference"):
            raise AdapterLoadError(_MISSING_MESSAGE) from error
        raise AdapterLoadError(f"could not import {name}: {error}") from error


def load_adapter(
    spec: Mapping[str, Any],
    snapshot: Path,
    *,
    hash_cache_dir: Path | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Build the Adapter named by ``spec`` from ``snapshot``.

    ``import_module`` is a seam for the tests: they pass a fake name or a function that raises
    ``ModuleNotFoundError``.
    """
    adapter_mod = _import(str(spec["adapter_module"]), import_module)
    provider_mod = _import(str(spec["provider_module"]), import_module)
    provider_cls = getattr(provider_mod, str(spec["provider_class"]))
    adapter_cls = getattr(adapter_mod, str(spec["adapter_class"]))
    cache = hash_cache_dir
    if cache is None:
        env = os.environ.get("U2_HASH_CACHE_DIR")
        cache = Path(env) if env else None
    kwargs: dict[str, Any] = {"snapshot_dir": snapshot, "hash_cache_dir": cache}
    model_id = spec.get("provider_model_id")
    provider = provider_cls(model_id, **kwargs) if model_id else provider_cls(**kwargs)
    provider._ensure_loaded_from(provider._load_target())
    if spec.get("needs_config"):
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        return adapter_cls(
            provider,
            n_layers=int(text["num_hidden_layers"]),
            dimension=int(provider.hidden_size() or 0),
        )
    return adapter_cls(provider)
