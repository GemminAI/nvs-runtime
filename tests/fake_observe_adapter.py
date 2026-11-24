"""A stand-in Adapter package for the observe-CLI tests. No mlx and no sensos_inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from u65_support import FakeCapture, make_capture, make_facts


class LlamaProvider:
    def __init__(
        self,
        snapshot_dir: Path | None = None,
        hash_cache_dir: Path | None = None,
        **_kwargs: Any,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.hash_cache_dir = hash_cache_dir
        self.loaded_from: str | None = None

    def _load_target(self) -> str:
        return str(self.snapshot_dir) if self.snapshot_dir is not None else "missing"

    def _ensure_loaded_from(self, target: str) -> None:
        self.loaded_from = target


class QwenProvider(LlamaProvider):
    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id

    def hidden_size(self) -> int:
        return 8


class LlamaAdapter:
    def __init__(self, provider: Any, **_kwargs: Any) -> None:
        self.provider = provider
        self._last_capture: Any = None
        self._capture: FakeCapture = make_capture(kind="llama")
        self._request: Any = None
        self._messages: Any = None

    def capture(self, messages: Any, request: Any, **_kwargs: Any) -> FakeCapture:
        self._messages = messages
        self._request = request
        self._last_capture = self._capture
        return self._capture

    def origin_facts(self) -> dict[str, Any]:
        return make_facts(self._capture)


class QwenAdapter(LlamaAdapter):
    def __init__(self, provider: Any, *, n_layers: int, dimension: int) -> None:
        super().__init__(provider)
        self.n_layers = n_layers
        self.dimension = dimension
        self._capture = make_capture(kind="qwen")
