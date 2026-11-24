"""The models the observe CLI can name, and the Adapter strings used to load them.

Keys and values match the real-machine tests (``test_hidden_trajectory_real_machine.MODELS``,
``GREEDY``, ``MAX_TOKENS``). This module holds data only: no import of ``sensos_inference``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

GREEDY: Final[dict[str, int]] = {"top_k": 1, "seed": 0}
MAX_TOKENS: Final = 48

# Same five names as ``QWEN_POINTS`` in the real-machine tests.
QWEN_POINTS: Final[tuple[str, ...]] = (
    "mlp_out:15",
    "mlp_out:19",
    "block_out:15",
    "block_out:19",
    "final_norm",
)

MODELS: Final[dict[str, dict[str, Any]]] = {
    "llama31": {
        "snapshot_env": "LLAMA31_SNAPSHOT",
        "points": ("final_norm",),
        "build_id": "llama3.1-8b-instruct-r241a666d-4bit",
        "compute": "mlx-metal",
        "adapter": "llama31-mlx-capture",
        "adapter_module": "sensos_inference.adapters",
        "adapter_class": "Llama31MLXAdapter",
        "provider_module": "sensos_inference.providers.llama31_mlx_capture",
        "provider_class": "Llama31MLXCaptureProvider",
        "provider_model_id": None,
        "needs_config": False,
    },
    "qwen35": {
        "snapshot_env": "QWEN35_SNAPSHOT",
        "points": QWEN_POINTS,
        "build_id": "qwen3.5-0.8b-orig-r2fc06364-bf16",
        "compute": "mlx-bf16-metal",
        "adapter": "qwen35-mlx-capture",
        "adapter_module": "sensos_inference.adapters",
        "adapter_class": "Qwen35MLXAdapter",
        "provider_module": "sensos_inference.providers.qwen35_mlx_capture",
        "provider_class": "Qwen35MLXCaptureProvider",
        "provider_model_id": "qwen35-real",
        "needs_config": True,
    },
}


def get(model: str) -> Mapping[str, Any]:
    """The catalog row for ``model``, or ``KeyError`` if the name is not one of the keys."""
    return MODELS[model]
