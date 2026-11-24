"""Fakes for the U6-5 tests: a capture shaped like U1's ``ObservationCapture`` and the
``origin_facts()`` of an Adapter. They import neither ``sensos_inference`` nor ``nvs_edge``.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

LLAMA_RULE = "llama31-mlx-final-norm/1"
QWEN_RULES = {
    "block_out": "qwen35-mlx-block-out/1",
    "mlp_out": "qwen35-mlx-mlp-out/1",
    "final_norm": "qwen35-mlx-final-norm/1",
}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
REVISION = "d" * 40


def ids_sha256(ids: Any) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeProvenance:
    """The fields of U1's ``CaptureProvenance`` (T-U65-02 compares the two lists)."""

    model_ref: str = "mlx-community/test-model"
    revision: str | None = REVISION
    weights_sha256: str | None = HASH_A
    tokenizer_sha256: str | None = HASH_B
    chat_template_sha256: str | None = HASH_C
    special_tokens_in: tuple[tuple[int, int], ...] | None = ((0, 1),)
    prompt_ids_sha256: str = ""
    gen_token_ids_sha256: str = ""
    runtime: Any = field(
        default_factory=lambda: MappingProxyType(
            {"backend": "mlx", "mlx": "0.32.4", "mlx_lm": "0.32.2", "device": "Device(gpu, 0)"}
        )
    )
    sampler: Any = field(
        default_factory=lambda: MappingProxyType(
            {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": None}
        )
    )
    max_tokens: int = 8
    chat_template_kwargs: Any = field(default_factory=lambda: MappingProxyType({}))
    eos_ids: tuple[int, ...] | None = (99,)


@dataclass(frozen=True, slots=True, kw_only=True)
class FakeCapture:
    """The fields of U1's ``ObservationCapture`` (``n_in``, ``n_gen``, ``n_rows``: properties)."""

    adapter: str
    model_build_id: str
    compute_environment: str
    dimension: int
    prompt_ids: Any
    token_ids: Any
    finish_reason: str
    points: tuple[str, ...]
    states: Any
    row_rule: Any
    row_pos: Any
    row_decides: Any
    final_norm_applied: Any
    state_dtype: str
    provenance: FakeProvenance
    unknowns: tuple[str, ...] = ()

    @property
    def n_in(self) -> int:
        return len(self.prompt_ids)

    @property
    def n_gen(self) -> int:
        return len(self.token_ids)

    @property
    def n_rows(self) -> int:
        return len(self.token_ids)


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def make_capture(
    *,
    kind: str = "llama",
    n_in: int = 6,
    n_gen: int = 4,
    dimension: int = 8,
    provenance: dict[str, Any] | None = None,
    **changes: Any,
) -> FakeCapture:
    """A consistent capture: Llama has ``final_norm``; Qwen has five points."""
    rng = np.random.default_rng(7)
    prompt_ids = _ro(np.arange(10, 10 + n_in, dtype=np.int64))
    token_ids = _ro(np.arange(50, 50 + n_gen, dtype=np.int64))
    if kind == "llama":
        points = ("final_norm",)
        rules = {"final_norm": LLAMA_RULE}
        adapter, dtype = "llama31-mlx-capture", "float16"
    else:
        points = ("block_out:3", "mlp_out:3", "block_out:7", "mlp_out:7", "final_norm")
        rules = {p: QWEN_RULES[p.split(":")[0]] for p in points}
        adapter, dtype = "qwen35-mlx-capture", "bfloat16"
    states = {p: _ro(rng.standard_normal((n_gen, dimension)).astype(np.float32)) for p in points}
    row_pos = {p: _ro(np.arange(n_in - 1, n_in - 1 + n_gen, dtype=np.int64)) for p in points}
    row_decides = {p: _ro(np.arange(n_gen, dtype=np.int64)) for p in points}
    prov = FakeProvenance(
        prompt_ids_sha256=ids_sha256(prompt_ids),
        gen_token_ids_sha256=ids_sha256(token_ids),
        **(provenance or {}),
    )
    values: dict[str, Any] = {
        "adapter": adapter,
        "model_build_id": "test-model-4bit",
        "compute_environment": "mlx-4bit",
        "dimension": dimension,
        "prompt_ids": prompt_ids,
        "token_ids": token_ids,
        "finish_reason": "length",
        "points": points,
        "states": MappingProxyType(states),
        "row_rule": MappingProxyType(rules),
        "row_pos": MappingProxyType(row_pos),
        "row_decides": MappingProxyType(row_decides),
        "final_norm_applied": MappingProxyType({p: p == "final_norm" for p in points}),
        "state_dtype": dtype,
        "provenance": prov,
        "unknowns": (),
    }
    values.update(changes)
    return FakeCapture(**values)


def make_facts(capture: FakeCapture, **overrides: Any) -> dict[str, Any]:
    """The ``origin_facts()`` of an Adapter that made ``capture`` (every value agrees with it)."""
    prov = capture.provenance
    runtime = dict(prov.runtime)
    runtime.update({"python": "3.12.13", "transformers": "5.22.2"})
    facts: dict[str, Any] = {
        "origin": {
            "capture": {
                "package": "sensos-inference",
                "version": "0.1.0",
                "adapter": capture.adapter,
                "alignment": "decides-next/1",
                "source_commit": COMMIT,
            },
            "runtime": runtime,
            "hardware": "Mac15,6/Apple M3 Pro/18GiB",
            "quantization": "4bit/group64",
            "model": prov.model_ref,
            "revision": prov.revision,
            "weights_sha256": prov.weights_sha256,
        },
        "capabilities_phase": "post_capture",
        "observed_state_dtype": capture.state_dtype,
        "version_policy": "package_version_only",
    }
    for dotted, value in overrides.items():
        target = facts
        *parents, last = dotted.split(".")
        for name in parents:
            target = target[name]
        if value is DELETE:
            del target[last]
        else:
            target[last] = value
    return facts


DELETE = object()


def unknown(source: str, reason: str = "unknown") -> dict[str, str]:
    return {"source": source, "reason": reason}


def with_provenance(capture: FakeCapture, **changes: Any) -> FakeCapture:
    return replace(capture, provenance=replace(capture.provenance, **changes))


def copy_of(value: Any) -> Any:
    return copy.deepcopy(value)
