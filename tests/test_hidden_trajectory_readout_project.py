"""``string_positions`` has no default projector: the kernel's projector comes from the caller."""

from __future__ import annotations

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.rsa import readout as ro


def test_string_positions_needs_a_projector() -> None:
    with pytest.raises(TypeError, match="project"):
        ro.string_positions({"a": "x"}, ["a"])  # type: ignore[call-arg]


def test_string_positions_uses_the_projector_it_is_given() -> None:
    out = ro.string_positions({"a": "x", "b": "yy"}, ["b", "a"], lambda text: [len(text), 0.0])
    assert np.array_equal(out, np.array([[2.0, 0.0], [1.0, 0.0]]))
