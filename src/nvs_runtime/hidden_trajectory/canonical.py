"""Functions shared by the K path and a repeated capture.

Standard library only. The Developer Tools use these names for the K files and the K checks.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """The canonical JSON of a K file: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def key_independent(wires: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The board records without their ``hmac``: what a second run with another key must repeat."""
    return [{k: v for k, v in dict(w).items() if k != "hmac"} for w in wires]
