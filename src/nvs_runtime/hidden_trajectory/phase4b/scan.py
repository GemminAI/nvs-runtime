"""The scan of an evidence document for values that may hold a position, a vector or a text.

A key of ``FORBIDDEN_KEYS`` is a violation only when its value can hold data: a number (a
difference, a count) and ``None`` (nothing) are allowed, so the name of a field group such as
``guidance_direction`` carrying "the largest difference" is not flagged, while a list, a string, a
dictionary or a boolean under such a key is.
"""

from __future__ import annotations

from typing import Any, Final

FORBIDDEN_KEYS: Final = frozenset({"position", "guidance_direction", "velocity", "text"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def violations(node: Any, path: str = "$") -> list[str]:
    """The paths whose key is forbidden and whose value is neither a number nor ``None`` (a list, a
    string, a dictionary or a boolean). Nested values are scanned too."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key in FORBIDDEN_KEYS and value is not None and not _is_number(value):
                found.append(here)
            found.extend(violations(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(violations(value, f"{path}[{index}]"))
    return found
