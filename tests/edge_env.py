"""Which environment the tests run in: with or without the NVS-Kernel edge package.

``nvs_observation.contract_check.edge_contract_available()`` is the one decision the package
itself makes (the edge package is installed and has contract v2); the tests use the same one.
A test of what a record says with the edge package present is marked ``requires_edge``; a test
of what it says without it is marked ``requires_no_edge``. Exactly one of the two runs.
"""

from __future__ import annotations

import pytest
from nvs_observation.contract_check import edge_contract_available

EDGE_AVAILABLE: bool = edge_contract_available()

requires_edge = pytest.mark.skipif(
    not EDGE_AVAILABLE, reason="needs the NVS-Kernel edge package (contract v2)"
)
requires_no_edge = pytest.mark.skipif(
    EDGE_AVAILABLE, reason="states the contract where the NVS-Kernel edge package is absent"
)
