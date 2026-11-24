"""The sensos contracts behind a JSON door, for ``nvs.hidden-trajectory`` 0.1.0 (design U6 8.2).

A THIN COMPATIBILITY LAYER since U6T-1c (DC-T1-1): the script itself lives in nvs-runtime's
``src`` as ``nvs_runtime/hidden_trajectory/_gate_child.py`` (the child process of ``GateClient``),
and this file only runs it, so that ``python gate_nvs_cli.py`` keeps working in ``GATE_PY``. The
operations (``hid_space_id``, ``seal``, ``session_id``, ``observation_id``, ``check``, and the
new ``gate``, ``catalog``, ``checker_files``, ``jcs_sha256``) are described there.

Run in ``GATE_PY`` (an interpreter with ``jsonschema``) with ``SENSOS_TOOLS`` set. Standard
library only; nothing of nvs-runtime is imported.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

CHILD = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "nvs_runtime"
    / "hidden_trajectory"
    / "_gate_child.py"
)

if __name__ == "__main__":
    sys.argv[0] = str(CHILD)
    runpy.run_path(str(CHILD), run_name="__main__")
