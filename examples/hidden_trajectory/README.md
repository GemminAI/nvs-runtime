# examples/hidden_trajectory

A thin layer from nvs-runtime to the sensos contracts for `nvs.hidden-trajectory` 0.1.0
(design U6 v0.1, section 8). Since U6T-1c the Gate is nvs-runtime's own, in `src/`
(`nvs_runtime/hidden_trajectory/gate_client.py`, with the child script `_gate_child.py`); these
files keep the U6-4 operations working over it (DC-T1-1).

| File | What it is |
|---|---|
| `gate_nvs_cli.py` | Runs `src/.../_gate_child.py` in `GATE_PY` (JSON in, JSON out). Kept so that `python gate_nvs_cli.py` still works |
| `sensos_door.py` | The caller's side: `SensosDoor.space_id_of`, `.seal` (for `build_envelopes`), `.check` (the U6-4 form) and `.verdict` (`GateClient.verdict`, a `GateVerdict`) |
| `run_door.py` | `run_through_door(door, captured, ...)`: `nvs_runtime.hidden_trajectory.run.run_capture_session` with the door's ids, `space_id_of`, `seal` and `verdict` as the Gate (U6-5d, U6T-1c) |

## Environment (no defaults)

| Variable | Points to |
|---|---|
| `SENSOS_TOOLS` | the `tools/` directory of a sensos checkout that has `contracts/gate_nvs.py` |
| `GATE_PY` | a Python with `jsonschema` (the Gate runs in it) |

```
export SENSOS_TOOLS=$HOME/src/sensos/tools
export GATE_PY=$HOME/venvs/gate/bin/python
pytest tests/test_hidden_trajectory_gate.py
```

Without both variables (or without nvs-kernel's `nvs_edge` contract v2) the tests in
`tests/test_hidden_trajectory_gate.py` are skipped.

`check` (and the `GateVerdict`'s layers) run the envelope rule (IC-00) first and the payload check only if the envelope rule
reports nothing; a window (`msr.trajectory-window` 0.10.0) also gets `check_window_space`.
Nothing is stored here: what may be stored after which verdict belongs to the Formal Store (U6T-2).
`examples/pb_path/` is not used and not changed.
