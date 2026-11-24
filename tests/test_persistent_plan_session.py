"""NVSRuntime._invoke_plan: a caller-supplied session_id stays open across repeated
plan_id invokes (one long-lived session, invoked once per token); a session this call
creates for itself (no session_id given) is still auto-completed, unchanged (the
single-shot pattern docs/runtime-guide.md and the README's own Quick Start use)."""

from __future__ import annotations

from nvs_runtime import NVSRuntime

_PLAN_REQUEST = {"goal": "echo", "input_type": "text", "target_outputs": ["text"]}


async def test_single_shot_invoke_with_no_session_id_still_auto_completes(
    runtime: NVSRuntime, token: str
) -> None:
    plan = runtime.plan({**_PLAN_REQUEST, "token": token})
    assert plan.executable

    result = await runtime.invoke(
        {"plan_id": plan.plan_id, "payload": {"text": "hi"}, "token": token}
    )
    assert result.ok and result.session_id is not None

    session = runtime.session({"action": "get", "session_id": result.session_id})
    assert session["state"] == "COMPLETED"


async def test_caller_supplied_session_id_stays_open_across_n_plan_invokes(
    runtime: NVSRuntime, token: str
) -> None:
    plan = runtime.plan({**_PLAN_REQUEST, "token": token})
    assert plan.executable

    created = runtime.session({"action": "create", "plan_id": plan.plan_id})
    session_id = created["session_id"]

    n_tokens = 129
    for i in range(n_tokens):
        result = await runtime.invoke(
            {
                "plan_id": plan.plan_id,
                "session_id": session_id,
                "payload": {"text": f"token-{i}"},
                "token": token,
            }
        )
        assert result.ok, result.detail
        assert result.output["text"] == f"token-{i}"  # order preserved, no cross-talk

        session = runtime.session({"action": "get", "session_id": session_id})
        assert session["state"] != "COMPLETED"  # still open -- the whole point of this test
        assert session["session_id"] == session_id  # no new session was created underneath

    runtime.session({"action": "complete", "session_id": session_id})
    final = runtime.session({"action": "get", "session_id": session_id})
    assert final["state"] == "COMPLETED"
