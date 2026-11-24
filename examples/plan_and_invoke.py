"""End-to-end pipeline example — offline, no API keys.

Demonstrates: discover → plan → session → invoke → observe → metrics → shutdown
"""

from __future__ import annotations

import asyncio

from nvs_runtime import build_runtime
from nvs_runtime.core.clock import ControllableClock


async def main() -> None:
    clock = ControllableClock()
    runtime = build_runtime(clock=clock)

    discovered = runtime.discover()
    print("discover:", len(discovered.capabilities), "capabilities")

    plan = runtime.plan(
        {
            "goal": "summarize text",
            "input_type": "text",
            "target_outputs": ["summary"],
            "optimization": "LOW_LATENCY",
        }
    )
    print("plan:", plan.plan_id, "steps:", [s.capability_id for s in plan.steps])

    session = runtime.session({"action": "create", "plan_id": plan.plan_id})
    result = await runtime.invoke(
        {
            "plan_id": plan.plan_id,
            "session_id": session["session_id"],
            "payload": {"text": "nvs runtime orchestrates providers"},
        }
    )
    print("invoke:", result.output)

    ckpt = runtime.checkpoint(
        {
            "action": "save",
            "session_id": session["session_id"],
            "state": {"output": result.output},
            "label": "example",
        }
    )
    print("checkpoint:", ckpt["checkpoint_id"])
    print("observe:", runtime.observe({"kind": "statistics"}).statistics)
    print("metrics:", runtime.metrics())

    await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
