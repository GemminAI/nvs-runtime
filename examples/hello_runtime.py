"""Minimal NVS Runtime example — offline, no API keys.

Demonstrates: discover → invoke → observe → metrics → shutdown
"""

from __future__ import annotations

import asyncio

from nvs_runtime import build_runtime


async def main() -> None:
    runtime = build_runtime()

    discovered = runtime.discover()
    print(f"providers: {len(discovered.providers)}")
    print(f"capabilities: {[c.id for c in discovered.capabilities]}")

    result = await runtime.invoke(
        {"capability_id": "local.echo", "payload": {"text": "hello nvs-runtime"}}
    )
    print("invoke:", result.output)
    print("observe:", runtime.observe({"kind": "statistics"}).statistics)
    print("metrics:", runtime.metrics())

    await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
