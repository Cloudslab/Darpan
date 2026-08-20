from __future__ import annotations

import asyncio

from .inventory import ClusterInventory


async def check_cluster(inventory: ClusterInventory) -> dict[str, dict]:
    async def check(node):
        try:
            return node.id, await inventory.client(node).ping()
        except Exception as exc:
            return node.id, {"error": str(exc)}

    return dict(await asyncio.gather(*(check(node) for node in inventory.nodes)))
