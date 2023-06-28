from typing import List

import aiohttp
from temporalio import activity

__all__ = ["activity_1"]


@activity.defn
async def activity_1() -> List[str]:
    async with aiohttp.ClientSession() as session:
        async with session.get("https://community.temporal.io/latest.json") as response:
            if not 200 <= int(response.status) < 300:
                raise RuntimeError(f"Status: {response.status}")
            post_ids = await response.json()

    return [str(topic["id"]) for topic in post_ids["topic_list"]["topics"]]
