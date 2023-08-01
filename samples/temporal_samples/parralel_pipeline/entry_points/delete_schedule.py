import asyncio
from pathlib import Path

from temporalio.client import Client

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)

from pipeline.config import SCHEDULE_1_NAME, HOST


async def main():
    client = await Client.connect(HOST)
    await client.get_schedule_handle(SCHEDULE_1_NAME).delete()


if __name__ == "__main__":
    asyncio.run(main())
