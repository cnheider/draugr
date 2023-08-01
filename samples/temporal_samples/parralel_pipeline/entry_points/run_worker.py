import asyncio
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)

from pipeline.config import TASK_QUEUE_1_NAME, HOST
from pipeline.activities.activity1 import activity_1
from pipeline.activities.activity2 import activity_2
from pipeline.workflows.workflow1 import Workflow1


async def main():
    client = await Client.connect(HOST)
    await Worker(
        client,
        task_queue=TASK_QUEUE_1_NAME,
        workflows=[Workflow1],
        activities=[activity_1, activity_2],
    ).run()


if __name__ == "__main__":
    asyncio.run(main())
