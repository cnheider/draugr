import asyncio
from datetime import timedelta
from pathlib import Path

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
)

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)
from pipeline.config import TASK_QUEUE_1_NAME, WORKFLOW_1_NAME
from pipeline.workflows.workflow1 import Workflow1


async def main():
    client = Client.connect(HOST)  # TODO: Make a context mananger for the client
    await client.create_schedule(
        SCHEDULE_1_NAME,
        Schedule(
            action=ScheduleActionStartWorkflow(
                Workflow1.run,
                id=WORKFLOW_1_NAME,
                task_queue=TASK_QUEUE_1_NAME,
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(hours=10))]
            ),
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
