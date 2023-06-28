import asyncio

from config import HOST, NAMESPACE
from definitions.activities.fizz_activity import fizz_activity
from definitions.workflows.fizz_workflow import FizzWorkflow
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "my-task-queue"


async def main():
    await Worker(
        await Client.connect(HOST, namespace=NAMESPACE),
        task_queue=TASK_QUEUE,
        workflows=[FizzWorkflow],
        activities=[fizz_activity],
    ).run()


if __name__ == "__main__":
    asyncio.run(main())
