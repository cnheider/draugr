import asyncio

from config import HOST, NAMESPACE
from definitions.activities.hello import say_hello
from definitions.workflows.hello import SayHello
from temporalio.client import Client
from temporalio.worker import Worker

TASK_QUEUE = "my-task-queue"


async def main():
    await Worker(
        await Client.connect(HOST, namespace=NAMESPACE),
        task_queue=TASK_QUEUE,
        workflows=[SayHello],
        activities=[say_hello],
    ).run()


if __name__ == "__main__":
    asyncio.run(main())
