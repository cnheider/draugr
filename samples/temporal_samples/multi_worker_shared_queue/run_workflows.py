import asyncio
from itertools import count

from config import HOST, NAMESPACE
from definitions.workflows.fizz_workflow import FizzWorkflow
from definitions.workflows.hello import SayHello
from temporalio.client import Client

TASK_QUEUE = "my-task-queue"
RUN_ID_PREFIX = "RUN"
HELLO_ARG = "Automation"
FIZZ_ARG = "buzz"


async def main():
    client = await Client.connect(HOST, namespace=NAMESPACE)

    counter = iter(count())

    result = await client.execute_workflow(
        SayHello.run,
        HELLO_ARG,
        id=f"{RUN_ID_PREFIX}_{next(counter)}",
        task_queue=TASK_QUEUE,
    )
    print(f"Result: {result}")

    result = await client.execute_workflow(
        FizzWorkflow.run,
        FIZZ_ARG,
        id=f"{RUN_ID_PREFIX}_{next(counter)}",
        task_queue=TASK_QUEUE,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
