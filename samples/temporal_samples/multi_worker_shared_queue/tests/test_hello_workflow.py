import uuid
from pathlib import Path

import pytest  # TODO: pip install pytest_asyncio
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)

from definitions.activities.hello import say_hello
from definitions.workflows.hello import SayHello


@pytest.mark.asyncio
async def test_execute_workflow():
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        """
            The time-skipping option starts a new environment that lets you test long-running Workflows without waiting for them to complete in real-time. You can use the start_local instead, which uses a full local insTance of the Temporal server instead. Both of these options download an instances of Temporal server on your first test run. This instance runs as a separate process during your test runs.

        The time-skipping option isn't a full implementation of the Temporal server, but it's good for basic tests like the ones in this tutorial.
        """

        async with Worker(
            env.client,
            task_queue=task_queue_name,
            workflows=[SayHello],
            activities=[say_hello],
        ):
            assert "Hello, World!" == await env.client.execute_workflow(
                SayHello.run,
                "World",
                id=str(uuid.uuid4()),
                task_queue=task_queue_name,
            )


@activity.defn(name="say_hello")
async def say_hello_mocked(name: str) -> str:
    return f"Hello, {name} from mocked activity!"


@pytest.mark.asyncio
async def test_mock_activity():
    task_queue_name = str(uuid.uuid4())
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=task_queue_name,
            workflows=[SayHello],
            activities=[say_hello_mocked],
        ):
            assert (
                "Hello, World from mocked activity!"
                == await env.client.execute_workflow(
                    SayHello.run,
                    "World",
                    id=str(uuid.uuid4()),
                    task_queue=task_queue_name,
                )
            )
