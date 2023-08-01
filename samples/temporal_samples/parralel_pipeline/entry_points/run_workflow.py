import asyncio
from pathlib import Path

import pandas
from temporalio.client import Client

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)

from pipeline.config import TASK_QUEUE_1_NAME, HOST, WORKFLOW_1_NAME
from pipeline.workflows.workflow1 import Workflow1


async def main():
    client = await Client.connect(HOST)

    stories = await client.execute_workflow(
        Workflow1.run,
        id=WORKFLOW_1_NAME,
        task_queue=TASK_QUEUE_1_NAME,
    )
    df = pandas.DataFrame(stories)
    df.columns = ["Title", "URL", "Views"]
    print("Top 10 stories on Temporal Community:")
    print(df)
    return df


if __name__ == "__main__":
    asyncio.run(main())
