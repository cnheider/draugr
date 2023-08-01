from datetime import timedelta
from pathlib import Path
from typing import List

from temporalio import workflow

from warg import ensure_in_sys_path

ensure_in_sys_path(Path(__file__).parent.parent)

with workflow.unsafe.imports_passed_through():
    from model import TemporalCommunityPost
    from activities.activity1 import activity_1
    from activities.activity2 import activity_2

__all__ = ["Workflow1"]


@workflow.defn
class Workflow1:
    @workflow.run
    async def run(self) -> List[TemporalCommunityPost]:
        return await workflow.execute_activity(
            activity_2,
            await workflow.execute_activity(
                activity_1,
                start_to_close_timeout=timedelta(seconds=15),
            ),
            start_to_close_timeout=timedelta(seconds=15),
        )
