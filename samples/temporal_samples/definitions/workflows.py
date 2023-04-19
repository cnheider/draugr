from datetime import timedelta
from temporalio import workflow

# Import our activity, passing it through the sandbox
with workflow.unsafe.imports_passed_through():
    from .activities import say_hello


@workflow.defn
class SayHello:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            say_hello, name, schedule_to_close_timeout=timedelta(seconds=5)
        )
