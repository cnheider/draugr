from datetime import timedelta

from temporalio import workflow


with workflow.unsafe.imports_passed_through():  # Import our activity, passing it through the sandbox
    from ..activities.hello import say_hello


@workflow.defn
class SayHello:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            say_hello, name, schedule_to_close_timeout=timedelta(seconds=5)
        )
