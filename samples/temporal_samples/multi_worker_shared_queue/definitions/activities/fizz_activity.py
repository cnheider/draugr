from temporalio import activity


@activity.defn
async def fizz_activity(name: str) -> str:
    return f"fizz, {name}!"
