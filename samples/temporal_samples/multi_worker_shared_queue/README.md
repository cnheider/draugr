# Shared Queue Example

- start worker1.py and worker2.py
- then run_workflows.py

## What are the minimum four pieces of a Temporal Application?

- A Workflow Definition.
- An Activity Definition.
- A Worker to host the Activity and Workflow code. (Maybe a subset)
- Some way to start the Workflow.

## How does the Temporal server get information to the Worker?

- The Temporal Server adds Tasks to a Task Queue, and the Worker polls the Task Queue.