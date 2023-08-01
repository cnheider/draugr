```command
poetry install
```

## Start the Workflow

Start and run the Workflow with the following commands:

```command
# terminal one
poetry run python run_worker.py

# terminal two
poetry run python run_workflow.py


# terminal three ( stop workflow )
temporal workflow terminate --workflow-id temporal-community-workflow
```

# MIIGR mapping

- Task -> Activity
- Runner -> Workers (Note plural), for scaling
- Pipeline Sequence -> Workflow