# Temporal Durable Workflow — User Guide

## Overview

The `temporal` capability provides crash-resilient workflow execution via Temporal.io. APG workflows — approval chains, multi-step business processes, human task queues — survive process restarts, network failures, and server crashes with automatic recovery.

## Quick Start

```bash
# Start Temporal (includes PostgreSQL)
docker run -d --name apg-temporal temporalio/auto-setup:1.26 \
  -e DB=postgres12 -e POSTGRES_SEEDS=localhost -p 7233:7233

export TEMPORAL_HOST=localhost:7233
export TEMPORAL_NAMESPACE=default
```

## Starting a Workflow

```python
from capabilities.common.temporal.service import TemporalService

svc = TemporalService(tenant_id="acme")
await svc.connect()

result = await svc.start_workflow(
    workflow_type="APGStateMachineWorkflow",
    input_data={
        "declaration_id": "wf_purchase_order",
        "tenant_id": "acme",
        "initiator_id": "user_abc",
    },
)
# {"workflow_id": "uuid7...", "status": "RUNNING"}
```

## Human Task Completion

```python
# Complete a pending human task (e.g., manager approval)
await svc.complete_task(
    task_token="token_from_workflow_event",
    result={"approved": True, "approver_note": "Looks good"},
)
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/temporal/workflows` | Start a workflow |
| GET | `/api/temporal/workflows` | List workflows |
| GET | `/api/temporal/workflows/{id}` | Get workflow status |
| DELETE | `/api/temporal/workflows/{id}` | Cancel workflow |
| POST | `/api/temporal/workflows/{id}/signal` | Send signal |
| GET | `/api/temporal/workflows/{id}/history` | Get event history |
| POST | `/api/temporal/tasks/complete` | Complete human task |
| POST | `/api/temporal/tasks/fail` | Fail a task |
| POST | `/api/temporal/schedules` | Create cron schedule |
| GET | `/api/temporal/schedules` | List schedules |
| DELETE | `/api/temporal/schedules/{id}` | Delete schedule |
| GET | `/api/temporal/health` | Health check |
| GET | `/api/temporal/metrics` | Workflow metrics |

## Workflow Status Values

| Status | Meaning |
|--------|---------|
| `RUNNING` | Workflow executing or waiting |
| `COMPLETED` | Successfully finished |
| `FAILED` | Unhandled error |
| `CANCELLED` | Cancelled by operator |
| `TERMINATED` | Forcibly stopped |
| `TIMED_OUT` | Exceeded execution timeout |

## Integration with APG Workflows

APG workflows defined in `.apg` files are compiled to `APGStateMachineWorkflow` definitions. When `TEMPORAL_HOST` is set, the `get_workflow_adapter()` factory returns a `TemporalWorkflowAdapter` that routes workflow starts/completions through Temporal.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TEMPORAL_HOST` | _(none)_ | `host:port` — if unset, stub mode |
| `TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `TEMPORAL_TASK_QUEUE` | `apg-workflows` | Default task queue |
