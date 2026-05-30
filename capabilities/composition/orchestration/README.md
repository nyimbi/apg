# APG Workflow Orchestration

`composition_orchestration` lets APG applications compose business processes from validated workflow definitions, governed releases, executable task graphs, human task assignments, and Bytewax lifecycle events.

## What It Provides

- Workflow definition lifecycle.
- Task graph validation.
- Workflow execution lifecycle.
- Human task coordination.
- Workflow release governance.
- Workflow rule enforcement.
- Workflow AI agents.

## How To Use It

Import the service for in-process generated applications:

```python
from capabilities.composition.orchestration import WorkflowOrchestrationService

service = WorkflowOrchestrationService()
definition = service.define_workflow(
    "order-fulfilment",
    "tenant-a",
    "Order Fulfilment",
    "ops-owner",
    "1.0.0",
    [
        {"id": "intake", "type": "automated", "handler": "orders.intake"},
        {"id": "review", "type": "human", "assignee": "ops-review", "depends_on": ["intake"]},
    ],
    "order.created",
    "completed",
)
```

Publishers and generated applications can inspect the package with:

```bash
./.venv/bin/apg capabilities inspect composition_orchestration --json
./.venv/bin/apg capabilities publish-plan capabilities/composition/orchestration --json
```

## Lifecycle

1. Define the workflow with tenant, owner, version, start event, terminal state, and tasks.
2. Validate handlers, assignments, approval policies, cross-capability contracts, retry limits, SLA escalation, dependencies, and graph cycles.
3. Release the workflow with validation evidence, dry-run success, rollback plan, and approval metadata.
4. Start executions with idempotency keys and Bytewax event-stream coordination.
5. Complete tasks to advance the graph.
6. Assign human tasks where people must approve, review, or perform business work.
7. Use AI agents to recommend or validate changes while preserving human approval for privileged actions.

## Screens

- Dashboard
- Workflow definitions
- Designer
- Executions
- Tasks
- Releases
- Rules
- Agents
- Settings

## Guardrails

The deterministic rule engine blocks missing tenant context, writes without policy, incomplete workflow definitions, unsafe tasks, missing release evidence, non-Bytewax execution coordination, missing idempotency keys, unsupported agent runtimes, unsupported agent roles, and privileged agent actions without human approval.

## AI Agent Support

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are workflow architect, BPML reviewer, release reviewer, incident reviewer, compliance reviewer, and optimization reviewer.
