# APG Workflow Orchestration Capability Specification

- **Capability Name**: Workflow Orchestration
- **Category**: Composition
- **Version**: 2.1.0
- **Capability ID**: `composition_orchestration`

## Purpose

Workflow Orchestration coordinates long-running APG business processes across capability boundaries. It gives generated applications a composable workflow definition library, graph validation, execution lifecycle, human task coordination, release governance, deterministic guardrails, operator screens, visual theme metadata, Bytewax event-stream evidence, and first-class AI agent participation.

## Capability Boundaries

The capability owns workflow definitions, task graph validation, release records, execution state, human task assignments, orchestration rule evaluation, and orchestration agent registration. It does not own authentication, audit storage, notification delivery, capability registry data, or event infrastructure; those remain adapter dependencies.

## Provides

- `workflow_definition_lifecycle`
- `workflow_graph_validation`
- `workflow_execution_lifecycle`
- `human_task_coordination`
- `workflow_release_governance`
- `workflow_rule_enforcement`
- `workflow_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `registry`
- `composition_events`
- `composition_config`

## Domain Model

Workflow definitions include owner, version, start event, terminal state, task graph, optional transaction flag, and optional compensation steps. Tasks include type, handler, dependencies, human assignment, approval policy, cross-capability contract, SLA, and escalation metadata. Releases include validation evidence, dry-run evidence, rollback plan, approval, and release status. Executions include idempotency key, current tasks, completed tasks, failed tasks, inputs, status, and Bytewax stream metadata.

## Lifecycle

1. Define workflow with owner, version, start event, terminal state, and task graph.
2. Validate task graph for missing handlers, missing assignments, missing approval policy, missing cross-capability contracts, unknown dependencies, and cycles.
3. Release workflow after validation evidence, dry-run success, rollback plan, and optional approval.
4. Start execution with tenant context, idempotency key, risk metadata, and Bytewax lifecycle stream.
5. Advance execution as tasks complete; ready tasks are derived from dependency completion.
6. Assign human tasks with accountable assignee and optional due date.
7. Record lifecycle evidence through audit/event adapters.
8. Allow AI agents to recommend, validate, and prepare workflow changes inside explicit runtime, role, and approval guardrails.

## Rule Engine

The rule engine is deterministic. It denies missing tenant context, writes without policy, incomplete workflow definitions, unsafe task definitions, non-Bytewax execution coordination, missing idempotency keys, releases without evidence, unbounded retry policy, missing compensation for transactional workflows, unsupported agent runtimes or roles, and privileged agent actions without human approval. High-risk execution starts and SLA tasks without escalation require review.

## UI Contract

The capability exposes generated-application screens for dashboard, workflow definitions, designer, executions, tasks, releases, rules, agents, and settings. Each route has a permission, component name, and navigation group. Theme metadata defines compact operational surfaces for graph design, execution lanes, task queues, release evidence, rule grids, and agent review lanes.

## Streaming

Workflow lifecycle events use the Bytewax processor and stream `apg.composition.orchestration.lifecycle`. The stream key is `tenant_id`. Events include workflow definition, validation, release, execution start, execution advancement, execution completion, task assignment, and workflow agent registration.

## AI Agent Composition

Workflow agents are first-class capability records. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are workflow architect, BPML reviewer, release reviewer, incident reviewer, compliance reviewer, and optimization reviewer. Agents may recommend and validate workflow changes, but privileged workflow actions require recorded human approval.
