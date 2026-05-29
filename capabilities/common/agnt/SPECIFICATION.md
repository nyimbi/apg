# AGNT Capability Specification

## Identity

- Capability ID: `agnt`
- Display name: AI Agent Composition
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `agnt_agent_ops`

## Purpose

AGNT makes AI agents first-class APG citizens. It lets applications register
provider-neutral agent runtimes, declare agents with models, tools, prompts,
memory, and IO contracts, compose teams through validated handoff graphs,
produce deterministic execution plans, and govern external agent providers as
they change.

The capability must support fast-moving providers such as local agents, Codex,
Claude Code, OpenCode, Pi, and future runtimes without tying APG source to one
vendor SDK. Provider calls remain adapter boundaries; the package lifecycle
must run locally without credentials or network access.

## Users And Outcomes

- Platform builders can declare reusable agents and teams in APG applications.
- Capability owners can compose agent teams without embedding provider code.
- Security reviewers can approve external runtimes before use.
- Operators can inspect execution plans, runtime assignments, handoffs, cost
  limits, memory policy, and approval state.
- Generated APG applications can expose agent registry, team builder, runtime
  manager, execution trace, memory policy, and settings screens.

## Domain Model

AGNT owns these records:

- `AgentRuntime`: provider-neutral runtime backend and sandbox/cost boundary.
- `AgentDefinition`: first-class agent declaration with model, runtime, tools,
  memory, prompts, and IO contracts.
- `AgentTeam`: tenant-owned team of declared agents.
- `HandoffEdge`: validated directed handoff between team members.
- `ExecutionPlan`: deterministic plan for running one team against an
  objective.
- `RuntimeApprovalRequest`: governed request to enable an external runtime.
- `AgentAuditEvent`: tenant-scoped evidence for runtime, agent, team, and plan
  lifecycle changes.

## Lifecycle

The primary lifecycle is:

1. Register or request approval for a runtime.
2. Approve external runtime requests before registration.
3. Register agents against approved runtimes with model, prompt, tools, memory,
   and IO contracts.
4. Register teams with one or more agents.
5. Validate handoff edges against declared team members.
6. Build deterministic execution plans with runtime assignments, tools,
   handoff targets, approval evidence, and cost-limit metadata.
7. Expose operational view models for registry, team builder, runtime manager,
   approval queue, and execution trace.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: write operations require tenant context.
- `agent_requires_model`: agents must declare models.
- `agent_runtime_must_be_registered`: agents must target registered runtimes.
- `team_requires_agent`: teams need at least one declared agent.
- `handoff_endpoint_must_resolve`: handoffs must reference team members.
- `workspace_runtime_requires_sandbox`: workspace-aware runtimes need sandbox
  policy.
- `external_runtime_requires_approval`: external providers require approval
  before direct registration/use.

Service methods must enforce these rules. Requesting approval is allowed for an
external runtime; using or registering that runtime is blocked until approval.

## UI And Theme

AGNT exposes route and view-model surfaces for:

- dashboard summary;
- agent registry;
- team builder;
- handoff graph;
- runtime manager;
- runtime approval queue;
- execution trace;
- memory policy;
- settings.

The `agnt_agent_ops` theme must provide semantic tokens and component metadata
for agent cards, team graphs, runtime matrices, approval bands, and execution
trace decisions.

## Adapter Boundaries

These integrations remain replaceable:

- Codex, Claude Code, OpenCode, Pi, and future agent runtimes;
- local shell, browser, IDE, and workspace tools;
- memory stores and vector databases;
- cost/billing providers;
- external audit, SIEM, and approval systems.

Local package tests must not require any of these systems.

## Acceptance Gates

Focused AGNT proof:

```bash
./.venv/bin/pytest -q capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
git diff --check -- capabilities/common/agnt
```
