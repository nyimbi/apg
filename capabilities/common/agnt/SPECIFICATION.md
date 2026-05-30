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
provider-neutral agent runtimes, request approval for external providers,
declare agents with models, tools, prompts, memory, and IO contracts, compose
teams through validated handoff graphs, produce deterministic execution plans,
and govern external agent providers as they change.

The capability supports fast-moving providers such as local agents, Codex,
Claude Code, OpenCode, Pi, and future runtimes without binding APG source to
one vendor SDK. Provider calls, shell execution, browser automation, memory
stores, and live stream workers remain adapter boundaries. The package
lifecycle must run locally without credentials or network access.

## Users And Outcomes

- Platform builders can declare reusable agents and teams in APG applications.
- Capability owners can compose agent teams without embedding provider code.
- Security reviewers can approve external runtimes before use.
- Operators can inspect execution plans, runtime assignments, handoffs, cost
  limits, memory policy, approval state, tenant boundaries, and audit evidence.
- Generated APG applications can expose agent registry, team builder, runtime
  manager, approval queue, execution trace, memory policy, audit, analytics,
  and settings screens.

## Domain Model

AGNT owns these records:

- `AgentRuntime`: provider-neutral runtime backend with tenant scope,
  sandbox policy, capabilities, approval state, and cost boundary.
- `RuntimeApprovalRequest`: governed request to enable an external runtime.
- `AgentDefinition`: first-class agent declaration with model, runtime, tools,
  memory, prompts, and IO contracts.
- `AgentTeam`: tenant-owned team of declared agents.
- `HandoffEdge`: validated directed handoff between team members.
- `ExecutionPlan`: deterministic plan for running one team against an
  objective.
- `AgentAuditEvent`: tenant-scoped evidence for runtime, agent, team, plan,
  and approval lifecycle changes.

## Lifecycle

The primary lifecycle is:

1. Use built-in approved runtimes or request approval for a tenant runtime.
2. Approve external runtime requests before registration/use.
3. Register agents against approved runtimes with model, prompt, tool
   allowlist, memory policy, and IO contracts.
4. Register teams with one or more tenant-local agents.
5. Validate handoff edges against declared team members.
6. Build deterministic execution plans with runtime assignments, tools,
   handoff targets, approval evidence, and cost-limit metadata.
7. Validate Bytewax-backed batch agent mutation metadata.
8. Expose operational view models for registry, team builder, runtime manager,
   approval queue, execution trace, audit trail, analytics, settings, and
   governance evidence.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: write operations require tenant context.
- `agent_requires_model`: agents must declare models.
- `agent_requires_system_prompt`: agents must declare system prompts.
- `agent_requires_tool_allowlist`: agents must declare tool allowlists.
- `agent_requires_io_contract`: agents must declare input and output
  contracts.
- `agent_requires_memory_policy`: agents must declare memory policy.
- `agent_runtime_must_be_registered`: agents must target registered runtimes.
- `runtime_requires_cost_limit`: runtime registration requires cost limits.
- `runtime_approval_requires_requester`: approval requests require requester
  identity.
- `runtime_approval_decision_requires_reviewer`: approval decisions require
  reviewer identity.
- `runtime_approval_decision_requires_notes`: approval decisions require
  notes.
- `team_requires_agent`: teams need at least one declared agent.
- `handoff_endpoint_must_resolve`: handoffs must reference team members.
- `workspace_runtime_requires_sandbox`: workspace-aware runtimes need sandbox
  policy.
- `external_runtime_requires_approval`: external providers require approval
  before direct registration/use.
- `execution_plan_requires_objective`: execution plans require an objective.
- `agnt_state_change_requires_audit`: lifecycle changes require audit
  evidence.
- `cross_tenant_agent_access_denied`: tenant records cannot cross boundaries.
- `batch_agent_mutation_requires_bytewax`: batch mutations must use Bytewax
  streams.

Service methods must enforce these rules. Requesting approval is allowed for
an external runtime; using or registering that runtime is blocked until
approval.

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
- audit trail;
- analytics;
- settings.

The `agnt_agent_ops` theme must provide semantic tokens and component metadata
for agent cards, team graphs, runtime matrices, approval queues, audit
timelines, and execution trace decisions.

## Adapter Boundaries

These integrations remain replaceable:

- Codex, Claude Code, OpenCode, Pi, and future agent runtimes;
- local shell, browser, IDE, and workspace tools;
- memory stores and vector databases;
- cost/billing providers;
- Bytewax workers;
- external audit, SIEM, approval, and workflow systems.

Local package tests must not require any of these systems.

## Acceptance Gates

Focused AGNT proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/agnt/__init__.py capabilities/common/agnt/models.py capabilities/common/agnt/agent_composition.py capabilities/common/agnt/service.py capabilities/common/agnt/api.py capabilities/common/agnt/views.py capabilities/common/agnt/capability_contract.py capabilities/common/agnt/app.py capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.agnt import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
git diff --check -- capabilities/common/agnt
```
