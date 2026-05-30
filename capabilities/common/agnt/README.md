# AGNT - AI Agent Composition

AGNT makes AI agents first-class APG citizens. It gives generated applications
a provider-neutral way to register agent runtimes, request approval for
external providers, declare agents with models and contracts, compose teams
through validated handoff graphs, build deterministic execution plans, expose
operational screens, and preserve audit evidence.

The package is intentionally side-effect-free. It does not invoke Codex,
Claude Code, OpenCode, Pi, shell tools, browsers, memory stores, or provider
SDKs directly. Those integrations attach through adapters after AGNT has
validated runtime approval, sandbox policy, cost limits, agent contracts,
handoffs, and tenant boundaries.

## What AGNT Provides

- Provider-neutral runtime registry for `local`, `codex`, `claude_code`,
  `opencode`, `pi`, and tenant-defined runtimes.
- External runtime approval lifecycle with requester, reviewer, decision,
  notes, sandbox policy, capabilities, and cost-limit evidence.
- First-class agent registry with model, runtime, system prompt, tool
  allowlist, input contract, output contract, memory policy, and status.
- Agent team registry with tenant-owned members and validated handoff edges.
- Deterministic execution plan generation with runtime assignments, tools,
  handoff targets, approval evidence, and estimated cost limits.
- Tenant-safe runtime, agent, team, approval, and event stores.
- Bytewax lifecycle stream metadata for batch agent mutation and generated app
  composition.
- UI/view-model surfaces for dashboards, agent registry, team builder,
  handoff graph, runtime manager, approval queue, execution trace, memory
  policy, audit trail, analytics, and settings.
- Visual theme tokens for compact AI-agent operations screens.

## Core Lifecycle

1. Use built-in approved runtimes or request approval for a tenant runtime.
2. Approve external runtime requests before registration/use.
3. Register agents against approved runtimes with model, prompt, tools, IO
   contracts, and memory policy.
4. Register teams with one or more tenant-local agents.
5. Validate handoff graph endpoints against declared team members.
6. Build execution plans for concrete objectives.
7. Use audit events and Bytewax stream metadata to compose AGNT into larger
   generated applications.

## Runtime Approval Example

```python
from capabilities.common.agnt.service import AgntService

service = AgntService()

request = service.request_runtime_approval(
    request_id="approval-1",
    tenant_id="tenant-a",
    runtime_name="future_agent",
    requested_by="platform-owner",
    kind="external",
    workspace_runtime=True,
    sandbox_policy="workspace-write",
    capabilities=["code", "analysis"],
    cost_limit=12.5,
)

service.decide_runtime_approval(
    request_id=request["id"],
    tenant_id="tenant-a",
    reviewer="security-reviewer",
    decision="approved",
    notes="Sandbox and cost limits accepted.",
)
```

## Agent Team Example

```python
agent = service.register_agent(
    agent_id="builder",
    tenant_id="tenant-a",
    name="Capability Builder",
    model="gpt-5.4",
    runtime="codex",
    system_prompt="Build governed APG capability slices.",
    tool_allowlist=["shell", "pytest"],
    input_contract={"objective": "string"},
    output_contract={"patch": "object"},
    memory_policy={"store": "tenant-vector", "retention_days": 30},
)

team = service.register_team(
    team_id="delivery",
    tenant_id="tenant-a",
    name="Delivery Team",
    agent_ids=[agent["id"]],
)

plan = service.plan_execution(
    team_id=team["id"],
    objective="Implement a capability package",
    tenant_id="tenant-a",
)
```

## Composition Contract

`get_capability_contract()` returns the executable APG contract:

- `configuration`: agent, team, runtime, memory, governance, observability,
  adapter, UI, and theme settings.
- `rule_engine`: deterministic guardrails for tenant context, model, system
  prompt, tool allowlist, IO contracts, memory policy, runtime registration,
  cost limits, requester/reviewer/notes, team membership, handoff resolution,
  sandbox policy, external runtime approval, execution objective, audit
  evidence, tenant isolation, and Bytewax batch mutation.
- `ui`: APG Python route metadata and view-model module.
- `theme`: AI-agent operations tokens and component metadata.
- `streaming`: Bytewax processor, topic, state collections, lifecycle events,
  and batch mutation guardrail.

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/agnt/__init__.py capabilities/common/agnt/models.py capabilities/common/agnt/agent_composition.py capabilities/common/agnt/service.py capabilities/common/agnt/api.py capabilities/common/agnt/views.py capabilities/common/agnt/capability_contract.py capabilities/common/agnt/app.py capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.agnt import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
```

Full platform suites, live provider CLIs, browser automation, shell execution,
memory stores, durable databases, live Bytewax workers, and performance/load
checks are separate integration concerns.
