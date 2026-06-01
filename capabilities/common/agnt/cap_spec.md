# AGNT Capability Packet

`agnt` is the APG AI Agent Composition capability. It makes AI agents
first-class, provider-neutral application components that generated APG
applications can compose with runtimes, tools, teams, handoffs, memory policy,
execution plans, execution run evidence, rules, screens, themes, and Bytewax
lifecycle streams.

## Purpose

AGNT lets applications register approved agent runtimes, request approval for
new external runtimes, declare agents with model/tool/prompt/memory/IO
contracts, compose tenant-owned teams, validate handoff graphs, build
deterministic execution plans, record governed execution runs, and preserve
review and audit evidence.

The package does not invoke provider CLIs or SDKs. Codex, Claude Code,
OpenCode, Pi, local shell/browser/IDE tools, memory stores, billing providers,
and live Bytewax workers are replaceable adapters above this dependency-light
capability packet.

## Executable Surfaces

- `capability_contract.py` publishes configuration, deterministic rules, UI
  routes, theme tokens, Bytewax stream metadata, and composition surfaces.
- `models.py` defines runtimes, approvals, agents, teams, handoffs, plans,
  execution runs, and audit events.
- `agent_composition.py` builds deterministic team execution plans.
- `service.py` implements tenant-scoped lifecycle behavior and enforces
  contract guardrails.
- `api.py` exposes dependency-light helper functions for generated apps.
- `views.py` exposes dashboard, team, runtime, approval, governance, audit,
  analytics, execution trace, execution-run console, and settings models.
- `app.py`, `semantic_model.json`, and `release_report.json` are generated
  package evidence derived from the executable contract.

## Core Lifecycle

1. Use built-in approved runtimes or request approval for a tenant runtime.
2. Decide external runtime requests with reviewer and notes.
3. Register agents against approved runtimes with model, system prompt, tool
   allowlist, IO contracts, and memory policy.
4. Register teams with tenant-local agents.
5. Validate handoff graph endpoints against declared team members.
6. Build deterministic execution plans for concrete objectives.
7. Record execution runs with requester identity, trace sink, status,
   side-effect approval evidence, and plan snapshot.
8. Persist runtime and execution-run review queues with matched rules, review
   reasons, and required actions.
9. Validate Bytewax-backed batch agent mutation metadata.
10. Compose AGNT screens, rules, theme, state, and Bytewax lifecycle metadata
   into larger generated APG applications.

## Guardrails

AGNT enforces these rule families:

- tenant context and tenant isolation;
- model, system prompt, tool allowlist, IO contract, and memory policy for
  every first-class agent;
- registered runtimes, cost limits, sandbox policies, and external runtime
  approval;
- runtime requester, reviewer, and decision notes;
- team membership and handoff endpoint resolution;
- execution objective, requester identity, trace sink, and side-effect approval
  evidence;
- audit evidence for lifecycle changes;
- Bytewax stream metadata for batch agent mutations.

## Provider-Neutral Execution Runs

`record_execution_run()` records the governance envelope for a run before any
adapter invokes a provider. The run stores a plan snapshot, requester, trace
sink, status, and side-effect approval state. This lets APG compose Codex,
Claude Code, OpenCode, Pi, local agents, or future providers without binding
the core package to one changing SDK or CLI.

Side-effecting runs without human approval are stored as `pending_review`
records with `decision`, `matched_rules`, `review_reasons`, and
`audit_evidence`. Runtime approval requests preserve the same policy evidence
with `policy_decision` so generated applications can render approval queues
without re-evaluating rules.

## Bytewax Composition

The streaming contract declares `bytewax` as the lifecycle processor and
`apg.agnt.lifecycle` as the topic. Batch agent mutations must carry Bytewax
metadata and fail when a non-Bytewax stream is used.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/agnt/__init__.py capabilities/common/agnt/models.py capabilities/common/agnt/agent_composition.py capabilities/common/agnt/service.py capabilities/common/agnt/api.py capabilities/common/agnt/views.py capabilities/common/agnt/capability_contract.py capabilities/common/agnt/app.py capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.agnt import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Known Adapter Gaps

This packet does not prove live provider invocation, shell/browser/IDE
automation, durable memory stores, billing providers, rendered UI, full workflow
engine integration, or live Bytewax execution. Those remain adapter and
integration layers above the local capability package.
