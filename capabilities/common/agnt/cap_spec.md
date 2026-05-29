# AI Agent Composition Capability Specification

- **Capability Name**: AI Agent Composition
- **Capability ID**: `agnt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

AGNT makes AI agents first-class APG citizens. A contributor can register
provider-neutral agent runtimes, define tenant-scoped agents with model/tool/
memory/IO contracts, compose those agents into validated teams, and produce a
deterministic execution plan before a runtime performs work.
External providers now move through an explicit approval lifecycle so fast
moving systems such as Codex, Claude Code, OpenCode, Pi, and future providers
can be introduced without bypassing sandbox, cost, and review guardrails.

## Provided Services

- `agent_registry`
- `runtime_registry`
- `runtime_approval_governance`
- `team_composition`
- `handoff_graph_validation`
- `execution_planning`
- `agent_operations_view_models`

## Required Services

- `tenant_context`
- `aicr`
- `auth`
- `wflo`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for every operation.
Runtime configuration recognizes local, Codex, Claude Code, OpenCode, Pi, and
future provider backends without coupling APG to one rapidly changing agent
toolchain.

## Rules

- `tenant_context_required`
- `agent_requires_model`
- `agent_runtime_must_be_registered`
- `team_requires_agent`
- `handoff_endpoint_must_resolve`
- `workspace_runtime_requires_sandbox`
- `external_runtime_requires_approval`

## Runtime Behavior

`service.py` owns the in-memory executable registry for runtimes, runtime
approval requests, agents, teams, and audit events. `agent_composition.py`
turns a valid team into a reviewable execution plan with runtime assignments,
handoff targets, approval requirements, and cost-limit evidence. The service
deliberately stays dependency-light so package tests, publish planning, and
generated applications can execute offline.

Primary lifecycle:

1. Request approval for an external runtime with sandbox and cost metadata.
2. Approve or reject the request with reviewer notes.
3. Register approved runtimes without provider SDK dependencies.
4. Register agents against approved runtimes.
5. Register teams and validate handoff endpoints.
6. Build deterministic execution plans with runtime assignments and trace
   metadata.
7. Emit audit events for approval, runtime, agent, team, and plan changes.

## UI

`views.py` exposes dashboard, team-builder, runtime-manager,
runtime-approval-queue, governance-evidence, and execution-trace view models
for the APG Python UI shell. The route contract comes from
`capability_contract.py`, while view models are derived from live service state.

## Theme

The package uses the `agnt_agent_ops` theme contract with compact operational
tokens and components for agent cards, handoff graphs, runtime matrices, and
execution traces.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/agnt/test_capability_contract.py capabilities/common/agnt/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/agnt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/agnt --json
git diff --check -- capabilities/common/agnt
```
