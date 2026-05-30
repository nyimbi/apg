# Custom Scripting Engine Capability Specification

## Purpose

`scpt` is the APG common capability for governed scripting and workflow
extension. It lets generated applications compose tenant-scoped script
definitions, package policies, sandboxes, approvals, deterministic execution
records, AI scripting agents, audit events, UI screens, visual theming, and
Bytewax event-stream policy.

## Scope

The capability must support:

- Tenant-local package policies with owner, allowed packages, blocked imports,
  secret access, filesystem access, network policy, approval actor, and state.
- Sandboxes with owner, runtime language, isolation mode, runtime limit, memory
  limit, network posture, health evidence, review state, block/retire reasons,
  and state-change audit.
- Script definitions with readable name, owner, language, source checksum,
  version, state, review status, requested permissions, detected dangerous
  permissions, package policy, sandbox policy, workflow bindings, publication
  actor, retirement reason, and tags.
- Approval records with type, approver, reason, evidence reference, status, and
  decision timestamps.
- Execution records with requested actor, Bytewax event stream, input/output,
  logs, status, runtime, memory, timeout flag, cancellation reason, completion
  evidence, start timestamp, and completion timestamp.
- AI scripting agents as first-class records, with supported runtime, role,
  scope, registration actor, status, and visible contribution disclosure.
- Bytewax-backed event-stream configuration for runtime events and batch script
  mutations.
- UI route contracts and dependency-light view models for generated
  applications.

## Dependencies

Required:

- `wflo` for workflow extension composition.
- `secu` for production security policy composition.
- `auth` for actor, approver, and permission composition.
- `audl` for durable scripting audit trails.

Optional:

- `schd`, `ncod`, `aicr`, `moni`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `scripts`
- `sandbox`
- `packages`
- `executions`
- `scripting_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- script owner, name, source, package policy, sandbox policy, review, publish
  approval, workflow-binding policy, and retirement reason
- package policy owner, allowlist, secret access approval, filesystem access
  approval, network policy, and blocked imports
- sandbox owner, positive resource limits, high-resource review, health
  evidence, block reason, and retirement reason
- published-script and ready-sandbox requirements for execution
- requesting actor, Bytewax stream, execution audit, runtime/memory counters,
  timeout status, cancellation reason, and completion evidence
- dangerous permission approval and network policy
- AI scripting agent registration, runtime, scope, role, and contribution
  disclosure
- scripting state-change audit
- tenant isolation
- Bytewax batch mutation enforcement

## Runtime

`service.ScptService` is the generated-application runtime. It stores
deterministic in-memory state for:

- package policies
- sandboxes
- script definitions
- approvals
- executions
- scripting agents
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine
and keeps live providers behind adapter boundaries.

## UI

The UI contract exposes:

- dashboard
- workbench
- scripts
- executions
- sandboxes
- packages
- approvals
- agents
- audit
- analytics
- settings

## Production Boundary

This packet does not execute arbitrary source, install packages, spawn
containers, start WASM runtimes, call external AI-agent CLIs, run production
security scanners, or start live Bytewax workers. Those are production adapters
behind the APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI,
  theme, streaming, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative
  guardrail behavior.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json`
  match the current contract.
- Focused compile, pytest, self-test, implementation audit, publish-plan,
  stale-marker scan, and diff check pass.
