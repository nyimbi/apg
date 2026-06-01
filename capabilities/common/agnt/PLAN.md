# AGNT Capability Development Plan

## Current State

AGNT is a domain-specific package with agent runtime registration, external
runtime approval, first-class agent declarations, teams, handoff validation,
execution planning, execution run evidence, API helpers, route metadata, theme
metadata, generated package evidence, and focused tests. The external-runtime
approval lifecycle is implemented. This packet deepens the execution lifecycle
by recording provider-neutral run requests with requester identity, trace sink,
side-effect approval evidence, status, and plan snapshots before adapters
invoke fast-changing providers.

## Packet: Governed Execution Run Lifecycle

Deliver a focused lifecycle packet:

- persist review-required runtime approvals and side-effecting execution runs
  with matched rules, review reasons, and audit evidence;
- add an `AgentExecutionRun` runtime model;
- add contract rules requiring requester identity, trace sink, and human
  approval for side-effecting runs;
- expose run recording and listing from service and API helpers;
- expose pending-review queues from service, API helpers, and view models;
- surface runs in dashboard, governance evidence, analytics, audit trail, and a
  dedicated run-console view model;
- keep run records provider-neutral by storing plan snapshots and trace sink
  metadata, not live provider invocations;
- refresh the package specification, README, `cap_spec.md`, semantic model,
  release evidence, tests, and progress log;
- run focused AGNT proof plus catalog lifecycle/publish gates.

## Packet: Agent Runtime And Tenant Guardrail Lifecycle

Deliver a focused lifecycle packet:

- add local README coverage for operators and generated-app composers;
- add provides/requires metadata and Bytewax lifecycle stream metadata;
- add deterministic rules for system prompt, tool allowlist, IO contract,
  memory policy, runtime cost limit, requester, reviewer, decision notes,
  execution objective, state-change audit evidence, tenant isolation, and
  Bytewax batch mutation;
- make runtime, approval, agent, and team records tenant-safe through
  tenant-qualified keys;
- keep built-in runtimes globally available while allowing tenant-specific
  runtime overrides;
- expose API helpers for Bytewax batch mutation validation;
- expose dashboard, audit-trail, analytics, and settings view-model data;
- refresh generated package evidence from the executable contract;
- update focused tests and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AgentExecutionRun`.
2. Extend `capability_contract.py` with run guardrails, UI route, theme
   component, provided surface, state, and lifecycle event metadata.
3. Extend `service.py` with tenant-scoped run recording, plan snapshots,
   requester/trace/side-effect approval enforcement, run listing, summaries,
   and audit events.
4. Extend `api.py` and `views.py` to expose run recording, listing, dashboard,
   governance, analytics, audit, and run-console data.
5. Align `SPECIFICATION.md`, `README.md`, and `cap_spec.md` with executable run
   behavior.
6. Regenerate `app.py`, `semantic_model.json`, and `release_report.json`.
7. Run focused AGNT compile, tests, self-test, implementation audit,
   publish-plan, lifecycle audit, strict package artifact audit, tooling audit
   if battery allows, and diff checks.

## Review Checklist

- External runtimes cannot be registered or used without approval.
- Runtime registrations require cost limits.
- Workspace-aware runtimes require sandbox policy.
- Agents cannot register without model, system prompt, tool allowlist, IO
  contracts, memory policy, and registered runtime.
- Tenant-local duplicate IDs do not overwrite another tenant's records.
- Teams cannot reference agents outside their tenant.
- Batch agent mutation requires Bytewax stream metadata.
- API helpers expose the same behavior as service methods.
- View models expose routes, rules, approvals, audit events, analytics,
  streaming, and theme state.
- Execution runs cannot be recorded without requester identity, trace sink, and
  side-effect approval evidence when side effects are requested.
- Execution run records store plan snapshots and audit events without invoking
  live provider adapters.
- Runtime approval and side-effect run records preserve review evidence after
  the human decision is recorded.
- Provider SDKs and live execution remain adapter boundaries, not package
  dependencies.
