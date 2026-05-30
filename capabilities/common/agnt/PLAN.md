# AGNT Capability Development Plan

## Current State

AGNT is a domain-specific package with agent runtime registration, external
runtime approval, first-class agent declarations, teams, handoff validation,
execution planning, API helpers, route metadata, theme metadata, generated
package evidence, and focused tests. The external-runtime approval lifecycle is
implemented. This packet brings AGNT up to the current common-capability
standard by adding local README coverage, Bytewax lifecycle metadata,
tenant-safe stores, stricter agent/runtime guardrails, and expanded evidence.

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

1. Extend `capability_contract.py` with stream metadata, provides/requires
   metadata, stricter rules, new routes, and richer configuration.
2. Extend `models.py` so runtimes carry tenant scope.
3. Harden `service.py` with tenant-qualified keys, runtime resolution,
   stricter agent/runtime validation, batch mutation validation, and
   tenant-local execution planning.
4. Extend `api.py` and `views.py` to expose the new lifecycle surfaces.
5. Add `README.md` and align `SPECIFICATION.md`/`cap_spec.md` with current
   behavior.
6. Regenerate `app.py`, `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
7. Run focused AGNT compile, tests, self-test, implementation audit,
   publish-plan, semantic-model, stale-marker, and diff checks.

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
- Provider SDKs and live execution remain adapter boundaries, not package
  dependencies.
