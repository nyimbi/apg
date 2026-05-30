# ACCS Capability Development Plan

## Current State

ACCS is a domain-specific accessibility governance package with deterministic
models, service behavior, API helpers, view models, contract rules, theme
metadata, generated package evidence, and focused tests. The critical-finding
review and closure lifecycle is implemented. This packet brings ACCS up to the
current common-capability standard by adding local README coverage, first-class
AI accessibility agents, Bytewax lifecycle metadata, tenant-safe stores, and
expanded verification evidence.

## Packet: Accessibility Agent And Tenant Guardrail Lifecycle

Deliver a focused lifecycle packet:

- add local README coverage for operators and generated-app composers;
- add first-class accessibility-agent configuration, supported runtimes, and
  supported roles;
- add deterministic rules for agent registration, runtime, role, scope,
  contribution disclosure, state-change audit evidence, tenant isolation, and
  Bytewax batch mutation;
- add an `AccessibilityAgent` runtime model;
- make standards, targets, audits, findings, remediation tasks, reviews, and
  agents tenant-safe through tenant-qualified keys;
- emit audit events for standard, target, audit, finding, review, remediation,
  closure, and agent lifecycle changes;
- expose API helpers for agent registration, agent listing, and Bytewax batch
  mutation validation;
- expose dashboard, agent, audit-trail, analytics, and settings view-model
  data;
- refresh generated package evidence from the executable contract;
- update focused tests and progress evidence.

## Implementation Steps

1. Extend `capability_contract.py` with agent configuration, Bytewax streaming,
   provides/requires metadata, new UI routes, and guardrails.
2. Extend `models.py` with `AccessibilityAgent`.
3. Harden `service.py` with tenant-qualified keys, agent registration, event
   emission, and batch mutation validation.
4. Extend `api.py` and `views.py` to expose the new lifecycle surfaces.
5. Add `README.md` and align `SPECIFICATION.md`/`cap_spec.md` with current
   behavior.
6. Regenerate `app.py`, `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
7. Run focused ACCS compile, tests, self-test, implementation audit,
   publish-plan, semantic-model, stale-marker, and diff checks.

## Review Checklist

- Tenant-local duplicate IDs do not overwrite another tenant's records.
- Critical findings cannot close without approved review and resolution
  evidence.
- Accessibility agents cannot register without supported runtime, supported
  role, explicit scope, and contribution disclosure.
- Batch accessibility mutation requires Bytewax stream metadata.
- API helpers expose the same behavior as service methods.
- View models expose routes, rules, agents, audit events, analytics,
  streaming, and theme state.
- Provider integrations remain adapter boundaries, not local dependencies.
