# ACCS Capability Development Plan

## Current State

ACCS is a domain-specific accessibility governance package with deterministic
models, service behavior, API helpers, view models, contract rules, theme
metadata, generated package evidence, and focused tests. The critical-finding
review, closure, AI-agent, Bytewax batch, and tenant-scope lifecycles are
implemented. The current packet deepens release governance by adding
approved, expiring accessibility exceptions with compensating controls.

## Packet: Accessibility Exception Governance

Deliver a focused lifecycle packet:

- add an `AccessibilityException` runtime model;
- add contract rules requiring expiry and compensating controls for
  accessibility exceptions;
- expose exception registration and listing from service and API helpers;
- surface exceptions in remediation, compliance, analytics, and a dedicated
  exception-board view model;
- make publication validation report active exceptions and
  `publishable_with_exception` when every open target finding has an approved,
  active exception;
- refresh the package specification, README, `cap_spec.md`, semantic model,
  package manifest, release evidence, tests, and progress log;
- run focused ACCS proof plus catalog lifecycle/publish gates.

## Packet: Accessibility Agent And Tenant Guardrail Lifecycle

Completed lifecycle packet:

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

1. Extend `models.py` with `AccessibilityException`.
2. Extend `capability_contract.py` with exception guardrails, UI route,
   theme component, provided surface, state, and lifecycle event metadata.
3. Extend `service.py` with tenant-scoped exception recording, expiry and
   compensating-control enforcement, publication-readiness exception
   reporting, compliance summaries, and audit events.
4. Extend `api.py` and `views.py` to expose exception registration, listing,
   dashboard/remediation/compliance/analytics data, and the exception board.
5. Align `SPECIFICATION.md`, `README.md`, and `cap_spec.md` with executable
   exception behavior.
6. Regenerate `app.py`, `semantic_model.json`, and `release_report.json`.
7. Run focused ACCS compile, tests, self-test, implementation audit,
   publish-plan, lifecycle audit, strict package artifact audit, tooling audit
   if battery allows, and diff checks.

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
- Accessibility exceptions cannot bypass expiry, approver, reason, or
  compensating-control evidence.
- Publication readiness distinguishes clean publication from temporary
  `publishable_with_exception` release governance.
