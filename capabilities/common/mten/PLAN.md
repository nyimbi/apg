# APG MTEN Capability Plan

## Slice

Implement the governed tenant provisioning, capacity approval, isolation
suspension, and live migration lifecycle as the first coherent MTEN guardrail
packet.

This turns MTEN into an executable APG foundation capability that generated
applications can compose without booting the full async production manager,
cloud orchestration, analytics, AI, or web stacks.

The next coherent packet extends that foundation with first-class tenant-agent
composition and Bytewax lifecycle-stream guardrails.

## Implementation Steps

1. Add dependency-light lifecycle records:
   - tenant environments
   - capacity approvals
   - isolation incidents
   - live migration requests
   - governance events

2. Add `mten_runtime.py`:
   - tenant-qualified stores
   - tenant registration
   - capacity approval request/decision
   - activation with DNS/capacity/isolation guardrails
   - isolation incident suspension and reactivation
   - live migration request/decision/execution evidence
   - summary and governance event helpers

3. Add package API and view helper surfaces:
   - `api_helpers.py`
   - `view_models.py`
   - shared default service state
   - dashboard, provisioning, capacity, isolation, migration, and governance
     models

4. Extend executable contract:
   - rules for capacity review, independent reviewers, encrypted isolation,
     isolation suspension, and live migration review
   - routes for capacity approvals, isolation incidents, live migrations, and
     audit timeline
   - theme components for approval queues, isolation incidents, migration
     runbooks, and governance timeline

5. Refresh package evidence:
   - contract-derived `semantic_model.json`
   - `release_report.json`
   - `package_manifest.json`
   - `cap_spec.md`

6. Replace stale package tests:
   - rename legacy generated-package test to package contract test
   - add positive tenant lifecycle coverage
   - add negative DNS, capacity, independent-review, suspension, runbook, and
     tenant-isolation coverage
   - add API-helper/view-model shared-state coverage

7. Review and focused proof:
   - py_compile package files
   - focused pytest package suite
   - implementation audit
   - publish-plan
   - stale marker search
   - diff whitespace check

## Packet 2: Tenant Agent Composition And Bytewax Guardrails

1. Extend `capability_contract.py` with:
   - supported tenant-agent runtimes `codex`, `claude_code`, `opencode`, and
     `pi`;
   - supported roles for provisioning, isolation, capacity, migration,
     optimization, compliance, and tenant support;
   - privileged-role approval rules;
   - Bytewax lifecycle stream metadata;
   - `/mten/agents` route and theme components.
2. Extend `models.py` and `mten_runtime.py` with `TenantAgentRecord`, tenant
   agent registration, tenant lifecycle batch validation, list helpers, and
   summary counts.
3. Extend `api_helpers.py` and `view_models.py` with tenant-agent and stream
   surfaces for generated applications.
4. Refresh semantic model, manifest evidence, release evidence, README,
   specification, and progress log.
5. Run focused compile, pytest, self-test, inspect, implementation-audit,
   publish-plan, service smoke, stale-marker, package-doc, and diff checks.

## Packet 3: Durable Review Evidence

1. Extend executable records with `policy_decision`, `matched_rules`,
   `review_reasons`, and `governance_evidence`.
2. Preserve denied tenant lifecycle batch validations as durable Bytewax
   evidence before raising `PermissionError`.
3. Preserve privileged tenant agents without human approval as
   `pending_review` records instead of dropping the registration attempt.
4. Expose `review_evidence` in the contract, registration metadata, semantic
   model, API helpers, view models, and package tests.
5. Add pending-review queues for capacity approvals, live migrations, tenant
   agents, and lifecycle batches.
6. Re-run focused MTEN compile, pytest, self-test, semantic JSON, implementation
   audit, lifecycle audit, publish plan, stale-marker, service-smoke, and diff
   checks.

## Non-Goals For This Slice

- Live cloud tenant provisioning.
- Live DNS validation.
- Live service-mesh or IAM mutation.
- Billing integration.
- Production persistence.
- AI optimization execution.
- Full repository test suite.

Those remain adapters after the executable lifecycle is stable.

## Review Risks

- Existing MTEN imports optional FastAPI and Flask-AppBuilder surfaces; package
  proof should exercise dependency-light helpers instead.
- Existing `app.py` embeds stale semantic evidence; replace it with
  contract-derived evidence.
- Existing tests use stale generated-package naming; rename and align with the
  current package contract terminology.
- Capacity approval must be backed by package state; caller booleans are not
  governance evidence.
- Tenant-agent registration must remain tenant-qualified and fail closed for
  unsupported runtimes, unsupported roles, and privileged roles without human
  approval.
- Bytewax metadata must stay sourced from the executable contract and appear in
  generated semantic evidence.
