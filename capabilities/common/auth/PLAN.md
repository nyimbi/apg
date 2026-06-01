# AUTH Capability Development Plan

## Current State

AUTH has a production-oriented authentication module plus a dependency-light
package surface for generated APG applications. The package already covers
identity registration, tenant-aware role governance, role-assignment approval,
session lifecycle, access decisions, privacy-budget approvals, audit events,
API helpers, view models, and generated package evidence.

The current packet closes the remaining composition gap for durable AUTH review
evidence: security agents must be first-class, governed participants; privileged
agent registration without human approval must create reviewable evidence
instead of disappearing as a transient exception; batch AUTH mutation must
declare the Bytewax lifecycle stream; denied non-Bytewax batch mutation attempts
must persist denial evidence; and package documentation must describe how
builders compose AUTH into executable applications.

## Packet: Security-Agent And Bytewax Governance

Deliver a focused lifecycle and guardrail packet:

- add a practical root `README.md`;
- keep `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` aligned with the
  executable package surface;
- add first-class AI security-agent configuration for Codex, Claude Code,
  OpenCode, and Pi style runtimes;
- require security-agent registration, supported runtime, supported role,
  explicit scope, owner, purpose, contribution disclosure, and human approval
  for privileged roles;
- add `AuthSecurityAgent` model state;
- add durable policy/review evidence to role approvals, privacy approvals,
  privacy queries, security agents, batch AUTH mutations, and audit events;
- add `AuthBatchMutationEvidence` model state;
- add service, API-helper, and view-model methods for security-agent
  registration, listing, pending reviews, and batch mutation evidence;
- add Bytewax stream metadata to the contract and generated semantic model;
- require Bytewax for batch AUTH mutation intent;
- expose security-agent, audit, analytics, and settings UI state;
- clean stale login/dashboard route names in the contract;
- refresh generated package evidence from the live contract;
- extend focused tests for the new rules and view/API surfaces;
- record progress evidence and commit the verified slice.

## Implementation Steps

1. Extend `capability_contract.py` with security-agent, governance,
   observability, adapter, UI, theme, provides/requires, and Bytewax stream
   metadata.
2. Extend the rule engine with security-agent registration/runtime/role/scope/
   disclosure/privileged-approval guardrails and Bytewax batch mutation
   enforcement.
3. Add `AuthSecurityAgent` to `models.py`.
4. Extend reviewable models with `policy_decision`, `matched_rules`,
   `review_reasons`, and `review_evidence`.
5. Add `AuthBatchMutationEvidence` to `models.py`.
6. Extend `AuthService` with tenant-qualified security-agent state,
   registration, listing, pending review queues, dashboard counts, token
   normalization, durable batch mutation validation, and audit evidence.
7. Extend `api_helpers.py` with security-agent, pending-review, and batch
   mutation helpers.
8. Extend `view_models.py` with security-agent, pending-review, analytics,
   audit, settings, review-evidence, and stream surfaces.
9. Update focused tests to cover contract shape, rule evaluation, service,
   API helpers, view models, generated package evidence, and Bytewax metadata.
10. Replace stale `cap_spec.md` content with a pointer to the active spec.
11. Add `README.md` and update `SPECIFICATION.md`.
12. Regenerate `app.py`, `semantic_model.json`, `package_manifest.json`, and
    `release_report.json` from the contract.
13. Run focused compile, package tests, self-test, implementation audit,
    publish-plan, stale-marker scan for touched files, and diff checks.

## Review Checklist

- Security-agent runtime and role values normalize predictable CLI/provider
  names.
- Unsupported agent runtimes fail closed.
- Unsupported agent roles fail closed.
- Missing security-agent scope fails closed.
- Undisclosed security-agent contribution fails closed.
- Privileged security-agent role registration without human approval is
  retained as `pending_review` evidence.
- Batch AUTH mutation fails unless `event_stream` is `bytewax` and denied
  validations persist evidence before `PermissionError`.
- `list_pending_reviews()` composes role approval, privacy approval, privacy
  query, security-agent, and batch review records for generated review consoles.
- Reviewable records expose `policy_decision`, `matched_rules`,
  `review_reasons`, and `review_evidence`.
- Tenant-qualified state remains intact for identity, role, approval,
  assignment, session, access, privacy, security-agent, and audit records.
- API helpers expose the same behavior as service methods.
- View models expose dashboard, agent, audit, analytics, settings, and stream
  state.
- Generated semantic model exposes the current route names, provides/requires
  metadata, first-class security-agent composition metadata, and Bytewax stream
  metadata.
- Production JWT, biometric, behavioral, cryptographic, federation, and web
  stacks remain adapter boundaries.

## Verification Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/auth/models.py capabilities/common/auth/service.py capabilities/common/auth/api_helpers.py capabilities/common/auth/view_models.py capabilities/common/auth/capability_contract.py capabilities/common/auth/app.py capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.auth import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/auth --json
./.venv/bin/apg capabilities publish-plan capabilities/common/auth --json
git diff --check -- capabilities/common/auth docs/progress_log.md
```
