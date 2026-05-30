# Accounts Receivable Implementation Plan

## Packet Goal

Build one coherent ARC lifecycle packet that can be composed by APG applications without importing optional providers. The packet must define the receivables contract, executable service lifecycle, API helpers, UI screen models, semantic metadata, focused tests, and documentation.

## Implementation Steps

1. Contract
   - Define supported customer types, currencies, invoice statuses, payment methods, dispute reasons, agent runtimes, and agent roles.
   - Define provides/requires lists for APG composition.
   - Define configuration sections for every lifecycle area.
   - Define rules for tenant safety, policy enforcement, lifecycle required fields, review gates, Bytewax routing, and agent approvals.
   - Define UI routes, theme tokens, and streaming metadata.

2. Runtime
   - Replace provider-bound service imports with `AccountsReceivableService`.
   - Implement in-memory customer, credit, invoice, payment, cash application, collection, dispute, agent, aging, and audit-event behavior.
   - Keep amounts as `Decimal`.
   - Raise `PermissionError` for denied or review-required guardrails.
   - Emit audit events with `processor = "bytewax"`.

3. API
   - Provide dependency-light function wrappers around the service.
   - Include a `create_record` smoke helper for APG tooling.
   - Expose `list_records` for composition and view models.

4. UI Models
   - Provide navigation and screen models for dashboard, customers, credit, invoices, payments, cash application, collections, disputes, aging, and agents.
   - Keep view models as plain dictionaries so renderers can consume them without framework imports.

5. App Entrypoint
   - Build semantic model dynamically from `capability_contract.py`.
   - Expose component manifest and self-test.
   - Verify Bytewax streaming, ARC agent provide, and agent screen presence.

6. Tests
   - Validate contract shape and metadata.
   - Test rule-engine denials.
   - Test a positive receivables lifecycle.
   - Test negative lifecycle guardrails.
   - Test API, views, app self-test, semantic model, and publishable manifest.

7. Documentation And Evidence
   - Replace README, add specification and plan, align `cap_spec.md`.
   - Regenerate `semantic_model.json`, `package_manifest.json`, and `release_report.json`.
   - Update `docs/progress_log.md` with focused verification evidence.

## Review Checklist

- No package-boundary imports require FastAPI, Flask, model providers, databases, or live external systems.
- Rule names and service contexts are synchronized.
- Payment and invoice amounts remain decimal-safe.
- UI route names match semantic model screens.
- Bytewax is the only declared stream processor.
- Agent runtimes and roles match the contract.
- Tests cover both successful lifecycle execution and guardrail rejection.

## Verification Plan

Battery-conscious verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/fin/arc/accounts_receivable/__init__.py capabilities/fin/arc/accounts_receivable/capability_contract.py capabilities/fin/arc/accounts_receivable/service.py capabilities/fin/arc/accounts_receivable/api.py capabilities/fin/arc/accounts_receivable/views.py capabilities/fin/arc/accounts_receivable/app.py capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/arc/accounts_receivable/app.py
./.venv/bin/apg capabilities inspect arc_accounts_receivable --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/arc/accounts_receivable --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fin/arc/accounts_receivable --json
git diff --check -- capabilities/fin/arc/accounts_receivable docs/progress_log.md
```

Full repository tests are deferred by user instruction while working on battery.
