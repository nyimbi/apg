# Fraud Detection Capability Plan

## Packet 1: Executable Local Capability

1. Replace the placeholder fraud package with a real APG package contract.
2. Define fraud configuration, dependencies, deterministic rules, UI routes,
   theme tokens, Bytewax lifecycle metadata, and provider-neutral agent roles.
3. Implement dependency-light models for fraud signals, decisions, cases, and
   evidence records.
4. Implement local runtime helpers for codes, amounts, currencies, risk scores,
   risk bands, and recommended decisions.
5. Implement `FraudDetectionService` with rule-enforced lifecycle methods:
   `score_signal`, `record_decision`, `open_case`, `resolve_case`,
   `register_fraud_agent`, `validate_batch`, and dashboards.
6. Add process-local API helpers and framework-neutral view models.
7. Add `app.py`, `semantic_model.json`, `package_manifest.json`, and
   `release_report.json` evidence.
8. Add focused package tests for contract, rule engine, service lifecycle,
   guardrails, API/views, and publishable app entrypoint.
9. Update fintech metadata, the catalog README, and the progress log.

## Review Plan

- Confirm service guardrails enforce the same rule names exposed by the
  contract.
- Confirm high-impact interventions require reason and human approval.
- Confirm step-up decisions require challenge references.
- Confirm fraud does not introduce Kafka or provider-specific AI coupling.
- Confirm UI routes, theme tokens, and view models are composable.
- Confirm local tests and APG audits pass for the package.

## Focused Verification

Run battery-conscious checks for the changed slice:

```bash
./.venv/bin/python -m py_compile capabilities/fintech/fraud/__init__.py capabilities/fintech/fraud/capability_contract.py capabilities/fintech/fraud/models.py capabilities/fintech/fraud/fraud_runtime.py capabilities/fintech/fraud/service.py capabilities/fintech/fraud/api.py capabilities/fintech/fraud/views.py capabilities/fintech/fraud/app.py capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/fraud/app.py
./.venv/bin/apg capabilities inspect fintech_fraud --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/fraud --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/fraud --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/fraud --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
git diff --check
```

Full repository tests, live provider flows, and durable Bytewax deployment are
deferred unless focused verification reveals a cross-cutting issue.
