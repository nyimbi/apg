# Anti Money Laundering Capability Plan

## Packet 1: Executable Local Capability

1. Replace the placeholder package with a real APG package contract.
2. Define AML configuration, dependencies, deterministic rules, UI routes,
   theme tokens, Bytewax lifecycle metadata, and provider-neutral agent roles.
3. Implement dependency-light models for monitored transactions, alerts, cases,
   SAR drafts, agent/evidence records, and audit events.
4. Implement local runtime helpers for codes, money, scores, risk bands, and
   typology flags.
5. Implement `AntiMoneyLaunderingService` with rule-enforced lifecycle methods:
   `monitor_transaction`, `create_alert`, `triage_alert`, `open_case`,
   `draft_sar`, `register_aml_agent`, `validate_batch`, and dashboards.
6. Add process-local API helpers and framework-neutral view models.
7. Add `app.py`, `semantic_model.json`, `package_manifest.json`, and
   `release_report.json` evidence.
8. Add focused package tests for the contract, rule engine, service lifecycle,
   guardrails, API/views, and publishable app entrypoint.
9. Update fintech metadata, the catalog README, and the progress log.

## Review Plan

- Confirm the exposed rule names match service enforcement paths.
- Confirm AML does not introduce Kafka or provider-specific AI coupling.
- Confirm SAR and privileged AI-agent operations require human approval.
- Confirm alerts and cases cannot be created without tenant and evidence.
- Confirm UI routes, theme tokens, and view models are composable.
- Confirm local tests and APG audits pass for the package.

## Focused Verification

Run battery-conscious checks for the changed slice:

```bash
./.venv/bin/python -m py_compile capabilities/fintech/aml/__init__.py capabilities/fintech/aml/capability_contract.py capabilities/fintech/aml/models.py capabilities/fintech/aml/aml_runtime.py capabilities/fintech/aml/service.py capabilities/fintech/aml/api.py capabilities/fintech/aml/views.py capabilities/fintech/aml/app.py capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/aml/app.py
./.venv/bin/apg capabilities inspect fintech_aml --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/aml --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/aml --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/aml --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
git diff --check
```

Full repository tests, live provider flows, and durable Bytewax deployment are
deferred unless focused verification reveals a cross-cutting issue.
