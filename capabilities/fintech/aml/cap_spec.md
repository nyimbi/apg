# Anti Money Laundering Executable Capability

## Runtime Contract

- Capability ID: `fintech_aml`
- Display name: `Anti Money Laundering`
- Version: `1.1.0`
- Target: `python`
- Event stream: `apg.fintech.aml.lifecycle`
- Stream processor: `bytewax`

## Provides

- `transaction_monitoring`
- `aml_alert_triage`
- `sanctions_pep_escalation`
- `suspicious_activity_case_management`
- `sar_workflow`
- `typology_rule_engine`
- `aml_agent_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`

## Executable Surface

- `capability_contract.py` exposes configuration, dependencies, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines transaction, alert, case, SAR, and evidence records.
- `aml_runtime.py` normalizes codes, currencies, money values, risk scores, and
  typology flags.
- `service.py` enforces deterministic AML rules during local lifecycle methods.
- `api.py` exposes process-local helper functions for generated apps.
- `views.py` exposes framework-neutral dashboard, alert, and rule view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()`.

## Rule Coverage

The package currently exposes 31 deterministic rules covering tenant context,
write policy, transaction evidence, KYC linkage, large transactions, velocity,
structuring, sanctions, high-risk KYC, alert evidence, triage disposition,
reviewer assignment, case evidence, SAR evidence, SAR human approval, Bytewax,
agent runtime/role support, and privileged-agent approval.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/aml/__init__.py capabilities/fintech/aml/capability_contract.py capabilities/fintech/aml/models.py capabilities/fintech/aml/aml_runtime.py capabilities/fintech/aml/service.py capabilities/fintech/aml/api.py capabilities/fintech/aml/views.py capabilities/fintech/aml/app.py capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/aml/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/aml/app.py
./.venv/bin/apg capabilities inspect fintech_aml --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/aml --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/aml --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/aml --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Deferred Integration Work

- Live sanctions/PEP/adverse-media providers.
- Regulator SAR/STR submission adapters.
- Durable Bytewax deployment.
- Entity-resolution graph analytics.
- Model-backed anomaly detection.
