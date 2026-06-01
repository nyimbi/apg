# Fraud Detection Executable Capability

## Runtime Contract

- Capability ID: `fintech_fraud`
- Display name: `Fraud Detection`
- Version: `1.1.0`
- Target: `python`
- Event stream: `apg.fintech.fraud.lifecycle`
- Stream processor: `bytewax`

## Provides

- `fraud_signal_scoring`
- `transaction_risk_decisioning`
- `account_takeover_detection`
- `device_risk_detection`
- `chargeback_evidence_workflow`
- `fraud_case_management`
- `fraud_agent_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_kyc`
- `fintech_aml`

## Executable Surface

- `capability_contract.py` exposes configuration, dependencies, rules, UI,
  theme, and Bytewax metadata.
- `models.py` defines fraud signals, decisions, cases, and evidence records.
- `fraud_runtime.py` normalizes codes, amounts, currencies, scores, risk bands,
  indicators, and recommended decisions.
- `service.py` enforces deterministic fraud rules during local lifecycle
  methods.
- `api.py` exposes process-local helper functions for generated apps.
- `views.py` exposes framework-neutral dashboard, signal, and rule view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()`.

## Rule Coverage

The package currently exposes 33 deterministic rules covering tenant context,
write policy, signal evidence, supported signal type/channel, KYC linkage,
money-bearing amount/currency, risk score range, high-risk review, velocity,
device anomaly, geography anomaly, AML linkage, account takeover, chargeback
evidence, supported decisions, step-up challenge evidence, hold/block reason
and human approval, case evidence, case resolution, Bytewax routing, agent
runtime/role support, and privileged-agent approval.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/fraud/__init__.py capabilities/fintech/fraud/capability_contract.py capabilities/fintech/fraud/models.py capabilities/fintech/fraud/fraud_runtime.py capabilities/fintech/fraud/service.py capabilities/fintech/fraud/api.py capabilities/fintech/fraud/views.py capabilities/fintech/fraud/app.py capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/fraud/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/fraud/app.py
./.venv/bin/apg capabilities inspect fintech_fraud --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/fraud --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/fraud --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/fraud --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Deferred Integration Work

- Live fraud model inference and training.
- Live device fingerprinting and behavioral biometric adapters.
- Live card-network chargeback submission.
- Durable Bytewax deployment.
- Cross-institution entity-resolution graph analytics.
