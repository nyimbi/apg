# FinTech Compliance Automation

`fintech_compliance` is the APG package-backed FinTech Compliance Automation
capability. It provides executable obligation, control, check, evidence,
attestation, issue, remediation, reporting, review, and AI-agent workflows for
generated APG fintech applications.

The package is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, tests, and
release evidence without requiring live regulator filing, document signing,
external GRC suites, payment rails, ledger posting, or AI vendors.

## What It Provides

- Regulatory obligation cataloging across PCI DSS, PSD2, Open Banking, GDPR,
  SOX, Basel III, MiFID II, AML, KYC, and data privacy frameworks.
- Control mapping with owner, evidence, and test frequency metadata.
- Compliance check recording with required failure evidence.
- Evidence vault metadata with source and retention controls.
- Attestation, issue, remediation, report, and review workflows.
- Provider-neutral compliance agents for Codex, Claude Code, OpenCode, and Pi.
- UI route metadata and theme tokens for generated compliance consoles.

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_compliance --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/compliance/app.py
```

Run focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/compliance/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.compliance import ComplianceAutomationService

service = ComplianceAutomationService()
obligation = service.register_obligation(
    "obl-1", "tenant-a", "pci_dss", "control", "Protect card data",
    "owner-a", "policy-1", "2026-06-01"
)
control = service.map_control(
    "control-1", "tenant-a", obligation["id"], "preventive",
    "control-owner-a", "control-evidence-1", "monthly"
)
service.record_check(
    "check-1", "tenant-a", obligation["id"], control["id"],
    "transaction", "payment-1", "compliant"
)
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and
enforced by `service.py`. Rules cover tenant context, write-policy evidence,
obligation framework/type/owner/evidence/effective date, control ownership and
frequency, compliance check subject/result/failure evidence, evidence retention,
attestation evidence, issue severity/owner/due date, remediation approval,
report approver/evidence, review evidence, Bytewax batch routing, supported
AI-agent runtimes and roles, and human approval for privileged agent actions.

## Composition

The capability depends on APG auth, audit, notifications, NLP, keys, payments,
wallets, KYC, AML, fraud, risk, and financial reporting contracts. Live
regulator filing, signed documents, external GRC suites, and durable Bytewax
workers remain adapter responsibilities.
