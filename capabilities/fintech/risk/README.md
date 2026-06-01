# FinTech Risk Management

`fintech_risk` is the APG package-backed FinTech Risk Management capability. It
provides executable risk appetite, profile, exposure, limit, control, stress
testing, breach, risk event, review, and AI-agent workflows for generated APG
fintech applications.

The package is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, tests, and
release evidence without requiring live core banking, payment, market-data,
actuarial, model-vendor, regulator-filing, or AI-vendor integrations.

## What It Provides

- Risk appetite registration across credit, market, liquidity, operational,
  fraud, compliance, model, and third-party domains.
- Tenant-scoped risk profiles for customers, merchants, wallets, accounts,
  portfolios, loans, agents, and counterparties.
- Exposure recording with currency, source, limit, and human-approved override
  guardrails.
- Control assurance with owner, evidence, and effectiveness scoring.
- Stress scenario recording with impact, probability, and mitigation evidence.
- Limit breach and risk event intake.
- Review workflows and provider-neutral risk agents for Codex, Claude Code,
  OpenCode, and Pi.
- UI route metadata and theme tokens for generated risk consoles.

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_risk --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/risk/app.py
```

Run focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/risk/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.risk import RiskManagementService

service = RiskManagementService()
appetite = service.register_appetite(
    "appetite-1", "tenant-a", "credit", 5000000, "KES", "cro-a", "board-approval-1"
)
profile = service.create_profile(
    "profile-1", "tenant-a", "customer-a", "customer", "kyc-a", 2000000, "KES", 54, "risk-engine-1"
)
exposure = service.record_exposure(
    "exposure-1", "tenant-a", profile["id"], "credit_limit", 1500000, "KES", appetite["threshold_minor"], "loan-ledger-1"
)
service.evaluate_control(
    "control-1", "tenant-a", profile["id"], "preventive", "control-owner-a", "control-evidence-1", 82
)
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and
enforced by `service.py`. Rules cover tenant context, write-policy evidence,
risk appetite evidence, profile KYC/source/score, exposure limits, control
assurance, stress scenarios, breach/event evidence, review evidence, Bytewax
batch routing, supported AI-agent runtimes and roles, and human approval for
privileged agent actions.

## Composition

The capability depends on APG auth, audit, notifications, NLP, keys, payments,
wallets, KYC, AML, fraud, analytics, and reporting contracts. Live banking
ledgers, market feeds, model engines, regulator filing, and durable Bytewax
workers remain adapter responsibilities.
