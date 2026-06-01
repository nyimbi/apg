# APG Mobile Banking

`fintech_mobile` is an executable APG capability for mobile-first banking:
customer enrollment, trusted devices, authentication factors, account links,
mobile payments, bill payments, airtime, service requests, notifications, fraud
events, and provider-neutral mobile-banking AI agents.

## What It Provides

- Mobile banking program governance by country, currency, and supported
  platform.
- Customer enrollment with KYC, consent, AML, and fraud evidence.
- Trusted device binding with attestation and risk tier.
- Authentication factor registration for passcodes, biometrics, device binding,
  OTP, and hardware keys.
- Account, wallet, and card linking.
- Mobile payment initiation for peer transfers, merchant payments, bills,
  airtime, loan repayment, savings transfer, card payment, and wallet cash-out.
- Bill payment and airtime purchase workflows.
- Service request intake and evidence.
- Notification preference and push campaign records.
- Fraud event intake with severity controls.
- First-class AI-agent composition for Codex, Claude Code, OpenCode, and Pi.

## How To Use It

Inspect the contract:

```bash
./.venv/bin/apg capabilities inspect fintech_mobile --json
```

Run the package self-test:

```bash
./.venv/bin/python capabilities/fintech/mobile/app.py
```

Use the service directly:

```python
from capabilities.fintech.mobile import MobileBankingService

service = MobileBankingService()
program = service.register_program(
	"program-1",
	"tenant-1",
	"Everyday Mobile",
	"mobile-ops",
	"KE",
	"KES",
	["ios", "android", "ussd"],
)
customer = service.enroll_customer(
	"customer-1",
	"tenant-1",
	"crm-1",
	"KE",
	"kyc-1",
	"consent-1",
	"aml-clear-1",
	"fraud-clear-1",
)
device = service.bind_device(
	"device-1",
	"tenant-1",
	customer["id"],
	"ios",
	"fingerprint-1",
	"attestation-1",
	"low",
)
factor = service.register_auth_factor(
	"factor-1",
	"tenant-1",
	customer["id"],
	device["id"],
	"biometric",
	"strength-1",
)
link = service.link_account(
	"link-1",
	"tenant-1",
	customer["id"],
	"deposit",
	"account-1",
	"KES",
	"neobank-link-1",
)
payment = service.initiate_payment(
	"payment-1",
	"tenant-1",
	customer["id"],
	device["id"],
	link["id"],
	"peer_transfer",
	2500,
	"KES",
	"recipient-1",
	"risk-clear-1",
)
```

## Composition Surfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- View models: `views.py`
- Semantic model: `semantic_model.json`
- Package manifest: `package_manifest.json`

## Guardrails

The deterministic rule engine checks tenant context, write policy, program
country/currency/platform support, customer evidence, trusted device evidence,
auth factor strength, account-link evidence, payment type/amount/currency/risk/
approval controls, biller and airtime evidence, notification consent, fraud
severity, Bytewax lifecycle processing, and privileged AI-agent approvals.

## Streaming

Lifecycle metadata uses Bytewax:

- stream: `apg.fintech.mobile.lifecycle`;
- processor: `bytewax`;
- key: `tenant_id`.

The package intentionally does not publish alternate broker settings.
