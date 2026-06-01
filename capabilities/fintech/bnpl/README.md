# APG Buy Now Pay Later

`fintech_bnpl` is an executable APG capability for merchant checkout credit,
consumer BNPL plans, installment scheduling, merchant settlement, dispute
servicing, and provider-neutral BNPL AI-agent workflows.

## What It Provides

- Merchant BNPL program governance.
- Consumer onboarding with KYC, AML, fraud, and consent evidence.
- Merchant profile registration by category, country, and settlement account.
- Checkout-session capture for web, mobile, POS, marketplace, and API flows.
- Affordability decisions with evidence, scoring, adverse reasons, and human
  approvals where needed.
- BNPL plan creation for pay-in-3, pay-in-4, monthly installments, and invoice
  split.
- Installment schedule state.
- Merchant settlement and reconciliation evidence.
- Dispute intake with reasons, reviewer, and supporting evidence.
- First-class AI-agent composition for Codex, Claude Code, OpenCode, and Pi.

## How To Use It

Inspect the contract:

```bash
./.venv/bin/apg capabilities inspect fintech_bnpl --json
```

Run the package self-test:

```bash
./.venv/bin/python capabilities/fintech/bnpl/app.py
```

Use the service directly:

```python
from capabilities.fintech.bnpl import BNPLService

service = BNPLService()
program = service.register_merchant_program(
	"program-1",
	"tenant-1",
	"Everyday BNPL",
	"ops-owner",
	"KE",
	"KES",
	"settlement-policy-1",
	"fee-disclosure-1",
	4,
)
consumer = service.onboard_consumer(
	"consumer-1",
	"tenant-1",
	"crm-1",
	"kyc-1",
	"KE",
	"consent-1",
	"aml-clear-1",
	"fraud-clear-1",
)
merchant = service.register_merchant(
	"merchant-1",
	"tenant-1",
	program["id"],
	"merchant-legal-1",
	"retail",
	"KE",
	"standard",
	"settlement-account-1",
)
checkout = service.create_checkout_session(
	"checkout-1",
	"tenant-1",
	merchant["id"],
	consumer["id"],
	"mobile",
	"retail",
	12000,
	"KES",
	"payment-ref-1",
	"fraud-clear-1",
	"aml-clear-1",
	"consent-1",
)
decision = service.record_affordability_decision(
	"decision-1",
	"tenant-1",
	checkout["id"],
	740,
	"approve",
	["income-1", "bureau-1"],
	"reviewer-1",
)
plan = service.create_bnpl_plan(
	"plan-1",
	"tenant-1",
	checkout["id"],
	decision["id"],
	"pay_in_4",
	12000,
	"KES",
	45,
	0,
	"fee-disclosure-1",
	"acceptance-1",
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

The deterministic rule engine checks tenant context, write policy, supported
countries and currencies, merchant and consumer evidence, checkout evidence,
affordability scoring, final decisions, plan terms, installment status,
settlement controls, dispute evidence, Bytewax lifecycle processing, and
privileged AI-agent approvals.

## Streaming

Lifecycle metadata uses Bytewax:

- stream: `apg.fintech.bnpl.lifecycle`;
- processor: `bytewax`;
- key: `tenant_id`.

The package intentionally does not publish alternate broker settings.
