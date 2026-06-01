# APG Agency Banking

`fintech_agency` is an executable APG capability for third-party agent networks:
program governance, outlet onboarding, teller accreditation, float management,
customer onboarding, agency transactions, liquidity movements, commission
settlement, disputes, supervision, and provider-neutral agency AI-agent
workflows.

## What It Provides

- Agency program governance by country, currency, service catalog, and
  settlement model.
- Outlet onboarding with business registration, license, location, security,
  channel, and initial float evidence.
- Individual agent/teller accreditation with training and background-check
  evidence.
- Float account controls for available balance, reserved balance, cash-in, and
  cash-out capacity.
- Customer onboarding with KYC, consent, AML, and fraud evidence.
- Cash-in, cash-out, transfers, bill payment, airtime, loan collection, loan
  disbursement, account opening, balance inquiry, mini-statement, card services,
  insurance, savings, and government-payment transaction workflows.
- Cash movement and liquidity rebalancing.
- Commission settlement with reconciliation evidence.
- Dispute and field-supervision workflows.
- First-class AI-agent composition for Codex, Claude Code, OpenCode, and Pi.

## How To Use It

Inspect the contract:

```bash
./.venv/bin/apg capabilities inspect fintech_agency --json
```

Run the package self-test:

```bash
./.venv/bin/python capabilities/fintech/agency/app.py
```

Use the service directly:

```python
from capabilities.fintech.agency import AgencyBankingService

service = AgencyBankingService()
program = service.register_program(
	"program-1",
	"tenant-1",
	"Rural Agent Network",
	"ops-owner",
	"KE",
	"KES",
	"real_time",
	["cash_in", "cash_out", "bill_payment"],
)
outlet = service.onboard_outlet(
	"outlet-1",
	"tenant-1",
	program["id"],
	"Village Shop",
	"retail_shop",
	"KE",
	"license-1",
	"location-1",
	"security-1",
	"pos_terminal",
	25000,
)
agent = service.accredit_agent(
	"agent-1",
	"tenant-1",
	outlet["id"],
	"Jane Teller",
	"id-check-1",
	"training-1",
	"background-1",
)
float_account = service.open_float_account(
	"float-1",
	"tenant-1",
	outlet["id"],
	"KES",
	50000,
	"neobank-float-1",
)
customer = service.onboard_customer(
	"customer-1",
	"tenant-1",
	"crm-1",
	"tier_2",
	"kyc-1",
	"consent-1",
	"aml-clear-1",
	"fraud-clear-1",
)
transaction = service.record_transaction(
	"txn-1",
	"tenant-1",
	outlet["id"],
	agent["id"],
	customer["id"],
	float_account["id"],
	"cash_in",
	2500,
	"KES",
	"pos_terminal",
	"customer-ref-1",
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
ownership, service catalog, settlement model, outlet licensing, location,
security, initial float, agent training/background checks, customer evidence,
transaction amount/limit/channel/service/currency, float sufficiency, commission
evidence, dispute evidence, supervision evidence, Bytewax lifecycle processing,
and privileged AI-agent approvals.

## Streaming

Lifecycle metadata uses Bytewax:

- stream: `apg.fintech.agency.lifecycle`;
- processor: `bytewax`;
- key: `tenant_id`.

The package intentionally does not publish alternate broker settings.
