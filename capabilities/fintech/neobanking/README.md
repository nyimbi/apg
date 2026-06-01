# Digital Neobanking

Digital Neobanking is the executable APG capability for neobank programs,
customer onboarding, deposit accounts, payment-rail links, account
transactions, savings pots, statements, customer-service cases, and
provider-neutral AI-agent-assisted banking operations.

It is dependency-light by design. Generated APG applications can import the
contract, service, API helpers, view models, and app entrypoint locally, while
live core banking, issuer processor, card network, payment rail, mobile money,
customer support, notification, audit, key-management, regulator filing, and
durable Bytewax worker integrations remain behind adapter boundaries.

## Capability ID

`fintech_neobanking`

## What It Provides

- `neobank_program_governance`
- `digital_customer_onboarding`
- `deposit_account_lifecycle`
- `payment_rail_linking`
- `account_transaction_posting`
- `savings_pot_workflow`
- `statement_workflow`
- `customer_service_case_workflow`
- `neobanking_agent_workflow`

## Runtime Surfaces

- `capability_contract.py` defines configuration, dependencies, rules, UI,
  theme, and Bytewax lifecycle metadata.
- `models.py` defines dependency-light neobanking records.
- `neobanking_runtime.py` contains normalization, account-number, date, amount,
  and transaction-direction helpers.
- `service.py` provides executable tenant-scoped neobanking behavior.
- `api.py` exposes process-local helper functions.
- `views.py` exposes route, dashboard, console, and rule view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and
  `self_test()`.

## Quick Use

```python
from capabilities.fintech.neobanking.service import NeobankingService

service = NeobankingService()
program = service.register_program(
	"program-1",
	"tenant-a",
	"Everyday Bank",
	"bank-ops",
	"KE",
	"KES",
	"settlement-1",
)
customer = service.onboard_customer(
	"customer-1",
	"tenant-a",
	"crm-1",
	"kyc-1",
	"KE",
	"consent-1",
	"aml-1",
	"fraud-1",
)
account = service.open_account(
	"account-1",
	"tenant-a",
	program["id"],
	customer["id"],
	"current",
	"KES",
	1000,
)
service.post_transaction(
	"txn-1",
	"tenant-a",
	account["id"],
	"deposit",
	2500,
	"KES",
	"deposit-1",
	"risk-clear-1",
)
```

## Rules And Guardrails

Digital Neobanking fails closed. Missing tenant context, missing write policy,
incomplete program setup, incomplete customer evidence, invalid account setup,
unsupported payment rails, incomplete rail references, unsupported transaction
types, missing risk references, high-impact transactions without approval,
invalid savings pots, missing statement periods, incomplete service cases,
unsupported agent runtimes, unsupported agent roles, privileged agent actions
without approval, and non-Bytewax batch routing are denied or routed to review.

Evaluate rules directly:

```bash
./.venv/bin/apg capabilities evaluate-rules fintech_neobanking \
  --context-json '{"tenant_context_present": true, "operation": "neobanking_batch", "event_stream": "bytewax"}' \
  --json
```

## UI Composition

The contract publishes routes for dashboard, programs, customers, accounts,
rails, transactions, savings, statements, cases, agents, and settings. The view
layer returns framework-neutral models that a generated APG ERP, neobank,
wallet, payments, or customer-operations application can mount into a larger
application shell.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/fintech/neobanking/__init__.py capabilities/fintech/neobanking/capability_contract.py capabilities/fintech/neobanking/models.py capabilities/fintech/neobanking/neobanking_runtime.py capabilities/fintech/neobanking/service.py capabilities/fintech/neobanking/api.py capabilities/fintech/neobanking/views.py capabilities/fintech/neobanking/app.py capabilities/fintech/neobanking/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/neobanking/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/neobanking/app.py
./.venv/bin/apg capabilities inspect fintech_neobanking --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/neobanking --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/neobanking --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/neobanking --json
```
