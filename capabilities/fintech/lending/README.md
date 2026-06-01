# Digital Lending

Digital Lending is the executable APG capability for credit products,
borrower onboarding, applications, underwriting, offers, disbursements,
repayments, collections, and lending-agent review workflows.

It is dependency-light by design. Generated APG applications can import the
contract, service, API helpers, view models, and app entrypoint locally, while
live credit-bureau, statement-analysis, payment, wallet, card, servicing,
collections, notification, audit, key-management, and Bytewax worker
integrations remain behind adapter boundaries.

## Capability ID

`fintech_lending`

## What It Provides

- `loan_product_governance`
- `borrower_lifecycle`
- `credit_application_workflow`
- `underwriting_decisioning`
- `loan_offer_workflow`
- `disbursement_control`
- `repayment_schedule_workflow`
- `collections_workflow`
- `lending_agent_workflow`

## Runtime Surfaces

- `capability_contract.py` defines configuration, dependencies, rules, UI,
  theme, and Bytewax lifecycle metadata.
- `models.py` defines dependency-light lending records.
- `lending_runtime.py` contains domain normalization and installment helpers.
- `service.py` provides executable tenant-scoped lending behavior.
- `api.py` exposes process-local helper functions for generated applications.
- `views.py` exposes route, dashboard, console, and rule view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and
  `self_test()`.

## Quick Use

```python
from capabilities.fintech.lending.service import LendingService

service = LendingService()
product = service.register_product(
	"product-1",
	"tenant-a",
	"SME Working Capital",
	"credit-ops",
	"term_loan",
	"KES",
	1000,
	200000,
	30,
	365,
	0.24,
	"monthly",
)
borrower = service.onboard_borrower(
	"borrower-1",
	"tenant-a",
	"customer-1",
	"kyc-1",
	"KE",
	"income-1",
	"consent-1",
)
application = service.submit_application(
	"application-1",
	"tenant-a",
	borrower["id"],
	product["id"],
	50000,
	"working_capital",
	"affordability-1",
	"statement-1",
	"aml-1",
	"fraud-1",
	"card-activity-1",
)
decision = service.record_underwriting(
	"uw-1",
	"tenant-a",
	application["id"],
	720,
	"approve",
	["scorecard-1"],
	"credit-manager-1",
)
offer = service.issue_offer(
	"offer-1",
	"tenant-a",
	application["id"],
	decision["id"],
	50000,
	0.24,
	180,
	"2026-07-01",
	"accepted",
	"acceptance-1",
)
```

## Rules And Guardrails

Digital Lending fails closed. Missing tenant context, missing write policy,
invalid product terms, incomplete borrower evidence, incomplete credit-file
evidence, unsupported underwriting decisions, missing adverse-action reasons,
final decisions without approval, accepted offers without borrower acceptance,
disbursements without accepted offers or approval, invalid repayment schedules,
incomplete collection cases, unsupported lending-agent runtimes, unsupported
lending-agent roles, and non-Bytewax batch processing are denied or routed to
review.

Evaluate rules directly:

```bash
./.venv/bin/apg capabilities evaluate-rules fintech_lending \
  --context-json '{"tenant_context_present": true, "operation": "lending_batch", "event_stream": "bytewax"}' \
  --json
```

## UI Composition

The contract publishes routes for dashboard, products, borrowers,
applications, underwriting, offers, disbursements, repayments, collections,
agents, and settings. The view layer returns framework-neutral models that a
generated APG application can mount into a larger ERP, banking, or credit
operations shell.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/fintech/lending/__init__.py capabilities/fintech/lending/capability_contract.py capabilities/fintech/lending/models.py capabilities/fintech/lending/lending_runtime.py capabilities/fintech/lending/service.py capabilities/fintech/lending/api.py capabilities/fintech/lending/views.py capabilities/fintech/lending/app.py capabilities/fintech/lending/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/lending/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/lending/app.py
./.venv/bin/apg capabilities inspect fintech_lending --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/lending --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/lending --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/lending --json
```
