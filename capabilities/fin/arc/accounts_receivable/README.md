# APG Accounts Receivable Capability

`arc_accounts_receivable` is the APG financial capability for customer receivables. It provides a composable lifecycle for customers, credit assessment, invoices, payment receipts, cash application, collections, disputes, aging, and receivables-focused AI agent review.

The package is intentionally dependency-light at its public boundary. Importing `capabilities.fin.arc.accounts_receivable` does not require FastAPI, Flask, databases, model providers, or live payment gateways. Live providers remain adapter work outside this executable lifecycle packet.

## Capability ID

- ID: `arc_accounts_receivable`
- Display name: `Accounts Receivable`
- Version: `2.1.0`
- Event stream: `apg.fin.arc.lifecycle`
- Stream processor: `bytewax`
- Primary package files: `capability_contract.py`, `service.py`, `api.py`, `views.py`, `app.py`

## What It Provides

- Customer receivables lifecycle.
- Credit assessment workflow with review gates for weak scores.
- Invoice lifecycle from draft to issued, partially paid, paid, disputed, and resolved.
- Payment receipt lifecycle with supported payment methods.
- Cash application workflow with overapplication blocking and unapplied cash review.
- Collection activity workflow for overdue receivables.
- Dispute opening and resolution workflow.
- Receivables aging summary.
- ARC agent registration and privileged-action guardrails.
- UI routes, screen models, theme tokens, semantic metadata, and publish evidence.

## Required Capabilities

The contract declares composition dependencies on:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `cash_management`
- `document_management`
- `business_intelligence`
- `customer_relationship_management`

The current package exposes adapter boundaries for these dependencies. It does not make live provider calls from the package import path.

## Quick Use

```python
from capabilities.fin.arc.accounts_receivable import AccountsReceivableService

svc = AccountsReceivableService()
customer = svc.create_customer(
	"cust-1",
	"tenant-1",
	"CUST-001",
	"Customer One",
	"business",
)
svc.assess_credit("credit-1", "tenant-1", customer["id"], 10000, 0.82)
invoice = svc.create_invoice(
	"inv-1",
	"tenant-1",
	customer["id"],
	"INV-001",
	"2026-05-31",
	"2026-06-30",
	[{"description": "Services", "quantity": 1, "unit_price": 500, "revenue_account": "4000"}],
)
svc.issue_invoice(invoice["id"], "tenant-1", "approver-1")
payment = svc.record_payment("pay-1", "tenant-1", customer["id"], "PAY-001", "2026-06-01", 500, "bank_transfer", "cash-1")
svc.apply_cash("apply-1", "tenant-1", payment["id"], invoice["id"], 500)
print(svc.dashboard_summary("tenant-1"))
```

## Rule And Guardrail Coverage

The deterministic rule engine enforces:

- tenant context for operations;
- policy attachment for writes;
- required customer code, legal name, and supported customer type;
- credit customer, limit, and review requirements;
- invoice customer, number, dates, due date, line, total, credit-hold, and issue approval requirements;
- payment customer, reference, date, amount, method, and cash account requirements;
- cash application payment, invoice, positive allocation, overapplication, and unapplied cash review requirements;
- collection overdue invoice, contact method, and priority requirements;
- dispute invoice, reason, owner, and resolution review requirements;
- Bytewax routing for ARC batches and lifecycle events;
- ARC agent runtime and role support;
- human approval for privileged agent actions.

## UI Surface

The contract and `views.py` expose these screens:

- Dashboard
- Customers
- Credit
- Invoices
- Payments
- Cash Application
- Collections
- Disputes
- Aging
- Agents
- Settings

The theme is `arc_accounts_receivable_control` and includes compact financial workflow tokens for panels, status chips, risk bands, queues, and review lanes.

## AI Agent Composition

ARC treats receivables agents as first-class records. Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `credit_reviewer`
- `invoice_reviewer`
- `cash_application_reviewer`
- `collections_reviewer`
- `dispute_reviewer`
- `revenue_recognition_reviewer`

Agents may prepare or review receivables work. Privileged changes require recorded human approval.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fin/arc/accounts_receivable/__init__.py capabilities/fin/arc/accounts_receivable/capability_contract.py capabilities/fin/arc/accounts_receivable/service.py capabilities/fin/arc/accounts_receivable/api.py capabilities/fin/arc/accounts_receivable/views.py capabilities/fin/arc/accounts_receivable/app.py capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fin/arc/accounts_receivable/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/arc/accounts_receivable/app.py
./.venv/bin/apg capabilities inspect arc_accounts_receivable --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/arc/accounts_receivable --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fin/arc/accounts_receivable --json
git diff --check -- capabilities/fin/arc/accounts_receivable
```

## Next Extensions

- Add durable persistence adapters.
- Wire live customer, cash, GL, document, BI, auth, audit, and notification providers.
- Add rendered browser validation for the UI shell.
- Add durable Bytewax topology deployment and replay checks.
- Add revenue recognition integration once the revenue capability is available.
