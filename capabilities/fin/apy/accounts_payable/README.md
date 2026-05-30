# Accounts Payable

`apy_accounts_payable` is the APG capability for composing vendor liability, invoice, matching, approval, payment, reimbursement, close, and AP-agent workflows into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

## What It Provides

- Vendor registration with owner, tax profile, payment method, and bank-change review controls.
- Invoice capture with vendor, invoice number, currency, document evidence, duplicate review, and amount validation.
- Matching workflow for PO-backed invoices, receipt evidence, and variance review.
- Approval workflow with high-value controls and separation of duties.
- Invoice hold and hold-release lifecycle.
- Payment scheduling and payment-batch release.
- Expense report capture with receipt and policy-exception review controls.
- AP aging and period-close guardrails.
- First-class AP agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for tenant, policy, financial, agent, and stream guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.fin.apy.accounts_payable import AccountsPayableService

service = AccountsPayableService()
vendor = service.register_vendor(
    "vendor-1",
    "tenant-a",
    "Northwind Supplies",
    "ap-operations",
    "vat-profile",
    "ach",
)
invoice = service.record_invoice(
    "invoice-1",
    "tenant-a",
    vendor["id"],
    "INV-001",
    5000,
    "USD",
    "doc-001",
)
service.match_invoice(
    "tenant-a",
    invoice["id"],
    po_backed=True,
    receipt_reference="receipt-001",
)
service.approve_invoice(
    "tenant-a",
    invoice["id"],
    approved_by="controller",
    requested_by="ap-operations",
)
payment = service.schedule_payment(
    "payment-1",
    "tenant-a",
    invoice["id"],
    5000,
    "operating-cash",
    "2026-06-05",
)
service.release_payment_batch("batch-1", "tenant-a", [payment["id"]], "treasury")
summary = service.dashboard_summary("tenant-a")
```

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.fin.apy.accounts_payable import get_capability_contract

contract = get_capability_contract("tenant-a")
print(contract["provides"])
print(contract["streaming"]["processor"])
```

The contract exposes:

- `configuration`
- `configuration_schema`
- `rule_engine`
- `ui`
- `theme`
- `streaming`

## Guardrails

The rule engine blocks or routes review for:

- Missing tenant context.
- Writes without policy attachment.
- Vendors without owner, tax profile, or payment method.
- Vendor bank changes without independent review.
- Invoices without vendor, invoice number, currency, positive amount, or document reference.
- Potential duplicate invoices without review.
- PO-backed invoices without receipt evidence.
- Matching variance above threshold without review.
- High-value approvals without approval evidence.
- Self-approval by invoice requesters.
- Payments against unapproved or held invoices.
- Payments without positive amount or cash account.
- Payment batches without review.
- Holds without reason and hold releases without approval.
- Expense reports without employee, positive amount, or receipt evidence.
- Expense policy exceptions without review.
- Period close with open exceptions, unposted invoices, or missing aging review.
- Batch and lifecycle events not routed through Bytewax.
- Unsupported AP-agent runtime or role.
- Privileged AP-agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/apy-accounts-payable/dashboard`
- `/apy-accounts-payable/vendors`
- `/apy-accounts-payable/invoices`
- `/apy-accounts-payable/matching`
- `/apy-accounts-payable/approvals`
- `/apy-accounts-payable/payments`
- `/apy-accounts-payable/expenses`
- `/apy-accounts-payable/aging`
- `/apy-accounts-payable/close`
- `/apy-accounts-payable/agents`
- `/apy-accounts-payable/settings`

The default theme is `apy_accounts_payable_control`. View helpers in `views.py` return dashboard, vendor, invoice, matching, approval, payment, expense, aging, close, and agent workbench models.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `vendor_risk_reviewer`
- `invoice_exception_reviewer`
- `matching_reviewer`
- `payment_run_reviewer`
- `cash_flow_reviewer`
- `close_reviewer`

Register an agent with `register_ap_agent()` and validate privileged proposals with `validate_agent_ap_action()`.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/apy/accounts_payable/__init__.py \
  capabilities/fin/apy/accounts_payable/capability_contract.py \
  capabilities/fin/apy/accounts_payable/service.py \
  capabilities/fin/apy/accounts_payable/api.py \
  capabilities/fin/apy/accounts_payable/views.py \
  capabilities/fin/apy/accounts_payable/app.py \
  capabilities/fin/apy/accounts_payable/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/apy/accounts_payable/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/apy/accounts_payable/app.py
```

Deferred live-system work includes durable stores, live GL, cash-management, document, audit, notification, and authorization adapters, durable Bytewax deployment, rendered browser UI, and performance testing.
