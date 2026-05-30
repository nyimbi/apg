# Accounts Payable Specification

## Intent

Accounts Payable (`apy_accounts_payable`) makes vendor liability processing a composable APG capability. It provides executable lifecycle surfaces for vendors, invoices, matching, approvals, holds, payments, expenses, aging, period close, AP-agent review, UI routes, theming, deterministic rules, and Bytewax lifecycle streaming.

The capability is designed for generated APG applications that need AP operations to be executable immediately while still exposing the contract, guardrails, and metadata required for later durable storage and adapter integration.

## Functional Requirements

- Register tenant-scoped vendors with owner, tax profile, payment method, and optional bank-change review.
- Capture tenant-scoped invoices with registered vendor, invoice number, positive amount, currency, document evidence, and duplicate review when needed.
- Match invoices with PO and receipt evidence requirements, variance review thresholds, and lifecycle status updates.
- Approve invoices with high-value approval evidence and separation-of-duties checks.
- Place invoice holds with a reason and release holds only with approval.
- Schedule payments only for approved, unheld invoices with positive payment amount and cash account.
- Release payment batches only after review and mark associated invoices/payments as paid.
- Record expense reports with employee identity, positive amount, receipt evidence, and policy-exception review.
- Close AP periods only after open exceptions, unposted invoices, and aging review controls pass.
- Register first-class AP agents for Codex, Claude Code, OpenCode, and Pi.
- Validate privileged AI-agent AP actions through a human approval guardrail.
- Expose dashboard, vendor, invoice, matching, approval, payment, expense, aging, close, agent, and settings UI route metadata.
- Emit lifecycle events through a Bytewax-backed stream named `apg.fin.apy.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates plain context dictionaries and returns `allow`, `deny`, or `require_review`. It enforces tenant context, write policy attachment, vendor owner/tax/payment evidence, vendor bank review, invoice vendor/number/currency/amount/document evidence, duplicate invoice review, PO receipt evidence, matching variance review, invoice approval and separation of duties, payment eligibility, payment batch review, hold controls, expense employee/receipt/policy review, AP close blockers, Bytewax routing, supported AP-agent runtime and role, and human approval for privileged agent actions.

## Configuration

The contract exposes explicit configuration sections:

- `vendors`
- `invoices`
- `matching`
- `approvals`
- `payments`
- `holds`
- `expenses`
- `close`
- `ap_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Tenant overrides are passed to `get_capability_contract(tenant_id, overrides)` and deep-merged into the default configuration.

## Composition Interfaces

Provides:

- `vendor_payables_lifecycle`
- `invoice_capture_and_matching`
- `approval_workflow`
- `payment_run_lifecycle`
- `expense_reimbursement_lifecycle`
- `ap_aging_and_close`
- `ap_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `cash_management`
- `document_management`

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, deterministic rules, UI routes, theme tokens, and Bytewax streaming metadata.
- Package import exposes `AccountsPayableService`, `APService`, contract helpers, streaming metadata, and registration metadata without requiring optional web or database dependencies.
- Service supports vendor, invoice, matching, approval, hold, payment, payment-batch, expense, period-close, AP-agent, dashboard, aging, audit, batch-validation, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes AP-agent metadata, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths, guardrail failures, API/view execution, app self-test, and semantic metadata.
