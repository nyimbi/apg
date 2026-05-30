# Accounts Payable User Guide

## Daily Workflow

1. Register or confirm the vendor with owner, tax profile, and payment method evidence.
2. Capture the invoice with vendor, invoice number, amount, currency, and document reference.
3. Match PO-backed invoices to receipt evidence.
4. Route variance, duplicate, bank-change, and policy-exception cases for review.
5. Approve invoices with separation of duties.
6. Schedule payments from an approved cash account.
7. Release payment batches after treasury or AP review.
8. Review AP aging and close the period only after exceptions and unposted invoices are cleared.

## Screens

- Dashboard: tenant summary, lifecycle counts, and stream metadata.
- Vendors: vendor onboarding and bank-change review.
- Invoices: invoice capture, duplicate handling, and status review.
- Matching: PO, receipt, and variance review.
- Approvals: approval queue and hold placement.
- Payments: payment scheduling and batch release.
- Expenses: employee reimbursement capture and policy review.
- Aging: open payable exposure.
- Close: period close blockers and close evidence.
- Agents: AP-agent registration and privileged-action review.

## AP Agents

AP agents can assist with review and preparation work. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Privileged AP actions still require human approval before execution.
