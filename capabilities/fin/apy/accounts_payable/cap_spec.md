# Accounts Payable Capability Summary

`apy_accounts_payable` provides the APG accounts payable packet for vendor, invoice, matching, approval, hold, payment, expense, aging, close, and AP-agent composition.

## Provides

- `vendor_payables_lifecycle`
- `invoice_capture_and_matching`
- `approval_workflow`
- `payment_run_lifecycle`
- `expense_reimbursement_lifecycle`
- `ap_aging_and_close`
- `ap_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `cash_management`
- `document_management`

## Execution Model

The package is executable without optional web or database dependencies. `AccountsPayableService` owns the in-memory lifecycle state, evaluates deterministic rules before writes, emits audit events with Bytewax stream metadata, and exposes summaries for generated applications.

## Composition Metadata

- Event processor: `bytewax`
- Stream: `apg.fin.apy.lifecycle`
- Theme: `apy_accounts_payable_control`
- UI shell: `apg_python`
- App target: `python`

## Deferred Integration

Durable storage, live GL posting, cash-management execution, document retrieval, authorization, notification, audit sinks, and durable Bytewax topologies remain adapter work after the executable package baseline.
