# Financial Management General Ledger Specification

## Purpose

`glr_general_ledger` is the APG ledger system of record. It must let composed
applications define accounts, collect dimensional context, open accounting
periods, create balanced journals, approve and post entries, reverse posted
entries, allocate balances, produce trial balances, and coordinate GLR-specific
AI review agents.

## Scope

In scope:

- Chart of accounts lifecycle.
- Ledger dimensions.
- Accounting periods.
- Journal batches.
- Balanced journal entries.
- Posting, reversal, and allocation workflows.
- Currency-rate guardrails for foreign-currency journals.
- Trial-balance reporting.
- Audit-event emission metadata.
- Bytewax lifecycle coordination metadata.
- APG UI route, permission, and theme metadata.
- GLR-agent composition for codex, claude_code, opencode, and pi.

Out of scope for this packet:

- Durable SQL storage.
- Live exchange-rate feeds.
- Rendered report generation.
- Browser-rendered UI.
- Live auth, audit, notification, document, BI, or financial-reporting adapters.
- Durable Bytewax topology deployment.

## Domain Model

Account:

- Tenant-scoped record with code, name, type, currency, parent, posting flag,
  status, and created timestamp.
- Supported types: asset, liability, equity, revenue, expense.

Ledger dimension:

- Tenant-scoped analysis tag with name, value, owner, and status.

Accounting period:

- Tenant-scoped period with name, fiscal year, start date, end date, and status.

Journal batch:

- Tenant-scoped source and currency container bound to an open period.

Journal entry:

- Tenant-scoped balanced entry with at least two lines, debit total, credit
  total, preparer, approver, poster, and status.

Posting:

- Posted journal evidence with idempotency control and Bytewax event metadata.

Reversal:

- Approved reversal record for a posted journal with a reason.

Allocation:

- Reviewed allocation record with source account, target accounts, and basis.

GLR agent:

- Tenant-scoped AI agent record with supported runtime, role, scope, and status.

## Guardrails

The rule engine must deny:

- Missing tenant context.
- Write operations without policy attachment.
- Accounts without code, name, supported type, or valid hierarchy.
- Periods without name, fiscal year, dates, or valid date range.
- Journal batches without open period, supported source, or supported currency.
- Journals without batch, description, at least two lines, valid posting
  accounts, balanced debits and credits, or required foreign-currency rate.
- Exchange rates that are zero or negative.
- Postings without journal, approval, open period, idempotency key, or
  segregation of duties.
- Closed-period adjustments without approval.
- Reversals without posted entry or reason.
- Unbalanced trial balances.
- GLR batches or events not routed through Bytewax.
- GLR agents with unsupported runtime or role.
- Privileged agent actions without human approval.

The rule engine must require review:

- Allocations without review evidence.

## UI Contract

Routes:

- `/glr-general-ledger/dashboard`
- `/glr-general-ledger/accounts`
- `/glr-general-ledger/dimensions`
- `/glr-general-ledger/periods`
- `/glr-general-ledger/batches`
- `/glr-general-ledger/journals`
- `/glr-general-ledger/postings`
- `/glr-general-ledger/trial-balance`
- `/glr-general-ledger/allocations`
- `/glr-general-ledger/reversals`
- `/glr-general-ledger/agents`
- `/glr-general-ledger/settings`

The theme name is `glr_general_ledger_control`.

## Event Contract

Processor: `bytewax`

Stream: `apg.fin.glr.lifecycle`

Key: `tenant_id`

Events:

- `account_created`
- `dimension_recorded`
- `period_opened`
- `journal_batch_created`
- `journal_entry_created`
- `journal_approved`
- `journal_posted`
- `journal_reversed`
- `trial_balance_generated`
- `allocation_created`
- `glr_agent_registered`

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with provides,
  requires, rules, UI, theme, and Bytewax metadata.
- `GeneralLedgerService` can run the account-period-journal-posting-trial
  balance lifecycle without optional framework imports.
- Guardrails deny invalid account, journal, posting, stream, and agent actions.
- API helpers and screen-model helpers are importable without Flask or database
  dependencies.
- `app.py` exposes a semantic model, component manifest, and self-test.
- Focused package tests pass.
