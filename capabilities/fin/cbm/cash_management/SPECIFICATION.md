# Cash Management Specification

## Purpose

`cbm_cash_management` provides APG treasury liquidity control. It must let
composed applications manage banks, cash accounts, positions, flows, forecasts,
reconciliations, treasury investments, payment-run funding checks, and
CBM-specific AI review agents.

## Scope

In scope:

- Bank relationship lifecycle.
- Cash account lifecycle.
- Cash position recording.
- Cash flow recording.
- Cash forecasting.
- Liquidity review.
- Bank reconciliation.
- Treasury investments.
- Payment-run funding validation.
- APG UI route, permission, and theme metadata.
- Bytewax lifecycle coordination metadata.
- CBM-agent composition for codex, claude_code, opencode, and pi.

Out of scope for this packet:

- Durable SQL storage.
- Live bank API connectivity.
- Live exchange-rate feeds.
- Rendered dashboards.
- Live auth, audit, notification, document, BI, GL, AP, or AR adapters.
- Durable Bytewax topology deployment.

## Domain Model

Bank:

- Tenant-scoped relationship with code, name, connectivity status, lifecycle
  status, and created timestamp.

Cash account:

- Tenant-scoped bank account with bank, account number, name, type, currency,
  minimum buffer, lifecycle status, and created timestamp.

Cash position:

- Tenant-scoped account balance snapshot with as-of date, available balance,
  ledger balance, optional liquidity review, and status.

Cash flow:

- Tenant-scoped expected or actual cash movement with account, flow type,
  amount, category, expected date, and status.

Cash forecast:

- Tenant-scoped horizon/scenario projection with confidence score, reviewed-by
  evidence, source flow count, projected net cash, and status.

Bank reconciliation:

- Tenant-scoped comparison of bank statement and ledger balances with variance
  and review evidence.

Treasury investment:

- Tenant-scoped investment with type, counterparty, principal, maturity, yield,
  approval, and status.

Payment run:

- Tenant-scoped funding validation with funding account, payment total,
  approval evidence, and funded status.

CBM agent:

- Tenant-scoped AI agent record with supported runtime, role, scope, and status.

## Guardrails

The rule engine must deny:

- Missing tenant context.
- Write operations without policy attachment.
- Banks without code or name.
- Cash accounts without bank, account number, name, supported account type, or
  supported currency.
- Cash positions without account, as-of date, or available balance.
- Cash flows without account, supported type, positive amount, or category.
- Forecasts without positive horizon or supported scenario.
- Reconciliations without bank statement balance or ledger balance.
- Treasury investments without supported type, counterparty, maturity, or
  approval.
- Payment runs without funding account or current position.
- Payment runs that create an unapproved projected deficit.
- CBM batches or events not routed through Bytewax.
- CBM agents with unsupported runtime or role.
- Privileged agent actions without human approval.

The rule engine must require review:

- Cash positions below minimum liquidity buffer.
- Forecasts below confidence threshold.
- Bank reconciliations with material variance.

## UI Contract

Routes:

- `/cbm-cash-management/dashboard`
- `/cbm-cash-management/banks`
- `/cbm-cash-management/accounts`
- `/cbm-cash-management/positions`
- `/cbm-cash-management/flows`
- `/cbm-cash-management/forecasts`
- `/cbm-cash-management/liquidity`
- `/cbm-cash-management/reconciliation`
- `/cbm-cash-management/investments`
- `/cbm-cash-management/payment-runs`
- `/cbm-cash-management/agents`
- `/cbm-cash-management/settings`

The theme name is `cbm_cash_management_control`.

## Event Contract

Processor: `bytewax`

Stream: `apg.fin.cbm.lifecycle`

Key: `tenant_id`

Events:

- `bank_created`
- `cash_account_created`
- `cash_position_recorded`
- `cash_flow_recorded`
- `cash_forecast_created`
- `liquidity_review_recorded`
- `bank_reconciliation_recorded`
- `treasury_investment_created`
- `payment_run_validated`
- `cbm_agent_registered`

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with provides,
  requires, rules, UI, theme, and Bytewax metadata.
- `CashManagementService` can run the bank-account-position-flow-forecast-
  reconciliation-investment-payment lifecycle without optional framework
  imports.
- Guardrails deny invalid bank, account, flow, payment, stream, and agent
  actions.
- API helpers and screen-model helpers are importable without FastAPI,
  SQLAlchemy, Redis, or bank SDK dependencies.
- `app.py` exposes a semantic model, component manifest, and self-test.
- Focused package tests pass.
