# Cash Management — User Guide

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## Overview

`cbm_cash_management` is the APG treasury liquidity packet.  It provides bank
relationship management, cash account lifecycle, position recording, forecasting,
reconciliation, investment management, payment controls, and AI agent workflows —
all in a dependency-light Python package callable from any APG composition.

---

## Quick Start

```python
from capabilities.fin.cbm.cash_management import CashManagementService
import asyncio

svc = CashManagementService(tenant_id="acme")

# Bank and account setup
bank = svc.create_bank("bank-1", "acme", "KCB", "KCB Bank Kenya")
account = svc.create_cash_account(
    "operating",
    "acme",
    bank["id"],
    "1001234567",
    "Operating Account",
    "operating",
    "KES",
    minimum_buffer=500_000,
)

# Record opening position
svc.record_cash_position("pos-1", "acme", account["id"], "2026-06-01", 12_000_000)

# Record flows
svc.record_cash_flow("flow-1", "acme", account["id"], "inflow",  3_500_000, "customer_receipt", "2026-06-05")
svc.record_cash_flow("flow-2", "acme", account["id"], "outflow", 1_200_000, "supplier_payment", "2026-06-07")

# Dashboard
print(svc.dashboard_summary("acme"))
```

---

## Core Workflows

### 1. Bank Relationships

```python
# Create a bank relationship
bank = svc.create_bank("bnk-eqty", "acme", "EQUITY", "Equity Bank")
# Connectivity status: 'manual' | 'swift' | 'open_banking'
```

### 2. Cash Accounts

```python
account = svc.create_cash_account(
    "acct-main",          # dedup key
    "acme",               # tenant
    bank["id"],
    "002-100-0001234",
    "Main USD Account",
    "operating",          # type: operating | savings | restricted | notional
    "USD",
    minimum_buffer=50_000,
)
```

Account types supported: `operating`, `savings`, `restricted`, `notional`, `escrow`,
`payroll`, `collection`, `disbursement`.

### 3. Cash Position Recording

Record end-of-day positions for each account.  Positions below `minimum_buffer` require
a `liquidity_reviewed_by` reviewer before the record is accepted.

```python
svc.record_cash_position(
    "pos-20260601",
    "acme",
    account["id"],
    "2026-06-01",
    available_balance=8_500_000,
    ledger_balance=8_600_000,
    liquidity_reviewed_by="jane.treasurer",  # required when below buffer
)
```

### 4. Cash Flow Management

```python
# Bulk import from a list
flows = [
    {"account_id": account["id"], "flow_type": "inflow",  "amount": 250_000, "category": "customer_receipt", "expected_date": "2026-06-10"},
    {"account_id": account["id"], "flow_type": "outflow", "amount": 80_000,  "category": "payroll",          "expected_date": "2026-06-25"},
]
svc.bulk_create_cash_flows(flows, tenant_id="acme")
```

Flow types: `inflow`, `outflow`, `transfer`.
Common categories: `customer_receipt`, `supplier_payment`, `payroll`, `tax_payment`,
`bank_fees`, `intercompany`, `capex`, `investing`, `financing`, `mobile_money`.

### 5. Forecasting

```python
# Deterministic forecast
forecast = svc.create_cash_forecast("fc-jun26", "acme", 30, "base", confidence_score=0.88)

# Probabilistic forecast (new — async)
async def run():
    prob = await svc.probabilistic_forecast(days=90, simulations=2000, scenario="pessimistic")
    print(prob["p5_ending_balance"], prob["p50_ending_balance"], prob["p95_ending_balance"])

asyncio.run(run())
```

Scenarios supported: `base`, `optimistic`, `pessimistic`, `stress`.

### 6. Liquidity Forecast

```python
liq = svc.liquidity_forecast(days=30, tenant_id="acme", scenario="base")
# Returns starting_balance, projected_inflows/outflows, projected_ending_balance,
# weekly_buckets
```

### 7. Bank Reconciliation

```python
# Import a statement
stmt = svc.import_bank_statement(
    "stmt-jun26",
    "acme",
    account["id"],
    raw_content=mt940_text,
    fmt="mt940",   # mt940 | camt053 | mpesa | manual
)

# Auto-reconcile
recon = svc.auto_reconcile_statement(stmt["id"], tenant_id="acme")

# Manual match when auto fails
svc.manual_match("GL-12345", "TXN-67890", tenant_id="acme", matched_by="reconciler")

# Period report
report = svc.reconciliation_report(account["id"], "2026-06", tenant_id="acme")
```

### 8. Treasury Investments

```python
inv = svc.create_treasury_investment(
    "tbill-jun26",
    "acme",
    "treasury_bill",
    "CBK",
    principal=5_000_000,
    maturity_date="2026-09-01",
    yield_rate=0.1325,
    approved_by="cfo",
)

# Maturity schedule
schedule = svc.investment_maturity_schedule(tenant_id="acme", horizon_days=180)

# Interest accrual
accrual = svc.interest_income_accrual("2026-06", tenant_id="acme")
```

### 9. Payment Runs

```python
run = svc.validate_payment_run("run-jun26", "acme", account["id"], 2_400_000, approved_by="cfo")

# Funding check at execution
check = svc.payment_run_funding_check(run["id"], tenant_id="acme")

# Priority scheduling (new — async)
async def schedule():
    sched = await svc.payment_scheduling(run["id"], urgency_threshold=500_000)
    return sched

asyncio.run(schedule())
```

### 10. FX Position and Intercompany

```python
fx = svc.fx_position("2026-06-01", tenant_id="acme", base_currency="KES")

settlement = svc.intercompany_settlement(
    "ACME_KE", "ACME_UG",
    amount=1_000_000, currency="USD",
    value_date="2026-06-10",
    approved_by="group.treasurer",
)
```

### 11. Cash Pooling

```python
# Zero-balance sweep
sweep = svc.cash_pooling_sweep("pool-group", "2026-06-30", tenant_id="acme", sweep_type="zero_balance")

# Notional pool interest (new — async)
async def pool_interest():
    pi = await svc.notional_pool_interest("pool-group", "2026-06-30", debit_rate=0.12, credit_rate=0.05)
    return pi

asyncio.run(pool_interest())
```

### 12. Mobile Money Reconciliation

```python
mpesa_txns = [
    {"date": "2026-06-01", "amount": 5000, "type": "credit",  "reference": "QGT7890ABC"},
    {"date": "2026-06-01", "amount": 2500, "type": "debit",   "reference": "QGT7891DEF"},
]
recon = svc.mobile_money_reconciliation("wallet-mpesa-001", "2026-06", tenant_id="acme", transactions=mpesa_txns)
```

---

## New Features (2026-06)

### Probabilistic Forecasting

Monte-Carlo cash-flow forecast with configurable number of simulation paths.
Returns P5/P50/P95 quantile ending balances and per-week confidence bands.
Shortfall probability (probability of ending balance below zero) is included.

```python
result = await svc.probabilistic_forecast(days=60, simulations=5000, scenario="base")
# result["p5_ending_balance"]   — worst-case 5th percentile
# result["p50_ending_balance"]  — median / most likely
# result["p95_ending_balance"]  — best-case 95th percentile
# result["shortfall_probability"] — probability of cash deficit
```

### Concentration Risk Monitoring

```python
report = await svc.concentration_risk_report(as_of_date="2026-06-01", threshold_pct=25.0)
# report["breaching_banks"]  — number of banks above threshold
# report["findings"]         — per-bank: concentration_pct, breach flag, recommendation
```

### Cash-Flow Categorisation

```python
suggestion = await svc.categorise_cash_flow(
    description="MPESA STK PUSH - Supplier ABC Ltd Invoice 1234",
    amount=87_500,
    account_id=account["id"],
)
# suggestion["suggested_category"] — "mobile_money" or "supplier_payment"
# suggestion["confidence"]          — 0.0 – 1.0
# suggestion["method"]              — "regex_pattern" | "amount_heuristic" | "fallback"
```

### Anomaly Detection

```python
anomalies = await svc.detect_anomalies(period="2026-06", sensitivity=3.0)
# anomalies["anomalies"]  — list sorted by z_score desc
# Each entry: flow_id, category, amount, z_score, recommended_action
```

### Working Capital Cycle Analytics

```python
cycle = await svc.working_capital_cycle(period="2026-06")
# cycle["dso_days"]                  — Days Sales Outstanding
# cycle["dpo_days"]                  — Days Payable Outstanding
# cycle["cash_conversion_cycle_days"] — CCC = DSO + DIO - DPO
# cycle["interpretation"]            — "efficient" | "average" | "slow_collection..."
```

### ESG Treasury Reporting

```python
esg = await svc.esg_treasury_report(
    period="2026-06",
    esg_ratings={"bank-1": "A", "bank-2": "B"},
)
# esg["esg_a_concentration_pct"] — % of deposits with ESG-A banks
# esg["scope3_kg_co2e"]          — Scope 3 payment emissions estimate
# esg["esg_policy_met"]          — True if >= 50% in ESG-A banks
```

### Basel III LCR Report

```python
lcr = await svc.lcr_report(
    as_of_date="2026-06-01",
    hqla_overrides={"acct-tbill": "L1", "acct-mmf": "L2A"},
)
# lcr["lcr_pct"]    — Liquidity Coverage Ratio percentage
# lcr["compliant"]  — True if LCR >= 100%
# lcr["hqla_total"] — Total HQLA after haircuts
```

### Payment Priority Scheduling

```python
schedule = await svc.payment_scheduling(run["id"], urgency_threshold=200_000)
# schedule["urgent_count"]   — RTGS same-day payments
# schedule["normal_count"]   — ACH next business day
# schedule["deferred_count"] — Batch/discretionary
# schedule["schedule"]       — Ordered list with channel and settlement_window
```

---

## Analytics and Reporting

```python
# Regulatory package
reg = svc.regulatory_reporting_package("cbk_liquidity", "2026-06", submitted_to="CBK")

# IAS 7 cash flow statement
ias7 = svc.ifrs_cash_flow_statement("2026-06", method="indirect")

# GAAP disclosure note
note = svc.gaap_disclosure_note("2026-06", framework="IFRS")

# Working capital ratios
wca = svc.working_capital_analysis("2026-06-01")

# Variance analysis
va = svc.cash_flow_variance_analysis("2026-06")

# Bank fee analysis
fees = svc.bank_fee_analysis("2026-06")

# Liquidity stress test
stress = svc.stress_test_liquidity("covid_shock", outflow_shock_pct=40)

# Covenant compliance
cov = svc.bank_covenant_compliance(
    "facility-001",
    "2026-06",
    covenants=[
        {"name": "DSCR",          "metric": "debt_service_coverage", "threshold": 1.25, "actual": 1.42, "direction": "min"},
        {"name": "Current Ratio", "metric": "current_ratio",          "threshold": 1.50, "actual": 1.38, "direction": "min"},
    ],
)
```

---

## AI Agents

Register autonomous review agents and gate privileged actions behind human approval:

```python
agent = svc.register_cbm_agent(
    "acme",
    name="ForecastAgent",
    runtime="claude_code",
    role="forecast_reviewer",
    scope="tenant:acme",
)

# Validate agent action — privileged actions require human_approval_recorded=True
result = svc.validate_agent_cbm_action(
    "acme",
    agent["id"],
    action="approve_payment_run",
    privileged_scope=True,
    human_approval_recorded=True,
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: `cash_position_reviewer`, `forecast_reviewer`, `liquidity_reviewer`,
`bank_reconciliation_reviewer`, `investment_reviewer`, `payment_run_reviewer`.

---

## Audit and Events

Every state-changing operation emits an audit event to the `apg.fin.cbm.lifecycle`
Bytewax stream.

```python
events = svc.audit_events("acme")
# events[0] keys: tenant_id, event_type, record_id, record_type, status,
#                 stream, processor, emitted_at
```

---

## Dashboard

```python
summary = svc.dashboard_summary("acme")
# Keys include:
#   total_cash_balance, bank_count, cash_account_count,
#   cash_position_count, cash_flow_count, forecast_count,
#   reconciliation_count, investment_count, payment_run_count,
#   bank_statement_count, intercompany_settlement_count,
#   mobile_money_recon_count, cbm_agent_count, audit_event_count
```

---

## Export

```python
export = svc.export_cash_flows("2026-06", format="csv")
# export["download_ref"] — path reference for file download adapter
```

---

## Error Reference

| Exception          | Cause                                                      |
|--------------------|------------------------------------------------------------|
| `PermissionError`  | Missing tenant context, rule engine deny/require_review    |
| `KeyError`         | Referenced entity (account, bank, payment run) not found   |
| `ValueError`       | Missing required field or invalid parameter value          |
| `AssertionError`   | Unsupported format or constraint violation                  |

---

## Composition Dependencies

The capability requires these APG platform capabilities at runtime:

- `auth` — tenant isolation
- `audl` — audit log persistence
- `ntfy` — alert notifications
- `general_ledger` — GL posting
- `accounts_payable` — AP integration
- `accounts_receivable` — AR integration
- `document_management` — statement storage
- `business_intelligence` — dashboard analytics

---

*© 2025 Datacraft. All rights reserved. www.datacraft.co.ke*
