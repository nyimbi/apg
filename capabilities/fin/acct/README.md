# Bank Account Management (`fin.acct`)

Regulatory bank account lifecycle engine for the APG platform.

## Features

- Open/close/freeze/unfreeze/dormant/reactivate lifecycle
- IBAN + account number generation
- Credit, debit, internal transfer (atomic)
- Fund locks (guarantees, holds, card authorisations)
- Overdraft facility with credit-approved limits
- Bulk payroll disbursement (`bulk_credit`)
- Savings sweep (`sweep_to_linked`)
- Statement generation (JSON / PDF-ready)
- Joint account signatories
- Full lifecycle audit trail
- GL journal events on every monetary operation
- NATS event emission on every state change
- CircuitBreaker on GL posting

## Quick Start

```python
from capabilities.fin.acct.service import BankAccountService
import asyncio
from decimal import Decimal

svc = BankAccountService()
acct = asyncio.run(svc.open_account("t1", "cust-001", "CURR001", "KES", opening_deposit=Decimal("5000")))
print(acct.iban, acct.available_balance)
```

## API

`POST /api/fin/acct/accounts` — see [docs/api_reference.md](docs/api_reference.md)

## Tests

```bash
python -m pytest capabilities/fin/acct/tests/ -v
```

## Docs

- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)
- [Installation](docs/installation_guide.md)
- [Troubleshooting](docs/troubleshooting_guide.md)

---

## World-Class Enhancements (v2.0)

Ten improvements that push `fin.acct` past commercial core banking systems. Full specs in [WORLD_CLASS_IMPROVEMENTS.md](WORLD_CLASS_IMPROVEMENTS.md).

**I1. Predictive Overdraft Scoring** — ML-driven overdraft limits using 90-day cash flow patterns via `common.pred`, reducing manual credit review ~70% [ML / Risk]

**I2. Real-Time Fraud Pattern Detection** — synchronous micro-model on every debit: velocity check, amount anomaly, geo-impossibility; auto-freeze at risk score > 0.9 [Security]

**I3. Interest Accrual Engine** — continuous accrual computed on `get_balance`; monthly capitalisation via `INTEREST` transaction type; eliminates batch EOD dependency [Core Banking]

**I4. Multi-Currency Sub-Accounts with FX Hedging Signals** — `get_customer_net_worth` aggregates all currency balances using live FX rates from `common.conn`; emits rebalance signals [Treasury]

**I5. Automated Dormancy-to-Escheatment Workflow** — `run_escheatment_sweep` notifies via `common.ntfy`, enforces 30-day grace period, files regulatory report via `fin.auc` [Compliance]

**I6. Account Segmentation & Behavioural Tagging** — periodic `tag_accounts` scores and labels each account (salary, high-velocity, low-balance-risk); feeds `common.recs` for cross-sell [CRM / Analytics]

**I7. Cascading Sweep with Tiered Rate Optimisation** — `tiered_sweep` cascades surplus across current → savings → FD tiers with configurable retain/max thresholds; built on existing `transfer_internal` [Treasury]

**I8. Transaction Enrichment via NLP** — payee normalisation and spend categorisation on every credit/debit via `common.nlpc`; stored in `transaction.metadata` [PFM / Reporting]

**I9. Atomic Multi-Account Journal** — `multi_ledger_transfer` validates then posts N-leg entries all-or-nothing; double-entry integrity enforced (debits == credits); required for payroll/supplier payments [Core Banking]

**I10. Customer-Visible Account Health Score** — `get_account_health_score` returns 0-100 score with grade and advice from four components: balance stability, overdraft utilisation, cash-flow ratio, positive-days [CX / Risk]

---

## New Methods

### `get_transaction_summary` — Monthly Cash Flow Snapshot

```python
summary = await svc.get_transaction_summary("t1", account_id, "2025-03")
# TransactionSummary(
#   period="2025-03", total_credits=Decimal("85000"), total_debits=Decimal("62000"),
#   net_movement=Decimal("23000"), transaction_count=41,
#   opening_balance=Decimal("12000"), closing_balance=Decimal("35000"), currency="KES"
# )
```

Use for monthly statements, PFM dashboards, and regulatory cash-flow reporting.

### `bulk_credit` — Payroll Disbursement

```python
result = await svc.bulk_credit("t1", [
    {"account_id": "acc-001", "amount": "45000", "reference": "PAY-2025-03"},
    {"account_id": "acc-002", "amount": "62000", "reference": "PAY-2025-03"},
    {"account_id": "acc-003", "amount": "38500", "reference": "PAY-2025-03"},
])
# BulkCreditResult(total=3, success_count=3, failure_count=0, succeeded=[...], failed=[])
# Failed items carry the original payload + error string — safe to retry selectively.
```

Partial failures do not roll back succeeded credits; inspect `result.failed` and retry individually.

### `get_account_stats` — Customer Portfolio Summary

```python
stats = await svc.get_account_stats("t1", "cust-001")
# AccountStats(
#   total_accounts=4, active_accounts=3, frozen_accounts=0, dormant_accounts=1,
#   total_book_balance=Decimal("247500"), total_available_balance=Decimal("231000"),
#   currencies=["KES", "USD"]
# )
```

Feeds relationship-manager dashboards and drives dormancy/upsell triggers when combined with I6 behavioural tags.
