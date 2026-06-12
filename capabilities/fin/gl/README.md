# General Ledger (`fin_gl`)

Double-entry accounting engine for APG core banking. Every monetary transaction on the platform produces balanced journal entries. Immutable, tenant-scoped, period-controlled.

## Key invariants

- `sum(debits) == sum(credits)` on every journal entry — raises `GLImbalanceError` otherwise
- Posted entries are **immutable** — reverse to correct, never edit
- Cannot post to a closed period — raises `PostingToClosedPeriodError`
- All amounts are `Decimal` — never `float`

## Quick start

```python
from capabilities.fin.gl.service import GLService

gl = GLService(tenant_id="acme_sacco")
gl.initialise_standard_coa()
await gl.open_period("2026-01", 2026, 1)

# Post a member deposit
await gl.post_journal_entry(
    entries=[
        {"account_code": "1001", "debit_amount": "10000.00", "credit_amount": "0", "narrative": "Cash received"},
        {"account_code": "2100", "debit_amount": "0", "credit_amount": "10000.00", "narrative": "Member FOSA deposit"},
    ],
    description="Member deposit — John Doe",
    reference="DEP-20260101-001",
    posting_date="2026-01-01",
    period_id="2026-01",
)
```

## API

| Endpoint | Description |
|----------|-------------|
| `POST /api/fin/gl/coa/initialise` | Seed standard SACCO/bank COA |
| `GET /api/fin/gl/accounts` | List accounts |
| `POST /api/fin/gl/accounts` | Create account |
| `GET /api/fin/gl/accounts/<code>/balance` | Get account balance |
| `POST /api/fin/gl/journal-entries` | Post journal entry |
| `POST /api/fin/gl/journal-entries/<id>/reverse` | Reverse entry |
| `GET /api/fin/gl/reports/trial-balance` | Trial balance |
| `GET /api/fin/gl/reports/profit-and-loss` | P&L |
| `GET /api/fin/gl/reports/balance-sheet` | Balance sheet |
| `GET /api/fin/gl/health` | Health check |

---

## World-Class Enhancements (v2.0)

- **I1.** General Ledger — World-Class Improvements
- **I2.** Accrual Reversal Scheduling
- **I3.** Cash Flow Statement Generation
- **I4.** Multi-Currency Revaluation with Gain/Loss Posting
- **I5.** Segment / Dimension Reporting
- **I6.** Intercompany Elimination Engine
- **I7.** Straight-Line Depreciation Scheduler
- **I8.** Budget vs Actual Variance Analysis
- **I9.** Period Locking with Hard/Soft Lock Distinction
- **I10.** Aging Analysis for Receivables and Payables
- **I11.** Audit Log with Tamper-Evidence Chaining
- **I12.** Recurring Journal Entry Templates
- **I13.** Interbank Reconciliation Statement
- **I14.** Deferred Revenue / Prepaid Expense Amortisation
- **I15.** Comparative Period Financial Statements

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
