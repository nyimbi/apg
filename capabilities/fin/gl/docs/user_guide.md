# General Ledger (`fin_gl`) — User Guide

**Capability**: `fin_gl`  
**Domain**: Financial  
**Version**: 1.1  
**Copyright**: © 2025 Datacraft  
**Author**: Nyimbi Odero  

---

## Overview

`fin_gl` is APG's double-entry accounting engine. Every monetary event on the platform — loan disbursement, deposit, fee, provision, interest accrual — must produce balanced journal entries through this capability. The engine enforces:

- Immutability: posted entries are never modified, only reversed
- Balance enforcement: `sum(debits) == sum(credits)` on every entry
- Period control: cannot post to a closed period
- Idempotency: batch postings keyed by `batch_id` prevent duplicates
- Audit trail: every entry carries a SHA-256 content hash

---

## Installation and Initialisation

```python
from capabilities.fin.gl.service import GLService

gl = GLService(tenant_id="acme_sacco")
gl.initialise_standard_coa()   # seeds 50+ SACCO/bank accounts
```

`tenant_id` must be a non-empty string. All data is scoped to the tenant.

---

## Chart of Accounts

### Standard COA

`initialise_standard_coa()` creates a SACCO-specific chart covering:

| Range | Type |
|-------|------|
| 1000–1999 | Assets (cash, loans, investments, fixed assets) |
| 2000–2999 | Liabilities (deposits, borrowings, payables) |
| 3000–3999 | Equity (share capital, reserves, retained surplus) |
| 4000–4999 | Income (interest, fees, recovery) |
| 5000–5999 | Expenses (interest expense, staff, admin, depreciation) |

### Create a Custom Account

```python
acc = await gl.create_account(
    code="4450",
    name="Mobile Banking Fees",
    account_type="INCOME",
    normal_balance="CREDIT",
    parent_code="4400",
    currency="KES",
)
```

`account_type` must be one of `ASSET`, `LIABILITY`, `EQUITY`, `INCOME`, `EXPENSE`.

### Browse Accounts

```python
# All active income accounts
income_accounts = await gl.list_accounts(account_type="INCOME")

# Search by name
cash_accounts = await gl.list_accounts(search="cash")

# Full hierarchy tree
tree = await gl.get_account_hierarchy()
```

---

## Period Management

Accounting periods gate all postings. A period must be open before you can post to it.

```python
await gl.open_period("2026-01", 2026, 1)
await gl.open_period("2026-02", 2026, 2)

# Check status
status = await gl.get_period_status("2026-01")  # "OPEN"

# Close at month end
await gl.close_period("2026-01", closed_by="finance_manager")
```

Once a period is closed, `post_journal_entry` raises `PostingToClosedPeriodError` for that period.

---

## Posting Journal Entries

Every entry must be balanced: total debits must equal total credits.

```python
await gl.post_journal_entry(
    entries=[
        {"account_code": "1001", "debit_amount": "50000.00", "credit_amount": "0",
         "narrative": "Cash received from member"},
        {"account_code": "2100", "debit_amount": "0", "credit_amount": "50000.00",
         "narrative": "FOSA deposit — Jane Wanjiku"},
    ],
    description="Member deposit — Jane Wanjiku",
    reference="DEP-20260115-002",
    posting_date="2026-01-15",
    period_id="2026-01",
    posted_by="teller_001",
)
```

### Reversals

Never edit a posted entry. Instead, reverse it:

```python
rev = await gl.reverse_journal_entry(
    journal_id="<original_entry_id>",
    reason="Duplicate posting error",
    reversed_by="ops_manager",
)
```

The reversal swaps all debit/credit amounts and prefixes the reference with `REV-`.

---

## Accrual Reversal Scheduling

Post accrual entries (e.g., month-end interest accruals) and have them auto-reverse at period open.

```python
# Month-end: accrue FOSA interest
je = await gl.post_accrual_entry(
    entries=[
        {"account_code": "5110", "debit_amount": "80000.00", "credit_amount": "0",
         "narrative": "FOSA interest accrual Jan 2026"},
        {"account_code": "2100", "debit_amount": "0", "credit_amount": "80000.00",
         "narrative": "FOSA interest payable"},
    ],
    description="January FOSA interest accrual",
    reference="ACCR-FOSA-2026-01",
    posting_date="2026-01-31",
    period_id="2026-01",
    reversal_date="2026-02-01",
)

# At start of February, process pending reversals
result = await gl.process_scheduled_reversals(as_of_date="2026-02-01")
print(result)
# {"as_of_date": "2026-02-01", "reversed_count": 1, "reversed_entry_ids": ["..."]}
```

---

## Recurring Journal Templates

Automate repetitive monthly, quarterly, or annual postings.

```python
# Create a monthly depreciation template
template = await gl.create_recurring_template(
    name="Monthly Computer Equipment Depreciation",
    lines=[
        {"account_code": "5500", "debit_amount": "8333.33", "credit_amount": "0",
         "narrative": "Computer equip depreciation"},
        {"account_code": "1310", "debit_amount": "0", "credit_amount": "8333.33",
         "narrative": "Accumulated depreciation"},
    ],
    frequency="MONTHLY",
    next_run_date="2026-01-31",
    period_template="%Y-%m",
    description_template="Depreciation — {period}",
    reference_prefix="DEPR",
)

# Run at month-end — processes all templates due
result = await gl.process_recurring_entries(as_of_date="2026-01-31")
print(result)
# {"posted_count": 1, "entry_ids": ["..."], "error_count": 0}
```

**Frequencies**: `MONTHLY`, `QUARTERLY`, `ANNUALLY`.

---

## Balance Queries

```python
# Current balance (O(1) — materialized)
bal = await gl.get_account_balance("1100")

# Historical balance as of a date (O(n) — scans entries)
hist_bal = await gl.get_account_balance("3300", as_of_date="2025-12-31")

# Movements within a period
movements = await gl.get_account_movements("4110", "2026-01")
# {"total_debit": "0.00", "total_credit": "125000.00", "net_movement": "-125000.00"}
```

---

## Financial Reports

### Trial Balance

```python
tb = await gl.get_trial_balance(as_of_date="2026-03-31")
# Last row is TOTALS with "balanced": True/False
```

### Profit and Loss

```python
pnl = await gl.get_profit_and_loss("2026-01-01", "2026-03-31")
print(pnl["net_surplus"])
```

### Balance Sheet

```python
bs = await gl.get_balance_sheet("2026-03-31")
print(bs["balanced"])   # True if assets == liabilities + equity
```

### Cash Flow Statement (Indirect Method)

```python
cf = await gl.get_cash_flow_statement("2026-01-01", "2026-03-31")
# {
#   "operating_activities": {"net_surplus": ..., "depreciation_addback": ..., ...},
#   "investing_activities": {"net_investing_cash_flow": ...},
#   "financing_activities": {"net_financing_cash_flow": ...},
#   "net_change_in_cash": "..."
# }
```

### Comparative Statements

Compare current period vs prior period with variance analysis:

```python
comp_pnl = await gl.get_comparative_pnl(
    current_from="2026-01-01", current_to="2026-03-31",
    prior_from="2025-01-01", prior_to="2025-03-31",
)
# Each income/expense row includes: current, prior, variance, variance_pct

comp_bs = await gl.get_comparative_balance_sheet("2026-03-31", "2025-12-31")
# Each balance row includes: current, prior, variance
```

---

## Segment / Dimension Reporting

Add a `segment` field to individual journal lines to enable branch or product-line P&L.

```python
# Post with segment tags
await gl.post_journal_entry(
    entries=[
        {"account_code": "4110", "debit_amount": "0", "credit_amount": "300000.00",
         "narrative": "BOSA interest — Mombasa", "segment": "MOMBASA"},
        {"account_code": "1110", "debit_amount": "300000.00", "credit_amount": "0",
         "narrative": "BOSA loans — Mombasa", "segment": "MOMBASA"},
    ],
    ...
)

pnl = await gl.get_segment_pnl("MOMBASA", "2026-01-01", "2026-03-31")
```

---

## Budget vs Actual

```python
# Set budgets for a period
await gl.set_account_budget("5310", "2026-01", "200000.00")   # Salaries budget
await gl.set_account_budget("5400", "2026-01", "50000.00")    # Admin budget

# Get variance report
report = await gl.get_budget_variance_report("2026-01")
for row in report["rows"]:
    print(f"{row['name']}: budget={row['budgeted']} actual={row['actual']} "
          f"variance={row['variance']} ({row['variance_pct']}%)")
```

---

## Aging Analysis

Classify outstanding balances into age buckets for provisioning and credit risk:

```python
aging = await gl.get_aging_report(
    account_code="1100",         # Member Loans - FOSA
    as_of_date="2026-03-31",
    buckets=[30, 60, 90],
)
# {
#   "buckets": {"0-30": "1200000.00", "31-60": "350000.00", "61-90": "80000.00", "91+": "20000.00"},
#   "total_outstanding": "1650000.00",
#   "lines": [...]
# }
```

---

## FX Revaluation

Revalue foreign-currency accounts and post unrealised gain/loss to the P&L:

```python
from decimal import Decimal

result = await gl.revalue_foreign_accounts_with_posting(
    fx_rates={"USD": Decimal("130.50"), "GBP": Decimal("165.20")},
    posting_date="2026-03-31",
    period_id="2026-03",
    gain_account="4400",   # Other Income
    loss_account="5600",   # Other Expenses
)
# {"entries_posted": 3, "total_gain": "12500.00", "total_loss": "2300.00",
#  "net_gain_loss": "10200.00", "journal_ids": [...]}
```

---

## Audit and Compliance

### Audit Trail

```python
trail = await gl.get_audit_trail(from_date="2026-01-01", to_date="2026-03-31")
# List of {id, posting_date, reference, description, total_debit, posted_by, posted_at}
```

### Hash Chain Verification

Verify no entries have been tampered with:

```python
result = await gl.verify_audit_chain()
if not result["chain_intact"]:
    print(f"ALERT: {result['broken_links']} broken links detected")
    for entry in result["broken_entries"]:
        print(f"  Entry {entry['entry_id']} ref={entry['reference']}")
```

### Suspense Account Check

```python
susp = await gl.check_suspense_accounts()
if not susp["clear"]:
    # Clear by posting to the correct account
    await gl.clear_suspense(
        account_code="1520",
        clearing_account="1010",
        posting_date="2026-03-31",
        period_id="2026-03",
        reason="Identified as bank fee posting error",
    )
```

---

## Year-End Close

```python
# Close the year — sweeps all income/expense accounts to retained earnings (3300)
result = await gl.close_year(year=2025, closed_by="cfo")
print(result["net_surplus"])
```

---

## Batch Postings (Idempotent)

Use `post_batch_entries` for bulk postings — safe to call multiple times with the same `batch_id`:

```python
result = await gl.post_batch_entries(
    entries_batch=[
        {
            "lines": [...],
            "description": "Loan disbursement batch",
            "reference": "DISB-BATCH-001-L1",
            "posting_date": "2026-01-20",
        },
        ...
    ],
    batch_id="DAILY-BATCH-20260120",
    period_id="2026-01",
    posted_by="system",
)
```

---

## Health Check

```python
status = await gl.health_check()
# {"status": "ok", "accounts": 52, "journal_entries": 1843, "periods": 3, "coa_balanced": True}
```

---

## Error Reference

| Exception | Cause |
|-----------|-------|
| `GLImbalanceError` | `sum(debits) != sum(credits)` in a journal entry |
| `PostingToClosedPeriodError` | Attempting to post into a closed period |
| `AccountNotFoundError` | Referenced account code does not exist in the COA |
| `ValueError` | Invalid account_type, negative amounts, zero-value entry |

---

## Composability

`fin_gl` is consumed by:

- `fin_loans` — disbursement and repayment postings
- `fin_deposits` — deposit and withdrawal postings
- `fin_interest` — interest accrual and capitalisation
- `fin_fees` — fee income recognition
- `fin_reporting` — regulatory and management reports
- `intel_dashboard` — real-time balance feeds

Inject a shared `GLService` instance via dependency injection to keep all postings within the same tenant context.
