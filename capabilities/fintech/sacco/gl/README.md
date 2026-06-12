# SACCO General Ledger (fintech_sacco_gl)

SASRA-compliant double-entry general ledger for SACCOs. Implements the ICPAK chart of accounts standard with 30 standard accounts across 5 categories.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/gl/health | Service health |
| POST | /api/fintech/sacco/gl/coa/init | Initialise standard COA |
| POST | /api/fintech/sacco/gl/transactions | Generic double-entry posting |
| POST | /api/fintech/sacco/gl/deposits | Post member deposit |
| POST | /api/fintech/sacco/gl/loans/disbursements | Post loan disbursement |
| POST | /api/fintech/sacco/gl/loans/repayments | Post loan repayment |
| POST | /api/fintech/sacco/gl/interest | Post interest earned |
| POST | /api/fintech/sacco/gl/dividends | Post dividend |
| POST | /api/fintech/sacco/gl/shares | Post share purchase |
| POST | /api/fintech/sacco/gl/withdrawals | Post withdrawal |
| POST | /api/fintech/sacco/gl/provisions | Post loan loss provision |
| POST | /api/fintech/sacco/gl/write-offs | Post loan write-off |
| GET | /api/fintech/sacco/gl/accounts/{code}/balance | Account balance |
| GET | /api/fintech/sacco/gl/trial-balance | Trial balance |
| GET | /api/fintech/sacco/gl/balance-sheet | Balance sheet |
| GET | /api/fintech/sacco/gl/income-statement | Income statement |
| GET | /api/fintech/sacco/gl/journal-entries | Journal entries |
| GET | /api/fintech/sacco/gl/summary | GL summary metrics |
| GET | /api/fintech/sacco/gl/validate | Validate double-entry |
| POST | /api/fintech/sacco/gl/periods/open | Open accounting period |
| POST | /api/fintech/sacco/gl/periods/close | Close accounting period |
| GET | /api/fintech/sacco/gl/periods/{year}/{month} | Period status |
| GET | /api/fintech/sacco/gl/reconciliation | Subsidiary reconciliation |

## Chart of Accounts

| Code | Name | Category |
|------|------|----------|
| 1001 | Cash | Asset |
| 1010 | Bank | Asset |
| 1100 | Member Loans - FOSA | Asset |
| 1110 | Member Loans - BOSA | Asset |
| 1120 | Non-Member Loans | Asset |
| 1125 | Provision for Loan Losses | Asset (contra) |
| 1200 | Investment Securities | Asset |
| 1300 | Fixed Assets | Asset |
| 1305 | Accumulated Depreciation | Asset (contra) |
| 1400 | Other Assets | Asset |
| 2100 | Member Deposits - FOSA | Liability |
| 2110 | Member Deposits - BOSA | Liability |
| 2200 | External Borrowings | Liability |
| 2300 | Dividends Payable | Liability |
| 2400 | Other Liabilities | Liability |
| 3100 | Institutional Capital | Equity |
| 3200 | Share Capital | Equity |
| 3300 | Retained Surplus | Equity |
| 3400 | Reserves | Equity |
| 4100 | Interest Income - Loans | Income |
| 4200 | Interest Income - Investments | Income |
| 4300 | Fee Income | Income |
| 4350 | Penalty Income | Income |
| 4400 | Other Income | Income |
| 5100 | Interest Expense | Expense |
| 5200 | Loan Loss Provisions | Expense |
| 5300 | Staff Costs | Expense |
| 5400 | Admin Expenses | Expense |
| 5500 | Depreciation | Expense |
| 5600 | Other Expenses | Expense |

## Quick Start

```python
from capabilities.fintech.sacco.gl.service import SACCOGLService
from decimal import Decimal

svc = SACCOGLService()
await svc.initialise_sacco_coa("sacco_001")
await svc.post_member_deposit("sacco_001", "M001", "FOSA", Decimal("50000"), "mpesa")
bs = await svc.get_balance_sheet("sacco_001", "2025-12-31")
```

## Authentication

Pass `X-Tenant-ID` header on all API requests for tenant isolation.

---

## World-Class Enhancements (v2.0)

Benchmarked against Temenos T24, Mambu, Finflux, and Mifos X. 15 improvements targeting SASRA/ICPAK compliance gaps.

**I1. Journal Entry Reversal Engine** — immutable counter-entry reversals with `reversal_of` linking; blocks re-reversal [Core Accounting Integrity]

**I2. Automated Accrual Engine** — daily interest accrual per loan type, DR Accrued Interest / CR Interest Income, itemised schedule [Revenue Recognition / IFRS 9]

**I3. Depreciation Posting Engine** — straight-line monthly charge from asset register, DR Depreciation (5500) / CR Accumulated Depreciation (1305) [Asset Management / IAS 16]

**I4. Multi-Currency GL Support** — FX revaluation at period-end closing rates, translation gain/loss per currency pair, KES base with FCY memo fields [Treasury / Regulatory]

**I5. SASRA PEARLS Ratios Calculator** — computes all 12 WOCCU-aligned ratios (P1, E1, A1, R8, L1, ...) with pass/fail against SASRA thresholds [Regulatory Compliance]

**I6. Bulk Transaction Import with Rollback** — two-pass validate-then-post with `dry_run` mode; unit-of-work semantics guarantee all-or-nothing atomicity [Operations / Data Integrity]

**I7. Audit Trail with Tamper-Evident Hashing** — SHA-256 chained `entry_hash` + `chain_hash` on every journal entry; `verify_chain` flag detects broken links [Audit & Compliance]

**I8. Loan Portfolio Ageing Analysis** — PAR bucket classification (current/1-30/31-90/91-180/181-365/>365), SASRA provisioning rates applied, incremental provision posted [Credit Risk / SASRA Reporting]

**I9. Intra-Period Reporting Snapshots** — O(1) point-in-time balance lookups via immutable date-keyed snapshots, avoiding full journal re-scan [Management Reporting]

**I10. Automated Closing Entries** — zeroes income/expense to Retained Surplus (3300) at year-end, validates net-surplus equality, locks the year [Period Close / IFRS]

**I11. Inter-Branch / Multi-Entity Elimination** — due-to/due-from postings with shared transaction ID; `eliminate_interentity` nets reciprocal balances at consolidation [Consolidation]

**I12. Regulatory Return Generator (SASRA Form-6)** — maps GL balances to Form 6 line items via configurable COA mapping, validates BS identity, outputs SASRA XML/XLS format [Regulatory Reporting]

**I13. Real-Time Liquidity Monitoring** — computes liquid ratio vs. SASRA 15% minimum, projects breach date from 30-day run-rate, alerts at 17% buffer [Treasury Risk]

**I14. Automated Withholding Tax on Interest** — deducts 15% KRA WHT at point of interest posting, tracks cumulative WHT Payable (2400) for P9 filing [Tax Compliance / KRA]

**I15. Configurable Approval Workflow** — maker-checker with configurable threshold; `approve_transaction` enforces approver != submitter rule before GL posting [Internal Controls / Fraud Prevention]

---

## New Methods

Three high-impact additions from v2.0.

### 1. `reverse_journal_entry` — audit-safe reversal

```python
from capabilities.fintech.sacco.gl.service import SACCOGLService

svc = SACCOGLService()

# Reverse a mis-posted journal entry
result = await svc.reverse_journal_entry(
    tenant_id="sacco_001",
    entry_id="jnl_abc123",
    reversal_date="2025-12-31",
    reversed_by="ops_manager",
    reason="Duplicate posting — batch ref TXN-9988",
)
# result["reversal_id"] — new counter-entry ID
# result["original_id"] — original entry, now is_reversed=True
```

### 2. `compute_sasra_pearls` — real-time PEARLS ratios

```python
pearls = await svc.compute_sasra_pearls(
    tenant_id="sacco_001",
    period="2025-12",
)
# Returns structured dict:
# {
#   "P1": {"value": 0.112, "threshold": 0.10, "status": "PASS"},
#   "E1": {"value": 0.043, "threshold": 0.05, "status": "PASS"},
#   "L1": {"value": 0.163, "threshold": 0.15, "status": "PASS"},
#   ...  # all 12 ratios
# }
for ratio, data in pearls.items():
    if data["status"] == "FAIL":
        print(f"BREACH: {ratio} = {data['value']:.1%} (min {data['threshold']:.1%})")
```

### 3. `post_bulk_transactions` — atomic batch posting

```python
from decimal import Decimal

transactions = [
    {
        "transaction_type": "salary_deduction",
        "entries": [
            {"account_code": "2100", "debit": Decimal("5000"), "credit": Decimal("0"), "narrative": "M001 deduction"},
            {"account_code": "1001", "credit": Decimal("5000"), "debit": Decimal("0"), "narrative": "Cash outflow"},
        ],
        "reference": "PAY-2025-12-001",
        "value_date": "2025-12-31",
        "posted_by": "payroll_sys",
    },
    # ... hundreds more
]

# Dry-run first — validates all entries without posting
validation = await svc.post_bulk_transactions(
    tenant_id="sacco_001",
    transactions=transactions,
    posted_by="payroll_sys",
    dry_run=True,
)
assert validation["errors"] == [], validation["errors"]

# Full atomic post — rolls back all if any entry fails
result = await svc.post_bulk_transactions(
    tenant_id="sacco_001",
    transactions=transactions,
    posted_by="payroll_sys",
    dry_run=False,
)
print(f"Posted {result['posted_count']} entries atomically")
```
