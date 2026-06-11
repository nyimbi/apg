# SACCO GL — User Guide

## Overview

The SACCO General Ledger (GL) is the accounting backbone for cooperative financial institutions in Kenya. It implements the ICPAK chart of accounts standard adapted to SASRA requirements, providing full double-entry accounting with period management and SASRA-compliant financial reporting.

## Getting Started

### 1. Initialise Your Chart of Accounts

Every SACCO tenant must initialise the standard COA before posting transactions.

```http
POST /api/fintech/sacco/gl/coa/init
X-Tenant-ID: my_sacco
```

Response: 30 accounts created across Assets, Liabilities, Equity, Income, and Expenses.

This is **idempotent** — safe to call on every deployment.

### 2. Post Transactions

All transactions are immutable once posted. Standard shortcuts:

**Member Deposit (FOSA via M-Pesa):**
```json
POST /api/fintech/sacco/gl/deposits
{"member_id": "M001", "account_type": "FOSA", "amount": "50000", "channel": "mpesa"}
```
GL effect: DR Bank 50,000 / CR Member Deposits - FOSA 50,000

**Loan Disbursement:**
```json
POST /api/fintech/sacco/gl/loans/disbursements
{"loan_id": "LN-001", "amount": "200000", "loan_type": "BOSA", "disbursement_channel": "savings_account"}
```

**Loan Repayment (principal + interest):**
```json
POST /api/fintech/sacco/gl/loans/repayments
{"loan_id": "LN-001", "principal": "10000", "interest": "1500", "penalty": "0"}
```

### 3. Financial Reports

**Balance Sheet:**
```
GET /api/fintech/sacco/gl/balance-sheet?as_of_date=2025-12-31
```

**Income Statement:**
```
GET /api/fintech/sacco/gl/income-statement?from_date=2025-01-01&to_date=2025-12-31
```

**Trial Balance:**
```
GET /api/fintech/sacco/gl/trial-balance?as_of_date=2025-12-31
```

**Management Summary:**
```
GET /api/fintech/sacco/gl/summary?period=2025-12
```
Returns: total assets, loan book (gross/net), deposit base, share capital, capital ratio %, NPA ratio %.

## Period Management

### Open a Period
```json
POST /api/fintech/sacco/gl/periods/open
{"year": 2025, "month": 12}
```

### Close a Period
Validates that all entries balance before closing. Closed periods reject new postings.
```json
POST /api/fintech/sacco/gl/periods/close
{"year": 2025, "month": 11, "closed_by": "finance_manager"}
```

## Subsidiary Ledger Reconciliation

Compares GL balances against member-level totals:
```
GET /api/fintech/sacco/gl/reconciliation?as_of_date=2025-12-31
```

Returns `reconciled: true/false` with itemised differences if any.

## Double-Entry Validation

```
GET /api/fintech/sacco/gl/validate
```

Returns `balanced: true` and the difference amount. Run this after bulk imports or migrations.

## Standard GL Entries Reference

| Transaction | Debit | Credit |
|-------------|-------|--------|
| Member deposit (cash) | 1001 Cash | 2100/2110 Deposits |
| Member deposit (mobile) | 1010 Bank | 2100/2110 Deposits |
| Loan disbursement | 1100/1110 Loans | 2110 Deposits / 1001 Cash |
| Loan repayment | 1001 Cash | 1100/1110 Loans + 4100 Interest |
| Interest on savings | 5100 Interest Expense | 2100/2110 Deposits |
| Share purchase | 1001 Cash | 3200 Share Capital |
| Dividend declaration | 3300 Retained Surplus | 2300 Dividends Payable |
| Loan provision | 5200 Provisions | 1125 Provision for Loan Losses |
| Write-off | 1125 Provision | 1100/1110 Loans |

## SASRA Compliance Notes

- **Institutional Capital (3100)**: Non-distributable per SASRA regulation. Never debit without regulatory approval.
- **Capital Ratio**: Minimum 10% of total assets required. Monitor via `/summary`.
- **NPA Ratio**: Provision balance / gross loans. SASRA watches this closely.
- **Period Closing**: Always close prior period before reporting to SASRA.
