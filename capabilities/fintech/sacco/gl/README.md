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
