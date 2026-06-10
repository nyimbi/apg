# SACCO Dividend & Distribution — User Guide

## Overview

Manages the annual financial cycle: surplus calculation, board-level dividend declaration, per-member dividend and rebate computation, payment processing, and KRA WHT filing.

## Annual Cycle

```
Create Financial Year → Update Income/Expenses → Allocate Surplus
→ Declare Dividend (board resolution) → Compute Member Distributions (bulk)
→ Run Payment Batch → Generate WHT Return → Close Financial Year
```

## Surplus Allocation

Gross surplus is split into pools by percentage:
- Statutory reserve (default 20%) — required by SACCO Societies Act
- Education fund (default 5%)
- Dividend pool (paid on share capital)
- Rebate pool (paid on savings/deposits)
- Retained surplus (remainder)

## Dividend & Rebate Formula

```
dividend_gross = share_capital × dividend_rate_pct / 100
rebate_gross   = savings_balance × rebate_rate_pct / 100
gross_total    = dividend_gross + rebate_gross
WHT            = gross_total × 5%   (KRA withholding tax)
net_payable    = gross_total - WHT
```

## API Reference

### Create a Financial Year

```
POST /api/fintech/sacco/div/years
X-Tenant-ID: sacco_abc

{
  "year_code": "FY2025",
  "start_date": "2025-01-01",
  "end_date": "2025-12-31"
}
```

### Allocate Surplus

```
POST /api/fintech/sacco/div/years/{year_id}/allocate
{
  "total_income": 12500000.00,
  "total_expenses": 8000000.00,
  "statutory_reserve_pct": 20,
  "education_fund_pct": 5,
  "dividend_pool_pct": 50,
  "rebate_pool_pct": 15,
  "allocation_approved_by": "board-chairman",
  "allocation_date": "2026-02-15"
}
```

### Declare Dividend

```
POST /api/fintech/sacco/div/declarations
{
  "year_id": "fy-...",
  "dividend_rate_pct": 12.0,
  "rebate_rate_pct": 4.0,
  "declared_by": "ceo-001",
  "board_resolution_ref": "BR-2026-001",
  "declaration_date": "2026-02-20",
  "payment_date": "2026-03-31"
}
```

### Bulk Compute & Pay

```
POST /api/fintech/sacco/div/distributions/bulk-compute
{
  "declaration_id": "decl-...",
  "members": [
    {"member_id": "mem-001", "member_number": "MEM-SACC-000001", "share_capital": 50000, "savings_balance": 200000, "payment_method": "savings_credit"},
    ...
  ]
}

POST /api/fintech/sacco/div/declarations/{id}/pay-all
{ "run_by": "finance-officer-01" }
```

### File WHT Return

```
POST /api/fintech/sacco/div/wht
{
  "declaration_id": "decl-...",
  "filed_by": "accountant-01",
  "kra_return_reference": "KRA-WHT-2026-001"
}
```
