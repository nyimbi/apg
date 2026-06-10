# SACCO Deposits & Savings — User Guide

## Overview

Manages savings products, member deposit accounts, deposit/withdrawal processing, minimum balance enforcement, and interest accrual.

## Key Use Cases

1. **Product Setup** — Define savings products (regular, fixed deposit, holiday, junior) with rates and rules
2. **Account Opening** — Open member savings accounts linked to a product
3. **Deposits** — Record cash, M-Pesa, or bank transfer deposits
4. **Withdrawals** — Process withdrawals with minimum balance enforcement
5. **Interest Accrual** — Run periodic interest posting with withholding tax calculation

## Account Lifecycle

```
Opened (active) → Dormant (inactivity) → Reactivated → Active
Active → Frozen (regulatory) → Reactivated → Active
Active → Closed
```

## API Reference

### Create a Savings Product

```
POST /api/fintech/sacco/dep/products
X-Tenant-ID: sacco_abc

{
  "product_code": "REG-SAV",
  "product_name": "Regular Savings",
  "product_type": "regular",
  "interest_rate_pa": 4.5,
  "min_balance": 1000.00,
  "min_opening_balance": 500.00,
  "interest_posting_frequency": "monthly"
}
```

### Open a Savings Account

```
POST /api/fintech/sacco/dep/accounts
{
  "member_id": "mem-...",
  "product_id": "prod-...",
  "opening_balance": 1000.00
}
```

### Make a Deposit

```
POST /api/fintech/sacco/dep/deposits
{
  "account_id": "dep-...",
  "amount": 5000.00,
  "payment_reference": "MPE-XYZ-789",
  "payment_method": "mpesa",
  "recorded_by": "teller-01"
}
```

### Run Interest Accrual

```
POST /api/fintech/sacco/dep/interest/accrue
{
  "period_start": "2025-05-01",
  "period_end": "2025-05-31",
  "posting_date": "2025-06-01",
  "run_by": "system"
}
```

## Interest Calculation

`net_interest = (balance × rate_pa / 365 × days) × (1 - 0.15_WHT)`

WHT is waived for tax-exempt products.
