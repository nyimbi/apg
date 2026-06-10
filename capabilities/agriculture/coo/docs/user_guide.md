# Cooperative Management — User Guide

## Overview

agr_coo manages agricultural cooperatives end-to-end: registering the society,
onboarding members with share purchases, pooling bulk input procurement, computing
dividend allocations proportional to shareholding, and filing annual returns.

## Key Use Cases

- **Member Registry**: Track all members with shares held, share value, and status.
  Share ledger records every purchase, transfer, and redemption.
- **Share Transfer**: Transfer shares between members within the same cooperative.
  Balances update atomically with ledger entry.
- **Pooled Inputs**: Aggregate bulk input orders across members for volume discounts.
  Allocate quantities to individual members from the pool.
- **Dividend Allocation**: Compute per-member dividends as a percentage of profit,
  distributed proportionally to shares held by active members.
- **Annual Returns**: File yearly financials with profit, ROE calculation, and member count.

## Example Workflows

### Register a Cooperative
```
POST /api/agriculture/coo/coops
{
  "name": "Rift Valley Maize Growers Cooperative",
  "registration_number": "BN/2024/001234",
  "region": "Rift Valley",
  "crop_focus": ["maize", "wheat"],
  "share_value": 1000,
  "currency": "KES"
}
```

### Add a Member
```
POST /api/agriculture/coo/members
{
  "coop_id": "coo-abc",
  "farmer_id": "farmer-001",
  "name": "John Kamau",
  "id_number": "12345678",
  "shares_purchased": 10,
  "join_date": "2025-01-15"
}
```

### Allocate Annual Dividends
```
POST /api/agriculture/coo/dividends
{"coop_id": "coo-abc", "financial_year": "2025", "total_profit": 5000000, "dividend_rate_pct": 40}
```
