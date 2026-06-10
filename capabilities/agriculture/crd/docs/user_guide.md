# Agricultural Credit Scoring — User Guide

## Overview

agr_crd provides a data-driven agricultural lending stack: farmer credit profiling,
yield-based scoring, seasonal loan management, group/solidarity lending, and a collateral registry.

## Credit Scoring Model

The scoring engine uses seven weighted factors (max 100 pts):

| Factor | Max Points | Basis |
|--------|-----------|-------|
| Farming experience | 25 | Years × 2.5 |
| Yield level | 25 | Avg annual yield / 500 kg |
| Annual revenue | 20 | Revenue / KES 10,000 |
| Crop diversity | 10 | 3 pts per crop type |
| Mobile money | 5 | Account present |
| Cooperative member | 10 | Membership |
| Repayment history | 5 | Settled / (settled + defaulted) |

Score → Rating → Max Loan (KES 500 per point) → Interest Rate (25% − 0.15 × score).

## Example Workflows

### Create a Credit Profile
```
POST /api/agriculture/crd/profiles
{
  "farmer_id": "farmer-001",
  "years_farming": 8,
  "crop_types": ["maize", "beans"],
  "avg_annual_yield_kg": 12000,
  "avg_annual_revenue": 350000,
  "mobile_money_account": "254712345678",
  "cooperative_member": true
}
```

### Score the Farmer
```
POST /api/agriculture/crd/score/farmer-001
```

### Apply for Seasonal Loan
```
POST /api/agriculture/crd/loans
{
  "farmer_id": "farmer-001",
  "amount": 50000,
  "purpose": "seed and fertilizer",
  "season": "2025A",
  "duration_months": 6
}
```
