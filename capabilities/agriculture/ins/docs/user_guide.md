# Crop Insurance — User Guide

## Overview

agr_ins delivers parametric (index-based) crop insurance: products define a trigger parameter,
threshold, and payout formula. Claims are auto-adjudicated by comparing verified satellite or
weather data against the trigger — no farm visit required.

## How Parametric Insurance Works

1. **Product** defines: trigger type (e.g., rainfall_deficit), threshold (e.g., 50mm),
   payout per unit deficit (e.g., KES 200/mm), max payout, and base premium rate.
2. **Policy** is issued to a farmer for a specific crop and coverage period; premium is computed.
3. **Claim** is submitted when a trigger event is observed with evidence (satellite NDVI, gauge data).
4. **Verify** endpoint checks the observed value against the threshold and auto-calculates payout.

## Example Workflows

### Create a Rainfall Deficit Product
```
POST /api/agriculture/ins/products
{
  "name": "Maize Drought Cover 2025",
  "trigger_type": "rainfall_deficit",
  "trigger_threshold": 60,
  "trigger_unit": "mm",
  "payout_per_unit": 300,
  "max_payout": 80000,
  "coverage_period_months": 4,
  "eligible_crops": ["maize"],
  "base_premium_rate_pct": 5.0
}
```

### Issue a Policy
```
POST /api/agriculture/ins/policies
{
  "farmer_id": "farmer-001",
  "product_id": "prd-abc",
  "crop_id": "crp-xyz",
  "farm_parcel_id": "par-001",
  "sum_insured": 80000,
  "coverage_start": "2025-03-01",
  "coverage_end": "2025-07-31",
  "season": "2025A"
}
```

### Submit and Verify a Claim
```
POST /api/agriculture/ins/claims
{"policy_id": "pol-abc", "trigger_event": "below_threshold_rainfall", "trigger_value": 35,
 "observed_at": "2025-05-15", "evidence_source": "CHIRPS"}

POST /api/agriculture/ins/claims/clm-xyz/verify
{"verified_value": 38, "source": "ENACTS_satellite"}
```
