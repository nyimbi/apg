# Results-Based Financing (ngo_rbf) — User Guide

## Overview

Manages RBF contracts where payments are unlocked only when verified results are achieved.
Supports output-based, outcome-based, and impact-based payment models.

## Core Workflow

```
Create RBF Contract → Define DLIs (Disbursement-Linked Indicators)
       ↓
  Activate Contract
       ↓
  Implement Programme (tracked via ngo_prg / ngo_me)
       ↓
  Submit Result Claim (with evidence)
       ↓
  Third-Party Verification
       ↓
  Trigger Payment → Confirm Payment
```

## Payment Models

| Model | Description |
|-------|-------------|
| output_based | Payment per measurable output unit |
| outcome_based | Payment when outcome thresholds met |
| impact_based | Payment on verified impact change |
| hybrid | Mix of output and outcome triggers |

## Verification Methods

`third_party`, `government`, `self_report`, `independent_audit`, `beneficiary_survey`

## API Examples

### Create an RBF Contract

```
POST /api/ngo/rbf/contracts
{
  "programme_id": "prg-001",
  "funder_reference": "WB-P12345",
  "title": "Health Sector RBF 2026",
  "total_value": 20000000,
  "currency": "KES",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "payment_model": "output_based"
}
```

### Define a DLI

```
POST /api/ngo/rbf/dlis
{
  "contract_id": "rbfc-xxx",
  "name": "Births attended by skilled personnel",
  "target_value": 5000,
  "price_per_unit": 2000,
  "due_date": "2026-12-31",
  "unit": "births",
  "verification_method": "third_party"
}
```

### Submit a Result Claim

```
POST /api/ngo/rbf/claims
{
  "contract_id": "rbfc-xxx",
  "dli_id": "dli-yyy",
  "claimed_value": 3200,
  "claim_date": "2026-07-01",
  "submitted_by": "programme_manager@org.ke",
  "evidence_references": ["HMIS-report-Q2.pdf"]
}
```

### Record Verification and Trigger Payment

```
POST /api/ngo/rbf/verifications
{
  "claim_id": "clm-zzz",
  "verifier": "Deloitte Kenya",
  "verification_date": "2026-07-15",
  "verified_value": 3050,
  "accepted": true,
  "methodology": "Random facility spot-checks"
}

POST /api/ngo/rbf/payment-triggers
{
  "contract_id": "rbfc-xxx",
  "claim_id": "clm-zzz",
  "verification_id": "ver-aaa",
  "amount": 6100000,
  "payment_date": "2026-08-01",
  "approved_by": "cfo@org.ke",
  "reference": "SWIFT-2026-0801"
}
```
