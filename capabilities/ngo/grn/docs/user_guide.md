# Grant Management (ngo_grn) — User Guide

## Overview

The Grant Management capability manages the full lifecycle of donor grants: from pipeline through
proposal, activation, disbursement, compliance reporting, and audit.

## Key Use Cases

- **Pipeline tracking**: Register incoming grant opportunities before formal application.
- **Proposal management**: Submit, review, approve or reject proposals against a grant.
- **Budget management**: Define budget lines per category; track utilisation.
- **Disbursement control**: Record and confirm disbursements with dual-approval.
- **Compliance reporting**: Submit narrative and financial reports; track approvals.
- **Audit management**: Log audit findings by severity; track resolution.

## Workflow

```
Pipeline → Proposal Submitted → Proposal Approved → Grant Activated
                                                         ↓
                                              Budget Lines Created
                                                         ↓
                                              Disbursements Recorded & Confirmed
                                                         ↓
                                              Compliance Reports Submitted
                                                         ↓
                                              Grant Closed
```

## API Reference

See `README.md` for the full endpoint table.

### Create a Grant

```
POST /api/ngo/grn/
{
  "title": "USAID Food Security Grant 2026",
  "donor_reference": "USAID-KE-2026-001",
  "amount": 5000000,
  "currency": "KES",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "sector": "food_security",
  "country": "KE"
}
```

### Activate a Grant

```
POST /api/ngo/grn/<grant_id>/activate
{ "approved_by": "ceo@org.ke" }
```

### Record a Disbursement

```
POST /api/ngo/grn/<grant_id>/disbursements
{
  "amount": 1000000,
  "disbursement_date": "2026-03-01",
  "reference": "WIRE-001",
  "approved_by": "finance@org.ke",
  "payment_method": "bank_transfer"
}
```
