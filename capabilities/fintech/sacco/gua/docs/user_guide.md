# SACCO Guarantor Management — User Guide

**Audience:** Loan officers, credit managers, SACCO operations staff  
**Capability:** `fintech_sacco_gua` | APG Platform

---

## What is a guarantee?

When a SACCO member takes a loan, the product may require one or more fellow members to pledge their savings as security. If the borrower defaults, the SACCO can recover the outstanding amount from the guarantor's savings. The `gua` capability manages this obligation end-to-end.

---

## Workflow overview

```
1. Loan officer identifies guarantors for a loan application
2. System sends a consent request to each guarantor
3. Each guarantor reviews, enters their PIN, and accepts or declines
4. On acceptance, the pledged amount is frozen in the guarantor's savings account
5. The loan proceeds to disbursement once all required guarantors have accepted
6. When the loan is fully repaid, frozen savings are automatically released
7. If the borrower defaults, the system calls the guarantee, deducts from savings,
   and posts the corresponding GL entry
```

---

## Step 1 — Check guarantor eligibility

Before requesting, verify the prospective guarantor qualifies.

**API:**
```
POST /api/fintech/sacco/gua/eligibility
X-Tenant-ID: <your-tenant>

{
  "member_id": "M-00123",
  "amount_to_guarantee": "50000"
}
```

**Response fields:**

| Field | Meaning |
|-------|---------|
| `eligible` | true/false |
| `reasons` | list of blocking conditions |
| `free_savings` | unencumbered savings available |
| `headroom` | how much more this member can guarantee before hitting their limit |
| `savings_cover_ratio` | free_savings / amount (should be >= 1.0) |

**Common ineligibility reasons:**

- `member_not_active` — member account is suspended or closed
- `member_is_defaulter` — member has an overdue loan
- `insufficient_savings` — free savings < 100% of requested guarantee amount
- `exposure_limit_exceeded` — total guarantees would exceed 3× share capital

---

## Step 2 — Request consent

```
POST /api/fintech/sacco/gua/requests
X-Tenant-ID: <your-tenant>

{
  "loan_id": "LN-T1-0000001",
  "guarantor_member_id": "M-00123",
  "amount_to_guarantee": "50000",
  "loan_applicant_message": "Please help me secure this school fees loan"
}
```

The guarantor receives an SMS/notification with details of the request.

---

## Step 3 — Guarantor accepts or declines

### Accept (guarantor's action)
```
POST /api/fintech/sacco/gua/requests/{request_id}/accept

{
  "guarantor_member_id": "M-00123",
  "pin_verified": true,
  "acceptance_notes": "Happy to support"
}
```

PIN verification is mandatory. On acceptance, KES 50,000 is immediately frozen in the guarantor's savings account. The guarantor cannot withdraw or use those funds until the guarantee is released.

### Decline
```
POST /api/fintech/sacco/gua/requests/{request_id}/decline

{
  "guarantor_member_id": "M-00123",
  "decline_reason": "I cannot commit at this time"
}
```

No savings are touched. The loan officer must find an alternative guarantor.

---

## Step 4 — View a guarantor's exposure

```
GET /api/fintech/sacco/gua/exposure/M-00123
X-Tenant-ID: <your-tenant>
```

Returns:
- `total_guaranteed` — total active obligations
- `frozen_savings` — savings currently locked
- `available_to_guarantee` — how much more can be pledged
- `at_risk_amount` — pledged on loans with > 30 days overdue

---

## Substituting a guarantor

If a guarantor needs to be replaced (e.g. leaving the SACCO):

```
POST /api/fintech/sacco/gua/guarantees/{guarantee_id}/substitute

{
  "new_guarantor_id": "M-00456",
  "reason": "Original guarantor emigrated",
  "approved_by": "LO-001"
}
```

This:
1. Releases the original guarantor's savings
2. Creates a new consent request for the replacement
3. Replacement must accept before the original is fully freed

---

## When a loan is repaid

The nightly job automatically releases all guarantees for fully repaid loans:

```
POST /api/fintech/sacco/gua/process-releases
X-Tenant-ID: <your-tenant>
```

Guarantors receive a release SMS. Their savings become available immediately.

---

## When a borrower defaults (calling the guarantee)

```
POST /api/fintech/sacco/gua/guarantees/{guarantee_id}/call

{
  "amount_called": "15000",
  "reason": "Borrower has not paid for 90 days"
}
```

This:
- Deducts KES 15,000 from the guarantor's frozen savings
- Posts: DR Guarantor Savings / CR Loan Recovery
- Sends a call notice to the guarantor

The amount cannot exceed the frozen amount.

---

## Portfolio monitoring

**At-risk guarantees** (loans > 30 DPD):
```
GET /api/fintech/sacco/gua/at-risk
```

**Aggregate metrics:**
```
GET /api/fintech/sacco/gua/metrics
```

Returns: total exposure, call rate %, release rate %, at-risk count.

---

## Notices

Send manual notices to guarantors at any stage:

```
POST /api/fintech/sacco/gua/guarantees/{id}/notice

{"notice_type": "warning"}   # early warning on overdue loan
{"notice_type": "call_notice"} # money has been taken
{"notice_type": "release"}   # obligation ended
```

---

## Setting a custom exposure limit

Override the default 3× share capital limit for a specific member:

```
POST /api/fintech/sacco/gua/exposure-limit

{
  "member_id": "M-00123",
  "limit": "300000",
  "set_by": "credit-manager-1"
}
```

---

## Viewing a member's full guarantee history

```
GET /api/fintech/sacco/gua/members/M-00123/history
```

Returns all past and present requests and guarantees, plus current exposure.

---

## GL entries

All guarantee calls generate auditable GL records:

```
GET /api/fintech/sacco/gua/gl-entries?guarantee_id={id}
```

| Field | Value |
|-------|-------|
| `debit_account` | Guarantor Savings |
| `credit_account` | Loan Recovery |
| `narrative` | Guarantee call reason |

---

© 2025 Datacraft — www.datacraft.co.ke
