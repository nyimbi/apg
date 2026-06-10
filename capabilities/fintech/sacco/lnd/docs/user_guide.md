# SACCO Lending — User Guide

## Overview

Manages the full loan lifecycle: product configuration, credit scoring, application, approval, disbursement, repayment tracking, arrears management, and CRB reporting.

## Loan Lifecycle

```
Application (pending) → Approved → Disbursed → Active
Active → Arrears (overdue) → Active (after payment) or Written Off
Active / Arrears → Closed (fully repaid)
Pending → Rejected / Cancelled
```

## Credit Scoring

Scores range 0–1000 with grades A–E. Factors:
- Savings adequacy vs share capital (30%)
- Membership tenure (20%)
- Repayment history (35%)
- Debt burden ratio (15%)

Grades A/B qualify for 3× savings multiplier; C for 2×; D/E for 1×.

## Repayment Schedule

Reducing-balance method: each installment covers accrued interest first, then principal. Grace period installments cover interest only.

## API Reference

### Apply for a Loan

```
POST /api/fintech/sacco/lnd/loans
X-Tenant-ID: sacco_abc

{
  "member_id": "mem-...",
  "product_id": "lprod-...",
  "amount_requested": 50000.00,
  "term_months": 12,
  "purpose": "Business working capital",
  "guarantor_ids": ["mem-...", "mem-..."]
}
```

### Approve & Disburse

```
POST /api/fintech/sacco/lnd/loans/{id}/approve
{ "approved_amount": 50000, "approved_term_months": 12, "approved_by": "officer-01" }

POST /api/fintech/sacco/lnd/loans/{id}/disburse
{ "disbursement_method": "mpesa", "disbursement_reference": "MPE-ABC", "disbursed_by": "cashier-01" }
```

### Record a Repayment

```
POST /api/fintech/sacco/lnd/repayments
{ "loan_id": "ln-...", "amount": 4500.00, "payment_reference": "MPE-REP-001", "recorded_by": "teller-01" }
```

### CRB Listing

```
POST /api/fintech/sacco/lnd/crb
{ "member_id": "mem-...", "report_type": "listing", "reason": "90+ days arrears", "reported_by": "manager-01" }
```

## PAR (Portfolio at Risk)

`PAR = total_arrears_amount / total_outstanding_balance × 100`

Available via `GET /api/fintech/sacco/lnd/summary`.
