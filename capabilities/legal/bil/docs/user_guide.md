# Legal Billing & Time Tracking (leg_bil) — User Guide

## Overview

Captures attorney time, records disbursements, generates VAT-inclusive invoices, tracks approval workflows, and maintains client trust accounts with full ledger history.

## Use Cases

- **Daily time capture**: attorneys log hours against ABA activity codes.
- **Invoice generation**: bundle approved time entries and disbursements into a client invoice.
- **Trust accounting**: receive client retainers, apply fees, refund balances.
- **Revenue reporting**: outstanding, collected, and write-off summaries by matter.

## Workflow

```
Time Entry (draft) → submit → approve → invoice → send → pay
Disbursement (pending) → billed (on invoice creation)
Trust Account → deposit → fee_application → withdrawal
```

## ABA Activity Codes (sample)

| Code | Description |
|------|-------------|
| L110 | Fact Investigation/Development |
| L210 | Pleadings |
| L310 | Written Discovery |
| L410 | Trial and Hearing Attendance |
| A101 | Plan and Prepare for |

## API Reference

### Log Time

```http
POST /api/legal/bil/time-entries
{
  "tenant_id": "acme",
  "matter_id": "mat-001",
  "attorney_id": "atty-007",
  "date": "2026-06-10",
  "hours": 3.5,
  "rate": 15000,
  "activity_code": "L210",
  "description": "Drafted Statement of Claim",
  "billable": true,
  "currency": "KES"
}
```

### Generate an Invoice

```http
POST /api/legal/bil/invoices
{
  "tenant_id": "acme",
  "matter_id": "mat-001",
  "client_id": "client-001",
  "billing_period_start": "2026-06-01",
  "billing_period_end": "2026-06-30",
  "due_date": "2026-07-30",
  "time_entry_ids": ["te-001", "te-002"],
  "disbursement_ids": ["dis-001"]
}
```

Returns invoice with fees, disbursements, 16% VAT, and total.

### Trust Transaction

```http
POST /api/legal/bil/trust-accounts/{id}/transactions
{
  "tenant_id": "acme",
  "transaction_type": "deposit",
  "amount": 500000,
  "date": "2026-06-01",
  "description": "Client retainer",
  "authorized_by_id": "partner-001"
}
```
