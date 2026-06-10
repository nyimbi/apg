# Contract Lifecycle Management (leg_ctr) — User Guide

## Overview

Manages the full contract lifecycle from drafting through execution, active obligation monitoring, and renewal management.

## Use Cases

- **NDA management**: draft, negotiate, sign, and track expiry.
- **Vendor contracts**: multi-level approval workflow with redline tracking.
- **Employment agreements**: obligation monitoring for notice periods, probation.
- **Lease agreements**: auto-renewal alerts and obligation calendars.

## Lifecycle States

`draft` → `under_review` → `approved` → `active` → `expired` | `terminated` | `archived`

## Key Features

| Feature | Description |
|---------|-------------|
| Version history | Every update creates a new version snapshot |
| Redlining | Section-level change proposals with accept/reject |
| Multi-level approval | Configurable approval chains |
| E-signature | Ordered signatory workflow |
| Obligation tracking | Recurring and one-time obligation reminders |
| Auto-renewal | Configurable notice window alerts |

## API Reference

### Draft a Contract

```http
POST /api/legal/ctr/contracts
{
  "tenant_id": "acme",
  "title": "Master Services Agreement — TechCorp",
  "contract_type": "msa",
  "counterparty_id": "party-techcorp",
  "owner_id": "atty-005",
  "effective_date": "2026-07-01",
  "expiry_date": "2027-06-30",
  "auto_renew": true,
  "renewal_notice_days": 60,
  "value": 5000000,
  "currency": "KES"
}
```

### Add a Redline

```http
POST /api/legal/ctr/redlines
{
  "tenant_id": "acme",
  "contract_id": "ctr-abc",
  "reviewer_id": "atty-009",
  "section_ref": "§ 12.3",
  "original_text": "30 days notice",
  "proposed_text": "60 days notice",
  "comment": "Industry standard is 60 days"
}
```

### Request Approval

```http
POST /api/legal/ctr/approvals
{
  "tenant_id": "acme",
  "contract_id": "ctr-abc",
  "approver_id": "general-counsel",
  "approval_level": 1
}
```

### List Expiring Contracts

```http
GET /api/legal/ctr/expiring?tenant_id=acme&days=30
```
