# Donor Relationship Management (ngo_don) — User Guide

## Overview

The Donor Relationship Management capability provides a complete CRM for NGO donors: registry,
communication tracking, pledge management, receipt generation, and stewardship planning.

## Key Use Cases

- **Donor registry**: Register individuals, corporates, foundations, governments.
- **Communication log**: Track every email, call, meeting by staff member.
- **Pledge tracking**: Record multi-year pledges with due-date alerting.
- **Receipt generation**: Auto-numbered receipts with tax-ID linkage.
- **Stewardship tiers**: Plan touchpoints for major and principal donors.

## Donor Types

`individual`, `corporate`, `foundation`, `government`, `bilateral`, `multilateral`

## Stewardship Tiers

| Tier | Touchpoints/Year |
|------|-----------------|
| standard | 4 |
| major | 6 |
| principal | 12 |
| legacy | 12+ |

## API Examples

### Register a Donor

```
POST /api/ngo/don/
{
  "name": "Rockefeller Foundation",
  "donor_type": "foundation",
  "email": "grants@rockefeller.org",
  "country": "US"
}
```

### Log a Communication

```
POST /api/ngo/don/<donor_id>/communications
{
  "subject": "Q1 2026 Programme Update",
  "body": "Dear Programme Officer...",
  "staff_member": "jane@org.ke",
  "communication_date": "2026-01-15",
  "channel": "email",
  "direction": "outbound"
}
```

### Generate a Receipt

```
POST /api/ngo/don/<donor_id>/receipts
{
  "amount": 500000,
  "receipt_date": "2026-02-01",
  "reference": "TRF-20260201",
  "issued_by": "finance@org.ke",
  "payment_method": "bank_transfer"
}
```
