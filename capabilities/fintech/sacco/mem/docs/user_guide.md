# SACCO Member Registry — User Guide

## Overview

The SACCO Member Registry capability manages the complete lifecycle of SACCO membership: onboarding, KYC verification, share capital, guarantor relationships, and member exit processing.

## Key Use Cases

1. **Member Onboarding** — Capture personal details, assign member numbers, collect entry fees
2. **KYC Verification** — Submit and verify identity documents (national ID, passport, etc.)
3. **Share Capital Management** — Record share purchases, track holdings, process transfers
4. **Guarantor Setup** — Link guarantors to beneficiaries for loan applications
5. **Member Exit** — Process resignations, deaths, expulsions with share capital refunds

## Member Lifecycle

```
Onboarded (pending) → KYC Submitted → KYC Verified + Entry Fee Paid → Active
Active → Suspended → Reinstated → Active
Active → Exit Initiated → Exit Completed → Exited
```

## API Reference

### Create a Member

```
POST /api/fintech/sacco/mem/members
Content-Type: application/json
X-Tenant-ID: sacco_abc

{
  "full_name": "Jane Wambui",
  "national_id": "12345678",
  "phone": "0712345678",
  "date_of_birth": "1985-03-15",
  "gender": "F",
  "county": "Nairobi",
  "membership_type": "ordinary",
  "entry_fee": 500.00
}
```

### Submit KYC

```
POST /api/fintech/sacco/mem/kyc
{
  "member_id": "mem-...",
  "document_type": "national_id",
  "document_number": "12345678",
  "document_front_ref": "s3://bucket/doc-front.jpg",
  "submitted_by": "officer-001"
}
```

### Purchase Shares

```
POST /api/fintech/sacco/mem/shares/purchase
{
  "member_id": "mem-...",
  "shares": 10,
  "share_value": 100.00,
  "payment_reference": "MPE-XYZ-123",
  "recorded_by": "officer-001",
  "payment_method": "mpesa"
}
```

## Tenant Isolation

All requests require the `X-Tenant-ID` header. Data is fully isolated per tenant.
