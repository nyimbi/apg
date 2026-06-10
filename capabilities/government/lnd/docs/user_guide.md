# Land Registry — User Guide

## Overview

The Land Registry capability (gov_lnd) provides a complete digital cadastre and title management system. It covers the full land administration lifecycle from parcel registration through title issuance, transfers, adjudication of claims, encumbrance management, and annual valuation rolls.

## Core Concepts

- **Parcel**: A surveyed unit of land identified by a unique parcel number
- **Title**: Legal document of ownership linked to a parcel
- **Transfer**: Conveyance of title ownership from one party to another
- **Adjudication**: Formal process to determine rightful ownership of disputed land
- **Encumbrance**: Charge, mortgage, or restriction registered against a title
- **Valuation Roll**: Annual assessment of land values for rating purposes

## Use Cases

### Register a New Parcel

```
POST /api/government/lnd/parcels
{
  "parcel_number": "NAIROBI/WESTLANDS/001",
  "county": "Nairobi",
  "sub_county": "Westlands",
  "location": "Parklands",
  "area_hectares": 0.125,
  "land_use": "residential",
  "tenant_id": "lands_kenya"
}
```

### Issue a Title Deed

```
POST /api/government/lnd/titles
{
  "parcel_id": "parcel-abc123",
  "title_number": "IR 12345",
  "owner_id": "owner-001",
  "owner_name": "John Doe",
  "issue_date": "2025-01-15",
  "tenure_type": "freehold",
  "issued_by": "Registrar of Titles",
  "tenant_id": "lands_kenya"
}
```

### Register a Mortgage

```
POST /api/government/lnd/encumbrances
{
  "title_id": "title-xyz789",
  "encumbrance_type": "mortgage",
  "holder_id": "bank-001",
  "holder_name": "Kenya Commercial Bank",
  "amount_kes": 5000000,
  "start_date": "2025-02-01",
  "instrument_reference": "MORT-2025-001",
  "registered_by": "Registrar of Titles"
}
```

### Initiate Land Transfer

```
POST /api/government/lnd/transfers
{
  "title_id": "title-xyz789",
  "transferor_id": "owner-001",
  "transferor_name": "John Doe",
  "transferee_id": "owner-002",
  "transferee_name": "Jane Smith",
  "consideration_kes": 8500000,
  "transfer_date": "2025-03-01",
  "instrument_number": "TRANS-2025-001",
  "approved_by": "Registrar"
}
```

## Supported Land Uses

residential, commercial, agricultural, industrial, mixed_use, public, conservation, institutional

## Supported Tenure Types

freehold, leasehold, community, government

## Supported Encumbrance Types

mortgage, caveat, charge, easement, restriction, lien, covenant, caution

## Error Codes

| Code | Meaning |
|------|---------|
| 422 | Validation error — business rule violated |
| 404 | Resource not found |
| 500 | Internal service error |
