# Entity & Corporate Secretary (leg_ent) — User Guide

## Overview

Manages the company registry, board composition, share register, statutory filings, and board/shareholder resolutions.

## Use Cases

- **Company formation**: register new entities and populate initial directors/shareholders.
- **Annual returns**: track due dates and record filings with the registrar.
- **Board changes**: appoint/remove directors with cessation date tracking.
- **Share transfers**: process and record share transfer instruments.
- **Resolutions**: maintain a searchable minute book.

## Entity Types

`limited_company`, `llp`, `branch`, `holding`, `subsidiary`, `ngo`, `trust`, `partnership`

## API Reference

### Register an Entity

```http
POST /api/legal/ent/entities
{
  "tenant_id": "acme",
  "legal_name": "Datacraft Limited",
  "entity_type": "limited_company",
  "registration_number": "PVT-12345",
  "jurisdiction": "Kenya",
  "incorporation_date": "2020-03-15",
  "registered_address": "14 Riverside Drive, Nairobi",
  "tax_pin": "P051234567X"
}
```

### Appoint a Director

```http
POST /api/legal/ent/directors
{
  "tenant_id": "acme",
  "entity_id": "ent-001",
  "full_name": "Jane Mwangi",
  "id_number": "29876543",
  "nationality": "Kenyan",
  "appointment_date": "2026-01-01",
  "role": "director"
}
```

### Transfer Shares

```http
POST /api/legal/ent/shareholders/transfer
{
  "tenant_id": "acme",
  "from_shareholder_id": "shr-001",
  "to_full_name": "Peter Kamau",
  "to_id_number": "12345678",
  "shares_transferred": 1000,
  "transfer_date": "2026-06-01",
  "consideration": 500000
}
```

### Schedule Annual Return

```http
POST /api/legal/ent/filings
{
  "tenant_id": "acme",
  "entity_id": "ent-001",
  "filing_type": "annual_return",
  "due_date": "2026-12-31",
  "filing_period": "2026"
}
```
