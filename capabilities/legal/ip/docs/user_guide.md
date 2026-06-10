# Intellectual Property Registry (leg_ip) — User Guide

## Overview

Manages the complete IP portfolio: patents, trademarks, copyrights, design rights, and domains. Tracks renewal deadlines, manages licensing deals, and records royalty payments.

## Use Cases

- **Trademark portfolio**: file applications, record registrations, monitor Nice class coverage.
- **Patent management**: track inventors, filing/grant dates, renewal windows.
- **Licensing revenue**: grant exclusive/non-exclusive licenses, compute royalties.
- **Renewal alerts**: identify assets expiring within a configurable window.

## Asset Types

`patent`, `trademark`, `copyright`, `trade_secret`, `design`, `domain`, `plant_variety`

## License Types

`exclusive`, `non_exclusive`, `sole`, `sublicense`, `compulsory`

## API Reference

### Register a Trademark

```http
POST /api/legal/ip/assets
{
  "tenant_id": "acme",
  "title": "DATACRAFT",
  "asset_type": "trademark",
  "owner_id": "entity-001",
  "jurisdiction": "Kenya",
  "application_number": "KE-TM-2026-00123",
  "filing_date": "2026-03-01",
  "classes": ["35", "42"],
  "expiry_date": "2036-03-01"
}
```

### Grant a License

```http
POST /api/legal/ip/licenses
{
  "tenant_id": "acme",
  "asset_id": "ip-001",
  "licensee_id": "partner-007",
  "license_type": "non_exclusive",
  "territory": "East Africa",
  "start_date": "2026-07-01",
  "royalty_rate": 5.0,
  "royalty_base": "revenue",
  "currency": "KES"
}
```

### Record Royalty Payment

```http
POST /api/legal/ip/royalties
{
  "tenant_id": "acme",
  "license_id": "lic-001",
  "period": "2026-06",
  "base_amount": 2000000,
  "submitted_by_id": "fin-001"
}
```

### Expiring Assets

```http
GET /api/legal/ip/expiring?tenant_id=acme&days=90
```
