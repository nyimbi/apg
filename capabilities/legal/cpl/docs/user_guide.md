# Legal Compliance Management (leg_cpl) — User Guide

## Overview

Tracks regulatory requirements across multiple regulations and jurisdictions, maintains a compliance calendar, collects evidence, and manages breach investigation and regulatory reporting.

## Supported Regulations

GDPR, AML, FATCA, POCAMLA, Companies Act, Employment Act, Data Protection Act, IFRS, SOX, OECD Pillar Two, and any custom regulation string.

## Risk Levels

`low` → `medium` → `high` → `critical`

## Compliance Status Flow

`active` → `compliant` | `non_compliant` → `exempted` | `archived`

## Breach Status Flow

`open` → `investigating` → `remediated` → `reported` → `closed`

## API Reference

### Register a Requirement

```http
POST /api/legal/cpl/requirements
{
  "tenant_id": "acme",
  "title": "Data Subject Access Request Response",
  "description": "Respond to DSARs within 30 days",
  "regulation": "Data Protection Act 2019",
  "jurisdiction": "Kenya",
  "category": "data_privacy",
  "frequency": "continuous",
  "owner_id": "dpo-001",
  "risk_level": "high"
}
```

### Schedule a Compliance Activity

```http
POST /api/legal/cpl/calendar
{
  "tenant_id": "acme",
  "requirement_id": "cpl-001",
  "scheduled_date": "2026-09-30",
  "title": "Annual Data Protection Audit",
  "assigned_to_id": "dpo-001",
  "reminder_days": [30, 14, 7]
}
```

### Report a Breach

```http
POST /api/legal/cpl/breaches
{
  "tenant_id": "acme",
  "requirement_id": "cpl-001",
  "title": "Unauthorized access to customer PII",
  "description": "Database misconfiguration exposed 2,400 records",
  "severity": "high",
  "discovered_by_id": "it-sec-001",
  "discovery_date": "2026-06-09",
  "affected_records": 2400,
  "notification_required": true,
  "notification_deadline": "2026-06-16"
}
```

### Risk Register

```http
GET /api/legal/cpl/risk-register?tenant_id=acme
```

Returns all non-compliant and active requirements sorted by risk level.
