# Legal Compliance Management (leg_cpl) — User Guide

## Overview

`leg_cpl` tracks regulatory requirements across multiple regulations and jurisdictions, maintains a compliance calendar, collects evidence with chain-of-custody, manages breach investigation with automatic 72-hour SLA countdown, quantifies financial penalty exposure, and produces trend data for board reporting.

## Supported Regulations

GDPR, AML, FATCA, POCAMLA, Kenya Data Protection Act 2019, Companies Act, Employment Act, IFRS, SOX, PCI DSS, HIPAA, NIS2, DORA, OECD Pillar Two, and any custom regulation string.

## Status Reference

### Compliance Status Flow

```
active → compliant
       → non_compliant → archived
       → exempted
       → archived
```

### Breach Status Flow

```
open → investigating → remediated → reported → closed
```

### SLA Status (Breach Notification)

| Status | Meaning |
|--------|---------|
| `green` | > 24 hours remaining |
| `amber` | 1–24 hours remaining |
| `red` | Overdue |
| `n/a` | Notification not required |

### Risk Levels

`low` → `medium` → `high` → `critical`

### Evidence Types

`document`, `screenshot`, `log`, `certificate`, `attestation`, `audit_report`, `policy`

---

## Core Workflows

### 1. Register a Compliance Requirement

```http
POST /api/legal/cpl/requirements
Content-Type: application/json

{
  "tenant_id": "acme",
  "title": "Data Subject Access Request Response",
  "description": "Respond to DSARs within 30 calendar days of receipt",
  "regulation": "Data Protection Act 2019",
  "jurisdiction": "Kenya",
  "category": "data_privacy",
  "frequency": "continuous",
  "owner_id": "dpo-001",
  "risk_level": "high",
  "tags": ["DSAR", "data_subject_rights"]
}
```

Response includes `id` (e.g., `cpl-a3b7f2c10d4e`) which is used in subsequent calls.

### 2. Schedule a Compliance Activity

```http
POST /api/legal/cpl/calendar
Content-Type: application/json

{
  "tenant_id": "acme",
  "requirement_id": "cpl-a3b7f2c10d4e",
  "scheduled_date": "2026-09-30",
  "title": "Annual Data Protection Audit",
  "assigned_to_id": "dpo-001",
  "description": "External auditor review of DSAR handling procedures",
  "reminder_days": [30, 14, 7]
}
```

### 3. Attach Evidence

```http
POST /api/legal/cpl/evidence
Content-Type: application/json

{
  "tenant_id": "acme",
  "requirement_id": "cpl-a3b7f2c10d4e",
  "evidence_type": "audit_report",
  "title": "2026 Q2 DSAR Process Audit Report",
  "description": "Independent auditor confirmed 100% on-time DSAR responses",
  "collected_by_id": "auditor-ext-001",
  "collection_date": "2026-06-01",
  "file_reference": "s3://acme-compliance/2026-q2-dsar-audit.pdf",
  "valid_until": "2027-06-01"
}
```

Each update to the evidence record appends a custody-chain entry automatically.

### 4. Report a Breach (with Auto-SLA and Auto-Remediation Plan)

```http
POST /api/legal/cpl/breaches
Content-Type: application/json

{
  "tenant_id": "acme",
  "requirement_id": "cpl-a3b7f2c10d4e",
  "title": "Unauthorised access to customer PII",
  "description": "Database misconfiguration exposed 2,400 records",
  "severity": "high",
  "discovered_by_id": "it-sec-001",
  "discovery_date": "2026-06-11T09:00:00",
  "affected_records": 2400,
  "notification_required": true
}
```

The response includes:
- `notification_sla_expires_at` — 72 hours after `discovery_date`
- `remediation_plan_id` — auto-generated 6-step plan with SLA-offset milestones

### 5. Monitor Breach Notification SLA

```http
GET /api/legal/cpl/breaches/{breach_id}/sla?tenant_id=acme
```

```json
{
  "breach_id": "brch-c5d8e1f0a2b3",
  "notification_required": true,
  "sla_expires_at": "2026-06-14T09:00:00Z",
  "hours_remaining": 43.2,
  "is_overdue": false,
  "sla_status": "green",
  "notification_filed": false
}
```

Set up polling or an alert trigger on `sla_status == "amber"` to initiate notification filing.

### 6. Risk Register

```http
GET /api/legal/cpl/risk-register?tenant_id=acme
```

Returns all non-compliant and active requirements sorted by risk level (critical first).

---

## Advanced Features

### Financial Penalty Exposure Report

Converts non-compliant requirements into board-ready financial figures using regulation-specific penalty schedules.

```http
POST /api/legal/cpl/penalty-exposure
Content-Type: application/json

{
  "tenant_id": "acme",
  "annual_turnover": "50000000.00",
  "currency": "EUR"
}
```

Response:
```json
{
  "aggregate_max_exposure": "2000000.00",
  "currency": "EUR",
  "non_compliant_count": 1,
  "line_items": [
    {
      "requirement_id": "cpl-a3b7f2c10d4e",
      "regulation": "GDPR",
      "max_exposure": "2000000.00",
      "likely_exposure": "1600000.00",
      "currency": "EUR"
    }
  ]
}
```

Built-in penalty schedules: GDPR (4% / EUR 20M), DPA Kenya (4% / KES 5M equivalent), HIPAA (USD 1.9M), AML (10% / USD 10M), PCI DSS (fixed USD 500K).

### Compliance Score Trend

Record a daily snapshot (call from a scheduled job):

```http
POST /api/legal/cpl/snapshot?tenant_id=acme
```

Retrieve 90-day trend:

```http
GET /api/legal/cpl/trend?tenant_id=acme&days=90
```

Response includes per-snapshot `delta_pct` and `direction` (`up` | `down` | `flat` | `baseline`).

### Evidence Chain-of-Custody

```http
GET /api/legal/cpl/evidence/{evidence_id}/chain?tenant_id=acme
```

Returns every mutation (create, update, archive) with actor_id, timestamp, and field-level delta. Presents a legally defensible custody record for regulatory enforcement proceedings.

### Evidence Gap Analysis

```http
GET /api/legal/cpl/evidence/gaps?tenant_id=acme
```

Returns per-requirement `has_valid_evidence`, `expired_count`, and `expiring_in_30d` IDs. Run before an audit to confirm you have zero gaps.

### Regulator Communication Log

Log every inbound or outbound correspondence:

```http
POST /api/legal/cpl/regulator-comms
Content-Type: application/json

{
  "tenant_id": "acme",
  "entity_id": "brch-c5d8e1f0a2b3",
  "regulator": "Kenya Office of the Data Protection Commissioner",
  "direction": "outbound",
  "summary": "72-hour breach notification filed per DPA s.43",
  "medium": "secure_portal",
  "reference": "ODPC-2026-0611-001",
  "actor_id": "dpo-001"
}
```

Retrieve all correspondence for a breach:

```http
GET /api/legal/cpl/regulator-comms?tenant_id=acme&entity_id=brch-c5d8e1f0a2b3
```

### Compliance Cost Tracking

```http
POST /api/legal/cpl/costs
Content-Type: application/json

{
  "tenant_id": "acme",
  "requirement_id": "cpl-a3b7f2c10d4e",
  "amount": "12500.00",
  "currency": "USD",
  "cost_type": "external_audit",
  "period": "2026-Q2",
  "recorded_by": "finance-001"
}
```

Get per-regulation cost summary:

```http
GET /api/legal/cpl/costs/summary?tenant_id=acme&currency=USD
```

### Owner Workload Balancing

```http
GET /api/legal/cpl/owner-workload?tenant_id=acme
```

Returns per-owner counts of `active_requirements`, `non_compliant`, `overdue_calendar`, `open_breaches`, and `compliance_rate_pct`. Use this to identify overburdened owners before they miss deadlines.

To reassign a requirement:

```http
POST /api/legal/cpl/requirements/{requirement_id}/reassign
Content-Type: application/json

{
  "tenant_id": "acme",
  "new_owner_id": "dpo-002",
  "reason": "dpo-001 on parental leave",
  "reassign_calendar": true
}
```

### Audit Trail Integrity Verification

```http
GET /api/legal/cpl/audit/verify?tenant_id=acme
```

Re-computes the SHA-256 chain hash for every audit event and returns:
```json
{
  "valid": true,
  "checked": 147,
  "first_broken_index": null
}
```

A `valid: false` response with a `first_broken_index` indicates tampering at that position.

---

## Integration with Other APG Capabilities

| Capability | How to Integrate |
|------------|-----------------|
| `intel_alerts` | Subscribe to `breach_reported`, `sla_status=red`, and `overdue_calendar` events to push platform alerts |
| `audl` | Forward `get_audit_events()` output to the platform audit ledger periodically for cross-capability queries |
| `auth_rbac` | Require `compliance_admin` role for `log_compliance_cost`, `reassign_requirement`, and `report_breach_to_regulator` |
| `leg_cntr` | When a contract containing regulatory obligations is signed, auto-create linked requirements via `create_requirement` |
| `notif` | Deliver amber/red SLA alerts and overdue calendar notifications to owners via email or SMS |
| `rep` | Pull `get_compliance_trend` and `get_compliance_cost_summary` into executive board packs |

---

## Developer Notes

- All monetary values use `Decimal`, never `float`. Pass amounts as strings when calling over HTTP to preserve precision.
- `tenant_id` is validated at every method entry via `guard_tenant_id`. Multi-tenant isolation is enforced at the data layer.
- `record_compliance_snapshot()` should be called once daily via a platform cron job to build the trend dataset.
- The `custody_chain` field on evidence records is append-only. Never mutate existing entries.
- `verify_audit_chain()` is O(n) over audit events per tenant; run it nightly rather than on every request.
