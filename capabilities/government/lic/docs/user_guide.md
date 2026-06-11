# Licensing and Permits — User Guide

**Capability ID**: `government_lic` | **Domain**: `government` | **Version**: `1.1.0`
**Company**: Datacraft | **Copyright**: © 2025

---

## Description

Business and professional licence applications, renewals, inspections, revocations, and fee collection with full compliance monitoring. Enforces that licences cannot be renewed if the last inspection failed, prevents duplicate licences, and requires formal notice before revocation.

v1.1.0 adds: risk-based compliance scoring, SLA tracking, late-fee auto-assessment, W3C VC digital credentials, revocation appeals, offline mobile inspector sync, scored inspection checklists, policy impact analysis, and a ranked compliance scorecard.

---

## Installation

```bash
pip install apg-government-lic
```

---

## Quick Start

```python
import asyncio
from apg_government_lic.service import LicensingService

svc = LicensingService(tenant_id="nairobi-county", actor_id="officer-001")

# Submit a new application
app = svc.apply_licence(
    applicant_id="citizen-123",
    licence_type="business",
    activity="retail_food",
    documents=["cert_of_registration.pdf", "health_cert.pdf"],
)
print(app["reference"])  # LIC-APP-20260611-A1B2C3

# Run background check
check = svc.background_check(app["id"])
print(check["recommendation"])  # proceed

# Schedule premises inspection
from datetime import datetime, timedelta
insp = svc.premises_inspection(
    application_id=app["id"],
    inspector_id="inspector-007",
    date=datetime.now() + timedelta(days=5),
)

# Issue the licence
lic = svc.issue_licence(
    licence_id="LIC-001",
    tenant_id="nairobi-county",
    application_id=app["id"],
    licence_type="business",
    licence_number="NBC-2026-001234",
    holder_id="citizen-123",
    issued_date="2026-06-11",
    expiry_date="2027-06-10",
    evidence_reference="doc-store://lic-001",
)
```

---

## Core Workflows

### 1. Licence Application

```python
# Full parameter form
result = svc.submit_application(
    application_id="APP-001",
    tenant_id="nairobi-county",
    licence_type="professional",
    applicant_id="doctor-456",
    business_registration="CPK-2026-789",
    evidence_reference="doc://evidence-001",
    fee_paid=True,
    policy_attached=True,
)

# Simplified form (auto-generates IDs and reference numbers)
result = svc.apply_licence(
    applicant_id="doctor-456",
    licence_type="professional",
    activity="medical_practice",
    documents=["medical_degree.pdf", "mmpk_cert.pdf"],
)
```

### 2. Inspections

```python
# Schedule an inspection
svc.schedule_inspection(
    inspection_id="INSP-001",
    tenant_id="nairobi-county",
    licence_id="LIC-001",
    inspection_type="routine",
    inspector_id="inspector-007",
    scheduled_date="2026-06-20",
    evidence_reference="",
)

# Record outcome manually
svc.record_inspection_outcome(
    inspection_id="INSP-001",
    tenant_id="nairobi-county",
    outcome="pass",
    findings="All checklist items satisfied.",
)

# Score a checklist (async, v1.1.0)
result = await svc.inspection_checklist_evaluate(
    inspection_id="INSP-001",
    responses={
        "fire_safety": True,
        "sanitation": True,
        "signage": True,
        "capacity": True,
        "equipment": False,
    },
)
print(result["score_pct"])   # 80.0
print(result["outcome"])     # pass
```

### 3. Licence Renewal

```python
# Simplified renewal
result = svc.licence_renewal(
    licence_id="LIC-001",
    renewal_documents=["renewal_form.pdf", "updated_insurance.pdf"],
)
print(result["new_expiry"])
print(result["inspection_required"])  # True if last inspection failed

# Full renewal form
svc.renew_licence(
    renewal_id="REN-001",
    tenant_id="nairobi-county",
    licence_id="LIC-001",
    renewal_type="standard",
    new_expiry_date="2028-06-10",
    evidence_reference="doc://renewal-evidence",
    renewal_fee_paid=True,
)
```

### 4. Suspension and Revocation

```python
# Suspend
svc.suspend_licence(
    licence_id="LIC-001",
    reason="Failure to maintain hygiene standards",
    period="30d",
)

# Revoke
svc.revoke_licence(
    revocation_id="REV-001",
    tenant_id="nairobi-county",
    licence_id="LIC-001",
    reason="Persistent non-compliance after two warnings",
    approval_reference="APPROV-2026-0041",
    evidence_reference="doc://revocation-evidence",
    notice_served=True,
)
```

### 5. Fee Collection

```python
receipt = svc.fee_collection(
    licence_id="LIC-001",
    amount=5000.0,
    payment_method="mpesa",
)
print(receipt["receipt"])  # RCT-202606110930-A1B2C3
```

---

## Async Methods

All async methods must be `await`-ed inside an async context.

### Citizen Portal

```python
# Lookup all licences for a citizen
licences = await svc.citizen_licence_lookup("citizen-123")

# Submit via citizen portal channel
result = await svc.online_application(
    applicant_id="citizen-123",
    licence_type="business",
    jurisdiction="nairobi-county",
)
print(result["tracking_ref"])
```

### Bulk Operations

```python
# Bulk renewal (up to 200 at once)
result = await svc.bulk_licence_renewal(["LIC-001", "LIC-002", "LIC-003"])
print(result["renewed"], result["failed"])

# Bulk status update
result = await svc.bulk_status_update([
    {"licence_id": "LIC-001", "status": "suspended"},
    {"licence_id": "LIC-002", "status": "active"},
])
```

### Compliance and Reporting

```python
audit   = await svc.compliance_audit()
report  = await svc.regulatory_reporting("2026-Q2")
kpi     = await svc.licence_kpi_summary("2026-06")
detail  = await svc.licence_analytics_detail("2026-06")
notifs  = await svc.expiry_notifications(days_ahead=30)
export  = await svc.export_licences(fmt="json")
```

---

## v1.1.0 Enhancement Methods

### Risk Scoring

Computes a 0–100 compliance score per licence. Score drives inspection targeting and fee-discount eligibility.

```python
score = await svc.risk_score_licence("LIC-001")
print(score["total_score"])   # e.g. 85
print(score["risk_tier"])     # low | medium | high
print(score["components"])
# {"inspection": 32, "fee_payment": 20, "renewal_timeliness": 20, "condition_adherence": 20}
```

### SLA Tracking

Reports application processing SLA compliance. Applications within 3 days of their deadline are flagged.

```python
sla = await svc.sla_status_report()
print(sla["breached"])            # count of breached applications
print(sla["approaching_breach"])  # count approaching breach (<=3 days)
# sla["details"] contains per-application breakdown
```

### Late-Fee Assessment

Automatically computes a late renewal penalty (KES 500/day overdue). Must be settled before the renewed licence is issued.

```python
penalty = await svc.late_fee_assessment("LIC-001")
print(penalty["days_overdue"])              # e.g. 15
print(penalty["penalty_amount"])            # 7500.0 KES
print(penalty["renewal_blocked_until_paid"])# True

# Settle the penalty before renewal
svc.fee_collection(
    licence_id="LIC-001",
    amount=penalty["penalty_amount"],
    payment_method="mpesa",
)
```

### Revocation Appeals

File a formal appeal against a revocation. Must be within 30 days of the revocation date.

```python
appeal = await svc.appeal_revocation(
    licence_id="LIC-001",
    appellant_id="citizen-123",
    grounds="Decision was disproportionate; first offence, corrective action taken.",
)
print(appeal["appeal_id"])
print(appeal["hearing_deadline"])
print(appeal["status"])   # appeal_filed
```

### Inspection Checklist Scoring

Score a completed inspection checklist. Pass threshold is 80% of items.

```python
result = await svc.inspection_checklist_evaluate(
    inspection_id="INSP-001",
    responses={
        "fire_safety": True,
        "sanitation": True,
        "signage": False,
        "capacity": True,
        "equipment": True,
    },
)
print(result["score_pct"])  # 80.0
print(result["outcome"])    # pass
```

### Policy Impact Analysis

Dry-run a fee or policy change before applying it. Returns affected count, revenue delta, and licences that would fail revalidation.

```python
analysis = await svc.impact_analysis({
    "change_type": "fee_schedule",
    "licence_type": "business",
    "new_fee": 6000.0,
    "delta_pct": 20,
})
print(analysis["affected_licence_count"])
print(analysis["projected_revenue_delta_kes"])
print(analysis["would_fail_revalidation_count"])
```

### Digital Licence Credentials (W3C VC)

Issue a cryptographically signable W3C Verifiable Credential for a licence for delivery to citizen wallet apps.

```python
credential = await svc.digital_licence_credential("LIC-001")
print(credential["credentialSubject"]["licence_number"])
# Wire credential["proof"]["proofValue"] to a signing service in production
```

### Offline Inspector Sync

Package all pending inspections for a given inspector as a self-contained offline payload for mobile apps (48-hour TTL).

```python
payload = await svc.inspection_sync_payload("inspector-007")
print(payload["inspection_count"])
print(payload["ttl_hours"])  # 48
# Load payload["inspections"] to local SQLite on the device
```

### Compliance Scorecard

Generate a ranked compliance scorecard for all active licences. Licences are listed lowest score first (highest risk first).

```python
scorecard = await svc.compliance_scorecard()
print(scorecard["high_risk_count"])
print(scorecard["rankings"][:5])  # highest-risk licences
```

---

## Audit Trail

```python
trail = await svc.audit_trail(
    from_date="2026-06-01",
    to_date="2026-06-30",
)
print(trail["event_count"])
```

---

## Health Check

```python
health = await svc.health_check()
print(health["status"])          # healthy
print(health["active_licences"])
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary("nairobi-county")
# Returns counts: applications, licences, inspections, renewals,
# fees, revocations, agents, suspensions, random_inspection_batches, audit_events
```

---

## Configuration Reference

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_LIC_`.

| Key | Type | Default | Description |
|---|---|---|---|
| `governance.licence_without_payment_denied` | bool | true | Block unpaid applications |
| `governance.expired_licence_operation_denied` | bool | true | Block operations on expired licences |
| `governance.inspection_fail_blocks_renewal` | bool | true | Block renewal after failed inspection |
| `governance.duplicate_licence_denied` | bool | true | Block duplicate active licences |
| `governance.late_fee_rate_per_day` | float | 500.0 | KES charged per overdue day |
| `governance.sla_days.business` | int | 21 | SLA for business licences |
| `governance.sla_days.professional` | int | 14 | SLA for professional licences |
| `governance.sla_days.temporary` | int | 5 | SLA for temporary permits |
| `governance.inspection_pass_threshold_pct` | float | 80.0 | Minimum checklist score to pass |
| `governance.appeal_window_days` | int | 30 | Days to file revocation appeal |

---

## Composability

Reference this capability in `.apg` source files:

```apg
use government_lic;
```

| Capability | Integration |
|---|---|
| `government_csr` | Applications submitted via citizen portal |
| `government_bud` | Licence fees credited to AIA vote accounts |
| `government_cas` | Complaints create cases; appeals trigger case records |
| `government_con` | Contractor registration requires professional licence |
| `government_per` | Building permits require valid contractor licence |
| `government_pay` | Payment gateway for online fee collection |

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — Detailed improvement catalogue
- `SPECIFICATION.md` — Formal capability specification
- `cap_spec.md` — Machine-readable capability spec
