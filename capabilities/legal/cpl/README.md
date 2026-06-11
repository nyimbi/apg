# leg_cpl — Legal Compliance Management

Regulatory requirement tracking, compliance calendar, evidence collection, breach reporting, and world-class GRC analytics.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/cpl/health | Health check |
| GET | /api/legal/cpl/requirements | List requirements |
| GET | /api/legal/cpl/requirements/{id} | Get requirement |
| POST | /api/legal/cpl/requirements | Create requirement |
| PUT | /api/legal/cpl/requirements/{id} | Update requirement |
| DELETE | /api/legal/cpl/requirements/{id} | Archive requirement |
| POST | /api/legal/cpl/requirements/{id}/compliant | Mark compliant |
| POST | /api/legal/cpl/requirements/{id}/non-compliant | Flag non-compliant |
| POST | /api/legal/cpl/requirements/{id}/reassign | Reassign owner |
| GET | /api/legal/cpl/calendar | List calendar entries |
| POST | /api/legal/cpl/calendar | Create calendar entry |
| PUT | /api/legal/cpl/calendar/{id} | Update calendar entry |
| POST | /api/legal/cpl/calendar/{id}/complete | Complete calendar entry |
| DELETE | /api/legal/cpl/calendar/{id} | Cancel calendar entry |
| GET | /api/legal/cpl/evidence | List evidence |
| POST | /api/legal/cpl/evidence | Attach evidence |
| PUT | /api/legal/cpl/evidence/{id} | Update evidence |
| DELETE | /api/legal/cpl/evidence/{id} | Archive evidence |
| GET | /api/legal/cpl/evidence/{id}/chain | Evidence chain-of-custody |
| GET | /api/legal/cpl/evidence/gaps | Evidence gap report |
| GET | /api/legal/cpl/breaches | List breaches |
| GET | /api/legal/cpl/breaches/{id} | Get breach |
| GET | /api/legal/cpl/breaches/{id}/sla | Breach notification SLA status |
| POST | /api/legal/cpl/breaches | Report breach |
| PUT | /api/legal/cpl/breaches/{id} | Update breach |
| DELETE | /api/legal/cpl/breaches/{id} | Close breach |
| POST | /api/legal/cpl/breaches/{id}/remediate | Remediate breach |
| POST | /api/legal/cpl/breaches/{id}/report | Report to regulator |
| GET | /api/legal/cpl/dashboard | Compliance dashboard |
| GET | /api/legal/cpl/risk-register | Risk register |
| GET | /api/legal/cpl/audit | Audit events |
| GET | /api/legal/cpl/audit/verify | Verify audit chain integrity |
| POST | /api/legal/cpl/penalty-exposure | Regulatory penalty exposure report |
| POST | /api/legal/cpl/snapshot | Record compliance score snapshot |
| GET | /api/legal/cpl/trend | Compliance score trend history |
| GET | /api/legal/cpl/owner-workload | Per-owner compliance workload |
| POST | /api/legal/cpl/regulator-comms | Log regulator communication |
| GET | /api/legal/cpl/regulator-comms | List regulator communications |
| POST | /api/legal/cpl/costs | Log compliance cost |
| GET | /api/legal/cpl/costs/summary | Compliance cost summary |

## Service Class

`LegalComplianceService` — multi-regulation tracking (GDPR, AML, POCAMLA, Kenya DPA, Companies Act), compliance calendar with reminders, evidence collection with chain-of-custody, breach investigation with 72-hour SLA countdown, auto-generated remediation plans, tamper-evident audit trail, and financial penalty exposure modelling.

## Key Features

### Core GRC

- **Requirement Management** — register obligations across any regulation and jurisdiction; track status through `active → compliant / non_compliant → archived`
- **Compliance Calendar** — schedule activities with reminder windows; auto-detect overdue items
- **Evidence Collection** — attach documents, certificates, attestations; track `valid_until` expiry
- **Breach Reporting** — full investigation workflow: `open → investigating → remediated → reported → closed`

### World-Class Enhancements

- **Regulatory Penalty Exposure Calculator** — converts non-compliant requirements into board-ready financial figures using regulation-specific penalty schedules (GDPR 4%/EUR 20M, etc.)
- **Compliance Score Trending** — daily snapshots with delta and direction indicators; 90-day trend window for board reporting
- **Evidence Chain-of-Custody** — every evidence mutation appended to an immutable custody log; legally defensible in enforcement proceedings
- **Breach Notification SLA Countdown** — real-time `hours_remaining` and `green/amber/red` status for 72-hour GDPR/DPA deadlines
- **Evidence Gap Analysis** — continuous audit-readiness scan identifying requirements with missing or expiring evidence
- **Regulator Communication Log** — litigation-ready record of all inbound/outbound correspondence with regulators
- **Compliance Cost Tracking** — per-regulation and per-category spend totals using `Decimal` arithmetic; CFO-ready budget defence
- **Owner Workload Balancing** — per-owner counts of active, non-compliant, overdue, and breach requirements; prevents silent deadline failures
- **Auto-Generated Remediation Plans** — on breach creation, a structured 6-step plan with SLA-offset milestones is auto-generated
- **Tamper-Evident Audit Trail** — SHA-256 chain hashing on every audit event; `verify_audit_chain()` detects manipulation

## Quick Usage Examples

### 1. Financial Penalty Exposure Report

```python
from decimal import Decimal
from capabilities.legal.cpl.service import LegalComplianceService

svc = LegalComplianceService(tenant_id="acme")

# Register a non-compliant GDPR requirement
req = await svc.create_requirement(
    tenant_id="acme",
    title="GDPR Article 30 — Records of Processing",
    description="Maintain a Record of Processing Activities (ROPA)",
    regulation="GDPR",
    jurisdiction="EU",
    category="data_privacy",
    frequency="annual",
    owner_id="dpo-001",
    risk_level="high",
)
await svc.flag_non_compliant("acme", req["id"], reason="ROPA not maintained")

# Get board-ready financial exposure
report = await svc.calculate_penalty_exposure(
    tenant_id="acme",
    annual_turnover=Decimal("50_000_000"),
    currency="EUR",
)
# report["aggregate_max_exposure"] => "2000000.00"
# report["line_items"][0]["max_exposure"] => "2000000.00"
```

### 2. Breach Notification SLA Monitoring

```python
# Report a breach that requires regulatory notification
breach = await svc.create_breach(
    tenant_id="acme",
    requirement_id=req["id"],
    title="Unauthorised access to customer PII",
    description="Credential stuffing attack exposed 1,200 records",
    severity="high",
    discovered_by_id="security-team",
    discovery_date="2026-06-11T09:00:00",
    affected_records=1200,
    notification_required=True,   # triggers 72-hour SLA
)

# Poll SLA status in real time
status = await svc.get_breach_sla_status("acme", breach["id"])
# {
#   "sla_status": "green",
#   "hours_remaining": 68.4,
#   "is_overdue": False,
#   "notification_filed": False
# }

# breach["remediation_plan_id"] is auto-populated — 6-step plan already created
```

### 3. Compliance Cost Summary

```python
# Log auditor fees against a requirement
await svc.log_compliance_cost(
    tenant_id="acme",
    requirement_id=req["id"],
    amount=Decimal("12500.00"),
    currency="USD",
    cost_type="external_audit",
    period="2026-Q2",
    recorded_by="finance-001",
)
await svc.log_compliance_cost(
    tenant_id="acme",
    requirement_id=req["id"],
    amount=Decimal("4200.00"),
    currency="USD",
    cost_type="staff_time",
    period="2026-Q2",
    recorded_by="finance-001",
)

summary = await svc.get_compliance_cost_summary("acme", currency="USD")
# {
#   "total": "16700.00",
#   "by_regulation": {"GDPR": "16700.00"},
#   "by_category": {"data_privacy": "16700.00"}
# }
```

## Integration Notes

| APG Capability | Integration Point |
|----------------|------------------|
| `leg_cntr` (Contract Management) | Link contract obligations as compliance requirements; contract expiry triggers calendar entries |
| `intel_alerts` (Alerts) | Emit critical breach and SLA-overdue events as platform alerts |
| `audl` (Audit Log) | Forward `_audit_events` to the platform audit ledger for cross-capability audit queries |
| `auth_rbac` (RBAC) | Gate `reassign_requirement`, `log_compliance_cost`, and `submit_attestation` behind compliance-admin roles |
| `notif` (Notifications) | Deliver `overdue_calendar` and `sla_amber/red` events via email/SMS to requirement owners |
| `rep` (Reporting) | Feed `get_compliance_trend` and `get_compliance_cost_summary` into executive dashboards |
