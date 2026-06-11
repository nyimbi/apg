# Healthcare Regulatory — User Guide

**Capability ID**: `healthcare_reg` | **Domain**: `healthcare` | **Version**: `1.1.0`

---

## Overview

`healthcare_reg` is the APG compliance runtime for healthcare facilities. It manages the full regulatory lifecycle: facility licensing, accreditation, patient safety incident reporting (including sentinel events with RCA enforcement), HIPAA compliance, regulatory submissions, corrective action tracking, and real-time compliance dashboards.

Version 1.1 adds AI-assisted ICD-10 coding, automated HIPAA Security Rule gap analysis, predictive license expiry risk scoring, multi-framework compliance matrix, structured RCA workflow engine, accreditation survey readiness scoring, multi-jurisdiction breach notification timelines, state-specific regulatory rule evaluation, and a regulatory intelligence feed.

---

## Installation

```bash
pip install apg-healthcare-reg
```

---

## Quick Start

```python
from apg_healthcare_reg.service import HealthcareRegulatoryService

svc = HealthcareRegulatoryService(tenant_id="facility_001", actor_id="compliance_officer")

# Compliance dashboard
dashboard = await svc.compliance_dashboard()
print(dashboard["compliance_score"], dashboard["risk_level"])
```

---

## Core Workflows

### 1. License Management

```python
from apg_healthcare_reg.models import LicenseCreate
from datetime import datetime

# Add a license
lic = await svc.add_license(LicenseCreate(
    tenant_id="facility_001",
    license_type="facility_operating",
    license_number="FAC-2025-001",
    issuing_authority="State Health Department",
    issued_date=datetime(2025, 1, 1),
    expiry_date=datetime(2026, 1, 1),
    holder_name="Riverside Medical Center",
    scope="General Acute Care",
    created_by="compliance_officer",
))

# Predictive risk score
risk = await svc.license_expiry_risk_score("facility_001", lic.id)
print(risk["risk_band"], risk["recommended_action_date"])
# Output: 'high'  '2025-10-12T00:00:00'

# Get expiring licenses (within 90 days)
expiring = await svc.get_expiring_licenses("facility_001", days=90)

# Initiate renewal
renewal = await svc.licence_renewal(lic.id)
```

The `license_expiry_risk_score` method uses historical renewal lead times by license type plus issuing authority SLA to produce a 0–100 risk score and recommended action date — earlier than static 90/30/7-day thresholds.

---

### 2. Accreditation Management

```python
from apg_healthcare_reg.models import AccreditationCreate

acc = await svc.add_accreditation(AccreditationCreate(
    tenant_id="facility_001",
    accreditation_body="joint_commission",
    program="Hospital Accreditation",
    award_date=datetime(2023, 6, 1),
    expiry_date=datetime(2026, 6, 1),
    certificate_reference="TJC-2023-HOSP-001",
    scope="All inpatient services",
    created_by="quality_director",
))

# Survey readiness scorecard — continuous monitoring
scorecard = await svc.survey_readiness_scorecard("facility_001", "joint_commission")
print(scorecard["readiness_band"])   # green / yellow / red
print(scorecard["recommendation"])
```

The readiness scorecard deducts points for open inspection findings (by severity), overdue corrective actions, and open sentinel events. Score bands map to Green (>=85), Yellow (70–84), Red (<70). Red triggers a NATS alert to compliance leadership.

---

### 3. Incident Reporting and RCA Workflow

```python
from apg_healthcare_reg.models import IncidentCreate

# Report a sentinel event
incident = await svc.report_incident(IncidentCreate(
    tenant_id="facility_001",
    incident_type="sentinel_event",
    severity="catastrophic",
    description="Wrong-site surgery on left knee",
    patient_id="PAT-001",
    department="Orthopedic Surgery",
    occurred_at=datetime(2026, 6, 10, 14, 30),
    reported_by="charge_nurse",
    immediate_actions="Surgery halted; patient stabilised; surgeon notified",
    witnesses=["RN Jones", "Anesthesiologist Lee"],
    created_by="quality_officer",
))
# Service automatically emits sentinel_event_reported to NATS
# 72-hour notification clock starts

# Create TJC RCA2 workflow (45-day deadline from occurrence)
rca = await svc.rca_workflow_create(incident.id, rca_type="tjc_rca2")
print(rca["tjc_45_day_deadline"], rca["days_remaining"])

# Advance stage by stage
await svc.rca_workflow_advance(
    incident_id=incident.id,
    workflow_id=rca["id"],
    stage="immediate_response",
    stage_data={
        "actions_taken": "Surgery halted, site marked correctly, patient transferred to ICU",
        "escalation_path": "CMO notified, Risk Management engaged",
    },
)

# Closing a sentinel event without RCA reference raises PolicyViolationError
await svc.close_incident(
    tenant_id="facility_001",
    incident_id=incident.id,
    rca_reference="RCA-2026-001",
    corrective_actions=["Time-out procedure revised", "Surgical site marking protocol updated"],
)
```

---

### 4. HIPAA Compliance

#### Risk Assessment

```python
assessment = await svc.hipaa_risk_assessment(period="2026-Q2")
print(assessment["overall_risk_level"], assessment["average_score"])
```

#### Automated Gap Analysis

```python
# Pass your system configuration snapshot for precision scoring
# Omit config_snapshot for default 75-point baseline scoring
gaps = await svc.hipaa_gap_analysis(
    tenant_id="facility_001",
    config_snapshot={
        "access_management": {"score": 62},
        "transmission_security": {"score": 58},
        "contingency_plan": {"score": 71},
    },
)
print(gaps["gap_count"], gaps["critical_gaps"])
for g in gaps["gaps"]:
    print(f"[{g['priority'].upper()}] {g['domain']} ({g['cfr_ref']}): {g['current_score']} → remediation: {g['remediation']}")
```

#### Data Breach Notification

```python
breach = await svc.data_breach_notification(
    tenant_id="facility_001",
    breach_type="unauthorized_access",
    records_affected=650,
    description="Employee accessed 650 patient records without authorisation",
    discovered_at=datetime(2026, 6, 1),
)
# breach["large_breach"] = True → media notice required

# Full multi-jurisdiction obligation timeline
timeline = await svc.breach_notification_timeline(
    tenant_id="facility_001",
    breach_id=breach["id"],
    records_affected=650,
    discovered_at=datetime(2026, 6, 1),
    jurisdictions=["us_hipaa", "gdpr"],
)
for ob in timeline["obligations"]:
    print(f"{ob['obligation']} — deadline: {ob['deadline']} ({ob['regulation']})")
```

---

### 5. AI-Assisted ICD-10 Coding

```python
# Uses locally-hosted Ollama model — no PHI leaves the facility
suggestions = await svc.suggest_icd_codes(
    clinical_text="Patient presents with acute chest pain, diaphoresis, and shortness of breath",
    max_suggestions=5,
)
for s in suggestions["suggestions"]:
    print(f"{s['code']} — {s['description']} (confidence: {s['confidence']:.0%})")
```

Suggestions require clinician confirmation before use in submissions. All confirmations and overrides are audit-logged against the submission record.

---

### 6. Regulatory Submissions

```python
# File a CMS IQR submission
from apg_healthcare_reg.models import RegulatorySubmissionCreate

sub = await svc.file_submission(RegulatorySubmissionCreate(
    tenant_id="facility_001",
    report_type="cms_iqr",
    title="CMS IQR Q2 2026",
    reporting_period_start=datetime(2026, 4, 1),
    reporting_period_end=datetime(2026, 6, 30),
    submitted_to="CMS",
    prepared_by="quality_analyst",
    data_references=["quality_measure_cache_q2_2026"],
    created_by="quality_director",
))

# Submit to agency
submitted = await svc.submit_submission("facility_001", sub.id)
print(submitted.submission_reference)
```

---

### 7. Compliance Matrix (Multi-Framework)

```python
matrix = await svc.compliance_matrix_status(
    tenant_id="facility_001",
    frameworks=["hipaa", "cms_conditions", "joint_commission"],
)
print(f"{matrix['deficiencies']} deficiencies across {len(matrix['frameworks_evaluated'])} frameworks")
for d in matrix["deficiency_detail"]:
    print(f"  {d['control']} / {d['framework']}: score={d['score']}, cross-risk: {d['cross_framework_risk']}")
```

---

### 8. State-Specific Regulatory Rules

```python
# Evaluate California-specific obligations for a breach notification
obligations = await svc.state_rules_evaluate(
    tenant_id="facility_001",
    state_code="CA",
    operation="data_breach_notification",
    context={"records_affected": 200},
)
for ob in obligations["obligations"]:
    print(f"{ob['rule']} — {ob['citation']} — deadline in {ob['deadline_days']} days")
```

---

### 9. Regulatory Intelligence Feed

```python
# Fetch the last 30 days of regulatory updates from CMS, FDA, and OIG
intel = await svc.regulatory_intelligence_fetch(
    tenant_id="facility_001",
    sources=["cms", "fda_medwatch", "oig_work_plan"],
    since_days=30,
)
print(f"{intel['items_found']} updates found, {intel['action_required_count']} require action")
for item in intel["items"]:
    if item["action_required"]:
        print(f"[{item['severity'].upper()}] {item['title']} → affects: {item['affected_areas']}")
```

---

### 10. Corrective Action Tracking

```python
from apg_healthcare_reg.models import CorrectiveActionCreate

ca = await svc.create_corrective_action(CorrectiveActionCreate(
    tenant_id="facility_001",
    incident_id=incident.id,
    source="sentinel_event_rca",
    description="Revise surgical time-out protocol to require verbal confirmation",
    assigned_to="surgical_services_director",
    due_date=datetime(2026, 8, 1),
    priority="high",
    created_by="quality_director",
))

# Complete and verify
completed = await svc.complete_corrective_action("facility_001", ca.id, verified_by="cmo")
```

---

## Compliance Dashboard

```python
dashboard = await svc.compliance_dashboard()
```

Returns:
- `compliance_score` — 0–100 weighted score
- `risk_level` — critical / high / medium / low
- Licence, accreditation, incident, corrective action, submission, and HIPAA assessment summaries

---

## Regulatory Calendar

```python
calendar = await svc.regulatory_calendar("facility_001")
for item in calendar["calendar"]:
    print(f"{item['type']} — {item['days_remaining']}d — {item['due_date']}")
```

---

## Streaming / Event Bus (NATS JetStream)

All mutations publish structured events to NATS JetStream subjects following the pattern:

```
apg.healthcare.reg.{event_category}.{tenant_id}
```

Bytewax pipelines consume the stream for real-time KPI aggregation. The notification capability (`ntfy`) subscribes to alert subjects for routing to email, pager, and SIEM integrations.

Key subjects:

| Subject | Events |
|---------|--------|
| `apg.healthcare.reg.alerts.{tenant}.critical` | Sentinel events, overdue MDRs, Red readiness band |
| `apg.healthcare.reg.alerts.{tenant}.breach` | Breach notification deadline escalations |
| `apg.healthcare.reg.intelligence.{tenant}` | New regulatory intelligence items |
| `apg.healthcare.reg.{tenant}.license` | License lifecycle events |
| `apg.healthcare.reg.{tenant}.incident` | Incident lifecycle events |
| `apg.healthcare.reg.{tenant}.rca` | RCA workflow state transitions |

---

## Configuration Reference

All keys are tenant-scoped. Set via the `conf` capability or prefix with `HEALTHCARE_REG_`:

```python
HEALTHCARE_REG_EXPIRY_WARNING_DAYS=90
HEALTHCARE_REG_SENTINEL_NOTIFICATION_HOURS=72
HEALTHCARE_REG_ICD_SUGGESTION_MODEL=llama3-medical
HEALTHCARE_REG_INTELLIGENCE_SOURCES=cms,fda_medwatch,oig_work_plan
HEALTHCARE_REG_STATE_RULES_ENABLED_STATES=CA,TX,NY
```

---

## Composability

```apg
use healthcare_reg;
use healthcare_ana;   // quality measure data for CMS submission auto-population
use healthcare_dev;   // device adverse events → FDA MDR pipeline
use healthcare_pha;   // controlled substance logs → DEA Schedule II submissions
use ntfy;             // license expiry and sentinel event alerts
use wflo;             // RCA and corrective action approval workflows
```

---

## Permissions Reference

| Permission | Scope |
|-----------|-------|
| `healthcare_reg:view` | Dashboard and calendar read access |
| `healthcare_reg:licenses` | License read/write |
| `healthcare_reg:accreditation` | Accreditation read/write/status |
| `healthcare_reg:incidents` | Incident read |
| `healthcare_reg:incidents_write` | Incident create, close, RCA |
| `healthcare_reg:submissions` | Submission read/write/submit |
| `healthcare_reg:corrective_actions` | CA read/write/complete |
| `healthcare_reg:hipaa` | HIPAA assessments, gap analysis, breach notifications |
| `healthcare_reg:compliance` | Compliance matrix and multi-framework evaluation |
| `healthcare_reg:icd` | ICD-10 suggestion endpoint |
| `healthcare_reg:intelligence` | Regulatory intelligence feed |
| `healthcare_reg:state_rules` | State-specific rule evaluation |

---

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Flask-AppBuilder views and schemas
- `capability_contract.py` — Supported types, business rules, and contract evaluation
- `README.md` — Quick reference and API route table
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement proposals with implementation detail
