# Mine Safety & Compliance — User Guide

**Capability ID**: `mining_saf` | **Domain**: `mining` | **Version**: `1.1.0`
**© 2025 Datacraft** | Author: Nyimbi Odero

---

## Description

`mining_saf` manages the full SHEQ lifecycle for mining operations: incident reporting and investigation, hazard identification, bowtie risk analysis, job/task risk assessments, permit-to-work with conflict detection and LOTO isolation tracking, stop-work authority management, corrective action tracking, emergency drills, critical control monitoring, safety culture surveys, statutory regulatory reporting, and ISO 45001 compliance gap assessment.

---

## Installation

```bash
pip install apg-mining-saf
```

---

## Quick Start

```python
import asyncio
from capabilities.mining.saf.service import SafService

svc = SafService(tenant_id="mine-site-001")

async def main():
    # Report a near-miss
    incident = await svc.incident_report(
        incident_type="near_miss",
        location="Open Pit - Level 4",
        injured_persons=[],
        lost_time=False,
        description="Dump truck came within 2m of drill rig at intersection",
        reported_by="emp-0042",
        mine_area="Open Pit",
    )
    print(incident["id"])

asyncio.run(main())
```

---

## Core Workflows

### 1. Incident Reporting and Investigation

```python
# Report a Lost Time Injury — triggers mandatory escalation log
incident = await svc.incident_report(
    incident_type="LTI",
    location="Underground Level 5",
    injured_persons=[{"name": "John Doe", "employee_id": "emp-0010", "injury": "fractured wrist"}],
    lost_time=True,
    description="Worker's hand caught in conveyor during maintenance",
    reported_by="sup-001",
    immediate_cause="Guard removed for maintenance not reinstated",
    root_cause="Maintenance procedure did not specify isolation before guard removal",
)

# Send statutory regulatory notification
await svc.send_regulatory_notification(incident["id"], sent_by="hse-mgr-001")

# Open investigation
await svc.open_investigation(incident["id"], investigation_id="inv-2026-001")

# Close only after investigation is linked (enforced for LTI, fatality, dangerous occurrence)
await svc.close_incident(
    incident["id"],
    close_notes="ICAM investigation complete. Control improvements implemented.",
    closed_by="hse-mgr-001",
)
```

**Rules enforced:**
- LTI/fatality/dangerous occurrence: `close_incident()` raises `PermissionError` if no `investigation_id`.
- Fatality and LTI: CRITICAL log entry emitted at report time.
- Statutory notification: tracked via `regulatory_notification_sent` flag.

---

### 2. Hazard Identification

```python
from capabilities.mining.saf.models import (
    HazardCreate, HazardCategory, RiskRating, ConsequenceLevel,
    LikelihoodLevel, ControlMeasureCreate, ControlType,
)
from datetime import datetime

hazard = await svc.identify_hazard(
    payload=HazardCreate(
        tenant_id="mine-site-001",
        hazard_category=HazardCategory.GROUND_INSTABILITY,
        location="Decline Ramp Level 3",
        mine_area="Underground",
        description="Fractured hangingwall above ore drive intersection",
        potential_consequence=ConsequenceLevel.CATASTROPHIC,
        likelihood=LikelihoodLevel.POSSIBLE,
        inherent_risk_rating=RiskRating.EXTREME,
        stop_work_invoked=True,  # mandatory for EXTREME
        identified_by="geo-001",
        identified_at=datetime.utcnow(),
        control_measures=[
            ControlMeasureCreate(
                control_type=ControlType.ENGINEERING,
                description="Install mesh bolting and cable anchors",
                responsible_person_id="ops-001",
            )
        ],
    ),
    created_by="geo-001",
)
```

**Rule enforced:** `RiskRating.EXTREME` with `stop_work_invoked=False` raises `PermissionError`.

---

### 3. Bowtie Risk Analysis

```python
bowtie = await svc.create_bowtie_analysis(
    material_unwanted_event="Fall of Ground",
    mine_area="Underground",
    threat_sources=[
        {"threat": "Fractured hanging wall", "category": "gravitational", "likelihood": "possible"},
        {"threat": "Poor blast design", "category": "mechanical", "likelihood": "unlikely"},
    ],
    prevention_controls=[
        {"control": "Geological mapping and face inspection", "type": "administrative",
         "is_critical": False, "owner_id": "geo-001"},
        {"control": "Ground support installation to standard", "type": "engineering",
         "is_critical": True, "owner_id": "ops-001"},
    ],
    mitigation_controls=[
        {"control": "Exclusion zone barricading", "type": "administrative",
         "is_critical": True, "owner_id": "shiftboss-001"},
        {"control": "Emergency response plan", "type": "administrative",
         "is_critical": False, "owner_id": "hse-001"},
    ],
    consequences=[
        {"consequence": "Worker fatality", "severity": "catastrophic", "receptor": "underground workers"},
        {"consequence": "Equipment damage", "severity": "major", "receptor": "fleet"},
    ],
    escalation_factors=["Wet ground conditions", "Seismic activity"],
    created_by="eng-001",
)
# bowtie["hierarchy_control_index"] — weighted score 0-1 favouring engineering controls
# bowtie["critical_control_count"] — count of is_critical=True controls
```

---

### 4. Permit to Work with Conflict Detection and LOTO

```python
# Check for conflicts before issuing
conflicts = await svc.check_permit_conflicts(
    mine_area="Smelter Bay 2",
    proposed_work_type="hot_work",
)
# conflicts["conflict_level"]: "none" | "warning" | "blocked"
if conflicts["conflict_level"] == "blocked":
    raise RuntimeError(f"Cannot issue PTW: {conflicts['conflicts']}")

# Issue the permit
permit = await svc.permit_to_work(
    work_type="hot_work",
    location="Smelter Bay 2 — pipe repair",
    hazards=["Burns", "Fire", "Fume inhalation"],
    precautions=["Fire watch assigned", "Fire extinguisher within 2m", "RPE worn"],
    issuer_id="ptw-auth-001",
    receiver_id="emp-0055",
    valid_hours=8,
    isolations=[{"device": "Gas supply valve V-14", "type": "mechanical"}],
)

# Register and verify LOTO isolation points
iso = await svc.register_isolation_point(
    permit_id=permit["id"],
    isolation_type="mechanical",
    device_id="V-14",
    location_description="Gas supply valve upstream of smelter bay 2 feed line",
    isolated_by="emp-0055",
)
# Second-person verification
await svc.verify_isolation_point(iso["id"], verified_by="ptw-auth-001")

# After work complete — reinstate isolation
await svc.reinstate_isolation_point(iso["id"], reinstated_by="emp-0055")
```

**Conflict matrix (hard blocks):** `hot_work` + `confined_space_entry` in same area → BLOCKED.
**Reinstatement rule:** `verify_isolation_point()` must precede `reinstate_isolation_point()` or `ValueError` is raised.

---

### 5. Stop-Work Authority

```python
# Invoke SWA — automatically suspends active permits in the area
swa = await svc.invoke_stop_work_authority(
    location="Open Pit Bench 5",
    mine_area="Open Pit",
    invoked_by="emp-0088",
    reason="Slope movement sensors exceeded threshold — potential slope failure",
)
# swa["suspended_permit_ids"] lists all permits automatically suspended

# Authorise resumption only after investigation
await svc.authorise_work_resumption(
    swa_id=swa["id"],
    authorised_by="mine-mgr-001",
    investigation_id="inv-2026-010",
    resumption_conditions=[
        "Geotechnical assessment confirms stability",
        "Slope monitoring frequency increased to 15-minute intervals",
    ],
)
# swa["hold_duration_minutes"] computed automatically
# Suspended permits are reinstated to "active" status
```

---

### 6. Corrective Actions

```python
from datetime import datetime, timedelta

ca = await svc.corrective_action(
    finding_id=incident["id"],
    action="Install proximity detection system on all heavy vehicles and drill rigs",
    responsible="fleet-eng-001",
    deadline=datetime.utcnow() + timedelta(days=90),
    priority="critical",
    source_type="incident",
    created_by="hse-mgr-001",
)

# Close with evidence reference
await svc.close_corrective_action_by_id(
    ca_id=ca["id"],
    closed_by="fleet-eng-001",
    evidence="PDS installation completion certificate — ref DOC-2026-0441",
)

# Scan and flag overdue CAs (call on schedule)
overdue = await svc.flag_overdue_corrective_actions()
```

---

### 7. Safety Statistics and Leading Indicators

```python
# Lagging indicators
stats = await svc.safety_statistics("2026-06")
# stats["ltifr"], stats["trifr"], stats["fatalities"], etc.

# Leading indicators — near-miss rate, CC pass rate, on-time CA closure
leading = await svc.get_leading_indicators("2026-06")
# leading["near_miss_reporting_rate_pct"]
# leading["critical_control_pass_rate_pct"]
# leading["corrective_action_on_time_closure_pct"]
# leading["overdue_ca_ratio_pct"]

# Area risk heat map for shift supervisor situational awareness
heatmap = await svc.get_area_risk_heatmap()
# Sorted highest-to-lowest composite_risk_score

# Incident pattern detection
patterns = await svc.analyse_incident_patterns(lookback_days=90, min_occurrences=2)
# patterns["recurring_patterns"] — each entry has occurrence_count, unresolved_pattern flag
```

---

### 8. Critical Control Monitoring

```python
result = await svc.critical_control_monitoring(
    control_id="CC-FOG-001",
    verification_result="ineffective",   # triggers CRITICAL escalation log
    verifier_id="shiftboss-002",
    control_description="Ground support installation — mesh bolting to standard",
    material_unwanted_event="Fall of Ground",
    deficiency_found=True,
    deficiency_detail="Bolts missing on 3m section of east drive 5 hangingwall",
)
# result["escalation_required"] == True when verification_result == "ineffective"
```

---

### 9. Regulatory Reporting

```python
# Generate draft report
report = await svc.regulatory_report_safety(
    period="2026-06",
    jurisdiction="Kenya Mines Department",
    submitted_by="hse-mgr-001",
    submission_deadline=datetime(2026, 7, 31),
)
# report["overdue_notifications"] — incidents requiring regulatory notification not yet sent

# Submit
submitted = await svc.submit_regulatory_report(
    report_id=report["id"],
    submitted_by="hse-mgr-001",
    submission_reference="KMD-2026-Q2-001",
)
# submitted["status"] == "submitted"

# List all submitted reports
reports = await svc.list_regulatory_reports(status="submitted", jurisdiction="Kenya Mines Department")
```

---

### 10. ISO 45001 Compliance Gap Assessment

```python
gap = await svc.compliance_report(standard="ISO_45001")
# gap["overall_compliance_index"] — 0 to 100
# gap["clause_scores"] — per-clause observable indicators:
#   incident_investigation_completion_pct
#   corrective_action_closure_pct
#   critical_control_pass_pct
#   emergency_drill_score
#   inspection_score
```

---

### 11. Safety Culture Survey

```python
survey = await svc.safety_culture_survey(
    period="2026-Q2",
    survey_instrument="Hearts and Minds",
    facilitated_by="hse-mgr-001",
    participation_rate_pct=72.5,
    responses=[
        {"question_id": "Q1", "dimension": "leadership", "score": 3.8, "max_score": 5.0},
        {"question_id": "Q2", "dimension": "reporting_culture", "score": 4.1, "max_score": 5.0},
        {"question_id": "Q3", "dimension": "learning", "score": 2.9, "max_score": 5.0},
    ],
)
# survey["culture_level"]: pathological | reactive | calculative | proactive | generative
# survey["overall_culture_index"]: 0.0 – 5.0
```

---

## Service Method Reference

### Incidents
| Method | Description |
|--------|-------------|
| `incident_report()` | Record incident with causal analysis fields |
| `report_incident()` | Structured Pydantic-model variant |
| `get_incident()` | Fetch by id |
| `list_incidents()` | Filter by type, status, date |
| `send_regulatory_notification()` | Mark statutory notification sent |
| `open_investigation()` | Link investigation record |
| `close_incident()` | Investigation-gated close |

### Hazards and Risk
| Method | Description |
|--------|-------------|
| `identify_hazard()` | Record hazard; blocks EXTREME without SWA |
| `get_hazard()` / `close_hazard()` / `list_hazards()` | CRUD operations |
| `add_risk_register_entry()` / `get_risk_register_entry()` / `list_risk_register()` | Risk register |
| `risk_assessment()` | JRA/JSA with validity window; blocks extreme residual risk |
| `get_active_risk_assessments()` | Filter by area, current validity |
| `create_bowtie_analysis()` | Bowtie with HCI scoring and critical control count |
| `get_bowtie_analysis()` | Fetch by id |

### Permits to Work
| Method | Description |
|--------|-------------|
| `permit_to_work()` / `issue_permit()` | Issue PTW |
| `get_permit()` / `close_permit()` / `check_permit_valid()` | Lifecycle |
| `list_active_permits()` | Non-expired permits, optional area filter |
| `check_permit_conflicts()` | Conflict matrix check for area + proposed work type |

### LOTO Isolation
| Method | Description |
|--------|-------------|
| `register_isolation_point()` | Register isolation against a PTW |
| `verify_isolation_point()` | Second-person verification |
| `reinstate_isolation_point()` | Record removal after PTW close |
| `list_isolation_points()` | Filter by permit or state |

### Stop-Work Authority
| Method | Description |
|--------|-------------|
| `invoke_stop_work_authority()` | Invoke SWA; suspends permits in area |
| `authorise_work_resumption()` | Investigation-gated resumption; reinstates permits |
| `list_stop_work_records()` | Filter by area, active-only |

### Corrective Actions
| Method | Description |
|--------|-------------|
| `corrective_action()` | Create CA from finding |
| `create_corrective_action()` | Pydantic-model variant |
| `close_corrective_action_by_id()` | Close with evidence reference |
| `close_corrective_action()` | Structured close |
| `flag_overdue_corrective_actions()` | Idempotent overdue scan |
| `list_corrective_actions()` | Filter by status, source type |

### Inspections and Drills
| Method | Description |
|--------|-------------|
| `safety_inspection()` | Record inspection with findings |
| `list_safety_inspections()` | Filter by area, status |
| `emergency_drill()` | Record drill outcome |
| `list_emergency_drills()` | Filter by drill type |

### Critical Controls
| Method | Description |
|--------|-------------|
| `critical_control_monitoring()` | Verify control; escalates on `ineffective` |
| `list_critical_control_verifications()` | Filter by control_id, ineffective-only |

### Analytics and Reporting
| Method | Description |
|--------|-------------|
| `get_safety_statistics()` | Aggregate LTIFR, counts, open CAs |
| `safety_statistics()` | Period-scoped LTIFR and TRIFR |
| `get_leading_indicators()` | Near-miss rate, CC pass rate, CA closure, etc. |
| `get_area_risk_heatmap()` | Composite risk score by mine area |
| `analyse_incident_patterns()` | Recurring pattern detection with CA linkage |
| `safety_culture_survey()` / `list_culture_surveys()` | Hearts and Minds culture index |
| `regulatory_report_safety()` | Generate statutory report draft |
| `submit_regulatory_report()` | Mark submitted with reference number |
| `list_regulatory_reports()` | Filter by period, status, jurisdiction |
| `compliance_report()` | ISO 45001 clause-level compliance gap assessment |
| `health_check()` | Service health and store sizes |
| `export_records()` | Tenant data bundle |

---

## Configuration Keys

| Key | Default | Description |
|-----|---------|-------------|
| `incidents.immediate_notification_required` | `true` | Fatality/LTI triggers immediate notification |
| `incidents.investigation_required_for_lti_and_above` | `true` | Mandatory investigation before closing LTI/fatality/DO |
| `hazards.risk_assessment_required` | `true` | Risk assessment mandatory for all hazards |
| `permits_to_work.issuer_qualification_required` | `true` | PTW issuer must hold statutory qualification |
| `permits_to_work.isolation_verification_required` | `true` | Isolation must be verified before PTW issue |
| `governance.open_extreme_risk_stop_work_trigger` | `true` | Extreme hazards require SWA before submission |
| `analytics.ltifr_hours_worked_source` | `estimate` | `estimate` (200k/month) or `mining_pro` integration |

---

## Permissions

| Permission | Grants |
|-----------|--------|
| `mining_saf:view` | Read all records |
| `mining_saf:write` | Create/update incidents, hazards, CAs, inspections, drills |
| `mining_saf:ptw_issue` | Issue and close permits, manage isolation points |
| `mining_saf:swa_invoke` | Invoke stop-work authority and authorise resumption |
| `mining_saf:reports` | Access statistics, compliance, regulatory reports |
| `mining_saf:admin` | Full access including delete (archived only) |

---

## Further Reading

- `service.py` — Complete async business logic
- `models.py` — Pydantic v2 data models and enums
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvement opportunities
- `SPECIFICATION.md` — Full capability specification
