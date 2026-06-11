# Telemedicine — User Guide

**Capability ID**: `healthcare_tel` | **Domain**: `healthcare` | **Version**: `2.0.0`

## Description

Full-featured telemedicine capability covering virtual consultation booking, video session management with consent and E-911 disclosure enforcement, remote patient monitoring, electronic prescription transmission and renewal, telehealth-specific billing, asynchronous encrypted patient-provider messaging, AI-assisted patient triage, structured chronic disease care programs, real-time video quality monitoring, and cross-jurisdiction compliance checking.

## Installation

```bash
pip install apg-healthcare-tel
```

## Quick Start

```python
from apg_healthcare_tel.service import TelemedicineService
from datetime import datetime

svc = TelemedicineService(tenant_id="clinic-001", actor_id="dr-jane")

# Book a teleconsultation
appt = await svc.book_teleconsult(
    patient_id="pat-abc",
    provider_id="dr-jane",
    appointment_type="video",
    preferred_time=datetime(2026, 6, 20, 10, 0),
)
print(appt["join_url"])

# Triage a patient before booking
triage = await svc.triage_patient(
    patient_id="pat-abc",
    symptoms=["chest_pain", "difficulty_breathing"],
    reported_vitals={"spo2": 88, "heart_rate": 130},
)
# triage["esi_score"] == 1 → immediate escalation

# Start and monitor video session
session = await svc.video_session_start(appt["id"], patient_token="tok-xyz")
qoe = await svc.record_session_qoe(
    session_id=session["session_id"],
    packet_loss_pct=3.5, jitter_ms=80, round_trip_ms=200,
)
# qoe["degraded"] True → recommended_resolution = "480p"

# Issue an e-prescription
rx = await svc.e_prescription(
    appointment_id=appt["id"],
    medications=[{"drug_name": "Metformin", "dose": "500mg", "frequency": "twice_daily", "quantity": 60, "refills": 3}],
)

# End session and bill
await svc.video_session_end(appt["id"], duration_mins=25, notes="Routine DM2 follow-up.")
await svc.teleconsult_billing(appt["id"])
```

## Core Features

### 1. Consultation Booking

```python
# Low-level structured booking
from apg_healthcare_tel.models import ConsultationCreate

payload = ConsultationCreate(
    tenant_id="clinic-001",
    patient_id="pat-abc",
    provider_id="dr-jane",
    consultation_type="video",           # video | audio_only | asynchronous | rpm | urgent_care
    scheduled_at=datetime(2026, 6, 20, 10, 0),
    duration_minutes=30,
    chief_complaint="Diabetes review",
    platform="zoom",                     # zoom | teams | webrtc_native | phone
    patient_consent_obtained=True,
    e911_disclosure_provided=True,
    created_by="front-desk",
)
consult = await svc.book_consultation(payload)

# Check jurisdiction compliance before booking
compliance = await svc.check_jurisdiction_compliance(
    provider_id="dr-jane",
    patient_jurisdiction="US-TX",
    consultation_type="audio_only",
)
if not compliance["compliant"]:
    raise ValueError(compliance["notes"])
```

### 2. Video Session Lifecycle

```python
# Create session (enforces consent + E-911 disclosure)
from apg_healthcare_tel.models import TeleSessionCreate

session_payload = TeleSessionCreate(
    tenant_id="clinic-001",
    consultation_id=consult.id,
    patient_id="pat-abc",
    provider_id="dr-jane",
    platform="webrtc_native",
    patient_consent_obtained=True,
    e911_disclosure_provided=True,
    technical_check_completed=True,
    created_by="dr-jane",
)
session = await svc.create_session(session_payload)

# Record QoE during session
qoe = await svc.record_session_qoe(
    session_id=session.id,
    packet_loss_pct=1.2,
    jitter_ms=35,
    round_trip_ms=120,
    resolution="720p",
)
# qoe["mos_score"] — 4.2 = good, <3.5 triggers NATS alert

# Complete session
await svc.complete_session(tenant_id="clinic-001", session_id=session.id)
```

### 3. Patient Triage

ESI (Emergency Severity Index) scoring at intake to prioritise the provider worklist. ESI-1 and ESI-2 emit a NATS `tel.triage.critical.{tenant_id}` event for immediate escalation.

```python
result = await svc.triage_patient(
    patient_id="pat-abc",
    symptoms=["fever", "cough"],
    reported_vitals={"spo2": 97, "heart_rate": 88},
)
# result["esi_score"]           — 1 (critical) to 5 (non-urgent)
# result["recommended_action"]  — immediate_teleconsult | same_day_teleconsult | next_available_slot
# result["escalation_required"] — True if ESI <= 2
```

### 4. Remote Patient Monitoring

```python
# Enrol patient and configure devices
enrollment = await svc.remote_monitoring_enrol(
    patient_id="pat-abc",
    device_ids=["glucometer-x1", "bp-cuff-y2"],
    vital_types=["blood_glucose", "blood_pressure_systolic"],
)

# Ingest readings (called by device hub or patient app)
reading = await svc.vital_reading_ingest(
    patient_id="pat-abc",
    device_id="glucometer-x1",
    vital_type="blood_glucose",
    value=210.0,
    timestamp=datetime.utcnow(),
)
# reading["alert_triggered"] == True → "warning" severity

# Trend analysis over 30 days
trend = await svc.vital_trend_analysis(
    patient_id="pat-abc",
    vital_type="blood_glucose",
    days=30,
)
# trend["trend"] — increasing | decreasing | stable
```

### 5. Prescription Management

```python
# Electronic prescription
rx = await svc.e_prescription(
    appointment_id=appt["id"],
    medications=[
        {"drug_name": "Amoxicillin", "dose": "500mg", "frequency": "three_times_daily",
         "quantity": 21, "refills": 0, "schedule": "OTC"},
    ],
)

# Request renewal (non-controlled auto-routes to review queue)
renewal = await svc.request_prescription_renewal(
    patient_id="pat-abc",
    original_rx_id=rx["id"],
    pharmacy_id="pharma-001",
    reason="Ongoing hypertension management",
)
# renewal["approval_status"] — pending_review | pending_consult | denied
# Schedule II → always denied; III-V → requires teleconsult within 30 days
```

### 6. Asynchronous Secure Messaging

For patients on low-bandwidth connections or asynchronous consultation types.

```python
msg = await svc.send_async_message(
    thread_id="thread-pat-abc-dr-jane",
    sender_id="pat-abc",
    recipient_id="dr-jane",
    content="My blood sugar has been high all week. Readings attached.",
    attachments=[{"name": "readings.pdf", "mime_type": "application/pdf", "size_bytes": 48000}],
    message_type="text",
)
# msg["encryption"] == "AES-256-GCM"
# msg["nats_event"]["subject"] == "tel.async.thread-pat-abc-dr-jane"
```

### 7. Chronic Disease Care Programs

```python
program = await svc.enroll_care_program(
    patient_id="pat-abc",
    program_type="diabetes",           # diabetes | hypertension | chf | copd | mental_health
    provider_id="dr-jane",
    duration_weeks=12,
)
# program["milestones"] — weekly check-in schedule
# First check-in auto-scheduled via schedule_follow_up()
# NATS event: tel.program.enrolled.{tenant_id}
```

### 8. Provider Performance Reporting

```python
report = await svc.provider_performance_report(
    provider_id="dr-jane",
    period="2026-Q2",
)
# report["consultations"]["completion_rate_pct"]
# report["prescriptions"]["controlled_rate_pct"]
# report["billing"]["total_billed_amount"]
```

### 9. Analytics & Quality

```python
# Telehealth-specific quality metrics
quality = await svc.telehealth_quality_metrics(period="2026-Q2")
# quality["consent_rate_pct"]
# quality["e911_disclosure_rate_pct"]

# Full analytics snapshot
analytics = await svc.teleconsult_analytics(period="2026-Q2")

# Dashboard summary
summary = await svc.dashboard_summary(tenant_id="clinic-001")
```

## Provides

- `virtual_consultation_booking`
- `video_session_management`
- `remote_patient_monitoring`
- `prescription_transmission`
- `telehealth_billing`
- `async_secure_messaging`
- `patient_triage`
- `care_program_management`
- `qoe_monitoring`
- `jurisdiction_compliance`
- `provider_performance_analytics`
- `ai_diagnosis_assist`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `schd`
- `comp`
- `moni`
- `mqeb` (NATS JetStream)

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-tel/dashboard` | `healthcare_tel:view` | Overview |
| `/healthcare-tel/schedule` | `healthcare_tel:schedule` | Scheduling |
| `/healthcare-tel/schedule/new` | `healthcare_tel:schedule_write` | Scheduling |
| `/healthcare-tel/schedule/<id>` | `healthcare_tel:schedule` | Scheduling |
| `/healthcare-tel/sessions` | `healthcare_tel:sessions` | Sessions |
| `/healthcare-tel/sessions/<id>/room` | `healthcare_tel:sessions` | Sessions |
| `/healthcare-tel/monitoring` | `healthcare_tel:monitoring` | Monitoring |
| `/healthcare-tel/monitoring/<patient_id>` | `healthcare_tel:monitoring` | Monitoring |
| `/healthcare-tel/messages` | `healthcare_tel:messages` | Messaging |
| `/healthcare-tel/triage` | `healthcare_tel:triage` | Triage |
| `/healthcare-tel/programs` | `healthcare_tel:programs` | Care Programs |
| `/healthcare-tel/analytics` | `healthcare_tel:analytics` | Analytics |
| `/healthcare-tel/providers/<id>/performance` | `healthcare_tel:analytics` | Analytics |

## All Service Methods

| Method | Description |
|--------|-------------|
| `describe()` | Return capability contract |
| `book_consultation()` | Structured consultation booking |
| `book_teleconsult()` | Quick teleconsult booking with availability check |
| `cancel_consultation()` | Cancel a consultation |
| `get_consultation()` | Fetch consultation by ID |
| `list_consultations()` | List consultations with optional filters |
| `update_consultation_status()` | Update consultation status |
| `create_session()` | Create video session (enforces consent) |
| `video_session_start()` | Start a video session |
| `video_session_end()` | End a session with notes |
| `complete_session()` | Mark session completed, compute duration |
| `get_session()` | Fetch session by ID |
| `list_sessions()` | List sessions with optional filters |
| `record_session_qoe()` | Record MOS-based QoE metrics |
| `enroll_monitoring()` | Enrol patient monitoring (structured) |
| `remote_monitoring_enrol()` | Quick monitoring enrolment |
| `vital_reading_ingest()` | Ingest a vital reading and evaluate thresholds |
| `telemonitoring_alert()` | Create and route a monitoring alert |
| `list_monitoring()` | List monitoring enrollments |
| `vital_trend_analysis()` | Trend statistics over N days |
| `active_monitoring_summary()` | Summary of active monitoring |
| `transmit_prescription()` | Structured prescription transmission |
| `e_prescription()` | Quick e-prescription issue |
| `request_prescription_renewal()` | Renewal with schedule-based routing |
| `list_prescriptions()` | List prescriptions |
| `create_billing_record()` | Structured billing record creation |
| `teleconsult_billing()` | Quick billing from appointment |
| `list_billing()` | List billing records |
| `record_patient_consent()` | Record consent event |
| `consent_check()` | Verify patient consent status |
| `create_clinical_note()` | Structured SOAP/progress note |
| `get_consultation_notes()` | Retrieve notes for a consultation |
| `create_referral()` | Specialist referral with urgency routing |
| `schedule_follow_up()` | Schedule follow-up appointment |
| `provider_availability()` | Available time slots for a specialty |
| `triage_patient()` | ESI 1–5 acuity scoring with NATS escalation |
| `send_async_message()` | Encrypted async message |
| `enroll_care_program()` | Chronic disease program enrollment |
| `check_jurisdiction_compliance()` | Jurisdiction licence validation |
| `provider_performance_report()` | Provider KPI report |
| `ai_diagnosis_assist()` | Differential diagnosis from symptoms |
| `teleconsult_analytics()` | Full analytics snapshot |
| `teleconsult_kpi_summary()` | Concise KPI card |
| `telehealth_quality_metrics()` | Quality and compliance metrics |
| `dashboard_summary()` | Summary for dashboard widget |
| `export_consultation_data()` | Export consultation records |
| `health_check()` | Service health and store sizes |

## NATS Integration

`healthcare_tel` publishes events using NATS subjects with the pattern `tel.{domain}.{tenant_id}`. Consume with NATS JetStream for durable, ordered delivery:

```python
import nats

nc = await nats.connect("nats://localhost:4222")
js = nc.jetstream()

# Subscribe to all vital alerts
await js.subscribe("tel.alerts.clinic-001", cb=handle_vital_alert)

# Subscribe to triage escalations
await js.subscribe("tel.triage.critical.clinic-001", cb=handle_critical_triage)

# Subscribe to QoE degradation
await js.subscribe("tel.qoe.alert.clinic-001", cb=handle_qoe_degradation)
```

For streaming dataflows, use bytewax to consume NATS subjects and apply window-based aggregations.

## Interoperability

`healthcare_tel` integrates with other APG capabilities through the composition engine:

```apg
use healthcare_tel;
use healthcare_emr;   -- consultation notes flow to EMR
use healthcare_pha;   -- prescriptions validated against formulary
use healthcare_pmt;   -- billing records sent to payment subsystem
use healthcare_ed;    -- ESI-1/2 triage events handed off to ED
use healthcare_qual;  -- care program milestones feed HEDIS quality scoring
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_TEL_`.

```bash
HEALTHCARE_TEL_CONSENT_REQUIRED=true
HEALTHCARE_TEL_CONTROLLED_SUBSTANCE_REQUIRES_IN_PERSON=true
HEALTHCARE_TEL_PLACE_OF_SERVICE_CODE=02
HEALTHCARE_TEL_QOE_MOS_THRESHOLD=3.5
HEALTHCARE_TEL_TRIAGE_ESCALATION_ENABLED=true
HEALTHCARE_TEL_NATS_URL=nats://localhost:4222
```

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Business rules and supported constants
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 roadmap improvements
