# Telemedicine

## Overview
Full-featured telemedicine capability covering virtual consultation booking, video session management with consent and E-911 disclosure enforcement, remote patient monitoring enrollment, electronic prescription transmission, telehealth-specific billing, asynchronous encrypted messaging, patient triage, chronic disease care programs, quality-of-experience tracking, and jurisdiction compliance checking. Schedule II/III prescription transmission is blocked without a prior in-person visit.

## Capability ID
`healthcare_tel`

## Provides
- `virtual_consultation_booking`: Book video, audio-only, asynchronous, RPM, and urgent care consultations with platform selection
- `video_session_management`: Create and manage WebRTC/platform sessions with join URL generation, QoE telemetry, and duration tracking
- `remote_patient_monitoring`: Enroll patients with devices (glucometer, BP cuff, oximeter, etc.) with alert thresholds and vital trend analysis
- `prescription_transmission`: Transmit and renew prescriptions via Surescripts, EPCS, or fax; blocks Schedule II without in-person visit
- `telehealth_billing`: Create billing records with telehealth-specific CPT codes (99201-99215, G2012, G2252, 99421-99423)
- `patient_consent_management`: Track informed consent and HIPAA authorization per consultation
- `async_secure_messaging`: End-to-end encrypted asynchronous text/image/voice messages between patient and provider
- `patient_triage`: ESI 1–5 acuity scoring from symptoms and self-reported vitals; ESI-1/2 triggers NATS escalation
- `care_program_management`: Structured longitudinal programs for diabetes, hypertension, CHF, COPD, mental health
- `qoe_monitoring`: Real-time MOS-based video quality monitoring with automatic downgrade recommendations
- `jurisdiction_compliance`: Provider licence check against patient jurisdiction before booking
- `provider_performance_analytics`: Per-provider KPI reports covering completion rate, prescribing, and billing productivity
- `technical_readiness_check`: Flag sessions where technical check was not completed
- `ai_diagnosis_assist`: Differential diagnosis suggestions from locally-served Ollama clinical LLM

## Requires
- auth: Provider and patient authentication
- audl: Audit trail for all session and prescription events
- mten: Multi-tenant isolation
- conf: Tenant-specific platform and billing configuration
- ntfy: Session reminders and monitoring alerts (NATS-backed)
- wflo: Consent and prescription approval workflows
- schd: Consultation scheduling integration
- comp: Regulatory compliance for telehealth licensing
- moni: Session quality and platform availability monitoring
- mqeb: Event emission for EMR and billing downstream (NATS JetStream)

## Streaming Events (NATS Subjects)
| Subject | Trigger |
|---------|---------|
| `tel.consultation.booked.{tenant_id}` | Consultation booked |
| `tel.consultation.cancelled.{tenant_id}` | Consultation cancelled |
| `tel.session.started.{tenant_id}` | Video session started |
| `tel.session.completed.{tenant_id}` | Video session completed |
| `tel.vitals.{tenant_id}.{vital_type}` | Vital reading ingested |
| `tel.alerts.{tenant_id}` | Telemonitoring alert triggered |
| `tel.rx.transmitted.{tenant_id}` | Prescription transmitted |
| `tel.rx.renewal_requested.{tenant_id}` | Prescription renewal requested |
| `tel.rx.controlled.{tenant_id}` | Controlled substance prescription |
| `tel.billing.created.{tenant_id}` | Billing record created |
| `tel.triage.critical.{tenant_id}` | ESI-1/2 triage escalation |
| `tel.triage.scored.{tenant_id}` | Triage score assigned |
| `tel.async.{thread_id}` | Async message sent |
| `tel.qoe.alert.{tenant_id}` | Video QoE degraded (MOS < 3.5) |
| `tel.qoe.metric.{tenant_id}` | QoE metric recorded |
| `tel.program.enrolled.{tenant_id}` | Care program enrollment |
| `tel.program.milestone.{tenant_id}` | Care program milestone completed |
| `tel.compliance.denied.{tenant_id}` | Jurisdiction compliance denied |

## Configuration

| Key | Description |
|-----|-------------|
| consultations.consent_required | Require patient consent before session start |
| sessions.recording_consent_required | Require explicit consent before recording |
| prescriptions.controlled_substance_requires_in_person | Block Schedule II Rx without prior in-person visit |
| billing.place_of_service_code | Default POS code for telehealth (default: 02) |
| qoe.mos_alert_threshold | MOS threshold for QoE degradation alerts (default: 3.5) |
| triage.escalation_enabled | Enable NATS escalation events for ESI-1/2 (default: true) |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/tel/schedule | List consultations | healthcare_tel:schedule |
| POST | /api/healthcare/tel/schedule | Book consultation | healthcare_tel:schedule_write |
| GET | /api/healthcare/tel/schedule/\<id\> | Consultation detail | healthcare_tel:schedule |
| POST | /api/healthcare/tel/schedule/\<id\>/cancel | Cancel | healthcare_tel:schedule_write |
| GET | /api/healthcare/tel/sessions | List sessions | healthcare_tel:sessions |
| POST | /api/healthcare/tel/sessions | Create session | healthcare_tel:sessions |
| GET | /api/healthcare/tel/sessions/\<id\> | Session detail | healthcare_tel:sessions |
| POST | /api/healthcare/tel/sessions/\<id\>/complete | Complete session | healthcare_tel:sessions |
| POST | /api/healthcare/tel/sessions/\<id\>/qoe | Record QoE metrics | healthcare_tel:sessions |
| GET | /api/healthcare/tel/monitoring | List monitoring | healthcare_tel:monitoring |
| POST | /api/healthcare/tel/monitoring | Enroll device | healthcare_tel:monitoring |
| GET | /api/healthcare/tel/prescriptions | List Rx | healthcare_tel:prescriptions |
| POST | /api/healthcare/tel/prescriptions | Transmit Rx | healthcare_tel:prescriptions |
| POST | /api/healthcare/tel/prescriptions/renewal | Request renewal | healthcare_tel:prescriptions |
| GET | /api/healthcare/tel/billing | Billing records | healthcare_tel:billing |
| POST | /api/healthcare/tel/billing | Create billing record | healthcare_tel:billing |
| POST | /api/healthcare/tel/triage | Triage patient | healthcare_tel:triage |
| GET | /api/healthcare/tel/messages/\<thread_id\> | List thread messages | healthcare_tel:messages |
| POST | /api/healthcare/tel/messages | Send async message | healthcare_tel:messages |
| GET | /api/healthcare/tel/programs | List care programs | healthcare_tel:programs |
| POST | /api/healthcare/tel/programs | Enroll in care program | healthcare_tel:programs |
| GET | /api/healthcare/tel/compliance/check | Jurisdiction check | healthcare_tel:compliance |
| GET | /api/healthcare/tel/providers/\<id\>/performance | Provider KPIs | healthcare_tel:analytics |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| patient_consent_required | operation=start_session, patient_consent_obtained=False | deny |
| e911_disclosure_required | operation=start_session, e911_disclosure_provided=False | deny |
| controlled_substance_telemedicine_restriction | drug_schedule=schedule_ii, in_person_visit_completed=False | deny |
| schedule_ii_renewal_denied | renewal request for schedule_ii prescription | deny |
| recording_requires_consent | operation=start_recording, recording_consent_obtained=False | deny |
| cancelled_consultation_not_startable | operation=start_session, consultation_status=cancelled | deny |
| monitoring_alert_threshold_required | operation=enroll_monitoring_device, alert_threshold_configured=False | deny |
| billing_code_supported | operation=create_billing_record, billing_code_supported=False | deny |
| technical_readiness_check_required | operation=start_session, technical_check_completed=False | warn |
| jurisdiction_compliance_check | booking in restricted jurisdiction | conditional/deny |
| esi1_escalation | triage ESI score <= 2 | emit NATS escalation |
| qoe_degradation_downgrade | session MOS < 3.5 | recommend resolution downgrade |

## Data Models
- ConsultationCreate/Response: consultation_type, platform, patient_consent_obtained, e911_disclosure_provided, status
- TeleSessionCreate/Response: consultation_id, join_url, started_at, ended_at, duration_seconds
- RemoteMonitoringEnrollmentCreate/Response: device_type, device_id, alert_thresholds, status
- PrescriptionTransmitCreate/Response: drug_schedule, transmission_method, confirmation_number
- TeleBillingCreate/Response: billing_code, place_of_service, diagnosis_codes, status

## New Service Methods (v2)

| Method | Description |
|--------|-------------|
| `enroll_care_program()` | Enrol patient in chronic disease program (diabetes, hypertension, CHF, COPD) |
| `triage_patient()` | ESI 1–5 acuity scoring with NATS escalation for critical cases |
| `send_async_message()` | Encrypted async message (text, image, voice note, document) |
| `record_session_qoe()` | Record MOS-based video quality metrics, recommend downgrade |
| `check_jurisdiction_compliance()` | Validate provider licence for patient jurisdiction |
| `request_prescription_renewal()` | Renewal workflow with schedule-based routing and DEA enforcement |
| `provider_performance_report()` | Provider KPI report (completion rate, prescribing, billing) |
| `ai_diagnosis_assist()` | Symptom-based differential diagnosis via Ollama clinical LLM |

## Edge Cases Handled
- Patient consent and E-911 disclosure are independent hard denies at session creation
- Schedule II prescriptions are blocked regardless of transmission method if in_person_visit_completed=False
- Schedule II renewals are always denied; Schedule III–V require teleconsult within 30 days
- Cancelled consultation sessions cannot be started; rebooking required
- Remote monitoring enrollment requires explicit alert_threshold_configured=True flag
- Session complete computes duration from started_at (or created_at as fallback) to ended_at
- ESI-1 triage emits NATS `tel.triage.critical` immediately regardless of appointment queue
- QoE MOS < 3.5 triggers automatic resolution downgrade recommendation and audit event

## Composability Notes
Consultation notes flow to `healthcare_emr` as progress notes. Prescriptions transmit to `healthcare_pha` for formulary validation. Monitoring alerts feed into `healthcare_cli` CDS alerts. Billing records integrate with `healthcare_pmt` billing subsystem. Triage events compose with `healthcare_ed` for emergency department handoff. Care program milestones integrate with `healthcare_qual` for HEDIS quality scoring.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. AI-Powered Real-Time Clinical Decision Support During Video Consultations** [Clinical Intelligence]
- **I2. Continuous Vital-Signs Inference from Video Feed (rPPG)** [Remote Monitoring]
- **I3. NATS JetStream-Backed Real-Time Vital Alert Pipeline** [Streaming Architecture]
- **I4. Structured SOAP Note Auto-Generation from Transcript** [Clinical Documentation]
- **I5. Cross-State Telehealth Licensure Compliance Engine** [Regulatory Compliance]
- **I6. Predictive No-Show & Cancellation Model** [Operational Intelligence]
- **I7. Federated Patient Data Aggregation via FHIR R4** [Interoperability]
- **I8. End-to-End Encrypted Asynchronous Messaging (Store-and-Forward)** [Async Communication]
- **I9. Automated Insurance Pre-Authorization Workflow** [Revenue Cycle]
- **I10. AI-Driven Patient Triage & Acuity Scoring at Intake** [Clinical Triage]
- **I11. Multi-Language Real-Time Translation for Consultations** [Accessibility & Reach]
- **I12. Smart Device Integration Hub (IoT Medical Devices)** [Remote Monitoring]
- **I13. Longitudinal Chronic Disease Management Programs** [Care Management]
- **I14. Session Quality of Experience (QoE) Analytics** [Platform Reliability]
- **I15. Blockchain-Anchored Prescription Audit Trail** [Compliance & Anti-Fraud]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
