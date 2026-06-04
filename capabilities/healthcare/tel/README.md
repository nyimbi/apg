# Telemedicine

## Overview
Full-featured telemedicine capability covering virtual consultation booking, video session management with consent and E-911 disclosure enforcement, remote patient monitoring enrollment, electronic prescription transmission, and telehealth-specific billing code management. Schedule II/III prescription transmission is blocked without a prior in-person visit.

## Capability ID
`healthcare_tel`

## Provides
- virtual_consultation_booking: Book video, audio-only, asynchronous, RPM, and urgent care consultations with platform selection
- video_session_management: Create and manage WebRTC/platform sessions with join URL generation and duration tracking
- remote_patient_monitoring: Enroll patients with devices (glucometer, BP cuff, oximeter, etc.) with alert thresholds
- prescription_transmission: Transmit prescriptions via Surescripts, EPCS, or fax; blocks Schedule II without in-person visit
- telehealth_billing: Create billing records with telehealth-specific CPT codes (99201-99215, G2012, G2252, 99421-99423)
- patient_consent_management: Track informed consent and HIPAA authorization per consultation
- technical_readiness_check: Flag sessions where technical check was not completed
- asynchronous_consultation: Support store-and-forward consultation type

## Requires
- auth: Provider and patient authentication
- audl: Audit trail for all session and prescription events
- mten: Multi-tenant isolation
- conf: Tenant-specific platform and billing configuration
- ntfy: Session reminders and monitoring alerts
- wflo: Consent and prescription approval workflows
- schd: Consultation scheduling integration
- comp: Regulatory compliance for telehealth licensing
- moni: Session quality and platform availability monitoring
- mqeb: Event emission for EMR and billing downstream

## Configuration

| Key | Description |
|-----|-------------|
| consultations.consent_required | Require patient consent before session start |
| sessions.recording_consent_required | Require explicit consent before recording |
| prescriptions.controlled_substance_requires_in_person | Block Schedule II Rx without prior in-person visit |
| billing.place_of_service_code | Default POS code for telehealth (default: 02) |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/tel/schedule | List consultations | healthcare_tel:schedule |
| POST | /api/healthcare/tel/schedule | Book consultation | healthcare_tel:schedule_write |
| GET | /api/healthcare/tel/schedule/<id> | Consultation detail | healthcare_tel:schedule |
| POST | /api/healthcare/tel/schedule/<id>/cancel | Cancel | healthcare_tel:schedule_write |
| GET | /api/healthcare/tel/sessions | List sessions | healthcare_tel:sessions |
| POST | /api/healthcare/tel/sessions | Create session | healthcare_tel:sessions |
| GET | /api/healthcare/tel/sessions/<id> | Session detail | healthcare_tel:sessions |
| POST | /api/healthcare/tel/sessions/<id>/complete | Complete session | healthcare_tel:sessions |
| GET | /api/healthcare/tel/monitoring | List monitoring | healthcare_tel:monitoring |
| POST | /api/healthcare/tel/monitoring | Enroll device | healthcare_tel:monitoring |
| GET | /api/healthcare/tel/prescriptions | List Rx | healthcare_tel:prescriptions |
| POST | /api/healthcare/tel/prescriptions | Transmit Rx | healthcare_tel:prescriptions |
| GET | /api/healthcare/tel/billing | Billing records | healthcare_tel:billing |
| POST | /api/healthcare/tel/billing | Create billing record | healthcare_tel:billing |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| patient_consent_required | operation=start_session, patient_consent_obtained=False | deny |
| e911_disclosure_required | operation=start_session, e911_disclosure_provided=False | deny |
| controlled_substance_telemedicine_restriction | drug_schedule=schedule_ii, in_person_visit_completed=False | deny |
| recording_requires_consent | operation=start_recording, recording_consent_obtained=False | deny |
| cancelled_consultation_not_startable | operation=start_session, consultation_status=cancelled | deny |
| monitoring_alert_threshold_required | operation=enroll_monitoring_device, alert_threshold_configured=False | deny |
| billing_code_supported | operation=create_billing_record, billing_code_supported=False | deny |
| technical_readiness_check_required | operation=start_session, technical_check_completed=False | warn |

## Data Models
- ConsultationCreate/Response: consultation_type, platform, patient_consent_obtained, e911_disclosure_provided, status
- TeleSessionCreate/Response: consultation_id, join_url, started_at, ended_at, duration_seconds
- RemoteMonitoringEnrollmentCreate/Response: device_type, device_id, alert_thresholds, status
- PrescriptionTransmitCreate/Response: drug_schedule, transmission_method, confirmation_number
- TeleBillingCreate/Response: billing_code, place_of_service, diagnosis_codes, status

## Streaming Events
- consultation_booked, consultation_cancelled
- session_started, session_completed, session_failed
- monitoring_alert_triggered
- prescription_transmitted
- consent_obtained, billing_record_created

## Edge Cases Handled
- Patient consent and E-911 disclosure are independent hard denies at session creation
- Schedule II prescriptions are blocked regardless of transmission method if in_person_visit_completed=False
- Cancelled consultation sessions cannot be started; rebooking required
- Remote monitoring enrollment requires explicit alert_threshold_configured=True flag
- Session complete computes duration from started_at (or created_at as fallback) to ended_at

## Composability Notes
Consultation notes flow to `healthcare_emr` as progress notes. Prescriptions transmit to `healthcare_pha` for formulary validation. Monitoring alerts feed into `healthcare_cli` CDS alerts. Billing records integrate with `healthcare_pmt` billing subsystem.
