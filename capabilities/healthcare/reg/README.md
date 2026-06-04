# Healthcare Regulatory

## Overview
Regulatory compliance management covering facility and professional licensing with expiry tracking, accreditation management (Joint Commission, DNV, CAP, etc.), incident reporting with sentinel event workflow enforcement, regulatory submission management (CMS IQR/OQR, HIPAA breach, FDA MDR), and corrective action tracking. Sentinel event closure requires a completed root cause analysis reference.

## Capability ID
`healthcare_reg`

## Provides
- facility_licensing_management: Track facility and professional licenses with 90-day expiry alerts
- accreditation_management: Manage accreditation cycles for TJC, DNV, CAP, AABB, and other bodies
- incident_reporting: Report patient safety incidents from near-miss to sentinel events with severity classification
- hipaa_compliance_tracking: Track HIPAA/HITECH compliance records and breach notification workflow
- regulatory_submission_management: Manage CMS, state, DEA, and FDA submission lifecycle
- audit_management: Internal, external, and mock survey audit tracking
- corrective_action_tracking: Open, assign, complete, and verify corrective actions linked to incidents
- compliance_dashboard: Cross-framework compliance status dashboard

## Requires
- auth: Role-based access for quality, compliance, and regulatory staff
- audl: Immutable audit trail for all regulatory records
- mten: Multi-tenant isolation
- conf: Tenant-specific regulatory framework configuration
- ntfy: License expiry alerts and sentinel event notifications
- wflo: Incident investigation and corrective action approval workflows
- comp: Regulatory compliance framework tracking
- moni: Submission deadline monitoring
- mqeb: Event emission for downstream quality analytics

## Configuration

| Key | Description |
|-----|-------------|
| licensing.expiry_warning_days | Days before license expiry to trigger alert (default: 90) |
| incidents.sentinel_event_notification_hours | Hours to notify after sentinel event (default: 72) |
| incidents.root_cause_analysis_required_for_sentinel | Block sentinel close without RCA reference |
| submissions.supported_types | Allowed regulatory report types |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/reg/licenses | List licenses | healthcare_reg:licenses |
| POST | /api/healthcare/reg/licenses | Add license | healthcare_reg:licenses |
| GET | /api/healthcare/reg/licenses/<id> | License detail | healthcare_reg:licenses |
| GET | /api/healthcare/reg/accreditation | List accreditations | healthcare_reg:accreditation |
| POST | /api/healthcare/reg/accreditation | Add accreditation | healthcare_reg:accreditation |
| PUT | /api/healthcare/reg/accreditation/<id>/status | Update status | healthcare_reg:accreditation |
| GET | /api/healthcare/reg/incidents | List incidents | healthcare_reg:incidents |
| POST | /api/healthcare/reg/incidents | Report incident | healthcare_reg:incidents_write |
| GET | /api/healthcare/reg/incidents/<id> | Incident detail | healthcare_reg:incidents |
| POST | /api/healthcare/reg/incidents/<id>/close | Close incident | healthcare_reg:incidents_write |
| GET | /api/healthcare/reg/submissions | List submissions | healthcare_reg:submissions |
| POST | /api/healthcare/reg/submissions | File submission | healthcare_reg:submissions |
| POST | /api/healthcare/reg/submissions/<id>/submit | Submit | healthcare_reg:submissions |
| GET | /api/healthcare/reg/corrective-actions | List CAs | healthcare_reg:corrective_actions |
| POST | /api/healthcare/reg/corrective-actions | Create CA | healthcare_reg:corrective_actions |
| POST | /api/healthcare/reg/corrective-actions/<id>/complete | Complete CA | healthcare_reg:corrective_actions |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| sentinel_event_requires_rca | operation=close_incident, incident_type=sentinel_event, rca_completed=False | deny |
| closed_submission_not_modifiable | operation=update_submission, submission_status=closed | deny |
| sentinel_event_notification_required | incident_type=sentinel_event, notification_sent=False | warn |
| hipaa_breach_requires_notification | incident_type=hipaa_breach, breach_notification_sent=False | warn |
| license_expiry_alert_required | days_to_expiry=90, alert_sent=False | warn |

## Data Models
- LicenseCreate/Response: license_type, license_number, expiry_date, days_to_expiry, renewal_initiated
- AccreditationCreate/Response: accreditation_body, program, award_date, expiry_date, status
- IncidentCreate/Response: incident_type, severity, rca_completed, rca_reference, corrective_actions
- RegulatorySubmissionCreate/Response: report_type, submission_reference, status, decision_at
- CorrectiveActionCreate/Response: source, assigned_to, due_date, status, verified_by

## Streaming Events
- license_added, license_expiring, accreditation_status_changed
- incident_reported, sentinel_event_reported, hipaa_breach_reported
- submission_filed, submission_accepted
- corrective_action_opened, corrective_action_completed

## Edge Cases Handled
- Sentinel event incidents cannot be closed without a non-empty RCA reference — hard deny at service layer
- License days_to_expiry computed at creation time; refresh via list_licenses for current value
- Closed submissions cannot be modified; an amendment submission must be filed
- Serious adverse events from device management feed directly into incident reporting

## Composability Notes
Quality indicators from `healthcare_ana` feed into CMS IQR/OQR submission data. Device adverse events from `healthcare_dev` map to FDA MDR incidents. Controlled substance logs from `healthcare_pha` underpin DEA Schedule II submissions. HIPAA breach incidents trigger the breach notification workflow through `ntfy`.
