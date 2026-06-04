# Patient Management

## Overview
Core patient lifecycle management covering registration with MRN generation, ADT (Admit/Discharge/Transfer) workflow enforcement, real-time bed board management, appointment scheduling, and insurance tracking. Enforces physician discharge orders, prevents admission of inactive patients, requires approval for patient merges, and enforces cancellation reason documentation.

## Capability ID
`healthcare_pmt`

## Provides
- patient_registration: Register patients with auto-generated MRN, demographic capture, and duplicate prevention
- adt_workflow: Admit, discharge, and transfer lifecycle with status transitions and disposition tracking
- bed_management: Real-time bed board with available/occupied/cleaning/maintenance status and housekeeping integration
- appointment_scheduling: Book, cancel, and check-in appointments across 8 appointment types with slot validation
- patient_billing: Insurance record management with primary/secondary payer tracking
- mrn_generation: Tenant-prefixed sequential MRN generation (MRN{PREFIX}{SEQ:06d})
- insurance_verification: Store and track insurance coverage with verification status
- patient_search: Search patients by last name or MRN
- visit_management: Track inpatient and outpatient visit history

## Requires
- auth: Patient PHI access authorization
- audl: Audit trail for all ADT and registration events
- mten: Multi-tenant isolation
- conf: Tenant-specific MRN prefix and bed configuration
- ntfy: Appointment reminders and bed status alerts
- wflo: Discharge order and patient merge approval workflows
- schd: Appointment scheduling integration
- mqeb: Event emission for EMR and billing downstream

## Configuration

| Key | Description |
|-----|-------------|
| registration.mrn_prefix | MRN prefix (default: MRN) |
| registration.duplicate_check_enabled | Enable MRN duplicate detection |
| adt.supported_admission_types | Allowed admission types |
| adt.supported_discharge_dispositions | Allowed discharge dispositions |
| appointments.reminder_hours_before | Hours before appointment to send reminders |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/pmt/patients | Search patients | healthcare_pmt:patients |
| POST | /api/healthcare/pmt/patients | Register patient | healthcare_pmt:patients_write |
| GET | /api/healthcare/pmt/patients/<id> | Patient detail | healthcare_pmt:patients |
| POST | /api/healthcare/pmt/patients/<id>/merge | Merge patients | healthcare_pmt:patients_write |
| GET | /api/healthcare/pmt/admissions | List admissions | healthcare_pmt:adt |
| POST | /api/healthcare/pmt/admissions | Admit patient | healthcare_pmt:adt |
| POST | /api/healthcare/pmt/admissions/<id>/discharge | Discharge | healthcare_pmt:adt |
| GET | /api/healthcare/pmt/beds | Bed board | healthcare_pmt:beds |
| POST | /api/healthcare/pmt/beds | Register bed | healthcare_pmt:beds |
| PUT | /api/healthcare/pmt/beds/<id>/status | Update bed status | healthcare_pmt:beds |
| GET | /api/healthcare/pmt/appointments | List appointments | healthcare_pmt:appointments |
| POST | /api/healthcare/pmt/appointments | Schedule appointment | healthcare_pmt:appointments |
| POST | /api/healthcare/pmt/appointments/<id>/cancel | Cancel | healthcare_pmt:appointments |
| POST | /api/healthcare/pmt/appointments/<id>/check-in | Check in | healthcare_pmt:appointments |
| GET | /api/healthcare/pmt/patients/<id>/insurance | Insurance list | healthcare_pmt:billing |
| POST | /api/healthcare/pmt/insurance | Add insurance | healthcare_pmt:billing |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| duplicate_mrn_denied | operation=register_patient, mrn_exists=True | deny |
| admission_type_supported | operation=admit_patient, admission_type_supported=False | deny |
| discharge_requires_physician_order | operation=discharge_patient, physician_order_present=False | deny |
| inactive_patient_adt_denied | operation=admit_patient, patient_status=inactive | deny |
| deceased_patient_modification_denied | operation=update_patient, patient_status=deceased | deny |
| appointment_slot_available | operation=schedule_appointment, slot_available=False | deny |
| appointment_cancel_requires_reason | operation=cancel_appointment, reason_present=False | deny |
| patient_merge_requires_approval | operation=merge_patients, approval_present=False | deny |
| bed_assignment_requires_available_bed | operation=assign_bed, bed_status=occupied | deny |

## Data Models
- PatientCreate/Response: mrn, first_name, last_name, date_of_birth, gender_code, status, merged_into
- AdmissionCreate/Response: admission_type, unit_id, bed_id, admit_time, discharge_time, discharge_disposition
- BedCreate/Response: unit_id, bed_number, bed_type, status, patient_id, admission_id
- AppointmentCreate/Response: appointment_type, scheduled_at, duration_minutes, status, checked_in_at
- InsuranceCreate/Response: insurance_type, payer_name, member_id, effective_date, primary, verification_status

## Streaming Events
- patient_registered, patient_updated, patient_merged
- patient_admitted, patient_discharged, patient_transferred
- bed_status_changed
- appointment_scheduled, appointment_updated
- billing_record_created

## Edge Cases Handled
- Inactive patients cannot be admitted; reactivation required first
- Discharge without physician_order_present=True is hard denied
- Bed is automatically set to "cleaning" on discharge (not immediately "available")
- Patient merge marks source patient as "merged" with merged_into pointer; approver required
- MRN is tenant-prefixed sequential to prevent cross-tenant leakage

## Composability Notes
Patient registration triggers encounter creation in `healthcare_emr`. ADT events feed into `healthcare_ana` for census and LOS analytics. Insurance records flow to billing in `healthcare_pmt` itself and are consumed by `healthcare_reg` for payer reporting.
