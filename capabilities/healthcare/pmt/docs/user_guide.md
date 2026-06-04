# Patient Management — User Guide

**Capability:** `healthcare_pmt` | **Version:** 1.0.0 | **Platform:** APG by Datacraft

---

## Overview

Patient Management (PMT) is the operational core of the APG Healthcare suite. It handles the complete patient lifecycle from first registration through discharge, billing, and portal access — across every care setting: inpatient, outpatient, emergency, and telemedicine.

Unlike legacy HIS systems, PMT is designed around real clinical workflows. Every rule is explicit, every edge case is handled, and the system degrades gracefully when external integrations are unavailable.

---

## Getting Started

### Tenant Setup

All operations are scoped to a `tenant_id`. Pass it as:
- HTTP header: `X-Tenant-ID: my_hospital`
- Query parameter: `?tenant_id=my_hospital`
- JSON body field: `"tenant_id": "my_hospital"`

### Quick Start (Standalone)

```bash
pip install apg-healthcare-pmt
apg-healthcare-pmt --port 8080
```

Visit `http://localhost:8080/health` to verify the service is running.

---

## Core Workflows

### 1. Patient Registration

**Path:** `POST /api/healthcare/pmt/patients`

Registers a new patient with automatic MRN generation and probabilistic duplicate detection.

- MRN format: `MRN{TENANT4}{SEQ:06d}` (e.g. `MRNNAIH000001`)
- Duplicate check: name + DOB + national ID — score ≥ 0.85 blocks registration
- Supports VIP flag for privacy-restricted records
- Paediatric patients require a `paediatric_guardian_id`

**Example:**
```json
{
  "tenant_id": "nairobi_hosp",
  "first_name": "Amina",
  "last_name": "Odhiambo",
  "date_of_birth": "1988-03-15T00:00:00",
  "gender_code": "female",
  "phone": "0712345678",
  "email": "amina@example.com",
  "created_by": "reception_001"
}
```

### 2. Triage

Triage uses the ESI (Emergency Severity Index) 5-level scale:

| Level | Name | Response Target |
|-------|------|----------------|
| 1 | Resuscitation | Immediate |
| 2 | Emergent | < 15 min |
| 3 | Urgent | < 30 min |
| 4 | Less Urgent | < 60 min |
| 5 | Non-Urgent | < 120 min |

Vitals trigger an automatic NEWS2-inspired Early Warning Score (EWS). An EWS ≥ 5 escalates to critical and prompts the charge nurse.

### 3. Bed Management

The Bed Board shows real-time status for every bed across all units:

| Status | Meaning |
|--------|---------|
| `available` | Ready for assignment |
| `occupied` | Patient in bed |
| `cleaning` | Post-discharge housekeeping |
| `maintenance` | Out of service |
| `blocked` | Reserved (e.g. isolation prep) |
| `isolation` | Active isolation precautions |

**Ward Overflow Protocol** activates automatically when available beds drop below 5% of unit capacity.

**Isolation Rules:** Infectious/immunocompromised patients must be assigned to `isolation_capable` beds. The system enforces this — you cannot accidentally assign an isolation patient to a standard bed.

**Paediatric Ward Age Limits:** Beds can specify `max_age_months`. The system blocks assignment of patients over the limit.

### 4. Admissions (ADT)

**Admit:** `POST /api/healthcare/pmt/admissions`

**Emergency Bypass:** Emergency and trauma admissions can bypass full registration (`emergency_bypass_registration: true`). A skeleton patient record is created and completed retrospectively.

**Transfer:** `POST /api/healthcare/pmt/admissions/{id}/transfer`
- Source bed set to `cleaning`
- Destination bed assigned automatically (first available in target unit)
- HL7 ADT A02 event recorded

**Discharge:** `POST /api/healthcare/pmt/admissions/{id}/discharge`
- Requires `physician_order_present: true`
- Calculates and stores Length of Stay (LOS)
- Bed released to `cleaning`
- Discharge summary shell created

### 5. Appointments

Appointments support 8 types: new patient, follow-up, annual wellness, urgent, procedure, telehealth, consultation, preventive.

**Reminders** are sent 24 hours before via SMS, email, push, or WhatsApp.

**No-Show Management:** Three or more no-shows flags the patient for care coordinator follow-up.

**Telemedicine bookings** require explicit `consent_obtained: true`.

### 6. Waiting List

The waiting list uses a weighted priority score:

```
score = base_weight + min(wait_hours, 48) + isolation_modifier + paediatric_modifier
```

| Priority | Base | Isolation +5 | Paediatric +3 |
|----------|------|-------------|---------------|
| Emergency | 100 | | |
| Urgent | 70 | | |
| Semi-urgent | 40 | | |
| Routine | 10 | | |

### 7. Insurance & Pre-Authorisation

- Add insurance records per patient (primary + secondary)
- Verify SHA/NHIF eligibility (`POST /patients/{id}/nhif-eligibility`)
- Submit pre-authorisation requests — valid for 30 days
- Pre-auth failure triggers `insurance_preauthorisation_failed` event for workflow escalation

### 8. Billing

**Bill Generation:** `POST /admissions/{id}/bill`
- Aggregates ward charges, professional fees, drugs, lab, imaging, theatre
- 16% VAT applied (Kenya)
- Draft status pending finance review

**Uninsured Patients:** A payment plan must be created before bill finalisation.

**Payment Plans:** Minimum 2 installments. Missed payments tracked separately.

**Copay Processing:** `POST /admissions/{id}/copay`
- Methods: cash, card, mobile money, insurance direct, waiver
- Receipt number generated on every transaction

### 9. Patient Portal

```
POST /patients/{id}/portal   → register
POST /patients/{id}/portal/activate
```

Portal users get MFA-capable login, language preference, and appointment self-booking.

### 10. Telemedicine

Telemedicine bookings integrate with the APG `tel` capability for video session management. A join URL is generated on confirmation.

---

## Edge Cases

### Emergency Admission Bypassing Registration
Set `emergency_bypass_registration: true` on the admission. The system creates a minimal patient record. Complete the record within 4 hours (configurable) or a care coordinator alert fires.

### Insurance Pre-Authorisation Failure
The `insurance_preauthorisation_failed` event triggers the `preauth_failure_workflow`. Options:
1. Resubmit with amended treatment plan
2. Switch to self-pay with payment plan
3. Escalate to clinical manager for medical necessity appeal

### Uninsured Patients — Payment Plan
The rule `uninsured_patient_must_have_payment_plan` blocks bill finalisation unless `payment_plan_eligible: true` and a plan is created. Finance can waive this with a recorded reason.

### Ward Overflow
When available beds < 5% of capacity, the `ward_overflow_risk` rule fires. The system:
1. Activates overflow protocol (configurable per unit)
2. Suggests transfer to adjacent units
3. Alerts bed manager and charge nurse

### Isolation Requirements
Beds with `isolation_capable: false` cannot receive isolation patients. The assignment will be blocked regardless of ward. Use the bed board to find the nearest available isolation bed.

### Paediatric Ward Age Limits
Set `max_age_months` on any paediatric bed. Patients older than the limit cannot be assigned. The system will suggest the nearest adult ward.

### VIP Patient Privacy
VIP patient records are only accessible to users with the `healthcare_pmt:vip` permission. Name, contact, and clinical details are masked for standard users.

---

## Reports

| Report | Endpoint |
|--------|---------|
| Bed occupancy | `GET /reports/bed-occupancy` |
| Admission summary | `GET /reports/admissions` |
| Billing collection | `GET /reports/billing` |
| Waitlist summary | `GET /reports/waitlist` |
| Triage summary | `GET /reports/triage` |
| Dashboard | `GET /dashboard` |

---

## Permissions

| Permission | Scope |
|-----------|-------|
| `healthcare_pmt:view` | Read patient and appointment data |
| `healthcare_pmt:register` | Register and update patients |
| `healthcare_pmt:adt` | Admissions, transfers, discharges |
| `healthcare_pmt:bed_management` | Manage bed inventory and status |
| `healthcare_pmt:billing` | Generate bills, process payments |
| `healthcare_pmt:insurance` | Manage insurance and claims |
| `healthcare_pmt:vip` | Access VIP patient records |
| `healthcare_pmt:admin` | System configuration and reports |

---

## Troubleshooting

**"duplicate_patient_detected"** — A patient with similar name, DOB, or ID already exists. Search by MRN or last name first. If genuinely different, use the merge bypass with supervisor approval.

**"discharge_requires_physician_order"** — The discharge request must include `physician_order_present: true`. Obtain and document the physician order first.

**"ward_overflow_risk"** — Available beds are critically low. Use the bed board to identify cleaning beds (available soon) or initiate a transfer to another ward.

**"isolation_bed_required"** — Filter the bed board by `isolation_capable: true` to find suitable beds.

**"preauth_expired"** — Pre-authorisations expire after 30 days. Resubmit the pre-auth request.

© 2025 Datacraft | www.datacraft.co.ke
