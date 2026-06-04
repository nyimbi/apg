# Patient Management — API Reference

**Base URL:** `/api/healthcare/pmt`
**Auth:** `X-Tenant-ID` header required on all requests.

---

## Patients

### List / Search Patients
```
GET /patients?last_name=Odhiambo&mrn=MRNNAIH000001
```
**Response 200:**
```json
{"items": [...], "count": 1}
```

### Register Patient
```
POST /patients
```
**Body:**
```json
{
  "tenant_id": "nairobi_hosp",
  "first_name": "Amina",
  "last_name": "Odhiambo",
  "date_of_birth": "1988-03-15T00:00:00",
  "gender_code": "female",
  "phone": "0712345678",
  "created_by": "reg_001"
}
```
**Response 201:** PatientResponse  
**Response 400:** Validation error  
**Response 403:** Duplicate patient detected / policy violation

### Get Patient
```
GET /patients/{patient_id}
```
**Response 200:** PatientResponse  
**Response 404:** Not found

### Merge Patients
```
POST /patients/{patient_id}/merge
```
**Body:** `{"target_id": "...", "approved_by": "supervisor_id"}`

### NHIF/SHA Eligibility
```
POST /patients/{patient_id}/nhif-eligibility
```
**Body:** `{"membership_number": "K001234567"}`

---

## Triage

### Triage Patient
```
POST /triage
```
**Body:**
```json
{
  "tenant_id": "nairobi_hosp",
  "patient_id": "...",
  "triage_level": "level_2_emergent",
  "chief_complaint": "Chest pain radiating to left arm",
  "vital_signs": {"bp_systolic": 85, "heart_rate": 130, "spo2": 92},
  "pain_score": 8,
  "isolation_required": false,
  "triaged_by": "nurse_001",
  "created_by": "nurse_001"
}
```

---

## Admissions (ADT)

### List Admissions
```
GET /admissions?patient_id=...&status=admitted
```

### Admit Patient
```
POST /admissions
```
**Body:** AdmissionCreate payload  
**Response 201:** AdmissionResponse

### Transfer Patient
```
POST /admissions/{admission_id}/transfer
```
**Body:**
```json
{
  "from_ward": "ED",
  "to_ward": "ICU",
  "transfer_reason": "Deteriorating vitals",
  "transferred_by": "dr_001"
}
```

### Discharge Patient
```
POST /admissions/{admission_id}/discharge
```
**Body:**
```json
{
  "disposition": "home",
  "physician_order_present": true,
  "discharge_type": "planned",
  "condition_on_discharge": "improved"
}
```
**Response 403:** `discharge_requires_physician_order`

---

## Beds

### List Beds
```
GET /beds?unit_id=ICU&status=available
```

### Register Bed
```
POST /beds
```
**Body:** BedCreate payload

### Update Bed Status
```
PUT /beds/{bed_id}/status
```
**Body:** `{"status": "cleaning"}`

### Bed Occupancy Summary
```
GET /beds/summary?unit_id=ICU
```

---

## Appointments

### List Appointments
```
GET /appointments?patient_id=...&provider_id=...&status=scheduled
```

### Schedule Appointment
```
POST /appointments
```
**Body:** AppointmentCreate payload

### Cancel Appointment
```
POST /appointments/{appt_id}/cancel
```
**Body:** `{"reason": "Patient request"}`

### Check In
```
POST /appointments/{appt_id}/check-in
```

### No-Show
```
POST /appointments/{appt_id}/no-show
```

### Appointment Reminder
```
POST /appointments/{appt_id}/reminder
```
**Body:** `{"channel": "SMS"}`

---

## Waiting List

### Add to Waitlist
```
POST /waitlist
```
**Body:** WaitlistCreate payload

### List Waitlist
```
GET /waitlist?unit_id=ICU&status=waiting
```

---

## Insurance

### List Patient Insurance
```
GET /patients/{patient_id}/insurance
```

### Add Insurance
```
POST /insurance
```
**Body:** InsuranceCreate payload

### Pre-Authorisation Request
```
POST /patients/{patient_id}/preauth
```
**Body:**
```json
{
  "insurer_id": "sha",
  "treatment_plan": {
    "diagnosis_codes": ["I21.9"],
    "procedure_codes": ["33512"],
    "admission_type": "urgent",
    "expected_los_days": 5
  },
  "estimated_cost": 250000,
  "requested_by": "dr_001"
}
```

---

## Billing

### Generate Bill
```
POST /admissions/{admission_id}/bill
```

### List Bills
```
GET /bills?patient_id=...
```

### Process Copay
```
POST /admissions/{admission_id}/copay
```
**Body:**
```json
{
  "copay_amount": 500,
  "payment_method": "mobile_money",
  "received_by": "cashier_001"
}
```

### Submit Insurance Claim
```
POST /admissions/{admission_id}/claim
```
**Body:**
```json
{
  "insurer_id": "sha",
  "claim_amount": 45000,
  "diagnosis_codes": ["I21.9"],
  "procedure_codes": ["33512"],
  "submitted_by": "billing_001"
}
```

---

## Deposits

### Record Deposit
```
POST /deposits
```
**Body:** DepositCreate payload

---

## Payment Plans

### Create Payment Plan
```
POST /payment-plans
```
**Body:** PaymentPlanCreate payload

---

## Patient Portal

### Register Portal
```
POST /patients/{patient_id}/portal
```
**Body:** PatientPortalCreate payload

---

## Telemedicine

### Book Telemedicine
```
POST /telemedicine
```
**Body:** TelemedicineBookingCreate payload (requires `consent_obtained: true`)

---

## Reports

### Bed Occupancy Report
```
GET /reports/bed-occupancy?unit_id=ICU
```

### Admission Summary
```
GET /reports/admissions
```

### Billing Collection
```
GET /reports/billing
```

### Waitlist Summary
```
GET /reports/waitlist
```

### Triage Summary
```
GET /reports/triage
```

---

## Dashboard
```
GET /dashboard
```
Returns full operational snapshot: patients, admissions, beds, appointments, billing, triage.

---

## Error Responses

All errors follow:
```json
{"error": "error_code_or_message"}
```

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Validation error |
| 403 | Policy violation (rule engine denied) |
| 404 | Resource not found |
| 500 | Internal server error |

Common 403 error codes:
- `duplicate_patient_detected`
- `discharge_requires_physician_order`
- `appointment_slot_not_available`
- `bed_not_available_for_assignment`
- `isolation_bed_required`
- `vip_patient_privacy_restriction`
- `ward_overflow_risk`
- `uninsured_patient_must_have_payment_plan`

---

## Capability Contract

```
GET /contract
```

Returns the full APG capability contract including rules, configuration, UI routes, and composition metadata.

```
POST /evaluate
```
**Body:** Rule evaluation context  
**Response:** `{"decision": "allow|deny", "actions": [...], "context": {...}}`

© 2025 Datacraft | www.datacraft.co.ke
