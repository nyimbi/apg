# Patient Management — Troubleshooting Guide

## Common Issues

### `duplicate_patient_detected`
**Cause:** A patient with name + DOB similarity score ≥ 0.85 already exists.  
**Fix:** Search by last name or MRN first. If a legitimate new patient, contact a supervisor to override. If a true duplicate, use the merge workflow.

### `discharge_requires_physician_order`
**Cause:** `physician_order_present` was `false` or omitted.  
**Fix:** Obtain the signed discharge order. Set `physician_order_present: true`.

### `appointment_slot_not_available`
**Cause:** The requested time slot is already booked.  
**Fix:** Query available slots for the provider/date. Set `slot_available: true` only after confirming availability.

### `bed_not_available_for_assignment`
**Cause:** Bed is `occupied`, `cleaning`, `maintenance`, or `blocked`.  
**Fix:** Filter beds by `status=available`. Check the bed board for cleaning beds becoming available.

### `isolation_bed_required`
**Cause:** Patient has `isolation_required: true` but assigned bed has `isolation_capable: false`.  
**Fix:** Query beds with `isolation_capable=true` in the target unit.

### `ward_overflow_risk`
**Cause:** Available beds in the target ward dropped below 5%.  
**Fix:** Activate overflow protocol. Use `POST /admissions/{id}/transfer` to route to a ward with capacity. Check the `effective_available_beds` metric — cleaning beds should turn over within 2 hours.

### `vip_patient_privacy_restriction`
**Cause:** Accessing a VIP patient record without `healthcare_pmt:vip` permission.  
**Fix:** Request VIP access from the system administrator.

### `preauth_expired`
**Cause:** Pre-authorisation is older than 30 days.  
**Fix:** Resubmit via `POST /patients/{id}/preauth` with an updated treatment plan.

### `uninsured_patient_must_have_payment_plan`
**Cause:** Bill finalisation blocked because patient has no insurance and no payment plan.  
**Fix:** Create a payment plan via `POST /payment-plans` before finalising the bill.

### `paediatric_age_limit_exceeded`
**Cause:** Patient age in months exceeds the bed's `max_age_months`.  
**Fix:** Find a bed in the adult ward. The paediatric ward maximum is set per-bed.

### `emergency_bypass_invalid_type`
**Cause:** `emergency_bypass_registration: true` on a non-emergency/trauma admission.  
**Fix:** Only set bypass for `emergency` or `trauma` admission types.

## Performance Issues

**Slow patient search:** In-memory store is O(n). For large datasets, enable PostgreSQL store.

**High memory usage:** In-memory store grows unbounded. Restart the service or migrate to PostgreSQL.

## Logs

All operations log to `pmt.*` logger namespace:
- `pmt.register_patient` — new patient registration
- `pmt.adt` — admission/discharge/transfer events
- `pmt.mrn_generated` — MRN generation
- `pmt.bed_occupancy` — bed status changes
- `pmt.claim` — insurance claim submission
- `pmt.rule_denied` — policy violation (includes rule name)

Set `APG_LOG_LEVEL=DEBUG` for verbose output.

© 2025 Datacraft
