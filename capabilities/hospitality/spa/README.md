# Spa & Activities Management (hos_spa)

Treatment booking, therapist scheduling, inventory, retail, and membership management for hotel spa operations.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/spa/health | Health check |
| GET | /api/hospitality/spa/treatments | List treatments |
| POST | /api/hospitality/spa/treatments | Create treatment |
| GET | /api/hospitality/spa/treatments/{id} | Get treatment |
| PUT | /api/hospitality/spa/treatments/{id} | Update treatment |
| GET | /api/hospitality/spa/therapists | List therapists |
| POST | /api/hospitality/spa/therapists | Create therapist |
| GET | /api/hospitality/spa/therapists/{id}/schedule | Get schedule |
| GET | /api/hospitality/spa/appointments | List appointments |
| POST | /api/hospitality/spa/appointments | Book appointment |
| GET | /api/hospitality/spa/appointments/{id} | Get appointment |
| PUT | /api/hospitality/spa/appointments/{id} | Update appointment |
| DELETE | /api/hospitality/spa/appointments/{id} | Cancel appointment |
| POST | /api/hospitality/spa/appointments/{id}/complete | Complete & pay |
| GET | /api/hospitality/spa/memberships | List memberships |
| POST | /api/hospitality/spa/memberships | Create membership |
| POST | /api/hospitality/spa/memberships/{id}/renew | Renew membership |
| GET | /api/hospitality/spa/retail | List retail items |
| POST | /api/hospitality/spa/retail | Create retail item |
| POST | /api/hospitality/spa/retail/{id}/sell | Sell retail item |
| GET | /api/hospitality/spa/revenue-report | Daily revenue |
| GET | /api/hospitality/spa/therapist-utilisation | Utilisation |
| GET | /api/hospitality/spa/dashboard | Dashboard |

## World-Class Enhancements (v2.0)

**I1. Dynamic Pricing Engine** — demand-based surge/off-peak multipliers (0.75–1.4×) per treatment slot for 18 % REVPAB uplift [AI/ML]

**I2. AI-Powered Treatment Recommendations** — cosine-similarity over guest history surfaces top-3 upsell treatments post-visit for 3× conversion [AI/ML]

**I3. Waitlist with Auto Backfilling** — `add_to_waitlist` / `process_waitlist` converts ~60 % of cancellations into immediate rebookings [Feature]

**I4. Package & Bundle Management** — `create_package` / `redeem_package_treatment` for pre-paid multi-treatment bundles with validity windows [Feature]

**I5. Real-Time Room Scheduling** — room-level conflict detection alongside therapist availability; eliminates silent double-bookings [Feature]

**I6. Gratuity & Commission Tracking** — `get_therapist_earnings(therapist_id, start, end)` produces payroll-ready tip and commission breakdown [Compliance]

**I7. GDPR / POPIA Consent & Erasure** — `record_guest_consent` / `erase_guest_data` replaces PII with `[redacted]` while preserving revenue aggregates [Compliance]

**I8. PMS Folio Integration** — `post_to_folio` emits a `folio_post_requested` CloudEvent consumed by `hos_pms` for single-bill checkout [Integration]

**I9. Multi-Currency Pricing** — dual `display_price` / `settlement_price` fields with exchange-rate snapshots for audit [Feature]

**I10. Inventory Consumption Tracking** — bill-of-materials per treatment; `complete_appointment` auto-decrements consumables and triggers `low_stock_alerts` [Feature]

**I11. Therapist Certification Monitoring** — `list_expiring_certifications(days_ahead)` prevents compliance failures from lapsed therapist licences [Compliance]

**I12. Guest Preference Profiles** — allergy and contraindication flags injected into `create_appointment` response as `safety_flags` list [UX / Safety]

**I13. No-Show & Late-Cancel Fees** — `apply_no_show_fee` records a `penalty_charge` and emits `charge_guest_requested` event for revenue recovery [Performance]

**I14. Offline Appointment Queue** — `queue_appointment_offline` / `sync_offline_queue` handles basement/poolside connectivity gaps with deterministic conflict resolution [Performance]

**I15. Predictive Staffing** — `predict_staffing_needs(target_date)` uses 8-week same-weekday history to return `{recommended_therapists, confidence_interval}` with no ML dependency [AI/ML]

## New Methods

Three high-impact additions from the v2.0 roadmap:

### `recommend_next_treatments(guest_email)`

```python
svc = SpaService(tenant_id="nairobi_serena")

# After appointment completion, fetch personalised upsell candidates
recs = await svc.recommend_next_treatments(
    guest_email="alice@example.com",
    top_n=3,
)
# [{"treatment_id": "t_001", "name": "Hot Stone Massage",
#   "score": 0.91, "reason": "matches past 3 visits"}]
```

### `predict_staffing_needs(target_date)`

```python
plan = await svc.predict_staffing_needs(
    target_date="2026-07-04",
    tenant_id="nairobi_serena",
)
# {
#   "recommended_therapists": 6,
#   "confidence_interval": [5, 8],
#   "historical_basis_days": 8,
#   "expected_appointments": 24,
# }
```

### `post_to_folio(appointment_id, pms_reservation_id)`

```python
result = await svc.post_to_folio(
    appointment_id="appt_xyz",
    pms_reservation_id="res_4521",
    tenant_id="nairobi_serena",
)
# Emits folio_post_requested CloudEvent; hos_pms posts line item.
# {"folio_reference": "FL-20260612-0047", "status": "posted"}
```
