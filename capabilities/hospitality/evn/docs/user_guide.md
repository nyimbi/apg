# Events & Venue Management (hos_evn) — User Guide

## Overview

The Events & Venue Management capability handles the complete MICE (Meetings, Incentives, Conferences, Events) lifecycle: from initial venue enquiry through contract signature, BEO generation, catering and AV coordination, payment tracking, and post-event NPS capture.

Version 2.0 adds 8 world-class improvements covering partial-day conflict detection, automatic waitlisting, HMAC contract tamper-evidence, payment milestone escalation, post-event NPS, tiered cancellation fees, AV inventory management, and forward revenue forecasting.

---

## Key Concepts

| Term | Definition |
|------|-----------|
| **Venue** | A physical space (ballroom, boardroom, outdoor garden) with capacity per setup style, AV, and catering attributes |
| **Setup Style** | Room configuration (theatre, classroom, cabaret, banquet, U-shape, boardroom) — each has its own seating capacity |
| **Event Booking** | A client request linking a venue to a date/time block with cost estimates computed in real-time |
| **Waitlist** | Automatic queue for overlapping time-slot requests; head promoted on cancellation |
| **BEO** | Banquet Event Order — operational document sent to kitchen, AV, and housekeeping with allergen matrix and setup details |
| **Contract** | Formal event agreement with deposit schedule and tiered cancellation policy; signed with HMAC tamper-evidence |
| **Payment Milestone** | Scheduled payment due dates; overdue milestones surface in escalation reminders |
| **NPS Record** | Post-event Net Promoter Score (0–10) with per-dimension ratings for continuous quality improvement |

---

## Standard Workflow

```
1. Configure venue(s) with capacity matrix + AV inventory
2. Receive enquiry → create event booking (status: tentative)
   - Partial-day conflict? → auto-added to waitlist
3. Issue contract → client signs (HMAC hash generated)
4. Record deposit → booking confirmed
5. Generate BEO (allergen validation enforced)
6. Finalise BEO → distribute to operations
7. Record interim / final payments
8. Event day → update booking status to completed
9. Send NPS survey → record response
```

---

## 1. Venue Setup

### Create a venue

```python
svc = EVNService(tenant_id="my_hotel")

hall = await svc.create_venue(
    name="Grand Ballroom",
    venue_type="ballroom",
    capacity_seated=400,
    capacity_standing=600,
    area_sqm=850.0,
    rental_rate_per_day=180000.0,   # KES — stored as Decimal internally
    av_included=False,
    catering_allowed=True,
    fire_code_capacity=450,          # enforced at booking creation
)
```

### Set per-setup-style capacity matrix

A ballroom may seat 400 in theatre but only 220 in cabaret. Without this, over-selling is a liability risk.

```python
await svc.set_venue_capacity_matrix(hall["id"], {
    "theatre":   400,
    "banquet":   280,
    "cabaret":   220,
    "classroom": 180,
    "u_shape":   80,
    "boardroom": 40,
})
```

### Register AV assets to the inventory pool

```python
await svc.register_av_asset("Christie 4K Projector", "projector", quantity_owned=3)
await svc.register_av_asset("JBL PA System", "pa_system", quantity_owned=2)
await svc.register_av_asset("75-inch LED Display", "display_screen", quantity_owned=6)
```

---

## 2. Event Booking

### Create a booking

Time-slot overlap detection runs automatically. On conflict, the request is waitlisted rather than rejected outright.

```python
booking = await svc.create_event_booking(
    venue_id=hall["id"],
    event_name="Safaricom Annual Conference",
    client_name="Safaricom PLC",
    client_email="events@safaricom.co.ke",
    event_type="conference",
    event_date="2026-09-15",
    start_time="08:00",
    end_time="18:00",
    expected_attendance=320,
    catering_required=True,
    av_required=True,
    setup_style="theatre",
)
# booking["status"] == "tentative"
# booking["total_estimate"] is a Decimal string e.g. "1,464,000.00"
```

If `expected_attendance` exceeds the matrix capacity for the chosen `setup_style`, a `ValueError` is raised. If the time slot overlaps an existing booking, the response is:

```json
{
  "status": "waitlisted",
  "waitlist_entry_id": "abc123",
  "conflicting_booking_id": "xyz789",
  "message": "venue_time_slot_conflict — added to waitlist position 1"
}
```

### Check AV availability before confirming

```python
av_check = await svc.check_av_availability(
    event_date="2026-09-15",
    equipment_requests=[
        {"category": "projector", "quantity": 2},
        {"category": "pa_system", "quantity": 1},
    ],
)
# Returns per-category: available, shortfall, conflicting_booking_ids
```

---

## 3. Contract & Signing

### Issue a contract

```python
contract = await svc.issue_contract(
    event_booking_id=booking["id"],
    deposit_pct=30.0,
    payment_terms="30% on signing, 40% 30 days before, balance on day",
    cancellation_policy="tiered",
)
```

### Sign with HMAC tamper-evidence

```python
signed = await svc.sign_contract(
    contract["id"],
    signed_by="Jane Wanjiku",
    signature_ip="41.89.24.10",
    user_agent="Mozilla/5.0 (Macintosh)",
)
# signed["signature_hash"] — HMAC-SHA256 of canonical contract body
```

### Verify the contract has not been altered

```python
result = await svc.verify_contract_signature(contract["id"])
# {"verified": True, "reason": "ok"}
```

If any field is changed after signing, `verified` will be `False` with reason `"hash_mismatch"`.

---

## 4. Payment Timeline & Overdue Escalation

### Generate a milestone schedule

```python
milestones = await svc.generate_payment_timeline(
    booking["id"],
    instalments=[
        {"due_date": "2026-07-01", "amount": "439200.00", "type": "deposit"},
        {"due_date": "2026-08-15", "amount": "585600.00", "type": "interim"},
        {"due_date": "2026-09-15", "amount": "439200.00", "type": "final"},
    ],
)
```

### Check overdue reminders

Called daily by a cron job or the `intel_alerts` capability:

```python
overdue = await svc.get_overdue_reminders()
# [{'event_booking_id': '...', 'amount': '439200.00', 'days_overdue': 14,
#   'client_name': 'Safaricom PLC', 'client_email': 'events@safaricom.co.ke', ...}]
```

---

## 5. BEO Generation

Allergen and dietary tags are mandatory on every menu line (Kenya Food, Drugs, and Chemical Substances Act / EU FIC 2014).

```python
beo = await svc.generate_beo(
    event_booking_id=booking["id"],
    menu_selections=[
        {
            "course": "breakfast",
            "item_name": "Continental Buffet",
            "quantity": 320,
            "allergens": ["gluten", "dairy", "eggs"],
            "dietary_tags": ["vegetarian_option"],
        },
        {
            "course": "lunch",
            "item_name": "Nyama Choma & Ugali",
            "quantity": 320,
            "allergens": [],
            "dietary_tags": ["gluten_free", "halal"],
        },
    ],
    av_requirements=["2x Christie 4K Projector", "JBL PA System", "4x LED Display"],
    setup_style="theatre",
    special_requirements="One prayer room required for afternoon Dhuhr",
)
# beo["dietary_summary"] == {
#     "allergens_present": ["dairy", "eggs", "gluten"],
#     "dietary_options": ["gluten_free", "halal", "vegetarian_option"]
# }
```

Finalise once approved by the F&B manager:

```python
await svc.finalise_beo(beo["id"], approved_by="Mary Otieno, F&B Manager")
```

---

## 6. Cancellation Fee Computation

Tiered forfeiture applies automatically based on days remaining to event:

| Days to Event | Forfeiture |
|---------------|-----------|
| >= 90 days | 0% |
| 60–89 days | 25% |
| 30–59 days | 50% |
| < 30 days | 100% |

```python
fee = await svc.compute_cancellation_fee(
    booking_id=booking["id"],
    cancellation_date="2026-08-20",  # 26 days before Sep 15 → 50% tier
)
# {
#   "days_to_event": 26,
#   "forfeiture_pct": "50",
#   "fee_amount": "732000.00",
#   "refund_amount": "0.00",
#   ...
# }
```

---

## 7. Post-Event NPS

Capture satisfaction immediately after the event closes:

```python
nps = await svc.record_event_nps(
    booking_id=booking["id"],
    nps_score=9,
    dimension_scores={"venue": 10, "catering": 9, "av": 8, "service": 9, "value_for_money": 8},
    comment="Excellent setup. AV team was professional and responsive.",
)
# nps["nps_category"] == "promoter"
```

Aggregate NPS across all events or filter by venue:

```python
summary = await svc.nps_summary(venue_id=hall["id"], date_from="2026-01-01")
# {
#   "total_responses": 24, "promoters": 18, "passives": 4, "detractors": 2,
#   "nps": "66.67", "avg_score": "8.50"
# }
```

NPS records emit `event_nps_recorded` audit events consumed by `hos_crm` for client lifetime-value scoring.

---

## 8. Revenue Forecast

```python
forecast = await svc.revenue_forecast(months_ahead=6)
# [
#   {"month": "2026-06", "contracted": "450000.00", "pipeline": "300000.00",
#    "weighted_total": "525000.00"},
#   {"month": "2026-07", "contracted": "720000.00", "pipeline": "180000.00",
#    "weighted_total": "720000.00"},
#   ...
# ]
```

- **contracted**: total of confirmed bookings in that month × 90% confidence
- **pipeline**: total of tentative bookings × 40% confidence
- **weighted_total**: blended forward revenue for P&L projection

---

## Integration with Other APG Capabilities

| Capability | Event / Data Flow |
|-----------|-------------------|
| **hos_crm** | `event_nps_recorded` → client satisfaction history for re-booking cadences |
| **hos_fnd** | Payment records and Decimal totals → financial ledger reconciliation |
| **hos_inv** | AV asset pool can reference equipment tracked in inventory |
| **intel_alerts** | `get_overdue_reminders()` results can trigger configurable alert rules |
| **intel_reporting** | `venue_utilisation_report` + `revenue_forecast` → BI dashboards |
| **hos_pms / hos_res** | `room_block_requested` events (planned I14) → PMS room block allocation |

---

## Coding Standards Quick Reference

- All monetary values stored as `str(Decimal)` — never `float`
- Convert with `_dec(value)` before arithmetic; round with `_round2(value)`
- All methods are `async def`; no `await` required for in-memory operations but the signature is correct for database-backed extensions
- Use tabs not spaces
- `guard_tenant_id` / `guard_non_empty_string` from `capabilities.common.reliability` at method entry for external-facing endpoints
- No bare `except:` — always catch specific exception types

---

## Error Reference

| Error | Meaning |
|-------|---------|
| `venue_not_found:{id}` | Venue does not exist or belongs to different tenant |
| `event_booking_not_found:{id}` | Booking not found or tenant mismatch |
| `attendance_exceeds_venue_capacity:{n}>{cap}` | Reduce attendance or choose different setup style |
| `attendance_exceeds_fire_code_capacity:{n}>{cap}` | Hard block — fire code violation |
| `cannot_modify_closed_event_booking` | Booking is completed or cancelled |
| `venue_has_upcoming_bookings` | Cannot deactivate venue with active bookings |
| `nps_score must be 0-10` | Invalid NPS score |
| `menu_item[n] missing 'allergens'` | BEO allergen compliance validation failed |
| `invalid_time_format` | start_time/end_time must be HH:MM format |
