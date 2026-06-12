# Events & Venue Management (hos_evn)

Event booking, venue configuration, catering BEO (Banquet Event Order), AV requirements, billing, and contract management.

Version 2.0 adds 8 world-class improvements: partial-day conflict detection with automatic waitlisting, tamper-evident HMAC contract signing, payment timeline and overdue escalation, post-event NPS capture, tiered cancellation fee engine, AV inventory conflict detection, per-setup-style capacity matrix, and weighted revenue forecasting.

---

## Feature List

### Core
| Feature | Method |
|---------|--------|
| Venue CRUD | `create_venue`, `get_venue`, `update_venue`, `delete_venue`, `list_venues` |
| Availability check (date + time-slot) | `create_event_booking` (built-in) |
| Per-setup-style capacity matrix | `set_venue_capacity_matrix`, `get_effective_capacity` |
| Event booking lifecycle | `create_event_booking`, `confirm_event_booking`, `update_event_booking`, `delete_event_booking` |
| Waitlist management | `get_waitlist` (auto-populated on conflict) |
| Banquet Event Order (BEO) | `generate_beo`, `get_beo`, `list_beos`, `finalise_beo` |
| Allergen/dietary validation on BEO | `generate_beo` (enforced) |
| Contract issuance & signing | `issue_contract`, `sign_contract`, `list_contracts` |
| HMAC tamper-evidence verification | `verify_contract_signature` |
| Payment recording | `record_event_payment`, `list_event_payments` |
| Payment timeline & overdue reminders | `generate_payment_timeline`, `get_overdue_reminders` |
| AV requirement capture | `set_av_requirements` |
| AV inventory & conflict detection | `register_av_asset`, `check_av_availability` |
| Tiered cancellation fee engine | `compute_cancellation_fee` |
| Post-event NPS & satisfaction | `record_event_nps`, `nps_summary` |
| Revenue forecast (contracted vs pipeline) | `revenue_forecast` |
| Venue utilisation report | `venue_utilisation_report` |
| Dashboard summary | `dashboard_summary` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/evn/health | Health check |
| GET | /api/hospitality/evn/venues | List venues |
| POST | /api/hospitality/evn/venues | Create venue |
| GET | /api/hospitality/evn/venues/{id} | Get venue |
| PUT | /api/hospitality/evn/venues/{id} | Update venue |
| DELETE | /api/hospitality/evn/venues/{id} | Deactivate venue |
| POST | /api/hospitality/evn/venues/{id}/capacity-matrix | Set per-style capacity matrix |
| GET | /api/hospitality/evn/venues/{id}/effective-capacity | Get effective capacity for a setup style |
| GET | /api/hospitality/evn/event-bookings | List event bookings |
| POST | /api/hospitality/evn/event-bookings | Create booking |
| GET | /api/hospitality/evn/event-bookings/{id} | Get booking |
| PUT | /api/hospitality/evn/event-bookings/{id} | Update booking |
| POST | /api/hospitality/evn/event-bookings/{id}/confirm | Confirm booking |
| DELETE | /api/hospitality/evn/event-bookings/{id} | Cancel booking (auto-promotes waitlist) |
| GET | /api/hospitality/evn/waitlist | Get waitlist for venue/date |
| POST | /api/hospitality/evn/beos | Generate BEO |
| GET | /api/hospitality/evn/beos | List BEOs |
| GET | /api/hospitality/evn/beos/{id} | Get BEO |
| POST | /api/hospitality/evn/beos/{id}/finalise | Finalise BEO |
| POST | /api/hospitality/evn/contracts | Issue contract |
| POST | /api/hospitality/evn/contracts/{id}/sign | Sign contract (HMAC hash stored) |
| GET | /api/hospitality/evn/contracts/{id}/verify | Verify contract signature |
| GET | /api/hospitality/evn/contracts | List contracts |
| POST | /api/hospitality/evn/event-bookings/{id}/payments | Record payment |
| GET | /api/hospitality/evn/event-bookings/{id}/payments | List payments |
| POST | /api/hospitality/evn/event-bookings/{id}/payment-timeline | Generate payment milestones |
| GET | /api/hospitality/evn/overdue-reminders | Get overdue payment milestones |
| POST | /api/hospitality/evn/event-bookings/{id}/av | Set AV requirements |
| POST | /api/hospitality/evn/av-assets | Register AV asset |
| GET | /api/hospitality/evn/av-assets/availability | Check AV availability for date |
| GET | /api/hospitality/evn/event-bookings/{id}/cancellation-fee | Compute tiered cancellation fee |
| POST | /api/hospitality/evn/event-bookings/{id}/nps | Record post-event NPS |
| GET | /api/hospitality/evn/nps-summary | NPS summary (filterable) |
| GET | /api/hospitality/evn/revenue-forecast | Revenue forecast (contracted vs pipeline) |
| GET | /api/hospitality/evn/utilisation-report | Venue utilisation report |
| GET | /api/hospitality/evn/dashboard | Dashboard summary |

---

## Quick Usage Examples

### 1. Revenue Forecast (contracted vs pipeline)

```python
svc = EVNService(tenant_id="datacraft")

# After bookings are in place
forecast = await svc.revenue_forecast(months_ahead=6)
# Returns per-month buckets:
# [{'month': '2026-06', 'contracted': '450000.00', 'pipeline': '180000.00',
#   'weighted_total': '477000.00', ...}, ...]
```

Confidence weights: confirmed bookings → 90%, tentative → 40%. Suitable for CFO-level P&L projections.

### 2. Partial-Day Conflict Detection & Automatic Waitlisting

```python
# First booking occupies the main hall 09:00–13:00
await svc.create_event_booking(
    venue_id=hall_id, event_name="Board Meeting",
    client_name="Acme Ltd", client_email="cfo@acme.co.ke",
    event_type="conference", event_date="2026-07-15",
    start_time="09:00", end_time="13:00", expected_attendance=80,
)

# Second booking overlaps → automatically waitlisted, no hard failure
result = await svc.create_event_booking(
    venue_id=hall_id, event_name="Product Launch",
    client_name="TechCo", client_email="events@techco.co.ke",
    event_type="product_launch", event_date="2026-07-15",
    start_time="11:00", end_time="15:00", expected_attendance=120,
)
# result == {'status': 'waitlisted', 'waitlist_entry_id': '...', ...}

# On cancellation of the first booking, the waitlist head is promoted automatically
await svc.delete_event_booking(booking_id, reason="client_request")
```

### 3. HMAC Contract Signing & Verification

```python
contract = await svc.issue_contract(booking_id, deposit_pct=30.0)

signed = await svc.sign_contract(
    contract["id"],
    signed_by="John Kamau",
    signature_ip="41.89.24.10",
    user_agent="Mozilla/5.0",
)
# signed["signature_hash"] is an HMAC-SHA256 hex digest

# Later — verify the document has not been altered
result = await svc.verify_contract_signature(contract["id"])
# {'verified': True, 'reason': 'ok'}
```

---

## Integration Notes

| APG Capability | Integration Point |
|----------------|-------------------|
| **hos_crm** | `event_nps_recorded` audit event feeds client satisfaction scores for lifetime-value modelling |
| **hos_pms / hos_res** | `room_block_requested` event (I14, planned) routes to PMS for guest room block allocation |
| **hos_fnd** | Payment records and Decimal-accurate totals feed into financial ledger reconciliation |
| **hos_inv** | AV asset inventory extends to general equipment tracked in inventory capability |
| **intel_alerts** | Overdue payment milestones (`get_overdue_reminders`) can trigger alert workflows |
| **intel_reporting** | `venue_utilisation_report` and `revenue_forecast` feed business intelligence dashboards |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Dynamic Yield-Based Pricing Engine** [AI/ML]
- **I2. Waitlist & Automatic Conflict-Resolution Queue** [Feature]
- **I3. Partial-Day Time-Slot Conflict Detection** [Reliability]
- **I4. Catering Actuals vs Estimate Variance Tracking** [Performance]
- **I5. Dietary & Allergen Matrix Validation on BEO** [Compliance]
- **I6. Digital Contract Signature with Tamper-Evidence Hash** [Security]
- **I7. Automated Payment Timeline & Overdue Escalation** [UX]
- **I8. Venue Capacity Matrix by Setup Style** [Feature]
- **I9. Post-Event NPS & Satisfaction Score Capture** [UX]
- **I10. Configurable Tiered Cancellation Fee Engine** [Compliance]
- **I11. AV Equipment Inventory & Conflict Detection** [Feature]
- **I12. Revenue Forecast — Contracted vs Pipeline Split** [Performance]
- **I13. Catering Menu Template Library with Per-Head Costing** [Feature]
- **I14. Cross-Capability Guest Room Block Request (PMS Link)** [Integration]
- **I15. Venue Floor Plan Layout Store for Diagramming** [UX]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
