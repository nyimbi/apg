# Spa & Activities Management — World-Class Improvements

Fifteen improvements that push `hos_spa` from feature-complete to market-leading.

---

### I1. Dynamic Pricing Engine with Demand-Based Surge and Off-Peak Discounts
**Category**: AI/ML
**Justification**: Static treatment prices leave 20–35 % revenue on the table. Real-time demand signals let the spa charge peak-hour premiums while filling slack capacity at off-peak discounts — exactly how airlines extract full yield from the same seat inventory. Competitors that do this report 18 % REVPAB uplift.
**Implementation**: Maintain hourly booking-rate counters per treatment per day-of-week; at booking time compute a multiplier (`1.0–1.4` for busy slots, `0.75–1.0` for sparse) and store `dynamic_price` alongside the base price so margin analysis stays clean.
**Competitive reference**: Mindbody platform (wellness scheduling SaaS)

---

### I2. AI-Powered Personalised Treatment Recommendations
**Category**: AI/ML
**Justification**: Upsell conversion at checkout is the single highest-ROI moment in a spa visit. Guests who receive a recommendation matched to their history convert at 3× the rate of generic upsell prompts.
**Implementation**: After each appointment completion, compute a cosine-similarity score over `(treatment_category, duration_mins, guest_history)` using a lightweight in-process vector approach, then surface top-3 recommendations via `recommend_next_treatments(guest_email)`.
**Competitive reference**: Zenoti (spa & salon enterprise platform)

---

### I3. Waitlist Management with Automatic Slot Backfilling
**Category**: Feature
**Justification**: Cancelled appointments represent pure lost revenue; a waitlist converts ~60 % of cancellations into rebookings within 30 minutes according to spa operator benchmarks.
**Implementation**: `add_to_waitlist` stores guest + treatment + date preference; `process_waitlist(appointment_id)` is called on every cancellation and auto-creates a new appointment for the highest-priority waitlist entry, emitting a notification event.
**Competitive reference**: Booker by Mindbody

---

### I4. Package & Bundle Management (Multi-Treatment Packages)
**Category**: Feature
**Justification**: Pre-paid packages increase average spend 40–60 % and lock in future visits, reducing churn. They are the primary revenue driver for destination spas.
**Implementation**: `create_package` links a list of treatment IDs with a bundled price and validity window; `redeem_package_treatment` decrements the remaining count and records which treatment was consumed against which package instance.
**Competitive reference**: Vagaro (wellness business management)

---

### I5. Real-Time Room / Treatment Bay Scheduling
**Category**: Feature
**Justification**: Therapist availability is necessary but not sufficient — room conflicts cause silent double-bookings that damage guest experience. Tracking rooms as a first-class resource reduces service failures to near zero.
**Implementation**: Introduce `treatment_rooms` dict; `create_appointment` gains a `room_id` field; conflict detection extends to room-level overlap in addition to therapist-level, returning `room_not_available` when both constraints cannot be satisfied simultaneously.
**Competitive reference**: Spasoft by Springer-Miller Systems

---

### I6. Gratuity & Commission Tracking per Therapist
**Category**: Compliance
**Justification**: Most jurisdictions require accurate tip records for payroll tax purposes; therapists demand commission transparency to trust the employer. Missing this forces manual spreadsheet reconciliation every pay period.
**Implementation**: `complete_appointment` accepts `gratuity_amount: Decimal`; a `therapist_commissions` ledger accumulates `(therapist_id, date, appointment_revenue, commission_pct, gratuity)` entries; `get_therapist_earnings(therapist_id, period_start, period_end)` aggregates and returns a payroll-ready breakdown.
**Competitive reference**: Zenoti Payroll & Commission module

---

### I7. GDPR / POPIA Consent and Guest Data Erasure
**Category**: Compliance
**Justification**: Hotels in Kenya (POPIA), EU (GDPR), and UK (UK-GDPR) face fines up to 4 % of global turnover for retaining guest personal data without consent. Automated right-to-erasure protects the property from regulatory exposure.
**Implementation**: `record_guest_consent(guest_email, consent_scope, expiry_date)` stores a consent ledger entry; `erase_guest_data(guest_email)` replaces all PII fields with `[redacted]` across appointments, memberships, and retail sales while preserving aggregate revenue figures for financial reporting.
**Competitive reference**: Mews Systems (PMS with GDPR toolkit)

---

### I8. Integration Bridge to PMS Reservation Folios
**Category**: Integration
**Justification**: Guests overwhelmingly prefer a single checkout. Without folio posting, spa revenue leaks through separate cash transactions that miss city-ledger and corporate billing integration, and night audit closes with unmatched charges.
**Implementation**: `post_to_folio(appointment_id, pms_reservation_id)` emits a structured `folio_post_requested` CloudEvent with charge code, amount, and VAT breakdown; the `hos_pms` capability consumes this event and posts the line item, returning a `folio_reference` stored on the appointment.
**Competitive reference**: Oracle OPERA Cloud Spa Interface

---

### I9. Multi-Currency Pricing with Base-Currency Settlement
**Category**: Feature
**Justification**: Resort spas in tourist destinations serve guests paying in 10+ currencies. Displaying prices in guest currency eliminates friction and reduces no-shows caused by sticker shock at checkout.
**Implementation**: Store treatment `base_currency` and `base_price: Decimal`; `create_appointment` accepts optional `display_currency` and `exchange_rate: Decimal`; returned records carry both `display_price` and `settlement_price` in base currency; exchange-rate snapshots are stored for audit.
**Competitive reference**: Agilysys Spa (global hotel tech)

---

### I10. Inventory Consumption Tracking per Treatment
**Category**: Feature
**Justification**: Spa consumables (oils, linens, single-use items) represent 15–25 % of treatment cost of goods. Without consumption tracking, reorder points are guesswork and stockouts disrupt service delivery mid-shift.
**Implementation**: `link_treatment_consumables(treatment_id, consumable_list)` records a bill-of-materials; `complete_appointment` auto-decrements inventory quantities for each consumable in the BoM, and `low_stock_alerts()` returns items below their reorder threshold.
**Competitive reference**: Vagaro Inventory module

---

### I11. Therapist Certification and Licence Expiry Monitoring
**Category**: Compliance
**Justification**: Operating with an expired therapist licence exposes the hotel to liability and regulatory shutdown. Automated expiry alerts prevent compliance failures that would otherwise only surface during an inspection.
**Implementation**: `update_therapist_certifications(therapist_id, certifications)` stores a list of `{name, authority, issued_date, expiry_date}` dicts; `list_expiring_certifications(days_ahead)` returns any certification expiring within the horizon, and `create_appointment` warns (but does not block) if the assigned therapist has an expired cert.
**Competitive reference**: Zenoti Compliance Management

---

### I12. Guest Preference Profiles with Allergy and Contraindication Flags
**Category**: UX / Safety
**Justification**: Applying a nut-oil product to a guest with a nut allergy is a medical emergency and a legal liability. Centralised preference profiles surface safety flags at booking and at session start, reducing adverse events to near zero.
**Implementation**: `upsert_guest_profile(guest_email, preferences, allergies, contraindications)` stores a profile record; `create_appointment` injects a `safety_flags` list into the returned appointment record by cross-referencing treatment ingredients against the guest's allergy list.
**Competitive reference**: Book4Time (enterprise spa platform)

---

### I13. Automated No-Show and Late-Cancellation Fee Application
**Category**: Performance
**Justification**: Industry average no-show rate is 8–12 %; late cancellations add another 6–9 %. Without a penalty policy, spas absorb 100 % of that dead time as pure lost revenue.
**Implementation**: `apply_no_show_fee(appointment_id)` checks whether the appointment was cancelled within the configured cancellation window or simply not attended; if so it records a `penalty_charge` record with amount equal to `cancellation_fee_pct * price` and emits a `charge_guest_requested` event.
**Competitive reference**: Mindbody late-cancel/no-show policy engine

---

### I14. Offline-Capable Appointment Queue with Sync-on-Reconnect
**Category**: Performance
**Justification**: Spa receptions in basement or poolside locations frequently lose connectivity. An offline queue prevents lost bookings and therapist idle time during network outages — a common pain point in resort properties.
**Implementation**: `queue_appointment_offline(payload)` appends to an `_offline_queue` list and returns a provisional booking reference; `sync_offline_queue()` replays queued payloads through normal `create_appointment` logic, reconciling conflicts deterministically (first-created wins) and returning a sync report.
**Competitive reference**: Vagaro offline mode

---

### I15. Predictive Staffing Recommendations Using Historical Demand
**Category**: AI/ML
**Justification**: Over-staffing wastes payroll; under-staffing generates negative reviews. Predictive staffing cuts scheduling time from hours to minutes and reduces labour cost variance by 12–20 %.
**Implementation**: `predict_staffing_needs(target_date)` aggregates same-weekday appointment counts from the past 8 weeks, computes mean ± 1 std-dev, divides by average treatments-per-therapist-per-day, and returns a `{recommended_therapists, confidence_interval, historical_basis_days}` dict — no external ML library required.
**Competitive reference**: Zenoti Analytics & Workforce Management
