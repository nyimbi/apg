# PMS World-Class Improvements

Fifteen targeted improvements that lift `hos_pms` from a functional baseline to a revenue-generating, compliance-ready, AI-augmented system on par with — or ahead of — best-in-class cloud PMS vendors.

---

### I1. Dynamic Pricing Engine
**Category**: AI/ML
**Justification**: Static `rate_per_night` leaves 20–40 % RevPAR on the table. A demand-aware rate engine competes directly with Oracle OPERA Cloud's Best Available Rate logic and Cloudbeds' revenue management module.
**Implementation**: Compute a demand score from occupancy rate, lead time, and day-of-week; apply a configurable multiplier band (e.g. 0.8×–1.5×) to the base rate and return the adjusted rate with its derivation reason.
**Competitive reference**: Oracle OPERA Cloud, Cloudbeds Revenue Management

---

### I2. Yield / RevPAR Analytics
**Category**: Performance
**Justification**: Operators need RevPAR, ADR, and TRevPAR in one call; today they must compute manually. Mews PMS surfaces these KPIs on its live dashboard.
**Implementation**: Add `get_revpar_analytics(date_from, date_to)` that aggregates room revenue, divides by total room-nights available, and returns ADR / RevPAR / TRevPAR with period-over-period delta.
**Competitive reference**: Mews Systems, Apaleo

---

### I3. Channel Manager Sync
**Category**: Integration
**Justification**: 60 %+ of bookings arrive via OTAs. Without two-way channel sync, double-bookings are inevitable. SiteMinder and Cloudbeds channel managers update inventory in real time across 400+ OTAs.
**Implementation**: Add `sync_channel_availability(channel_id, room_type, allotment, rate)` that writes to a per-channel allotment register and returns a reconciliation diff; expose a `pull_channel_reservations(channel_id, since)` ingestion method.
**Competitive reference**: SiteMinder, Cloudbeds

---

### I4. Digital Key / Mobile Check-In
**Category**: UX
**Justification**: Guests increasingly expect keyless entry and express check-in via mobile. Hilton's Digital Key program reduced front-desk wait time by 35 %. Hoteliers using self-check-in report 12 % higher guest satisfaction.
**Implementation**: Add `issue_digital_key(reservation_id, device_token)` that generates a time-bounded token (valid check_in→check_out), stores it against the reservation, and returns a deep-link URI; add `validate_digital_key(token)` for door-controller callbacks.
**Competitive reference**: ASSA ABLOY VingCard, Hilton Digital Key

---

### I5. Loyalty Points Integration
**Category**: Feature
**Justification**: Loyalty programmes drive repeat bookings; properties without them lose direct-booking share to OTAs. Marriott Bonvoy attributing 50 % of revenue to loyalty members is the benchmark.
**Implementation**: Add `accrue_loyalty_points(guest_id, reservation_id, spend_amount)` that applies a configurable earn rate (pts per currency unit), persists a `loyalty_balance` on the guest record, and emits a `loyalty_accrual` event; add `redeem_loyalty_points(guest_id, points, reservation_id)` that converts points to a folio credit.
**Competitive reference**: Marriott Bonvoy PMS hooks, WorldHotels

---

### I6. Overbooking Walk Management
**Category**: Feature
**Justification**: Properties routinely overbook by 5–10 % to absorb no-shows. Unmanaged walks are legally and reputationally costly. IHG's PMS tracks walk inventory and auto-selects relocation partners.
**Implementation**: Add `walk_reservation(reservation_id, relocation_property: str, reason: str, covered_costs: Decimal)` that cancels the reservation with a `walked` status, creates a compensation folio credit, logs the walk event, and returns a relocation record.
**Competitive reference**: IHG Concerto PMS

---

### I7. Maintenance Work Orders
**Category**: Feature
**Justification**: Housekeeping tasks today conflate cleaning with engineering work orders. A separate work-order lifecycle (reported → assigned → in_progress → verified → closed) with SLA tracking aligns with what Mews and Clock PMS offer as distinct maintenance modules.
**Implementation**: Add `create_work_order(room_id, category, description, priority, reported_by)` with `assign_work_order`, `start_work_order`, and `close_work_order` verbs; track SLA breach flag when resolution exceeds priority-based thresholds.
**Competitive reference**: Mews Maintenance, Clock PMS Maintenance Module

---

### I8. Revenue Forecasting
**Category**: AI/ML
**Justification**: Operators need 30/60/90-day revenue projections for staffing and procurement. Duetto's GameChanger offers predictive forecasting; a lightweight version embedded in the PMS removes the need for a separate BI tool.
**Implementation**: Add `forecast_revenue(horizon_days: int)` that projects room revenue from confirmed reservations plus a no-show-adjusted probability for tentative ones, returning daily buckets with a confidence interval derived from historical occupancy variance.
**Competitive reference**: Duetto GameChanger, IDeaS G3 RMS

---

### I9. Rate Plan Management
**Category**: Feature
**Justification**: Current reservations store `rate_plan` as a free-text label; there is no rate plan catalogue, restriction management, or minimum-stay enforcement. OPERA Cloud's Rate Manager is the industry gold standard.
**Implementation**: Add `create_rate_plan(code, name, base_rate, min_stay, advance_purchase_days, restrictions)` and `get_applicable_rate(room_id, check_in, check_out, channel)` that evaluates active plans against restriction rules and returns the best rate.
**Competitive reference**: Oracle OPERA Cloud Rate Manager

---

### I10. PCI-DSS Payment Tokenisation
**Category**: Security
**Justification**: Storing raw card data is a PCI-DSS Level 1 violation carrying fines up to $500K. Stripe and Adyen provide tokenisation vaults; the PMS must store only tokens, never PANs.
**Implementation**: Add `tokenise_payment_method(reservation_id, pan_token: str, card_brand: str, expiry_month: int, expiry_year: int, last4: str)` that persists a `payment_tokens` record (never the PAN), and update `post_payment` to accept a token reference rather than a raw amount alone.
**Competitive reference**: Stripe Terminal, Adyen Hospitality

---

### I11. Guest Preference Profiles
**Category**: UX
**Justification**: Returning guests expect their preferences (pillow type, floor preference, dietary needs, newspaper) to be remembered. Accor's ALL loyalty platform surfacing preference data at check-in is a key differentiator.
**Implementation**: Add `upsert_guest_preferences(guest_id, preferences: dict)` that merges new preference key-value pairs into a `preferences` field on the guest record, and add `get_pre_arrival_briefing(reservation_id)` that returns a structured summary of room + guest preferences for front-desk use.
**Competitive reference**: Accor ALL, StayNTouch PMS

---

### I12. Revenue Audit Trail with Fiscal Compliance
**Category**: Compliance
**Justification**: Kenya's VAT Act 2013 and the Tax Procedures Act 2015 require tamper-evident fiscal records. Germany's GoBD and France's NF525 impose even stricter requirements. A fiscal sequence counter prevents after-the-fact manipulation.
**Implementation**: Add `post_fiscal_receipt(reservation_id, receipt_lines: list, tax_rate: Decimal)` that computes VAT, assigns a monotonically incrementing `fiscal_sequence_number`, signs the record with a SHA-256 hash chained to the previous receipt, and stores it in a write-once `fiscal_receipts` dict.
**Competitive reference**: DATEV fiscal integration, FISKALY cloud TSS

---

### I13. Automated Housekeeping Assignment
**Category**: AI/ML
**Justification**: Manual task assignment wastes supervisor time. ALICE Technologies (acquired by Actabl) uses an optimisation algorithm to balance staff workload and room priority. Automating this yields 15 % faster room turnover.
**Implementation**: Add `auto_assign_housekeeping(date: str, staff_roster: list[dict])` that groups pending tasks by floor/section, scores each task (checkout > occupied stayover > vacant inspect), and distributes them across available staff to minimise cross-floor travel.
**Competitive reference**: ALICE by Actabl, Knowcross KNOW Housekeeping

---

### I14. OTA Booking Ingestion (PMS-to-Channel)
**Category**: Integration
**Justification**: Properties using Booking.com, Expedia, and Airbnb need reservations to flow directly into the PMS without manual re-entry. Cloudbeds' direct API integrations eliminate human error and overbooking.
**Implementation**: Add `ingest_ota_reservation(channel: str, payload: dict)` that normalises channel-specific payloads (Booking.com, Expedia schema variants) into the standard PMS reservation format, creates or matches a guest record by email, and returns the created reservation with a `channel_confirmation_number` field.
**Competitive reference**: Cloudbeds, RoomRaccoon

---

### I15. Sustainability / Carbon Tracking
**Category**: Compliance
**Justification**: EU's Corporate Sustainability Reporting Directive (CSRD) and ESG investor mandates require hotels to report per-guest carbon footprint. IHG Green Engage tracks energy per occupied room night. Embedding this in the PMS avoids a separate ESG tool.
**Implementation**: Add `record_resource_consumption(date: str, electricity_kwh: Decimal, water_litres: Decimal, waste_kg: Decimal)` and `get_sustainability_report(date_from: str, date_to: str)` that normalise consumption per occupied room night, compute a CO2e estimate (using configurable grid emission factors), and return a period report.
**Competitive reference**: IHG Green Engage, Sustainable Hospitality Alliance Hotel Footprint Tool
