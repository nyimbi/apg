# Revenue Management & Rates — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

---

### I1. ML-Driven Demand Forecasting with Pickup Curves
**Category**: AI/ML
**Justification**: Naive occupancy-tier bucketing leaves 8–15% RevPAR on the table. Pickup-curve models incorporate booking pace, lead-time distribution, and external event signals to produce date-specific demand estimates with quantified confidence intervals — precisely how IDeaS G3 and Duetto GameChanger beat legacy systems.
**Implementation**: Implement an exponential-smoothing pickup model that maintains a rolling booking-pace table per room type / day-of-week cell; on each new reservation event the model updates pace, recomputes the forecast, and triggers a rate recommendation if the delta exceeds a configurable threshold.
**Competitive reference**: IDeaS G3 RMS (SAS Institute)

---

### I2. Real-Time OTA Price Scraping & Channel Monitoring
**Category**: Integration
**Justification**: Manual competitor rate entry is stale within hours. Automated channel monitoring keeps parity checks accurate and allows sub-hour reaction windows, matching the capability offered by OTA Insight (now Lighthouse) and RateGain.
**Implementation**: Async HTTP scraper per configured channel URL with rotating user-agents; results are normalised to a canonical `ChannelRateSnapshot` and fed directly into `_check_parity`, with exponential-backoff retry and a configurable scrape interval stored in `channel_monitors`.
**Competitive reference**: OTA Insight / Lighthouse RMS

---

### I3. Decimal-Accurate Revenue Accounting
**Category**: Compliance
**Justification**: `float` arithmetic accumulates rounding errors that fail IFRS 15 and PCI-DSS reporting audits; every hospitality ERP (Opera Cloud, Maestro) stores monetary values as `DECIMAL(12,4)`. A single misrouted cent in a 500-room nightly audit triggers a reconciliation cascade.
**Implementation**: Replace every `float` monetary field with `Decimal`; introduce `_to_decimal(v)` helper that raises `TypeError` on non-numeric input; store `base_rate`, `effective_rate`, `recommended_rate`, `target_revpar`, and `target_adr` as `Decimal` throughout.
**Competitive reference**: Oracle Opera Cloud PMS

---

### I4. Length-of-Stay (LOS) Pricing Matrix
**Category**: Feature
**Justification**: Flat nightly rates fail to capture the value of multi-night stays. LOS restrictions (MinLOS / MaxLOS / CTLOS) increase RevPAR by 4–9% and are table-stakes in GDS-connected properties. Marriott Bonvoy's pricing engine assigns a different ADR to every LOS bracket.
**Implementation**: Add `los_matrix` per rate plan: a dict mapping `(min_nights, max_nights)` tuples to rate multipliers; `get_effective_rate` resolves the correct bracket given `length_of_stay` param; matrix is validated on creation to ensure no overlapping brackets.
**Competitive reference**: Marriott Bonvoy Revenue Strategy Platform

---

### I5. Multi-Channel Distribution Control (CRS/GDS/OTA)
**Category**: Integration
**Justification**: A rate plan that is not channel-mapped is invisible to 60–70% of demand. Full channel attribution — including GDS (Amadeus, Sabre), OTA (Booking.com, Expedia), and direct — is the core differentiator of Duetto GameChanger vs. older RMS products.
**Implementation**: Add `channel_restrictions: list[str]` to rate plans; introduce `channel_distribution` store mapping plan → channel availability windows; `list_rate_plans` accepts a `channel` filter; a new `push_to_channel` method emits a channel-specific pricing event.
**Competitive reference**: Duetto GameChanger

---

### I6. Group & Negotiated Rate Contracts
**Category**: Feature
**Justification**: Corporate and group business comprises 30–45% of hotel revenue but requires contract-based pricing, volume thresholds, and blackout dates that flat-rate plans cannot model. Amadeus Central Reservations treats negotiated rates as first-class entities.
**Implementation**: Add `NegotiatedContract` records with `account_id`, `volume_commitment`, `contracted_rate: Decimal`, `valid_from`, `valid_to`, `blackout_dates: list[str]`, and `rooms_blocked: int`; `get_effective_rate` checks for an active contract before applying seasonal multipliers.
**Competitive reference**: Amadeus CRS / Central Reservations

---

### I7. RevPAR / ADR / TRevPAR KPI Tracking
**Category**: Feature
**Justification**: Yield reports without standardised KPIs are unactionable. RevPAR, ADR, and TRevPAR are the universal language of hospitality finance; STR benchmarking data integrates against these metrics. Without them, the dashboard is a display, not a decision tool.
**Implementation**: New `compute_kpis` method ingests `actual_rooms_sold`, `total_rooms_available`, `total_room_revenue: Decimal`, `total_ancillary_revenue: Decimal` and returns `revpar`, `adr`, `trevpar`, `occupancy_pct`, `goppar` as `Decimal` values with period tagging.
**Competitive reference**: STR Benchmarking / CoStar

---

### I8. Dynamic Minimum-Stay Enforcement
**Category**: Feature
**Justification**: During peak demand, unrestricted single-night bookings displace higher-value multi-night guests. MinLOS enforcement — standard in Mews, Cloudbeds, and every tier-1 PMS — drives a measured 6–11% uplift in high-demand periods.
**Implementation**: Add `min_stay_rules` store with date-range, room-type, and `min_nights` fields; `enforce_min_stay` method checks an incoming booking's `check_in` / `check_out` against active rules and raises `MinStayViolation` with the required minimum.
**Competitive reference**: Mews PMS / Cloudbeds

---

### I9. Overbooking & Walk Management Model
**Category**: AI/ML
**Justification**: Zero-overbooking policy leaves 2–5% of nightly capacity unmonetised due to no-shows and same-day cancellations. Every tier-1 RMS (IDeaS, Duetto) models no-show probability per segment and recommends an overbooking allotment with a controlled walk-risk budget.
**Implementation**: Maintain a `no_show_history` table keyed by `(room_type, day_of_week, lead_time_bucket)`; `compute_overbooking_allotment` uses a Poisson approximation to recommend the allotment that keeps walk probability below a configurable `max_walk_pct` threshold.
**Competitive reference**: IDeaS G3 Overbooking Module

---

### I10. Closed-to-Arrival (CTA) & Open-to-Arrival Controls
**Category**: Feature
**Justification**: Hard inventory controls that block arrivals on specific dates are mandatory for managing sell-through on shoulder days. Hilton Revenue Management uses CTA restrictions to prevent low-value single-night arrivals before high-demand weekends.
**Implementation**: Add `arrival_controls` store with `date`, `room_type`, `control_type` (`CTA` | `OTA`), and `reason`; `check_arrival_allowed` raises `ArrivalBlockedError` for CTA dates; controls are surfaced in `dashboard_summary` as `active_cta_dates`.
**Competitive reference**: Hilton Revenue Management System

---

### I11. Rate Fencing & Restriction Engine
**Category**: Compliance
**Justification**: Undifferentiated rate access exposes corporate-rate arbitrage; proper fencing (advance purchase, non-refundable, loyalty-tier, geographic) is required for GDS participation and IATA compliance. Rate fencing rules are audited during GDS certification.
**Implementation**: Add `rate_fences: list[dict]` to each rate plan specifying fence type (`advance_purchase`, `non_refundable`, `loyalty_tier`, `geo_restriction`) and parameters; `validate_booking_eligibility` checks all active fences and returns a structured pass/fail with fence ID.
**Competitive reference**: GDS / Amadeus Rate Fencing Standards

---

### I12. Event-Driven Demand Spike Detection
**Category**: AI/ML
**Justification**: Local events (conferences, concerts, sports) predictably spike demand by 40–300% but are invisible to occupancy-history models. Automated event ingestion combined with demand adjustment is how Duetto and Cendyn outperform generic forecasters.
**Implementation**: Add `local_events` store with `event_date`, `event_name`, `expected_impact_pct`, `radius_km`, `source`; `get_demand_adjusted_rate` adds a weighted `event_premium` on top of the base seasonal multiplier; events can be ingested from Eventbrite/PredictHQ webhooks.
**Competitive reference**: Duetto GameChanger / Cendyn Guestrev

---

### I13. Multi-Property Portfolio RevPAR Benchmarking
**Category**: Feature
**Justification**: Chains operating >1 property need cross-property comparison to identify outliers and reallocate demand. AccorHotels and IHG both operate portfolio dashboards that rank properties by RevPAR index against a comp set.
**Implementation**: New `portfolio_benchmark` method aggregates `compute_kpis` across all tenants in a `portfolio_id` group; returns a ranked list with `revpar_index` (property RevPAR / portfolio mean RevPAR) and deviation flags; access is gated to a `portfolio_admin` role check.
**Competitive reference**: AccorHotels RMS / IHG One Rewards Revenue Platform

---

### I14. Automated Rate Recommendation Workflow with Human-in-the-Loop Approval
**Category**: UX
**Justification**: Fully automated rate changes with no human review expose the property to runaway pricing (e.g., the Amazon book-pricing incident). Best-in-class RMS (IDeaS, Duetto) generate recommendations that a revenue manager approves or rejects, with an audit trail.
**Implementation**: Add `rate_recommendations` store; `generate_rate_recommendation` creates a pending recommendation from yield and forecast signals; `approve_recommendation` applies the rate as a `price_override`; `reject_recommendation` records the reason; all actions are emitted as audit events.
**Competitive reference**: IDeaS G3 Automated Pricing Workflow

---

### I15. PMS Two-Way Integration via Event Bus
**Category**: Integration
**Justification**: An RMS that cannot push rates back to the PMS in real time is a reporting tool, not a pricing engine. Mews, Opera Cloud, and Cloudbeds all expose webhooks/APIs for rate push; without two-way sync the RMS recommendation loop is broken.
**Implementation**: Add `pms_push_queue: list[dict]` and `push_rates_to_pms` method that serialises approved rate recommendations into a canonical `RatePushEvent` payload; supports pluggable adapters (`mews`, `opera`, `cloudbeds`) selected by `pms_adapter` config; failed pushes are retried with exponential backoff and logged.
**Competitive reference**: Mews PMS / Oracle Opera Cloud
