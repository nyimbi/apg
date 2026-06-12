# Revenue Management & Rates (hos_rms)

Dynamic pricing, demand forecasting, rate parity, yield optimisation, and competitor rate monitoring.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/rms/health | Health check |
| GET | /api/hospitality/rms/rate-plans | List rate plans |
| POST | /api/hospitality/rms/rate-plans | Create rate plan |
| GET | /api/hospitality/rms/rate-plans/{id} | Get rate plan |
| PUT | /api/hospitality/rms/rate-plans/{id} | Update rate plan |
| DELETE | /api/hospitality/rms/rate-plans/{id} | Deactivate rate plan |
| GET | /api/hospitality/rms/rate-plans/{id}/effective-rate | Compute effective rate |
| GET | /api/hospitality/rms/forecasts | List demand forecasts |
| POST | /api/hospitality/rms/forecasts | Create forecast |
| GET | /api/hospitality/rms/competitor-rates | List competitor rates |
| POST | /api/hospitality/rms/competitor-rates | Add competitor rate |
| GET | /api/hospitality/rms/parity-alerts | List parity alerts |
| POST | /api/hospitality/rms/parity-alerts/{id}/resolve | Resolve alert |
| POST | /api/hospitality/rms/yield-optimisation | Run yield optimisation |
| GET | /api/hospitality/rms/yield-reports | List yield reports |
| POST | /api/hospitality/rms/seasonal-rules | Create seasonal rule |
| POST | /api/hospitality/rms/revenue-targets | Set revenue target |
| GET | /api/hospitality/rms/parity-report | Rate parity report |
| GET | /api/hospitality/rms/dashboard | Dashboard |

---

## World-Class Enhancements (v2.0)

**I1. ML-Driven Demand Forecasting with Pickup Curves** — Exponential-smoothing pickup model per room-type/DOW cell; updates on each reservation event and triggers rate recommendations on pace delta. [AI/ML]

**I2. Real-Time OTA Price Scraping & Channel Monitoring** — Async per-channel HTTP scraper with rotating user-agents; normalises results to `ChannelRateSnapshot` and feeds `_check_parity` with exponential-backoff retry. [Integration]

**I3. Decimal-Accurate Revenue Accounting** — All monetary fields replaced with `Decimal` (IFRS 15 / PCI-DSS compliance); `_to_decimal()` helper raises `TypeError` on non-numeric input. [Compliance]

**I4. Length-of-Stay (LOS) Pricing Matrix** — `los_matrix` per rate plan maps `(min_nights, max_nights)` tuples to rate multipliers; `get_effective_rate` resolves the correct non-overlapping bracket. [Feature]

**I5. Multi-Channel Distribution Control (CRS/GDS/OTA)** — `channel_restrictions` on rate plans; `push_to_channel` emits channel-specific pricing events; `list_rate_plans` accepts a `channel` filter. [Integration]

**I6. Group & Negotiated Rate Contracts** — `NegotiatedContract` records with volume commitment, contracted rate, blackout dates, and rooms blocked; evaluated before seasonal multipliers in `get_effective_rate`. [Feature]

**I7. RevPAR / ADR / TRevPAR KPI Tracking** — `compute_kpis` ingests rooms-sold, rooms-available, room revenue, and ancillary revenue; returns `revpar`, `adr`, `trevpar`, `occupancy_pct`, `goppar` as `Decimal`. [Feature]

**I8. Dynamic Minimum-Stay Enforcement** — `min_stay_rules` store keyed by date-range and room type; `enforce_min_stay` raises `MinStayViolation` with required minimum on breach. [Feature]

**I9. Overbooking & Walk Management Model** — Poisson-approximation model over `no_show_history`; `compute_overbooking_allotment` recommends allotment keeping walk probability below configurable `max_walk_pct`. [AI/ML]

**I10. Closed-to-Arrival (CTA) & Open-to-Arrival Controls** — `arrival_controls` store; `check_arrival_allowed` raises `ArrivalBlockedError` on CTA dates; active CTA dates surfaced in `dashboard_summary`. [Feature]

**I11. Rate Fencing & Restriction Engine** — `rate_fences` per plan (advance purchase, non-refundable, loyalty tier, geo restriction); `validate_booking_eligibility` returns structured pass/fail with fence ID. [Compliance]

**I12. Event-Driven Demand Spike Detection** — `local_events` store with impact percentage and radius; `get_demand_adjusted_rate` adds weighted `event_premium`; supports Eventbrite/PredictHQ webhook ingestion. [AI/ML]

**I13. Multi-Property Portfolio RevPAR Benchmarking** — `portfolio_benchmark` aggregates `compute_kpis` across a `portfolio_id` group; returns ranked list with `revpar_index` and deviation flags; gated to `portfolio_admin` role. [Feature]

**I14. Automated Rate Recommendation Workflow with Human-in-the-Loop Approval** — `generate_rate_recommendation` creates pending recommendations from yield + forecast signals; `approve_recommendation` / `reject_recommendation` with full audit trail. [UX]

**I15. PMS Two-Way Integration via Event Bus** — `push_rates_to_pms` serialises approved recommendations into `RatePushEvent`; pluggable adapters (`mews`, `opera`, `cloudbeds`); failed pushes retry with exponential backoff. [Integration]

---

## New Methods

### `compute_kpis` — RevPAR/ADR/TRevPAR computation

```python
svc = RMSService(tenant_id="hotel-001")

kpis = await svc.compute_kpis(
    actual_rooms_sold=180,
    total_rooms_available=200,
    total_room_revenue=Decimal("54000.00"),
    total_ancillary_revenue=Decimal("8200.00"),
    period="2026-06",
)
# kpis["revpar"]       → Decimal("270.00")
# kpis["adr"]          → Decimal("300.00")
# kpis["occupancy_pct"] → Decimal("90.00")
# kpis["goppar"]       → Decimal("311.00")
```

### `generate_rate_recommendation` / `approve_recommendation` — Human-in-the-loop pricing

```python
rec = await svc.generate_rate_recommendation(
    rate_plan_id="plan-uuid",
    target_date="2026-07-04",
    recommended_rate=Decimal("285.00"),
    rationale="Pickup pace +22% vs. comp set; event premium applied.",
)

# Revenue manager reviews rec["id"] in dashboard, then:
applied = await svc.approve_recommendation(
    recommendation_id=rec["id"],
    approved_by="rev-mgr@hotel.com",
)
# Emits audit event; writes price_override for the date.
```

### `push_rates_to_pms` — Two-way PMS sync

```python
svc = RMSService(tenant_id="hotel-001")

result = await svc.push_rates_to_pms(
    pms_adapter="opera",           # "mews" | "opera" | "cloudbeds"
    recommendation_ids=["rec-uuid-1", "rec-uuid-2"],
)
# result["pushed"]  → 2
# result["failed"]  → 0
# Failed pushes queued in pms_push_queue with exponential-backoff retry.
```
