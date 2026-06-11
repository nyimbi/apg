# Hospitality Analytics — World-Class Improvements

15 targeted improvements that elevate `hos_ana` from a KPI store to a full revenue intelligence platform.

---

### I1. Revenue Strategy Displacement Forecasting
**Category**: AI/ML
**Justification**: Knowing *current* RevPAR is table stakes. Knowing which business to displace to maximise total yield — replacing low-rate corporate blocks with transient before a compression event — is what separates IDeaS G3 from basic BI tools. This closes the single largest gap between a data recorder and a revenue manager co-pilot.
**Implementation**: For each future date with constrained inventory, rank open segments by marginal revenue contribution using a linear displacement model (`displaced_value = (best_available_rate - segment_avg_rate) * remaining_capacity`). Surface the top-N displacement recommendations with confidence levels.
**Competitive reference**: IDeaS G3 RMS (displacement optimiser), Duetto GameChanger

---

### I2. Unconstrained Demand Estimation
**Category**: AI/ML
**Justification**: Actual occupancy data is always censored — you can only observe rooms sold, never rooms *refused*. Hoteliers making pricing decisions from constrained actuals permanently underestimate true demand, leading to rate timidity. Competitors like Duetto publish unconstrained demand figures as a first-class metric.
**Implementation**: Apply a truncated-normal deconvolution over historical sell-out dates: for every night at 100% occupancy, estimate unconstrained demand as `observed + (arrivals_on_day_of / average_booking_window_days)`. Expose `unconstrained_demand`, `demand_capture_rate`, and `lost_revenue_estimate` per period.
**Competitive reference**: Duetto GameChanger, OTA Insight Rate Insight

---

### I3. Rate Parity Monitoring & Leakage Detection
**Category**: Compliance
**Justification**: A single OTA undercutting BAR by 5% can shift 15–20% of bookings to the highest-commission channel, destroying net RevPAR. Rate parity monitoring is mandatory under most franchise agreements (Marriott, Hilton, IHG) and the absence of it exposes properties to brand audit failures and financial penalties.
**Implementation**: Accept rate snapshots per channel per date. Compute parity delta `(channel_rate - bar_rate) / bar_rate * 100`. Flag violations where `|delta| > threshold_pct` (default 1%). Persist violations with a severity tier (minor / major / critical based on channel commission spread) and expose `parity_score` (0–100) per period.
**Competitive reference**: OTA Insight Parity, RateGain RateIntelligence, Duetto

---

### I4. Dynamic Forecast with Pickup Velocity
**Category**: Feature
**Justification**: Static 30/60/90-day forecasts are rendered stale within 48 hours in compression markets. Pickup velocity (rooms booked per day at N days out) provides a leading indicator that lets revenue managers update rate fences intraday — functionality central to Rainmaker and Duetto's commercial pitch.
**Implementation**: For each future date, maintain a rolling pickup curve: `pickup_velocity_7d = (current_otb - otb_7_days_ago) / 7`. Compare velocity against the same-date-last-year curve to emit a `pace_signal` enum: `ahead | on_pace | behind | critical`. Include projected final occupancy assuming constant velocity.
**Competitive reference**: Rainmaker guestrev, IDeaS G3, Duetto GameChanger

---

### I5. GOPPAR Cost-Side Decomposition
**Category**: Feature
**Justification**: GOPPAR without cost decomposition is meaningless for asset managers and ownership groups. STR's HOST study and CBRE benchmarks both report departmental expense ratios; any analytics tool competing for hotel asset manager mindshare must match this granularity or be relegated to operational reporting.
**Implementation**: Accept departmental expense records (rooms, F&B, admin, sales, maintenance, management fees). Compute `departmental_profit_margin`, `flow_through_ratio = (GOP_change / RevPAR_change)`, and `expense_per_occupied_room` per department. Flow-through below 40% triggers a `margin_compression_alert`.
**Competitive reference**: STR HOST benchmarking, CBRE Hotels Americas Research, M3 Accounting

---

### I6. Sentiment-Driven Reputation Score (AI Text Analysis)
**Category**: AI/ML
**Justification**: Raw NPS is a lagging indicator. Extracting topic-level sentiment (cleanliness, staff, value, location) from free-text comments with local NLP produces actionable department-level scores 4–6 weeks earlier than composite NPS trends — matching ReviewPro's "Global Review Index" which is used as a performance KPI in Marriott franchise agreements.
**Implementation**: Apply keyword/phrase extraction over survey `comments` field using a rule-based sentiment lexicon (no external API required). Score each topic (-1 to +1), aggregate into a `reputation_index` (0–100). Surface `top_positive_themes` and `top_complaint_themes` with occurrence counts.
**Competitive reference**: ReviewPro Global Review Index, TrustYou Semantic Analysis, Medallia

---

### I7. Length-of-Stay (LOS) Optimisation Reporting
**Category**: Feature
**Justification**: Minimum LOS restrictions are one of the most powerful yield levers available outside pricing, yet most analytics tools report LOS only as a descriptive average. Identifying LOS patterns that create "shoulder night" holes — where a 3-night stay blocks two profitable weekend nights — is a primary use case in IDeaS G3's restriction engine.
**Implementation**: For each date band, aggregate `avg_los`, `median_los`, `los_distribution` histogram (1/2/3/4/5+ nights), and `shoulder_night_waste_pct = rooms_with_single_night_gaps / total_rooms`. Flag date bands where waste exceeds 8% as `restriction_opportunity`.
**Competitive reference**: IDeaS G3 (LOS restriction optimiser), Duetto, SiteMinder Channel Manager

---

### I8. Volatility-Adjusted RevPAR (vRevPAR)
**Category**: Feature
**Justification**: Two properties with identical average RevPAR can have radically different business quality — one driven by consistent corporate demand, the other by volatile event-driven spikes. Asset managers and lenders use volatility-adjusted metrics to assess cash-flow risk; this is explicitly discussed in CBRE and JLL hotel investment reports.
**Implementation**: Compute rolling 30-day standard deviation of daily RevPAR. `v_revpar = avg_revpar * (1 - coefficient_of_variation)` where `cv = std_dev / mean`. Attach `demand_stability_tier` (stable / moderate / volatile) using cv thresholds (cv < 0.15 / 0.15–0.30 / > 0.30). Include in `kpi_period_summary`.
**Competitive reference**: JLL Hotel Investment Outlook, CBRE Hotels Research, STR Chain Scales

---

### I9. Booking Window & Lead-Time Analytics
**Category**: Feature
**Justification**: Understanding booking window compression (rising share of same-week reservations) is critical for OTA bid strategy and rate fence timing. Expedia and Booking.com both surface booking window data to hotel partners; any analytics platform lacking this forces revenue managers to context-switch into OTA extranets.
**Implementation**: Accept `booking_date` and `arrival_date` on pace records. Derive `booking_window_days`. Aggregate `avg_window`, `pct_0_7d`, `pct_8_30d`, `pct_31_90d`, `pct_90d_plus` per period. Emit `window_compression_alert` when 0–7 day share grows >5pp month-over-month.
**Competitive reference**: Expedia Partner Central, Booking.com Pulse, OTA Insight

---

### I10. Geo-Demand Heatmap Data (Origin Market Analytics)
**Category**: Feature
**Justification**: Knowing that 40% of weekend guests originate within 150km (drive market) versus 60% from international origins drives fundamentally different marketing spend allocation. IHG and Marriott both mandate origin-market reporting from managed properties; this data feeds into corporate negotiation segmentation.
**Implementation**: Accept `origin_country`, `origin_region`, `origin_city` on reservation records. Compute `origin_mix` percentages, `avg_adr_by_origin`, `avg_los_by_origin`, and `top_feeder_markets` ranked by room-night contribution. Flag anomalous origin-market shifts vs. trailing-90-day baseline.
**Competitive reference**: Marriott Revenue Management, IHG Merlin system, Sabre Hospitality

---

### I11. Event & Compression Calendar Overlay
**Category**: Integration
**Justification**: RevPAR fluctuations are incomprehensible without event context. Revenue managers manually cross-reference STR reports with local event calendars daily — a workflow that costs 30–45 minutes per property per day. Integrating event overlays directly into pace and KPI views is a core differentiator cited in every IDeaS and Duetto sales deck.
**Implementation**: Maintain an event registry (name, date, category, expected_demand_impact_pct). When computing pace comparisons and KPI summaries, annotate each date with `events[]` and adjust the `pace_signal` baseline by the estimated demand impact. Expose `events_this_period` in the executive dashboard.
**Competitive reference**: IDeaS Event Score, Duetto Event Score, STR Custom Calendars

---

### I12. Multi-Property Portfolio Rollup
**Category**: Feature
**Justification**: Asset managers and management companies operate 10–500 properties; they need portfolio-level RevPAR, blended NPS, and consolidated channel mix in a single query. This is the primary reason hotel groups pay for enterprise licenses of Duetto and IDeaS instead of using property-level tools — and the capability that justifies pricing at 3–5x per-seat cost.
**Implementation**: Accept a `property_ids: list[str]` parameter on `executive_dashboard` and `kpi_period_summary`. Fan out to all matching tenant records, aggregate with weighted averaging (weights = `total_rooms` per property), and return `portfolio_summary` alongside `property_breakdown` array. Guard requires a portfolio-level tenant scope.
**Competitive reference**: Duetto Portfolio Dashboard, IDeaS Enterprise, Marriott MRDW

---

### I13. Anomaly Detection & Automated KPI Alerts
**Category**: AI/ML
**Justification**: Revenue managers check dashboards reactively; proactive anomaly alerts reduce response latency from days to hours. Specifically, an unexpected RevPAR drop of >15% versus pace should trigger an alert within the same business day — capability explicitly marketed by OTA Insight's "Alert" product and Duetto's "Strategy" module.
**Implementation**: After each `record_kpi_snapshot`, compute z-score of `revpar` against the trailing 28-day distribution. If `|z| > 2.5`, append an `AnomalyAlert` to a `kpi_alerts` store with `severity` (warning / critical), `metric`, `observed_value`, `expected_range`, and `suggested_action` text. Expose via `list_kpi_alerts`.
**Competitive reference**: OTA Insight Alert, Duetto Strategy, Rainmaker guestrev Alerts

---

### I14. Carbon & Sustainability KPI Tracking (ESG Reporting)
**Category**: Compliance
**Justification**: ESG reporting is now mandatory for listed hotel REITs (EPRA, GRI index) and required by major corporate travel buyers (banks, tech firms) as a supplier qualification criterion. Accor, Marriott, and IHG all publish annual TCFD-aligned sustainability metrics; properties without tracking tooling are excluded from RFPs.
**Implementation**: Accept `energy_kwh`, `water_litres`, `waste_kg`, `carbon_kg_co2e` per occupied-room-night. Compute `energy_intensity`, `water_intensity`, `carbon_intensity` (consumption / occupied room nights). Compare against AHLA sustainability benchmarks. Emit `esg_score` (0–100) and `vs_benchmark_delta` per period.
**Competitive reference**: Duetto ESG module, Accor Planet 21, IHG Green Engage, Hapi Hotels ESG

---

### I15. Total Revenue Optimisation (TRevPAR) Cross-Department Attribution
**Category**: Feature
**Justification**: Rooms-only RevPAR optimisation is zero-sum in a full-service hotel — discounting rooms to fill F&B covers may maximise total property profit even at lower room rates. Marriott's "Total Hotel Revenue Management" programme and Accor's pricing system both operate on TRevPAR; any tool that ignores non-rooms revenue is structurally incompatible with full-service hotel strategy.
**Implementation**: Accept per-department revenue inputs (F&B, spa, parking, events, resort fees). Compute `trevpar = total_revenue / total_rooms`, `rooms_revenue_share_pct`, and `ancillary_revenue_per_occupied_room` by department. Model the `rooms_discount_crossover_point` where ancillary uplift from incremental occupancy exceeds the rate discount cost.
**Competitive reference**: Marriott Total Hotel Revenue Management, Accor Revenue++, Duetto TotalRevPAR
