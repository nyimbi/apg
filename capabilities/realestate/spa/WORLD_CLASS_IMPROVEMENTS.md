# Space Planning & Management — World-Class Improvements

**Capability**: `realestate_spa` | **Date**: 2026-06-11 | **Author**: Nyimbi Odero

---

## 1. Real-Time Occupancy Stream via WebSocket / SSE

Replace the current pull-based `calculate_occupancy_metrics` with a push channel. Sensor gateways emit `OccupancyDataCreate` events to a message broker (MQTT → Redis Streams); a new `stream_occupancy` async generator yields live floor-level counts to clients over SSE or WebSocket. Benefits: sub-second awareness for room booking screens, eliminates repeated polling, enables emergency evacuation head-counts.

---

## 2. CAD/BIM Layer Extraction Engine

The current `upload_floor_plan` stores files as opaque references. Add a `parse_floor_plan` method that calls a locally-hosted microservice (LibreCAD headless / IfcOpenShell) to extract room polygons, wall centrelines, door locations, and area calculations automatically. Output feeds directly into `SpaceCreate` records, eliminating manual space registration. Supports DWG, DXF, IFC, and Revit formats already declared in the model.

---

## 3. ML-Driven Space Utilisation Forecasting

Add `forecast_utilisation(space_id, horizon_days)` using a locally-hosted time-series model (Chronos or Prophet via Ollama-compatible GGUF). Train on `occupancy_data` readings per space. Output: predicted occupancy curve with confidence intervals for the next N days. Feeds scenario planning and proactive booking nudges. No cloud model dependency — aligns with project AI strategy.

---

## 4. Neighbourhood / Zone Grouping

Introduce a `Zone` model grouping related spaces (e.g., "Sales Floor", "Executive Suite"). `ZoneCreate` carries a polygon boundary in GeoJSON for CAD overlay rendering. `get_zone_analytics` aggregates utilisation, headcount, and density across member spaces. Enables department-led space budgeting and zone-level chargeback without enumerating individual spaces.

---

## 5. Hot-Desk Demand Forecasting & Auto-Release

Add `forecast_hot_desk_demand(property_id, date)` that combines historical booking patterns with calendar events (imported from `schd` capability) to predict tomorrow's desk demand. If predicted demand < available desks × threshold, auto-release reserved desks back to pool. Eliminates "ghost" reservations, increases effective utilisation by 15–25%.

---

## 6. Space Request & Approval Workflow

Add `submit_space_request` / `approve_space_request` / `reject_space_request` forming a structured approval chain that sits above raw allocation. Requests carry business justification, headcount forecast, and duration. Approvers are resolved from the `auth` capability by space budget role. Connects to `wflo` for SLA tracking and `ntfy` for escalation. Replaces informal Slack-based space negotiation.

---

## 7. Digital Twin Synchronisation

Add `sync_digital_twin(building_id)` that exports the current space graph (spaces, allocations, floor plans, sensor readings) as an IFC-annotated JSON payload and pushes it to a twin registry endpoint. Downstream: BMS, energy management, and emergency systems read live occupancy from the twin rather than bespoke integrations. Single source of truth for the built environment.

---

## 8. Energy & Sustainability Correlation

Add `calculate_energy_per_occupant(building_id, period)` that joins occupancy readings with energy meter data (imported via `moni` capability) to compute kWh/person/day by zone. Surfaces wasteful zones — e.g., a server-room corridor consuming 80 kWh/day with zero occupancy. Output feeds ESG dashboards and informs decommissioning decisions. Aligns with RICS Whole Life Carbon Measurement standard.

---

## 9. Conflict-Resolution Engine for Simultaneous Allocation Changes

Current `allocate_space` uses in-memory iteration without optimistic locking. Under concurrent writes (multiple HR systems or a bulk import) races produce double-allocations. Add an async `allocate_space_atomic` using `SELECT … FOR UPDATE SKIP LOCKED` on PostgreSQL row-level locks, with a 3-retry back-off. Eliminates ghost allocations without serialising the entire allocation table.

---

## 10. Lease-Linked Space Expiry & Renewal Alerts

Add `check_allocation_expiries(tenant_id, lookahead_days)` that scans allocations with `end_date` within the window and emits `AllocationExpiryEvent` events. The `ntfy` capability converts these to targeted emails/Slack messages to space managers and department heads. Prevents departments from squatting on spaces after lease expiry, recovering an average of 8% of portfolio area per annum (per CBRE 2024 benchmarks).

---

## 11. Multi-Tenancy Space Sharing (Sub-Let Tracking)

Add `create_sublease_arrangement` / `terminate_sublease_arrangement` to track spaces sub-let between tenants in a multi-tenancy property. Introduces `SubleaseArrangement` model with licensor/licensee tenant IDs, agreed rate, and term. Chargeback engine gains a `sublease_pass_through` mode. Required for co-working operators and large corporate campuses with internal charge-back.

---

## 12. Space Portfolio Benchmarking

Add `benchmark_against_portfolio(tenant_id, metric)` that computes where a building ranks against peer buildings in the same tenant portfolio for metrics like sqm/person, utilisation rate, booking adherence, and chargeback yield. Returns percentile ranks and outlier flags. Gives portfolio directors instant prioritisation signals without bespoke reporting.

---

## 13. Accessibility & Compliance Space Tagging

Extend `SpaceCreate` with `accessibility_features: list[str]` (e.g., `["wheelchair_accessible", "hearing_loop", "quiet_for_neurodivergent"]`). Add `find_accessible_spaces(tenant_id, required_features)` that filters on the intersection. Feeds `list_bookings` with a `requires_accessible_space` flag so bookers are auto-routed to compliant spaces. Supports PSED (Public Sector Equality Duty) audit evidence.

---

## 14. Predictive Maintenance Trigger from Occupancy Spikes

Add `detect_overuse_events(space_id, period)` that identifies periods where `occupant_count > capacity * 1.1` (overcrowding). Emits `OveruseEvent` to `mqeb`; subscribed maintenance workflows schedule deep-cleaning, HVAC filter replacements, and furniture audits. Closes the loop between space management and facilities maintenance without a separate FM integration.

---

## 15. Audit-Grade Immutable Allocation History

Replace the mutable `is_active` flag pattern with an append-only allocation log: each state transition (create, modify, deallocate) writes a new `AllocationHistoryEntry` with actor, timestamp, delta, and previous state hash (SHA-256). `get_allocation_history(allocation_id)` returns the full chain. Satisfies FRC, RICS, and SOX audit requirements for demonstrable space cost attribution, and makes forensic investigation of billing disputes trivial.
