# Transport Scheduling — World Class Improvement Proposals

© 2025 Datacraft | capability: `transport_sch`

---

## 1. Persistent Database Backend (PostgreSQL)

Current: all state lives in in-memory dicts — lost on restart.

Replace with async SQLAlchemy 2.x sessions over PostgreSQL. The `database/store.py` scaffold exists; wire it into `TransportSchedulingService.__init__` behind a `store:` adapter. Benefit: survives restarts, supports horizontal scaling, enables native SQL window-function analytics.

---

## 2. Event-Sourced Audit Trail via Bytewax Streaming

All `_audit()` calls append to a list. Replace with a proper Bytewax-backed event sourcer: every state mutation emits a CloudEvent onto `apg.transport.scheduling.lifecycle`, consumed by a Bytewax dataflow that materialises read-side projections (KPIs, conflict counts). This decouples writes from reads and gives full replayability.

---

## 3. Tachograph Integration with Real Regulatory Data

`_MAX_DAILY_DRIVE_HOURS` / `_MAX_WEEKLY_DRIVE_HOURS` are module-level constants. Replace with a `comp` (compliance) adapter that fetches the live applicable regulation (EC 561/2006 baseline + country-specific overrides) per driver's licence jurisdiction. Required for multi-country fleets operating under different HOS regimes.

---

## 4. Real-Time Conflict Detection via WebSocket Push

`schedule_conflict_check` is a pull scan. Add a WebSocket/SSE endpoint that fires conflict alerts to the ops dashboard the moment a shift or vehicle assignment is recorded. Eliminates the need for operators to manually trigger checks before publish.

---

## 5. ML-Backed Demand Forecasting with Ollama

`passenger_load_forecast` applies a fixed compound growth formula. Replace with a locally-hosted Ollama model (e.g. `llama3` or `mistral`) fine-tuned on historical trip telemetry. Feed seasonality vectors, weather signals, and event calendars to produce probabilistic load bands rather than point estimates.

---

## 6. Multi-Objective Schedule Optimisation (NSGA-II / OR-Tools)

`schedule_optimise_ml` uses a greedy overlap heuristic. Integrate Google OR-Tools CP-SAT solver for genuine multi-objective optimisation across cost, CO2, driver wellbeing, and SLA dimensions. Expose Pareto frontier results so planners can make informed trade-off decisions.

---

## 7. Driver Preference & Wellbeing Engine

No driver preference modelling exists. Add a `DriverPreference` model (preferred shift type, max consecutive days, days-off requests) and enforce soft constraints in the shift planner. Track a `driver_wellbeing_score` based on shift spread, rest periods, and overtime exposure.

---

## 8. Charter Dynamic Pricing with Surge Model

`charter_booking` uses static per-km rates. Replace with a dynamic pricing engine: base rate × demand_index × lead_time_factor × vehicle_utilisation_factor. Publish price changes to a `charter_pricing` topic so revenue management and CRM systems can react.

---

## 9. Automated Shift Swap Marketplace

`shift_swap_approve` requires ops to manually orchestrate swaps. Add a `shift_swap_request` flow: requesting driver posts a swap, the service matches willing replacement drivers against HOS constraints, ranks candidates, notifies them, and auto-approves when a match accepts — with human override preserved for privileged routes.

---

## 10. Vehicle Maintenance Blackout Integration

No awareness of maintenance windows. Integrate with `transport_mai` via a `MaintenanceBlockAdapter`: before assigning a vehicle, query the maintenance schedule for blackout periods. Block assignment during planned servicing and propagate downtime disruptions to active schedules automatically.

---

## 11. GTFS / NeTEx Schedule Import/Export

`export_schedule_data` returns a stub. Implement full GTFS Static and NeTEx profile serialisation, enabling interoperability with journey planners (Google Maps, Navitia), regulatory bodies (NTSA), and partner operators. Import from GTFS feeds to bootstrap schedules from existing timetables.

---

## 12. Geo-Aware Route Conflict Detection

Current conflict detection is resource-count-based — it cannot detect two vehicles on the same physical road segment at the same time. Add a geo-conflict checker that intersects route geometries (GeoJSON LineStrings) using PostGIS `ST_Intersects` with time-window overlap, raising `route_conflict` type conflicts.

---

## 13. Role-Based Access Control (RBAC) at Method Level

`_enforce` evaluates rules from the capability contract but has no per-method permission matrix. Add a method-level RBAC decorator (`@requires_permission("transport_sch:shifts_write")`) that checks the actor's roles against a permission store before evaluating policy rules. Eliminates over-privileged service calls from downstream integrators.

---

## 14. Schedule Versioning and Rollback

Schedules are mutable in place — no history of who changed what when. Add `ScheduleVersion` records on every write: snapshot the schedule dict with a `version_number`, `changed_by`, and `changed_at`. Expose `schedule_rollback(schedule_id, version)` to revert to any prior snapshot without touching the audit log.

---

## 15. SLA-Aware Notification Escalation

`send_notification` fires one message and moves on. Build an escalation ladder: if a notification (e.g. shift reminder) goes unacknowledged within N minutes, escalate to the next channel (SMS → call → supervisor alert → incident ticket). Track `notification_ack` events and cancel pending escalations on acknowledgement.

---

*Generated by Claude Code | Datacraft © 2025*
