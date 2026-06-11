# Dispatch Operations — World-Class Improvements

**Capability**: `transport_dis` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Dynamic Driver Re-Allocation on Live Dispatch Failure

When an active driver becomes unavailable mid-route (breakdown, HOS violation, medical), the system currently has no automated re-allocation path. Add `async reassign_driver_in_flight()` that atomically swaps driver assignments on a live dispatch, notifies the original and replacement driver via preferred channel, creates an audit trail, and triggers an `eta_recalculation` tracking update — all within a single idempotent transaction boundary.

**Impact**: Eliminates dead time when a driver goes offline; estimated +15% on-time delivery rate for fleets >20 vehicles.

---

## 2. Geofence-Triggered Automatic Status Transitions

GPS pings are ingested but status transitions (`at_stop`, `departed_stop`, `completed`) are driven by manual operator input. Implement `async process_geofence_event()` that accepts a geofence entry/exit event, matches it to the nearest stop on the dispatch's load plan, and automatically advances dispatch and stop status. Store geofence polygons per stop in load plan metadata.

**Impact**: Removes operator click-through for every stop completion; reduces ops headcount for large fleets by 30–40%.

---

## 3. Hours-of-Service Predictive Violation Alert

The existing `compliance_hours_check` is reactive — it checks after the fact. Add `async predict_hos_violation()` that projects remaining drive time against the assigned dispatch's estimated duration (from ETA data) and raises a pre-emptive alert when remaining HOS margin falls below a configurable threshold (default 90 min). Integrates with the `ntfy` capability.

**Impact**: Prevents unplanned driver swaps; reduces regulatory exposure for operations in HGV, FMCSA, and EU tachograph regimes.

---

## 4. Multi-Stop ETA Cascade Recalculation

When a delay event occurs at one stop (traffic hold, customs, missed time window), all downstream stop ETAs in the same dispatch need recalculation. Add `async recalculate_stop_etas()` that takes a delay delta in minutes, propagates it forward across all remaining stops using per-stop service-time estimates, and generates a tracking update per stop. Composes with `transport_rou` for live distance data.

**Impact**: Keeps customer notification accuracy above 90% even during in-route disruptions.

---

## 5. Intelligent Dispatch Consolidation (Load Merging)

When two partial-load dispatches share an origin and overlapping route corridor, they can be merged into a single full-truck-load dispatch. Add `async consolidate_dispatches()` that identifies candidate pairs using route proximity scoring, validates combined weight/volume against the target vehicle's capacity, creates a new merged dispatch, cancels the originals with an audit link, and returns the consolidation savings report.

**Impact**: 8–12% reduction in per-unit transport cost through vehicle utilisation improvement.

---

## 6. Driver Performance Scoring

No per-driver performance signal exists beyond compliance flags. Add `async score_driver_performance()` that aggregates: on-time delivery rate, exception rate, stop completion rate, and average speed-limit adherence score (from GPS data) into a weighted composite score (0–100). Store per-driver score history to support driver incentive programs and AI-based assignment optimisation.

**Impact**: Enables data-driven driver tiering; top-quartile drivers assigned to high-value or time-sensitive dispatches.

---

## 7. Cargo Integrity Monitoring Integration

Temperature-controlled and hazmat loads require continuous cargo state monitoring beyond GPS position. Add `async ingest_cargo_sensor_event()` that accepts telematics sensor readings (temperature, humidity, door-open events, impact-g-force), validates against per-load-type thresholds defined in the load plan, auto-raises a `cargo_damage` or `hazmat_spill` exception when a threshold is breached, and sends immediate customer and depot notifications.

**Impact**: Reduces cargo loss claims; essential for pharma, food, and hazmat compliance regimes.

---

## 8. Time-Window Optimisation with Penalty Functions

The nearest-neighbour route optimiser ignores customer time-window constraints. Replace/augment it with a time-window aware insertion heuristic inside `optimise_dispatch()` that respects earliest/latest arrival windows per stop and assigns penalty weights for early arrivals (waiting cost) and late arrivals (SLA breach cost). Expose a `time_window_penalty_config` parameter in the load plan.

**Impact**: SLA compliance rates improve from ~72% (NN-only) to ~91% in benchmark mixed-window scenarios.

---

## 9. Automated Proof-of-Delivery Capture

After a stop is completed, drivers must submit PoD (signature, photo, or barcode scan). Add `async record_proof_of_delivery()` that accepts a PoD payload (type: signature | photo_ref | barcode), links it to the relevant stop on the dispatch, marks the stop as `pod_captured`, and emits a `stop_pod_recorded` event on the dispatch stream. Triggers customer notification with PoD reference.

**Impact**: Eliminates disputed deliveries; integrates directly into AR/billing workflows via the `invoic` capability.

---

## 10. Real-Time Fleet Heatmap Aggregation

Currently hub operations returns point-in-time counts. Add `async fleet_position_snapshot()` that iterates all active dispatches for a tenant, collects last-known GPS positions from tracking updates, and returns a position array suitable for map rendering — with per-vehicle speed, status, and ETA annotations. Supports sub-10-second polling intervals for live ops dashboards.

**Impact**: Replaces manual radio check-ins for dispatchers; single-screen situational awareness for fleets up to 500 vehicles.

---

## 11. SLA Breach Prediction and Pre-Emptive Escalation

Add `async predict_sla_breach()` that scores each active dispatch's probability of missing its committed delivery window given current ETA, remaining stops, and historical exception rate for the route corridor. When breach probability exceeds a threshold (default 0.7), automatically escalates to ops manager via the `ntfy` capability and creates a `time_window_missed` exception in draft state for operator confirmation.

**Impact**: Moves exception management from reactive to predictive; reduces customer-visible SLA breaches by 25–35%.

---

## 12. Carrier Capacity Pooling (Spot Freight Integration)

When no owned vehicle is available for a load plan, integrate with spot freight APIs to source third-party carrier capacity. Add `async request_spot_capacity()` that broadcasts a load tender to registered spot carriers, collects quotes, ranks by cost-time trade-off, and returns the top-N options for operator selection. Accepted quotes create a dispatch with `assignment_type = temp_assignment` and external carrier metadata attached.

**Impact**: Eliminates "no capacity" dispatch failures; critical for peak-season and last-minute load coverage.

---

## 13. Shift-Aware Dispatch Scheduling

Dispatches created outside a driver's scheduled shift window fail silently or require manual override. Add `async schedule_dispatch_for_shift()` that queries the `schd` capability for the assigned driver's next available shift window, validates the dispatch's planned departure against it, and either confirms or reschedules to the earliest valid departure slot — returning a shift-aligned dispatch plan with estimated departure time.

**Impact**: Eliminates shift-boundary dispatch failures; improves driver rest compliance by keeping scheduling in sync with shift data.

---

## 14. Automated Return-Trip Planning (Backhaul Optimisation)

After a dispatch completes, the vehicle is typically empty on the return leg. Add `async plan_backhaul()` that queries pending loads near the vehicle's final stop, scores feasibility (weight, volume, direction alignment), and if a viable backhaul is found, creates a new load plan and dispatch for the return leg — reducing empty-km by pairing return trips with compatible loads.

**Impact**: 10–18% reduction in empty-vehicle kilometres; direct fuel and carbon cost reduction.

---

## 15. Audit Event Streaming with Replay Support

Audit events are stored in-memory as a flat list with no replay capability. Refactor `_audit()` to emit structured CloudEvents-compatible payloads on the `bytewax` stream, keyed by `(tenant_id, dispatch_id)`. Add `async replay_audit_trail()` that reconstructs the full state history of a dispatch from its audit events, enabling forensic analysis, regulatory reporting, and bi-temporal query support (what-was-known-when).

**Impact**: Unlocks regulatory audit trails required for EU tachograph, FMCSA ELD, and ISO 9001 logistics compliance; also enables event-sourced dispatch state reconstruction for resilience.
