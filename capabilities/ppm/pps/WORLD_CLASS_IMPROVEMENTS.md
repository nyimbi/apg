# World-Class Improvements — Project Planning & Scheduling (ppm_pps)

## 1. Monte Carlo Schedule Risk Simulation

Replace the stub PERT mention with a full Monte Carlo engine. Each task carries three-point estimates (optimistic, most-likely, pessimistic). Run N simulations sampling duration from PERT/triangular distributions, compute P50/P80/P90 project-completion dates, and return a probability distribution. This transforms the service from deterministic to probabilistic — the gold standard for high-value project risk management.

## 2. Earned Value Management (EVM) Integration

Track Planned Value (PV), Earned Value (EV), and Actual Cost (AC) per task and roll them up to project level. Derive SPI (Schedule Performance Index), CPI (Cost Performance Index), EAC (Estimate at Completion), and TCPI (To-Complete Performance Index). Surface S-curves. Without EVM, schedule progress is an opinion; with it, it is a measurement.

## 3. PERT Three-Point Estimation Engine

Add `optimistic_days`, `most_likely_days`, and `pessimistic_days` fields to `Task`. Compute PERT expected duration `(O + 4M + P) / 6` and standard deviation `(P - O) / 6` automatically on `add_task`. Propagate uncertainty through the network using variance summation on the critical path so stakeholders see confidence intervals on delivery dates, not false precision.

## 4. Calendar-Aware Date Arithmetic

The current CPM implementation counts calendar days. Replace raw `timedelta` arithmetic with a working-calendar engine that skips weekends, public holidays, and resource-specific non-working days. Integrate with `ProjectCalendar` (already modelled) so ES/EF/LS/LF are expressed in real calendar dates, not floating-point offsets from day 0.

## 5. Resource-Constrained Critical Path (RCCP)

Extend the CPM network pass to account for resource availability. When a resource is over-allocated, tasks compete for it according to a priority rule (minimum slack first, total float, user-assigned priority). The result is a resource-feasible schedule that may extend beyond the time-only critical path — a mandatory input to serious project commitments.

## 6. Persistent PostgreSQL-Backed Store

The in-memory `dict` store is test-only infrastructure. Replace it with an async `asyncpg`/SQLAlchemy 2.0 store behind a repository interface (`AbstractScheduleRepository`). Keep the in-memory implementation for unit tests. This eliminates the single biggest gap between the current service and production-grade scheduling software.

## 7. Earned Schedule (ES) Metrics

Earned Schedule is the more reliable replacement for SPI(t) when projects slip. Compute the time at which the current EV equals the Baseline PV, yielding schedule variance in time units rather than cost units. Couple with IEAC(t) (Independent Estimate at Completion in time) for reliable forecast-to-complete.

## 8. Critical Chain Project Management (CCPM)

Implement buffer management: aggregate feeding buffers at merge points and a project buffer at the end of the critical chain. Track buffer consumption (green/yellow/red fever-chart zones). CCPM out-performs CPM on resource-constrained projects by protecting the chain rather than individual tasks.

## 9. Schedule Variance Trend Analysis

Store daily snapshots of SPI, total float of critical path, and percent complete. Expose a `schedule_variance_trend(project_id, lookback_days)` method that returns a time-series enabling teams to detect deteriorating trajectories before they become missed milestones.

## 10. Automated Schedule Quality Score

Compute a composite Schedule Quality Index: penalise for tasks with no predecessor/successor links (dangling logic), tasks longer than a configurable threshold (usually 2× the reporting period), missing resource assignments, and zero-float tasks that are not on the documented critical path. Return a 0–100 score with itemised findings. This is the DCMA 14-point health check as a service.

## 11. Multi-Baseline Variance Reporting

Extend `schedule_baseline_save` to diff the current schedule against any stored baseline and return task-level variance: start slip days, finish slip days, duration change, and float erosion. Support side-by-side comparison of baseline vs. current vs. latest replan — essential for contract change-order evidence.

## 12. Dependency Impact Propagation

When a task's actual finish is recorded later than its planned finish, forward-propagate the slip through all dependent tasks in the network and return a ripple-impact report: which successors are affected, by how many days, and which of them are on the critical path. This replaces the manual "what does this mean for the project?" analysis.

## 13. AI-Assisted Task Duration Estimation

Add an `estimate_task_duration(task_name, description, historical_tasks)` async method that uses a locally-hosted Ollama LLM (per the project's technology strategy) to suggest duration estimates based on task semantics and analogous historical tasks. Return confidence scores and comparable historical references so estimators can calibrate judgement against evidence.

## 14. Schedule Import/Export (MPP/XER/iCal)

Implement `import_mpp_xml(xml_bytes)` and `import_xer(xer_text)` parsers to ingest Microsoft Project XML and Primavera P6 XER formats. Export to iCal (`VTODO` components with dependencies encoded in `RELATED-TO`) for calendar integration. This removes the data-entry burden that prevents adoption by teams already running schedules in incumbent tools.

## 15. Real-Time WebSocket Schedule Updates

Add a `subscribe_schedule_updates(project_id)` async generator that yields schedule-change events over a WebSocket/SSE channel whenever tasks are updated, dependencies are added, or the critical path changes. Consumers (e.g., Gantt front-ends) get live updates without polling, enabling multi-user collaborative scheduling with sub-second latency.
