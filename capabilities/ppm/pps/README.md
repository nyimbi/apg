# Project Planning & Scheduling

## Overview
Project Planning & Scheduling (pps) manages the full project schedule lifecycle: WBS decomposition, task definition, dependency linking with circular-dependency prevention, critical path calculation (CPM/PERT/CCPM/Monte Carlo), resource levelling, calendar management, and milestone tracking. Retroactive edits are blocked to maintain schedule integrity.

## Capability ID
`ppm_pps`

## Provides
| Service | Description |
|---------|-------------|
| wbs_creation_and_management | Hierarchical WBS up to 10 levels with coded elements |
| critical_path_analysis | CPM, PERT, CCPM, and Monte Carlo methods |
| resource_levelling | Priority-based, EDF, minimum slack, critical chain, and genetic algorithms |
| dependency_management | FS, SS, FF, SF dependency types with lag/lead and cycle detection |
| timeline_management | Multi-calendar project timelines with constraint management |
| schedule_optimisation | Auto-levelling and what-if schedule compression |
| project_calendar_management | Standard 5x8, 7x24, custom, and resource-specific calendars |
| milestone_tracking | Milestone status with earned-value linkage |
| schedule_risk_analysis | Monte Carlo and PERT probabilistic schedule analysis |
| gantt_chart_generation | Task bar data for Gantt rendering |
| schedule_baseline_export | Export approved schedule for ppm_pbl baseline creation |

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Audit logging of schedule changes |
| mten | Tenant scoping |
| conf | Configuration and feature flags |
| ntfy | Milestone and critical-path alert notifications |
| wflo | Change request workflow for retroactive edits |
| schd | External scheduling engine integration |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| tasks.supported_types | 7 | work_package, milestone, summary, deliverable, gate, buffer, hammock |
| scheduling.supported_critical_path_methods | 4 | CPM, PERT, CCPM, Monte Carlo |
| scheduling.supported_levelling_algorithms | 5 | Priority, EDF, slack, CC, genetic |
| wbs.max_depth | 10 | Maximum WBS hierarchy levels |
| governance.wbs_circular_dependency_denied | true | Schedule integrity |
| governance.critical_path_manipulation_denied | true | Data integrity |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-pps/projects | GET/POST | Project list | ppm_pps:projects |
| /ppm-pps/projects/<id>/wbs | GET/POST | WBS editor | ppm_pps:wbs |
| /ppm-pps/projects/<id>/gantt | GET | Gantt chart data | ppm_pps:gantt |
| /ppm-pps/projects/<id>/critical-path | GET/POST | Critical path | ppm_pps:critical_path |
| /ppm-pps/projects/<id>/dependencies | GET/POST | Dependency graph | ppm_pps:dependencies |
| /ppm-pps/projects/<id>/levelling | POST | Resource levelling | ppm_pps:levelling |
| /ppm-pps/projects/<id>/pert | POST | PERT three-point estimation | ppm_pps:analysis |
| /ppm-pps/projects/<id>/monte-carlo | POST | Monte Carlo simulation | ppm_pps:analysis |
| /ppm-pps/projects/<id>/evm | POST | Earned Value metrics | ppm_pps:analysis |
| /ppm-pps/projects/<id>/quality | GET | Schedule quality index | ppm_pps:analysis |
| /ppm-pps/projects/<id>/impact | POST | Dependency impact propagation | ppm_pps:analysis |
| /ppm-pps/projects/<id>/baseline-variance | GET | Baseline variance report | ppm_pps:analysis |
| /ppm-pps/projects/<id>/trend | GET | Schedule variance trend | ppm_pps:analysis |
| /ppm-pps/projects/<id>/ccpm-buffers | GET | CCPM buffer management | ppm_pps:analysis |
| /ppm-pps/milestones | GET | Milestone tracker | ppm_pps:milestones |
| /ppm-pps/calendars | GET/POST | Calendar manager | ppm_pps:calendars |
| /ppm-pps/agents | GET/POST | Agent workbench | ppm_pps:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| wbs_circular_dependency_denied | circular_dependency=True | deny |
| critical_path_manipulation_denied | critical_path_manipulation=True | deny |
| retroactive_edit_requires_change_request | retroactive=True on edit | deny |
| task_duration_must_be_positive | duration_positive=False | deny |
| wbs_element_required | wbs_element_present=False | deny |
| dependency_predecessor_required | predecessor_present=False | deny |
| cross_tenant_schedule_access_denied | cross_tenant_access=True | deny |
| schedule_batch_requires_bytewax | event_stream != "bytewax" | deny |

## Data Models
- **Project** — id, name, status, methodology, owner_id, start_date, end_date
- **WbsElement** — id, project_id, parent_id, level, code, name
- **Task** — id, project_id, wbs_element_id, task_type, status, duration_days, scheduling_mode, progress_pct
- **TaskDependency** — id, predecessor_id, successor_id, dependency_type, lag_days
- **CriticalPathResult** — id, project_id, method, critical_task_ids, total_float_days, project_duration_days
- **ResourceLevellingResult** — id, project_id, algorithm, over_allocations_resolved, schedule_extension_days
- **ProjectCalendar** — id, name, calendar_type, working_hours_per_day, working_days
- **ScheduleAgent** — id, name, runtime, role, scope

## Advanced Analysis Methods (v1.1)
| Method | Description |
|--------|-------------|
| `pert_estimate(project_id, task_estimates)` | Three-point PERT estimation with P50/P80/P90 per task |
| `monte_carlo_simulation(project_id, simulations)` | Probabilistic completion date distribution |
| `earned_value_metrics(project_id, pv, ac)` | Full EVM: SPI, CPI, EAC, VAC, TCPI |
| `schedule_quality_index(project_id)` | DCMA-style 0–100 health score with findings |
| `dependency_impact_propagation(project_id, task_id, slip_days)` | Forward-propagate slip through network |
| `baseline_variance_report(project_id, baseline_name)` | Task-level diff vs. stored baseline |
| `record_schedule_snapshot(project_id)` | Record daily SPI/progress snapshot for trend tracking |
| `schedule_variance_trend(project_id, lookback_days)` | SPI trend with deterioration detection |
| `critical_chain_buffers(project_id, buffer_pct)` | CCPM project + feeding buffers with fever zones |

## World-Class Enhancements (v2.0)

1. **Monte Carlo Schedule Risk Simulation** — Full probabilistic engine: sample PERT/triangular distributions per task, return P50/P80/P90 completion dates.
2. **Earned Value Management (EVM)** — PV/EV/AC tracking rolled up to project level; SPI, CPI, EAC, TCPI, and S-curves.
3. **PERT Three-Point Estimation Engine** — `optimistic_days`/`most_likely_days`/`pessimistic_days` fields on `Task`; variance propagated along the critical path.
4. **Calendar-Aware Date Arithmetic** — Replace raw `timedelta` with a working-calendar engine: weekends, public holidays, resource non-working days all honoured in ES/EF/LS/LF.
5. **Resource-Constrained Critical Path (RCCP)** — CPM network pass respects resource availability; tasks compete via minimum-slack priority rule.
6. **Persistent PostgreSQL-Backed Store** — `asyncpg`/SQLAlchemy 2.0 repository behind `AbstractScheduleRepository`; in-memory impl retained for unit tests.
7. **Earned Schedule (ES) Metrics** — SPI(t)-based schedule variance in time units; IEAC(t) forecast-to-complete.
8. **Critical Chain Project Management (CCPM)** — Feeding and project buffers with green/yellow/red fever-chart consumption zones.
9. **Schedule Variance Trend Analysis** — Daily SPI/float snapshots; `schedule_variance_trend` detects deteriorating trajectories early.
10. **Automated Schedule Quality Score** — DCMA 14-point health check as a service: dangling logic, oversized tasks, missing resources; returns 0–100 with itemised findings.
11. **Multi-Baseline Variance Reporting** — Task-level start/finish slip, duration change, float erosion vs. any stored baseline; supports baseline vs. current vs. replan.
12. **Dependency Impact Propagation** — Record an actual finish; receive a ripple-impact report of all affected successors, slip days, and critical-path membership.
13. **AI-Assisted Task Duration Estimation** — `estimate_task_duration` calls a local Ollama LLM, returns duration suggestions with confidence scores and analogous historical references.
14. **Schedule Import/Export (MPP/XER/iCal)** — Parse MS Project XML and Primavera P6 XER; export to iCal `VTODO` components with `RELATED-TO` dependency encoding.
15. **Real-Time WebSocket Schedule Updates** — `subscribe_schedule_updates(project_id)` async generator yields live change events (task updates, new deps, CP recalc) to Gantt front-ends.

## New Methods

### `monte_carlo_simulation` — Probabilistic delivery date

```python
result = await svc.monte_carlo_simulation(
    project_id="proj-abc",
    simulations=10_000,
)
# result["p50_date"], result["p80_date"], result["p90_date"]
# result["completion_histogram"]  — {date_str: frequency}
```

### `dependency_impact_propagation` — Slip ripple analysis

```python
impact = await svc.dependency_impact_propagation(
    project_id="proj-abc",
    task_id="task-123",
    slip_days=5,
)
# impact["affected_tasks"]  — list of {task_id, name, new_finish, on_critical_path}
# impact["project_slip_days"]
```

### `schedule_variance_trend` — Detect deteriorating schedule health

```python
trend = await svc.schedule_variance_trend(
    project_id="proj-abc",
    lookback_days=30,
)
# trend["snapshots"]     — [{date, spi, total_float, pct_complete}]
# trend["deteriorating"] — True if SPI slope < 0 over lookback window
# trend["latest_spi"]
```

## Streaming Events
- `project_created`, `project_updated`, `wbs_element_added`, `task_status_changed`
- `dependency_linked`, `critical_path_recalculated`, `resource_levelling_completed`
- `milestone_status_changed`, `schedule_risk_assessed`, `baseline_exported`, `agent_registered`

## Edge Cases Handled
- Circular dependency detection uses graph traversal before any write is committed
- Zero-duration tasks are rejected unless type is "milestone" (duration check is on add_task)
- Retroactive task edits are blocked at the rule-engine level; change requests must be used
- Critical path manipulation (manual overrides of float/duration) is explicitly denied
- Calendar type must be in the supported set; custom calendars need explicit working-day specification

## Composability Notes
- Schedule data is exported to **ppm_pbl** to create approved schedule baselines
- Resource assignments feed back into **ppm_res** for allocation and utilisation tracking
- Milestone events trigger **ntfy** notifications and **wflo** status transitions
- EVM metrics feed **ppm_ctr** for contract performance reporting
- Monte Carlo P80/P90 dates feed **risk** capability for schedule risk register entries
- CCPM fever-chart zones feed **ntfy** for proactive buffer-breach alerts
