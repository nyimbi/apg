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
