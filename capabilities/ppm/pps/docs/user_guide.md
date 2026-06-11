# Project Planning & Scheduling — User Guide

**Capability ID**: `ppm_pps` | **Domain**: `ppm` | **Version**: `1.1.0`

## Description

Project Planning & Scheduling (pps) manages the full project schedule lifecycle: WBS decomposition, task definition, dependency linking with circular-dependency prevention, critical path calculation (CPM/PERT/CCPM/Monte Carlo), resource levelling, calendar management, and milestone tracking. Retroactive edits are blocked to maintain schedule integrity.

## Installation

```bash
pip install apg-ppm-pps
```

## Quick Start

```python
import asyncio
from capabilities.ppm.pps.service import ProjectPlanningService

svc = ProjectPlanningService(tenant_id="acme", actor_id="alice")

# 1. Create a project
svc.create_project(
    project_id="p1", tenant_id="acme", name="Bridge Refurb",
    status="active", methodology="waterfall", owner_id="alice",
    start_date="2026-07-01", end_date="2026-12-31",
    evidence_reference="SOW-2026-001",
)

# 2. Add WBS elements
svc.add_wbs_element("wbs1", "acme", "p1", None, "phase", "1", "Design", "")
svc.add_wbs_element("wbs2", "acme", "p1", "wbs1", "work_package", "1.1", "Structural Survey", "")

# 3. Add tasks
svc.add_task("t1", "acme", "p1", "wbs2", "work", "not_started",
             "Initial Survey", 5.0, "auto", "asap", "percent_complete", 0.0,
             "2026-07-01", "2026-07-05")
svc.add_task("t2", "acme", "p1", "wbs2", "work", "not_started",
             "Load Analysis", 10.0, "auto", "asap", "percent_complete", 0.0,
             "2026-07-06", "2026-07-15", predecessors=["t1"])

# 4. Compute critical path
result = asyncio.run(svc.critical_path_analysis("p1"))
print(result["critical_task_ids"])   # ['t1', 't2']

# 5. Generate Gantt data
gantt = asyncio.run(svc.gantt_chart_data("p1"))
for bar in gantt["bars"]:
    print(bar["task_name"], bar["start_date"], "→", bar["end_date"], "critical:", bar["critical"])
```

## Core Service Methods

### Projects

| Method | Description |
|--------|-------------|
| `create_project(...)` | Create a project with scheduling metadata |
| `get_project(project_id, tenant_id)` | Fetch a single project |
| `list_projects(tenant_id)` | List all projects for a tenant |

### Work Breakdown Structure

| Method | Description |
|--------|-------------|
| `add_wbs_element(...)` | Add a single WBS node |
| `create_wbs(project_id, wbs_elements)` | Bulk-create WBS hierarchy |
| `list_wbs_elements(tenant_id, project_id)` | List WBS nodes |

### Tasks

| Method | Description |
|--------|-------------|
| `add_task(...)` | Add a task, optionally auto-linking predecessors |
| `update_task_status(task_id, tenant_id, status, progress_pct)` | Update progress |
| `list_tasks(tenant_id, project_id)` | List tasks |
| `bulk_create_tasks(project_id, task_specs)` | Bulk-create from spec list |

### Dependencies

| Method | Description |
|--------|-------------|
| `link_dependency(...)` | Link FS/SS/FF/SF dependency with lag |

Circular dependency detection runs a graph traversal before any write is committed — the write is rejected if a cycle would form.

### Scheduling

| Method | Description |
|--------|-------------|
| `schedule_network(project_id)` | CPM forward/backward pass; returns ES/EF/LS/LF/float per task |
| `critical_path_analysis(project_id)` | Identifies critical tasks and stores a `CriticalPathResult` |
| `schedule_compression(project_id, technique)` | Fast-track or crash the critical path |
| `gantt_chart_data(project_id)` | Gantt-ready bars + dependency links with absolute dates |
| `what_if_analysis(project_id, scenario)` | Simulate duration overrides without mutating the live plan |
| `schedule_baseline_save(project_id, baseline_name)` | Snapshot current schedule as a named baseline |

### Resources

| Method | Description |
|--------|-------------|
| `resource_levelling(project_id, resource_constraints)` | Level allocations given capacity caps |
| `resource_histogram(project_id)` | Effort-by-resource loading histogram |

### Calendars and Agents

| Method | Description |
|--------|-------------|
| `create_calendar(...)` | Define a working calendar (5x8, 7x24, custom) |
| `register_agent(...)` | Register a scheduling agent with runtime and role |
| `validate_agent_action(...)` | Enforce human-approval requirement for privileged agent actions |

### Reporting

| Method | Description |
|--------|-------------|
| `schedule_analytics(project_id)` | SPI, completion trend, float distribution, milestone status |
| `milestone_tracker(project_id)` | Milestone completion and overdue count |
| `export_schedule(project_id, format)` | Export tasks + deps as JSON or CSV |
| `schedule_compliance_check()` | Unassigned tasks, compliance rate |
| `health_check()` | Service health status |
| `dashboard_summary(tenant_id)` | Aggregate counts for dashboard |

---

## Advanced Analysis Methods (v1.1)

### PERT Three-Point Estimation

```python
result = await svc.pert_estimate("p1", [
    {"task_id": "t1", "optimistic": 3, "most_likely": 5, "pessimistic": 10},
    {"task_id": "t2", "optimistic": 7, "most_likely": 10, "pessimistic": 18},
])
print(result["p80_project_days"])  # P80 completion estimate
```

Computes PERT mean `(O + 4M + P) / 6` and standard deviation `(P - O) / 6` per task. Updates the task's `duration_days` to the PERT mean. Aggregates variance on the critical path to produce P50/P80/P90 project completion estimates.

### Monte Carlo Schedule Risk Simulation

```python
sim = await svc.monte_carlo_simulation("p1", simulations=2000, seed=42)
print(sim["p80_date"])    # e.g. "2026-12-14"
print(sim["distribution_buckets"])  # 10-bin frequency histogram
```

Runs N forward-pass simulations sampling each task duration from a triangular distribution parameterised by `duration_days` ± 15% (or PERT estimates if available). Returns P50/P80/P90 completion dates, mean/min/max, and a frequency histogram. Seed is optional but recommended for reproducible CI tests.

### Earned Value Management (EVM)

```python
evm = await svc.earned_value_metrics("p1", planned_value=50000.0, actual_cost=48000.0)
print(evm["spi"], evm["cpi"], evm["eac"])
```

Derives EV from task progress percentages, computes SPI, CPI, EAC (cost estimate to complete), VAC (variance at completion), and TCPI (to-complete performance index). Returns a `status` of `on_track`, `at_risk`, or `critical`.

### Schedule Quality Index

```python
qi = await svc.schedule_quality_index("p1")
print(qi["score"], qi["rating"])   # e.g. 82, "good"
print(qi["findings"])              # itemised check results
```

DCMA-inspired 0–100 health score. Checks: isolated tasks (missing logic), open starts/ends, tasks longer than 10 days, and stale in-progress tasks (0% progress). Each finding deducts points; score is clamped to 0.

### Dependency Impact Propagation

```python
impact = await svc.dependency_impact_propagation("p1", "t1", slip_days=3.0)
for t in impact["impacted_tasks"]:
    print(t["task_name"], t["inherited_slip_days"], "critical:", t["is_critical"])
```

BFS forward-propagation from the slipped task. Float absorbs slip at each hop; residual slip is passed to successors. Returns per-task inherited slip, absorbed float, and criticality. The `project_level_slip_days` field shows the overall project impact.

### Baseline Variance Report

```python
# First save a baseline
await svc.schedule_baseline_save("p1", "original")

# ... work progresses, tasks change ...

# Then diff
report = await svc.baseline_variance_report("p1", "original")
for v in report["variances"]:
    if v["finish_slip_days"] > 0:
        print(v["task_name"], "slipped by", v["finish_slip_days"], "days")
```

Task-level diff between current schedule and any stored baseline: `start_slip_days`, `finish_slip_days`, `duration_delta_days`, and `float_erosion_days`. Results sorted by finish slip descending.

### Schedule Variance Trend

```python
# Record a snapshot each day (e.g., via cron)
await svc.record_schedule_snapshot("p1")

# Query trend
trend = await svc.schedule_variance_trend("p1", lookback_days=14)
print(trend["trend"])                    # "stable" | "deteriorating" | "improving"
print(trend["deteriorating_trajectory"]) # True if SPI declining 3 consecutive days
```

Snapshots store SPI, average progress, and critical task count. `schedule_variance_trend` detects a deteriorating trajectory when SPI has declined for three or more consecutive snapshots — triggering proactive intervention before a milestone is missed.

### CCPM Buffer Management

```python
buffers = await svc.critical_chain_buffers("p1", buffer_pct=25.0)
print(buffers["project_buffer_days"])
print(buffers["project_buffer_fever_zone"])  # "green" | "yellow" | "red"
for fb in buffers["feeding_buffers"]:
    print(fb["feeding_task_name"], fb["fever_zone"])
```

Computes a project buffer (25% of critical chain by default) and feeding buffers at merge points where non-critical chains join the critical chain. Each buffer has a fever-chart zone (green < 33% consumed, yellow < 66%, red ≥ 66%) for at-a-glance status.

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-pps/dashboard` | `ppm_pps:view` | Overview |
| `/ppm-pps/projects` | `ppm_pps:projects` | Projects |
| `/ppm-pps/projects/<id>` | `ppm_pps:projects` | Projects |
| `/ppm-pps/projects/<id>/wbs` | `ppm_pps:wbs` | Planning |
| `/ppm-pps/projects/<id>/gantt` | `ppm_pps:gantt` | Planning |
| `/ppm-pps/projects/<id>/critical-path` | `ppm_pps:critical_path` | Analysis |
| `/ppm-pps/projects/<id>/dependencies` | `ppm_pps:dependencies` | Planning |
| `/ppm-pps/projects/<id>/levelling` | `ppm_pps:levelling` | Resources |
| `/ppm-pps/projects/<id>/pert` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/monte-carlo` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/evm` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/quality` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/impact` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/baseline-variance` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/trend` | `ppm_pps:analysis` | Analysis |
| `/ppm-pps/projects/<id>/ccpm-buffers` | `ppm_pps:analysis` | Analysis |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `wbs_circular_dependency_denied` | `circular_dependency=True` | deny |
| `critical_path_manipulation_denied` | `critical_path_manipulation=True` | deny |
| `retroactive_edit_requires_change_request` | `retroactive=True` | deny |
| `task_duration_must_be_positive` | `duration_positive=False` | deny |
| `wbs_element_required` | `wbs_element_present=False` | deny |
| `dependency_predecessor_required` | `predecessor_present=False` | deny |
| `cross_tenant_schedule_access_denied` | `cross_tenant_access=True` | deny |
| `schedule_batch_requires_bytewax` | `event_stream != "bytewax"` | deny |

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PPS_`.

| Key | Default | Description |
|-----|---------|-------------|
| `tasks.supported_types` | 7 | work_package, milestone, summary, deliverable, gate, buffer, hammock |
| `scheduling.supported_critical_path_methods` | 4 | CPM, PERT, CCPM, Monte Carlo |
| `scheduling.supported_levelling_algorithms` | 5 | Priority, EDF, slack, CC, genetic |
| `wbs.max_depth` | 10 | Maximum WBS hierarchy levels |
| `monte_carlo.default_simulations` | 1000 | Default simulation count |
| `ccpm.default_buffer_pct` | 25 | Default CCPM buffer percentage |

## Interoperability

```apg
use ppm_pps;
```

- Schedule baselines export to **ppm_pbl** for contract baseline management
- Resource assignments feed **ppm_res** for utilisation tracking
- EVM metrics feed **ppm_ctr** for contract performance reporting
- Monte Carlo P80/P90 dates feed the **risk** capability's schedule risk register
- CCPM fever-chart zone changes trigger **ntfy** buffer-breach alerts
- Milestone status changes trigger **wflo** status transitions

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of planned enhancements
