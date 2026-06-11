# Project Baseline Management — User Guide

**Capability ID**: `ppm_pbl` | **Domain**: `ppm` | **Version**: `2.0.0`

## Description

Project Baseline Management (pbl) establishes and protects the scope, schedule, and cost baselines for projects. It enforces formal change control, calculates earned value metrics (including Earned Schedule), detects variance threshold breaches, prevents retroactive baseline manipulation, and provides portfolio-level baseline health scoring — delivering the performance measurement baseline required for EVMS/EVM compliance.

---

## Installation

```bash
pip install apg-ppm-pbl
```

---

## Quick Start

```python
from capabilities.ppm.pbl.service import ProjectBaselineService

svc = ProjectBaselineService(tenant_id="acme", actor_id="pmo.lead")

# Lock scope baseline
scope_bl = await svc.set_scope_baseline(
    project_id="prj-101",
    scope_document={
        "deliverables": ["System Design", "Integration Module", "UAT Report"],
        "inclusions": ["API development", "Testing"],
        "exclusions": ["Hosting infrastructure"],
        "assumptions": ["Client provides test data by Week 4"],
        "constraints": ["Go-live by 2026-09-30"],
    },
    approved_by="sponsor.alice",
)

# Lock schedule baseline
sched_bl = await svc.set_schedule_baseline(
    project_id="prj-101",
    baseline_schedule={
        "project_duration_days": 120,
        "tasks": [{"id": "T1", "name": "Design", "duration": 20}],
        "dependencies": [{"from": "T1", "to": "T2"}],
    },
    approved_by="pm.bob",
)

# Lock cost baseline
cost_bl = await svc.set_cost_baseline(
    project_id="prj-101",
    budget={
        "total_budget": 250000.0,
        "budget_lines": [{"code": "LA", "amount": 150000}, {"code": "HW", "amount": 100000}],
        "contingency": 15000.0,
        "currency": "USD",
    },
    approved_by="cfo.carol",
)
```

---

## Service Methods

### Baseline Lifecycle

| Method | Description |
|--------|-------------|
| `set_scope_baseline(project_id, scope_document, approved_by)` | Lock the scope baseline. Increments version if called again. |
| `set_schedule_baseline(project_id, baseline_schedule, approved_by)` | Lock the schedule baseline with task/dependency snapshot. |
| `set_cost_baseline(project_id, budget, approved_by)` | Lock the cost baseline with budget-at-completion (BAC). |
| `baseline_comparison(project_id, baseline_name, current)` | Compare a named baseline against current project data; returns deltas. |
| `baseline_restore(project_id, baseline_name, approved_by)` | Restore project data from a saved baseline snapshot. |
| `baseline_rebase(project_id, change_request_id, approved_by)` | Rebase all baselines following an approved change request. |
| `baseline_analytics(project_id)` | Summary analytics: baseline health, change velocity, EV trend. |

### Baseline Governance

| Method | Description |
|--------|-------------|
| `lock_baseline(project_id, baseline_type, locked_by, reason)` | Write-protect an approved baseline. |
| `unlock_baseline(project_id, baseline_type, unlocked_by)` | Remove a baseline lock. |
| `set_freeze_period(project_id, start_date, end_date, freeze_scope, reason)` | Block CR submissions for a project during a date range. Emergency CRs bypass. |
| `get_baseline_version_history(project_id, baseline_type)` | Return full immutable version history for a project baseline. |

### Integrated Baseline Review

| Method | Description |
|--------|-------------|
| `integrated_baseline_review(project_id)` | Cross-validate all three baselines; returns IBR health index (0–100) and dimension pass/fail. |

### Change Control

| Method | Description |
|--------|-------------|
| `change_request(project_id, change_type, description, impact_assessment, requested_by)` | Raise a change request. |
| `approve_change(change_request_id, approved_by, decision)` | Approve / reject / defer a CR. |
| `change_log(project_id)` | Full change log with CR summary statistics. |
| `change_impact_summary(project_id)` | Rolled-up cost and schedule impact across all CRs. |
| `change_request_analytics(period)` | CR KPIs: submission rate, approval rate. |
| `link_change_requests(cr_id, related_cr_id, relationship)` | Record a directed relationship between two CRs. |
| `get_cr_dependency_graph()` | Return the CR dependency DAG (nodes + edges + adjacency list). |

### Earned Value Management

| Method | Description |
|--------|-------------|
| `take_ev_snapshot(snapshot_id, tenant_id, baseline_id, snapshot_date, pv, ev, ac, bac, forecasting_method, eac, etc)` | Record a new EV data point. |
| `variance_analysis(project_id, baseline_name, period)` | SV, CV, SPI, CPI with threshold colour. |
| `earned_value_trend(project_id)` | SPI/CPI time series for a project. |
| `forecast_completion(project_id)` | EAC (typical/atypical/scheduled), TCPI, VAC with method recommendation. |
| `earned_schedule_metrics(project_id, planned_duration_days)` | ES, SPI(t), SV(t), IEAC(t) — time-domain EVM metrics. |

### Portfolio & Reporting

| Method | Description |
|--------|-------------|
| `portfolio_baseline_summary()` | Cross-project CPI/SPI rollup with risk-tiered project list (red/amber/green). |
| `baseline_deviation_scores()` | Composite BDS (0–100) for all projects, ranked worst-first. |
| `baseline_compliance_check()` | Verify all baselines have evidence and approved status. |
| `dashboard_summary(tenant_id)` | Record counts for the management dashboard. |
| `export_baselines(format)` | Export baseline records as JSON or CSV. |
| `health_check()` | Service liveness check. |
| `get_audit_events()` | Retrieve raw audit log entries. |

---

## Earned Schedule Explained

Traditional EVM SPI converges to 1.0 at project completion even when the project is late. Earned Schedule (ES) corrects this:

```python
metrics = await svc.earned_schedule_metrics(
    project_id="prj-101",
    planned_duration_days=120,
)
# Returns:
# {
#   "es_days": 72.0,       # Work accomplished expressed in time units
#   "at_days": 80.0,       # Actual time elapsed (proxy)
#   "spi_t": 0.900,        # True time-based performance index
#   "sv_t": -8.0,          # 8 days behind schedule
#   "ieac_t_days": 133.3,  # Independent EAC in days
#   "schedule_status": "behind",
# }
```

---

## CR Dependency Tracking

```python
# CR-002 cannot be approved until CR-001 is resolved
await svc.link_change_requests("cr-002", "cr-001", relationship="depends_on")

# CR-003 supersedes CR-001
await svc.link_change_requests("cr-003", "cr-001", relationship="supersedes")

graph = await svc.get_cr_dependency_graph()
# graph["edges"], graph["adjacency"], graph["nodes"]
```

---

## Baseline Locking

```python
# Lock cost baseline before month-end reporting
await svc.lock_baseline("prj-101", "cost", locked_by="pmo.lead",
                         reason="Month-end reporting lock")

# Unlock after reporting window
await svc.unlock_baseline("prj-101", "cost", unlocked_by="pmo.lead")
```

---

## Freeze Periods

```python
# Block non-emergency CRs during contract reporting window
await svc.set_freeze_period(
    project_id="prj-101",
    start_date="2026-07-01",
    end_date="2026-07-05",
    freeze_scope=["cost", "schedule"],
    reason="CDRL reporting window",
    set_by="pmo.lead",
)
```

---

## Integrated Baseline Review

```python
ibr = await svc.integrated_baseline_review("prj-101")
# {
#   "ibr_health_index": 80.0,
#   "overall": "warning",
#   "dimensions": {
#     "completeness": {"pass": True, ...},
#     "scope_integrity": {"pass": True, "deliverable_count": 3},
#     "schedule_integrity": {"pass": True, "task_count": 12},
#     "cost_integrity": {"pass": True, "total_budget": 250000.0},
#     "scope_schedule_alignment": {"pass": False, "ratio": 4.0},
#   }
# }
```

---

## Portfolio Health

```python
portfolio = await svc.portfolio_baseline_summary()
# {
#   "project_count": 8,
#   "portfolio_cpi": 0.94,
#   "portfolio_spi": 0.97,
#   "red_projects": 1,
#   "amber_projects": 3,
#   "green_projects": 4,
#   "projects": [...]   # sorted worst-first
# }

bds = await svc.baseline_deviation_scores()
# {
#   "scores": [
#     {"project_id": "prj-105", "bds": 72.3, "tier": "red", ...},
#     {"project_id": "prj-101", "bds": 18.5, "tier": "green", ...},
#   ]
# }
```

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| retroactive_baseline_edit_denied | retroactive=True on edit | deny |
| change_control_bypass_denied | change_control_bypass=True | deny |
| ev_manipulation_denied | ev_manipulation=True | deny |
| baseline_approval_requires_designated_approver | designated_approver=False | deny |
| change_baseline_required | baseline_present=False | deny |
| change_impact_required | impact_present=False | deny |
| change_approval_required | approval_present=False on implement | deny |
| cross_tenant_baseline_access_denied | cross_tenant_access=True | deny |

---

## Composability

```apg
use ppm_pbl;
```

- Receives cost data from **ppm_pac** for cost baseline creation
- Receives schedule from **ppm_pps** for schedule baseline creation
- EV snapshots feed **ppm_pan** portfolio performance dashboards
- Change requests trigger notifications via **ntfy** and approval steps via **wflo**
- IBR results feed **comp** for EVMS compliance reporting

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and supported enumerations
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 architectural enhancements
- `README.md` — Quick reference
