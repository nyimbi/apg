# Succession Planning — User Guide

## Overview

The Succession Planning (SCP) capability enables organisations to identify, assess, and develop talent for critical roles. It covers talent pools, readiness assessments, the nine-box grid, succession scenarios, critical role management, and a suite of advanced analytics including succession depth scoring, bench strength index, vacancy simulation, and retention risk alerting.

---

## Talent Pools

Create pools via `POST /api/hcm/scp/talent-pools`. Set `min_readiness_level` (`developing`, `ready_in_1_year`, or `ready_now`) to gate entry. Employees below the minimum readiness level are rejected when added.

```json
POST /api/hcm/scp/talent-pools
{
  "name": "Senior Leadership Pipeline",
  "min_readiness_level": "ready_in_1_year",
  "target_roles": ["vp-finance", "vp-operations"]
}
```

Add employees via `POST /api/hcm/scp/talent-pools/{id}/members`. The readiness gate is enforced server-side. A pool with active members cannot be deleted.

---

## Readiness Assessments

Provide `performance_rating` (1–5) and `potential_rating` (1–5) along with `readiness_level`. The nine-box quadrant is auto-computed from the ratings and stored on the assessment record.

```json
POST /api/hcm/scp/readiness-assessments
{
  "employee_id": "emp-001",
  "target_role_id": "vp-finance",
  "readiness_level": "ready_in_1_year",
  "performance_rating": 4.2,
  "potential_rating": 3.8,
  "assessed_by": "mgr-007",
  "development_needs": ["financial-modelling", "stakeholder-management"]
}
```

### Bulk Import

Use `POST /api/hcm/scp/readiness-assessments/bulk` to import an array of assessment objects in one call. Partial failures are tolerated — the response includes per-record `status` and `error` fields.

```json
POST /api/hcm/scp/readiness-assessments/bulk
{
  "assessments": [
    { "employee_id": "emp-001", "target_role_id": "vp-finance", ... },
    { "employee_id": "emp-002", "target_role_id": "vp-ops", ... }
  ]
}
```

Response includes `succeeded`, `failed`, and a `results` array indexed to the input.

---

## Nine-Box Grid

Place employees on the grid via `POST /api/hcm/scp/nine-box` using axes of 1.0–3.0 (low / medium / high):

```json
POST /api/hcm/scp/nine-box
{
  "employee_id": "emp-001",
  "performance_axis": 2.8,
  "potential_axis": 2.9,
  "review_cycle": "2026-H1",
  "reviewer_id": "mgr-007"
}
```

Retrieve a grouped grid view for a review cycle:

```
GET /api/hcm/scp/nine-box/grid?review_cycle=2026-H1
```

### Nine-Box Movement History

Track an employee's quadrant trajectory across review cycles:

```
GET /api/hcm/scp/nine-box/emp-001/history
```

Returns chronological `placements` and `movements` arrays with `performance_delta`, `potential_delta`, and `quadrant_changed` flags.

**Nine-box quadrant labels**: `star`, `high_performer`, `solid_contributor`, `high_potential`, `core_employee`, `inconsistent_player`, `enigma`, `average_performer`, `underperformer`.

---

## Succession Scenarios

Each scenario lists ranked successors with `employee_id`, `readiness`, and `rank`. Scenarios flow `draft → active`. Activate via:

```
PUT /api/hcm/scp/scenarios/{id}/activate
{ "approved_by": "ceo-001" }
```

Only draft scenarios can be deleted or activated.

### Vacancy Simulation

Run a what-if without persisting any changes:

```json
POST /api/hcm/scp/scenarios/simulate-vacancy
{
  "role_id": "vp-finance",
  "incumbent_employee_id": "emp-001"
}
```

Returns a simulation snapshot including `depth_score_post_vacancy`, `ready_now` count, and `risk_tier`.

---

## Critical Roles

Flag roles as critical via `POST /api/hcm/scp/critical-roles`:

```json
POST /api/hcm/scp/critical-roles
{
  "role_id": "vp-finance",
  "role_title": "VP Finance",
  "rationale": "Single point of financial control",
  "impact_if_vacant": "critical",
  "identified_by": "ceo-001",
  "time_to_fill_estimate_days": 120,
  "review_due_date": "2027-01-01"
}
```

Set `review_due_date` to activate cadence enforcement — stale active roles appear in `GET /api/hcm/scp/overdue-reviews`.

---

## Analytics

### Succession Coverage Report

```
GET /api/hcm/scp/coverage-report
```

Returns `covered_roles`, `uncovered_roles`, `coverage_pct`, and a list of uncovered role details.

### Succession Depth Score

Weighted 0–10 score per role. Weights: ready_now×3, ready_in_1_year×1.5, developing×0.5. Normalised against an ideal slate of 3 ready_now successors.

```
GET /api/hcm/scp/depth-score/vp-finance
```

Returns `score`, `risk_tier` (`low` / `medium` / `high`), and successor breakdown.

### Bench Strength Index (BSI)

```
GET /api/hcm/scp/bench-strength               # org-wide
GET /api/hcm/scp/bench-strength?pool_id=tp-01 # specific pool
```

BSI = `(ready_now + 0.5 × ready_in_1_year) / total_members × 100`. Graded A–D.

| Grade | BSI Range | Interpretation |
|-------|-----------|----------------|
| A | ≥ 70 | Strong bench |
| B | 50–69 | Adequate |
| C | 30–49 | Developing |
| D | < 30 | At risk |

### Overdue Reviews

```
GET /api/hcm/scp/overdue-reviews
GET /api/hcm/scp/overdue-reviews?as_of=2026-12-31
```

Returns scenarios and critical roles where `review_due_date` is in the past and status is `active`.

### Retention Risk Alerts

```
GET /api/hcm/scp/retention-risk-alerts
GET /api/hcm/scp/retention-risk-alerts?stale_months_threshold=12&depth_score_threshold=4.0
```

Three alert categories:

| Alert Type | Severity | Trigger |
|------------|----------|---------|
| `stale_ready_now_successor` | high | ready_now member in pool > 18 months without progression |
| `low_succession_depth` | critical/high | depth score for critical role below threshold |
| `star_not_reassessed` | medium | nine-box star not re-placed in > 12 months |

### Role Risk Registry

```
GET /api/hcm/scp/role-risk-registry
```

Composite risk score = `impact_weight × 0.4 + (10 - depth_score) × 0.4 + fill_time_factor × 0.2`. Roles sorted descending by composite risk. Use to prioritise succession remediation backlog.

### Dashboard

```
GET /api/hcm/scp/dashboard
```

Aggregated counts plus embedded coverage and readiness reports.

---

## API Quick Reference

```
GET  /api/hcm/scp/health
POST /api/hcm/scp/talent-pools
POST /api/hcm/scp/readiness-assessments
POST /api/hcm/scp/readiness-assessments/bulk
POST /api/hcm/scp/nine-box
GET  /api/hcm/scp/nine-box/grid?review_cycle=2026-H1
GET  /api/hcm/scp/nine-box/{employee_id}/history
POST /api/hcm/scp/scenarios
POST /api/hcm/scp/scenarios/simulate-vacancy
POST /api/hcm/scp/critical-roles
GET  /api/hcm/scp/coverage-report
GET  /api/hcm/scp/depth-score/{role_id}
GET  /api/hcm/scp/bench-strength
GET  /api/hcm/scp/overdue-reviews
GET  /api/hcm/scp/retention-risk-alerts
GET  /api/hcm/scp/role-risk-registry
GET  /api/hcm/scp/dashboard
```
