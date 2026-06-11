# Succession Planning (hcm_scp)

Capability for talent management and succession planning: talent pools, readiness assessments, nine-box grid placement, succession scenarios, critical role identification, and advanced analytics.

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/health | Health check |
| GET | /api/hcm/scp/describe | Capability contract |
| GET | /api/hcm/scp/audit-events | Audit trail |

### Talent Pools

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/talent-pools | List talent pools |
| POST | /api/hcm/scp/talent-pools | Create talent pool |
| GET | /api/hcm/scp/talent-pools/{id} | Get talent pool |
| PUT | /api/hcm/scp/talent-pools/{id} | Update talent pool |
| DELETE | /api/hcm/scp/talent-pools/{id} | Delete talent pool (empty only) |
| GET | /api/hcm/scp/talent-pools/{id}/members | List pool members |
| POST | /api/hcm/scp/talent-pools/{id}/members | Add member (readiness gate enforced) |
| DELETE | /api/hcm/scp/talent-pools/{id}/members/{eid} | Remove member |

### Readiness Assessments

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/readiness-assessments | List assessments |
| POST | /api/hcm/scp/readiness-assessments | Create assessment (nine-box quadrant auto-computed) |
| GET | /api/hcm/scp/readiness-assessments/{id} | Get assessment |
| PUT | /api/hcm/scp/readiness-assessments/{id} | Update assessment |
| DELETE | /api/hcm/scp/readiness-assessments/{id} | Delete assessment |
| POST | /api/hcm/scp/readiness-assessments/bulk | Bulk import (batch with per-record status) |

### Nine-Box Grid

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/nine-box | List nine-box entries |
| POST | /api/hcm/scp/nine-box | Place employee on grid |
| GET | /api/hcm/scp/nine-box/grid | Grid grouped by quadrant (filter by review_cycle) |
| GET | /api/hcm/scp/nine-box/{employee_id}/history | Movement history across review cycles |

### Succession Scenarios

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/scenarios | List succession scenarios |
| POST | /api/hcm/scp/scenarios | Create scenario |
| GET | /api/hcm/scp/scenarios/{id} | Get scenario |
| PUT | /api/hcm/scp/scenarios/{id} | Update scenario |
| PUT | /api/hcm/scp/scenarios/{id}/activate | Activate scenario (draft → active) |
| DELETE | /api/hcm/scp/scenarios/{id} | Delete scenario (draft only) |
| POST | /api/hcm/scp/scenarios/simulate-vacancy | Vacancy simulation (no state mutation) |

### Critical Roles

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/critical-roles | List critical roles |
| POST | /api/hcm/scp/critical-roles | Identify critical role |
| GET | /api/hcm/scp/critical-roles/{id} | Get critical role |
| PUT | /api/hcm/scp/critical-roles/{id} | Update critical role |
| DELETE | /api/hcm/scp/critical-roles/{id} | Remove critical role |

### Analytics & Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/coverage-report | Succession coverage for critical roles |
| GET | /api/hcm/scp/readiness-report | Readiness distribution across pools |
| GET | /api/hcm/scp/dashboard | Dashboard summary |
| GET | /api/hcm/scp/depth-score/{role_id} | Succession depth score (0–10) for a role |
| GET | /api/hcm/scp/bench-strength | Bench Strength Index (BSI) org-wide or per pool |
| GET | /api/hcm/scp/overdue-reviews | Stale scenarios and critical roles past review date |
| GET | /api/hcm/scp/retention-risk-alerts | Retention and succession risk alerts |
| GET | /api/hcm/scp/role-risk-registry | Prioritised role risk registry |

## Key Concepts

**Readiness Levels**: `developing` | `ready_in_1_year` | `ready_now`

**Nine-Box Quadrants**: `star`, `high_performer`, `solid_contributor`, `high_potential`, `core_employee`, `inconsistent_player`, `enigma`, `average_performer`, `underperformer`

**Scenario Types**: `planned`, `emergency`, `voluntary`, `retirement`

**Impact Levels**: `low`, `medium`, `high`, `critical`

## Analytics Methods

| Method | Description |
|--------|-------------|
| `succession_depth_score(role_id)` | Weighted 0–10 score: ready_now*3 + ready_in_1_year*1.5 + developing*0.5 |
| `bench_strength_index(pool_id?)` | BSI = (ready_now + 0.5*ready_in_1_year) / total * 100, graded A–D |
| `get_nine_box_movement_history(employee_id)` | Chronological placements + movement vectors across cycles |
| `simulate_vacancy(role_id, incumbent_id)` | What-if snapshot with incumbent removed; no state mutation |
| `get_overdue_reviews(as_of?)` | Scenarios and critical roles past `review_due_date` |
| `bulk_create_readiness_assessments(assessments)` | Atomic-ish batch import with per-record success/failure |
| `get_retention_risk_alerts(...)` | Stale ready_now, low depth score, and unassessed star alerts |
| `role_risk_registry()` | Composite risk-scored and sorted registry of critical roles |
