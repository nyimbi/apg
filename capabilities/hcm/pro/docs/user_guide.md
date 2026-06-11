# Professional Development — User Guide

## Overview

The Professional Development (PRO) capability supports employee growth through structured development plans, skill gap analysis, mentoring relationships, certification tracking, career pathing, learning activity management, 360-degree feedback, automated nudges, and the Professional Development Index (PDI).

---

## Development Plans

Plans flow: `draft → active → completed`.

1. Create a plan: `POST /api/hcm/pro/development-plans` with `employee_id`, `plan_year`, `objectives`, and `focus_areas`.
2. Activate: `PUT /api/hcm/pro/development-plans/{id}/activate` — requires `reviewed_by`.
3. Track progress: `PUT /api/hcm/pro/development-plans/{id}/progress` with `progress_pct` (0–100). At 100% the plan auto-transitions to `completed`.
4. Clone for a new year: `POST /api/hcm/pro/development-plans/{id}/clone` — copies objectives and focus areas into a new draft for `new_plan_year`, resetting progress.

### Plan Templates

Avoid authoring plans from scratch for common roles.

1. Create a template: `POST /api/hcm/pro/plan-templates` with `name`, `target_role`, `department`, default `objectives`, and `focus_areas`.
2. Apply a template: `POST /api/hcm/pro/plan-templates/{id}/apply` with `employee_id` and `plan_year`. Returns a seeded draft plan. The template's `usage_count` increments automatically.

---

## Skill Gap Analysis

1. Add skills to the catalogue: `POST /api/hcm/pro/skills` — provide `name` and `category` (technical|leadership|communication|analytical|domain|soft).
2. Assess an employee: `POST /api/hcm/pro/skill-assessments` with `current_level` and `target_level`. `gap_exists` is computed automatically.
3. Individual gap report: `GET /api/hcm/pro/skill-gap-report/{employee_id}` — returns `overall_readiness` (ready|partial|needs_development).
4. Team gap report: `POST /api/hcm/pro/team-skill-gap-report` with a list of `employee_ids`. Runs all individual reports in parallel and returns a `heat_map` by skill category and `top_skill_gaps`.

Proficiency levels (ascending): `beginner → intermediate → advanced → expert`.

### Skill Endorsements

Peers can validate expertise independently of manager assessments.

- Endorse: `POST /api/hcm/pro/skill-endorsements` — provide `endorsee_employee_id`, `endorser_employee_id`, `skill_id`, and `endorsed_level`. Self-endorsement is rejected.
- Summary: `GET /api/hcm/pro/skill-endorsements/{employee_id}/{skill_id}` — returns `endorsement_count`, `highest_endorsed_level`, and a level frequency breakdown.

---

## 360-Degree Feedback

Collect structured multi-rater observations on a specific skill.

1. Create a request: `POST /api/hcm/pro/feedback-requests` — specify `subject_employee_id`, `skill_id`, `rater_employee_ids`, and `rater_types` (self|peer|manager|skip_level|report).
2. Raters respond: `POST /api/hcm/pro/feedback-requests/{id}/respond` with `observed_level` and optional `comments`. The request moves to `complete` when all raters respond.
3. Aggregate: `GET /api/hcm/pro/feedback-requests/{id}/aggregate` — returns `consensus_level`, `variance_levels`, `consensus_quality` (high/medium/low), and a per-rater-type breakdown.

---

## Mentoring Programmes

1. Create a pairing: `POST /api/hcm/pro/mentoring-programmes` with `mentee_employee_id`, `mentor_employee_id`, `programme_name`, `start_date`, and `meeting_frequency` (weekly|fortnightly|monthly|quarterly).
2. Log sessions: `POST /api/hcm/pro/mentoring-programmes/{id}/sessions` with `session_date`, `topics_covered`, and `action_items`.
3. The programme's `sessions_completed` counter increments automatically.

---

## Certifications

- Add: `POST /api/hcm/pro/certifications` — `certification_name`, `issuing_body`, `issue_date`, optional `expiry_date`.
- The field `days_to_expiry` is computed dynamically on every read.
- Filter upcoming expirations: `GET /api/hcm/pro/certifications?expiring_within_days=30`.
- Status is `active` or `expired` based on `expiry_date`.

---

## Career Paths

1. Create: `POST /api/hcm/pro/career-paths` with `current_role`, `target_role`, `target_timeline_months`, and a `milestones` array. Each milestone object should have a `title` key.
2. Complete a milestone: `POST /api/hcm/pro/career-paths/{id}/milestones/{idx}/complete`. When all milestones are completed the path status moves to `achieved`.

---

## Learning Activities

Track every formal and informal learning event.

- Add: `POST /api/hcm/pro/learning-activities` — `employee_id`, `title`, `activity_type` (course|workshop|conference|book|elearning|coaching|webinar), optional `plan_id`, `provider_id`, `hours_cpe`, `cost`, `currency`, `scheduled_date`.
- Complete: `POST /api/hcm/pro/learning-activities/{id}/complete` with `completed_date`.
- Filter by `employee_id`, `plan_id`, `provider_id`, `activity_type`, or `status`.

### Training Providers

Register external vendors to aggregate spend and CPE hours.

- Add provider: `POST /api/hcm/pro/training-providers`.
- Stats: `GET /api/hcm/pro/training-providers/{id}/stats` — returns `total_activities`, `completed_activities`, `completion_rate_pct`, `total_spend`, `total_cpe_hours`.

### Learning Budget

- Set: `POST /api/hcm/pro/learning-budgets` — `employee_id`, `fiscal_year`, `amount`, `currency`.
- Utilisation: `GET /api/hcm/pro/learning-budgets/{employee_id}/{year}` — returns `allocated`, `spent`, `remaining`, and `utilisation_pct`. Spend is derived dynamically from completed activities.

---

## Automated Nudges

`GET /api/hcm/pro/nudges` scans all entities for the tenant and returns a prioritised list of actionable alerts:

| Nudge Type | Trigger | Priority |
|---|---|---|
| `stale_plan` | Active plan with < 10% progress older than 60 days | high |
| `cert_expired` | Certification past its expiry_date | high |
| `cert_expiring` | Certification expiring within 30 days | medium |
| `mentoring_inactive` | Active programme with no session in 45 days | medium |

All thresholds are configurable via query parameters.

---

## Professional Development Index (PDI)

A single composite score (0–100) summarising an employee's development health.

| Component | Weight | Basis |
|---|---|---|
| Plan completion | 25% | Average progress_pct across active/completed plans |
| Skill gap closure | 25% | % assessments with no gap |
| Certifications | 20% | Active certs / total certs (50 if no data) |
| Career milestones | 20% | Milestones completed / total milestones |
| Mentoring | 10% | 100 if any active programme, else 0 |

- Compute: `GET /api/hcm/pro/pdi/{employee_id}` — returns `pdi`, `sub_scores`, and `computed_at`. Each call stores a snapshot.
- Trend: `GET /api/hcm/pro/pdi/{employee_id}/trend?last_n=8` — returns snapshots most-recent-first with a `trend` indicator (improving|declining|stable).

Run `compute_pdi` at regular intervals (e.g., end of each quarter) to build meaningful trend data.

---

## Full Report & Dashboard

- Employee report: `GET /api/hcm/pro/report/{employee_id}` — aggregates all entities for one employee in parallel.
- Tenant dashboard: `GET /api/hcm/pro/dashboard` — counts across all entities with expiry and activity summaries.

---

## API Quick Reference

```
GET  /api/hcm/pro/health
POST /api/hcm/pro/development-plans
POST /api/hcm/pro/plan-templates/{id}/apply
POST /api/hcm/pro/development-plans/{id}/clone
POST /api/hcm/pro/skill-assessments
GET  /api/hcm/pro/skill-gap-report/{employee_id}
POST /api/hcm/pro/team-skill-gap-report
POST /api/hcm/pro/skill-endorsements
POST /api/hcm/pro/feedback-requests
POST /api/hcm/pro/feedback-requests/{id}/respond
GET  /api/hcm/pro/feedback-requests/{id}/aggregate
POST /api/hcm/pro/mentoring-programmes
POST /api/hcm/pro/certifications
POST /api/hcm/pro/career-paths
POST /api/hcm/pro/learning-activities
POST /api/hcm/pro/training-providers
POST /api/hcm/pro/learning-budgets
GET  /api/hcm/pro/nudges
GET  /api/hcm/pro/pdi/{employee_id}
GET  /api/hcm/pro/pdi/{employee_id}/trend
GET  /api/hcm/pro/report/{employee_id}
GET  /api/hcm/pro/dashboard
```
