# World-Class Improvements: Professional Development (hcm_pro)

## Overview

15 high-impact improvements to elevate the PRO capability from functional CRUD to a production-grade professional development platform.

---

## 1. Learning Activity Registry

**Current gap:** Development plans exist as objective lists with no linkage to concrete learning activities (courses, workshops, books, e-learning).

**Improvement:** Add a `learning_activities` store. Each activity carries `type` (course|workshop|conference|book|elearning|coaching), `provider`, `hours_cpe`, `cost`, `completion_status`, and a link to a `development_plan_id`. Enables ROI analysis per plan.

**New method:** `add_learning_activity`, `list_learning_activities`, `complete_learning_activity`

---

## 2. 360-Degree Feedback Integration

**Current gap:** Skill assessments are single-rater (assessed_by), hiding rater bias and providing no peer signal.

**Improvement:** Add a `feedback_requests` store supporting multi-rater collection (self, peer, manager, skip-level). Aggregate scores per skill using weighted averaging. Surface a `consensus_level` field (high/medium/low) based on rater variance.

**New method:** `request_360_feedback`, `submit_feedback_response`, `aggregate_360_results`

---

## 3. Individual Development Plan Templates

**Current gap:** Every development plan is created from scratch with no institutional knowledge reuse.

**Improvement:** Add a `plan_templates` store keyed by role or department. Templates carry default objectives, focus_areas, and recommended skill targets. `create_development_plan` accepts an optional `template_id` to seed the plan.

**New method:** `create_plan_template`, `list_plan_templates`, `apply_template_to_plan`

---

## 4. Skill Endorsements

**Current gap:** Assessments are point-in-time and manager-driven. Peer expertise is invisible.

**Improvement:** Add a `skill_endorsements` store. Peers endorse an employee's skill at a level with optional evidence. Weight endorsements by endorser seniority. Surface `endorsement_count` and `highest_endorsed_level` on assessments.

**New method:** `endorse_skill`, `list_endorsements`, `get_endorsement_summary`

---

## 5. Certification Renewal Workflow

**Current gap:** Certifications expire silently. There is no structured renewal tracking distinct from a plain update.

**Improvement:** Add explicit `initiate_renewal`, `complete_renewal` transitions on certifications. Track `renewal_attempts` list with dates and outcomes. Emit `certification_renewal_initiated` and `certification_renewed` events for downstream alerting.

**New method:** `initiate_certification_renewal`, `complete_certification_renewal`

---

## 6. Mentoring Programme Effectiveness Scoring

**Current gap:** Mentoring sessions are logged but there is no measure of programme quality or mentee progress.

**Improvement:** After each session, capture `mentee_satisfaction` (1–5) and `mentor_rating` (1–5). Compute a rolling `effectiveness_score` on the programme record. Surface in the dashboard as `avg_mentoring_effectiveness`.

**New method:** `rate_mentoring_session`, `get_programme_effectiveness`

---

## 7. Bulk Skill Gap Analysis Across Teams

**Current gap:** Gap reports are per-employee only; team-wide patterns require N separate API calls.

**Improvement:** Add `get_team_skill_gap_report(tenant_id, employee_ids)` that gathers all assessments in parallel via `asyncio.gather`, computes per-skill gap prevalence, identifies the top-N most critical gaps, and surfaces a `heat_map` dict keyed by skill category.

**New method:** `get_team_skill_gap_report`

---

## 8. Career Path Matching & Recommendations

**Current gap:** Career paths are manually defined. There is no mechanism to suggest paths based on existing role transitions in the tenant's data.

**Improvement:** Analyse completed career paths to extract role transition pairs with typical timelines and milestone patterns. Given an employee's current role, recommend matching templates ranked by `success_rate` (paths that reached `achieved` status).

**New method:** `recommend_career_paths`, `get_role_transition_stats`

---

## 9. CPE / Continuing Education Credit Tracking

**Current gap:** Certifications track expiry but not the continuing professional education credits required for renewal.

**Improvement:** Add `cpe_requirements` (hours per cycle, cycle_months) per certification. Track `cpe_records` — each record ties a `learning_activity_id` to a certification and logs CPE hours earned. Surface `cpe_hours_remaining` on the certification record.

**New method:** `log_cpe_credit`, `get_cpe_status`

---

## 10. Automated Nudge / Alert Generation

**Current gap:** Stale plans, approaching certification expiries, and overdue mentoring sessions require manual monitoring.

**Improvement:** Add `generate_nudges(tenant_id)` that scans all entities and returns a prioritised list of actionable alerts: plans with progress < 10% and age > 60 days, certifications expiring within 30 days, mentoring programmes with no session in 45 days, milestones overdue by timeline estimate.

**New method:** `generate_nudges`

---

## 11. Competency Framework Linkage

**Current gap:** Skills exist in isolation. There is no grouping into competency frameworks (e.g., leadership framework, engineering ladder) used for promotion decisions.

**Improvement:** Add a `competency_frameworks` store and `competency_profiles` that group skills with minimum proficiency targets by role. Employees can be evaluated against a profile to produce a `readiness_score` (0–100) for a target role.

**New method:** `create_competency_framework`, `evaluate_against_framework`

---

## 12. Learning Budget Tracking

**Current gap:** Training costs are recorded on learning activities but there is no budget envelope per employee or department.

**Improvement:** Add a `learning_budgets` store (employee or department scope, fiscal year, currency, amount). When a learning activity with a cost is added, deduct from the applicable budget and surface `budget_remaining`. Reject activities exceeding the remaining budget (configurable hard/soft limit).

**New method:** `set_learning_budget`, `get_budget_utilisation`

---

## 13. Development Plan Cloning

**Current gap:** Employees in similar roles repeat plan authoring work each year with no carryover mechanism.

**Improvement:** `clone_development_plan(tenant_id, source_plan_id, new_plan_year, employee_id)` creates a new draft plan pre-populated from the source plan's objectives, focus_areas, and target_role_id but resets progress and status. Useful for annual rollover.

**New method:** `clone_development_plan`

---

## 14. External Training Provider Catalogue

**Current gap:** Learning activities are free-text, leading to inconsistent naming, duplicate records, and inability to aggregate spend by provider.

**Improvement:** Add a `training_providers` store with `name`, `website`, `specialisations`, `average_rating`. When logging a learning activity, optionally link to a `provider_id`. Enable `list_learning_activities(provider_id=...)` to aggregate per-provider spend and completion rates.

**New method:** `add_training_provider`, `list_training_providers`, `get_provider_stats`

---

## 15. Professional Development Index (PDI)

**Current gap:** There is no single composite metric summarising an employee's overall professional development health.

**Improvement:** Compute a `professional_development_index` (0–100) from weighted sub-scores:
- Plan completion rate (25%)
- Skill gap closure rate (25%)
- Active certifications vs. target (20%)
- Career path milestone progress (20%)
- Mentoring engagement (10%)

Surface in `professional_development_report` and on the dashboard. Trend over quarters when historical snapshots exist.

**New method:** `compute_pdi`, `get_pdi_trend`
