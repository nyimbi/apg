# Timetabling — World-Class Improvement Opportunities

**Capability**: `education_ttbl` | **Domain**: `education` | **Version**: `1.0.0`

---

## 1. Genetic Algorithm with Local-Search Hybrid

**Category**: Core Algorithm

**Justification**: Pure genetic algorithms plateau after ~500 generations on large instances (200+ teachers, 40+ rooms). Industry benchmarks (UniTime, FET) show a memetic GA — GA crossover/selection + local-search hill-climbing at the mutation step — reduces residual soft-constraint violations by 30–60% over pure GA on standard Curriculum-Based Course Timetabling (CB-CTT) benchmark datasets.

**Implementation**: Add `async optimise_timetable_genetic_ls()` that runs the GA loop with a configurable `local_search_rounds` parameter. Each mutated chromosome undergoes `n` rounds of 2-opt period swaps before fitness evaluation. Store the Pareto-frontier of (hard_violations=0, soft_score) solutions.

**Competitor Reference**: UniTime (open-source, used by 100+ universities) — hybrid GA + simulated annealing solver; Mimosa 2.0 (commercial) — memetic algorithm with tabu search.

---

## 2. Soft-Constraint Weighted Scoring Engine

**Category**: Constraint Modelling

**Justification**: The current system distinguishes hard vs. soft constraints by a boolean flag but accumulates no aggregate soft score. Without a numeric penalty surface, optimisation algorithms have no gradient to descend. Every production timetabling system (FET, UniTime, Mimosa) computes a per-timetable soft penalty score at each iteration.

**Implementation**: Add `async compute_soft_score(tenant_id, timetable_id) -> dict` that sums `(1 - satisfied) * weight` across all soft constraints for each entry. Cache the score on the timetable record. Expose it on `validate_timetable()` and `dashboard_summary()`.

**Competitor Reference**: FET (Free Timetabling Software) — weighted soft-constraint penalty as the primary optimisation objective.

---

## 3. Teacher Availability Matrix with Preferred-Period Encoding

**Category**: Resource Modelling

**Justification**: Institutions schedule around teacher availability (part-time staff, off-campus days, union-mandated free periods). Without an explicit availability matrix the system can only detect conflicts after the fact; it cannot pre-screen candidate slots during generation, forcing costly backtracking.

**Implementation**: Add `async set_teacher_availability(tenant_id, teacher_id, availability_matrix: dict[str, list[int]]) -> dict` where the matrix maps day → list of available period numbers. Surface as a `teacher_availability` constraint with parameters. Expose `async get_teacher_availability_gaps(tenant_id, timetable_id) -> dict` to identify under-constrained teachers.

**Competitor Reference**: Mimosa, Asc Timetables — availability grid editor is the first data-entry step before any generation run.

---

## 4. Exam Timetable Clash-Free Guarantor with Student-Group Intersection Graph

**Category**: Exam Scheduling

**Justification**: Exam timetabling is NP-hard but can be solved near-optimally by first building a graph where nodes are exams and edges represent shared students, then applying graph colouring (Welsh-Powell or DSATUR). No two adjacent exams may share a slot. Current implementation has no exam-specific logic beyond timetable_type="exam".

**Implementation**: Add `async generate_exam_schedule(tenant_id, timetable_id, exam_entries: list[dict], group_memberships: dict[str, list[str]]) -> dict`. Build the conflict graph, colour it, map colours to slots, return the assignment plan with clash-freedom proof.

**Competitor Reference**: IntelliTimetable, UniTime Exam Scheduling module — both implement graph-colouring-based exam scheduling.

---

## 5. Multi-Objective Pareto Dashboard

**Category**: Analytics / Reporting

**Justification**: Decision-makers need to compare timetable variants across competing objectives (soft-constraint satisfaction, teacher travel distance between buildings, room utilisation, substitution risk). A Pareto frontier view makes trade-offs explicit and defensible.

**Implementation**: Add `async pareto_frontier_report(tenant_id, academic_year) -> dict` that, for each published/approved timetable in the year, computes a vector of normalised metrics: `soft_score`, `avg_room_utilisation`, `avg_teacher_load_variance`, `conflict_rate`. Returns the non-dominated set and a JSON payload suitable for parallel coordinates rendering.

**Competitor Reference**: UniTime Analytics module — multi-KPI comparison view across solver runs.

---

## 6. Real-Time Conflict Heat-Map Stream

**Category**: Observability

**Justification**: Administrators need immediate visual feedback during timetable build sessions. Batch conflict reports after every `assign_entry` call are too coarse; a streaming heat-map aggregated over rolling windows enables live resolution during interactive editing sessions.

**Implementation**: Add `async conflict_heatmap(tenant_id, timetable_id) -> dict` returning a matrix `{day: {period: conflict_count}}` computed from current unresolved conflicts. Wire to the existing bytewax `apg.education.ttbl.lifecycle` stream via a `conflict_heatmap_updated` event on every `_detect_conflicts` invocation.

**Competitor Reference**: Asc Timetables — live grid highlighting conflicting cells during drag-drop scheduling.

---

## 7. Substitution Recommendation Engine (Availability + Qualification Scoring)

**Category**: Substitution Management

**Justification**: Current substitution flow requires a human to nominate the substitute. In large institutions (500+ teachers) this is a bottleneck. A ranked-list recommender that scores candidates by: (1) no conflict at that period, (2) qualification match for the subject, (3) lowest current-day load, (4) proximity of home room — reduces mean substitution assignment time from hours to minutes.

**Implementation**: Add `async recommend_substitutes(tenant_id, original_entry_id, date, top_k=5) -> dict`. Score each teacher in the tenant's pool and return a ranked list with score breakdown. Record the recommendation run in audit.

**Competitor Reference**: Substitution management modules in Veracross, iSAMS — both offer ranked substitute suggestions based on availability and qualification.

---

## 8. Academic Calendar-Aware Generation Window Enforcement

**Category**: Calendar Integration

**Justification**: Generating a timetable that spans non-teaching days (public holidays, exam periods, school breaks) wastes slots and introduces phantom conflicts. Enforcing generation within valid academic windows from the `schd` capability prevents these artefacts at the source.

**Implementation**: Add `async set_academic_calendar(tenant_id, timetable_id, teaching_days: list[str], excluded_dates: list[str]) -> dict`. Integrate with `create_time_slot` to reject slot creation on excluded dates. Add `async get_teaching_day_coverage(tenant_id, timetable_id) -> dict` to report slot distribution vs. calendar.

**Competitor Reference**: Lantiv Timetabler — calendar exclusions are first-class inputs to the scheduling engine.

---

## 9. Room Allocation Fairness Auditor

**Category**: Equity / Compliance

**Justification**: Room allocation bias (e.g., certain departments consistently getting premium rooms or early slots) can be a compliance and morale issue. Automated equity audits flag statistically significant allocation imbalances before publication.

**Implementation**: Add `async room_allocation_fairness_audit(tenant_id, timetable_id) -> dict`. Compute per-department/subject-group: mean room capacity, mean period quality score (early-day slots scored higher), Gini coefficient of room-quality distribution. Flag outliers beyond 1.5σ.

**Competitor Reference**: IntelliTimetable equitability report; UniTime solution comparison view with fairness metrics.

---

## 10. Batch Constraint Import via Structured CSV/JSON

**Category**: Usability / Onboarding

**Justification**: Initial setup of a 200-teacher institution requires hundreds of availability constraints. Manual API calls are impractical. Batch import from structured CSV (teacher_id, constraint_type, parameters JSON) reduces onboarding from weeks to hours.

**Implementation**: Add `async batch_import_constraints(tenant_id, timetable_id, records: list[dict], created_by: str) -> dict`. Validate each record against supported constraint types, accumulate errors, commit accepted records transactionally, return a per-record status report.

**Competitor Reference**: FET — bulk data import from XML/CSV is the primary onboarding path for large schools.

---

## 11. Timetable Diff and Change-Log View

**Category**: Change Management

**Justification**: When a timetable is revised mid-term (room change, teacher reassignment), affected students and teachers need a precise diff, not a full re-publication. Current `compare_timetables()` only counts deltas; it does not enumerate them.

**Implementation**: Add `async timetable_diff(tenant_id, timetable_a, timetable_b) -> dict` returning a list of `{type: added|removed|changed, entry_id, field, old_value, new_value}` change records, grouped by entity type (teacher, room, student_group). Generate a human-readable change summary.

**Competitor Reference**: Asc Timetables — full change log with before/after views for each session modification.

---

## 12. Constraint Sensitivity Analysis

**Category**: Decision Support

**Justification**: Before removing a hard constraint under pressure (e.g., "teacher X needs Fridays free"), administrators should know which other entries would be impacted. Sensitivity analysis exposes the blast radius of constraint relaxation, preventing cascading conflicts post-removal.

**Implementation**: Add `async constraint_sensitivity(tenant_id, constraint_id) -> dict`. Temporarily remove the constraint from in-memory evaluation, re-run conflict detection across all affected entries, return the count and IDs of newly introduced conflicts. Do not persist the removal.

**Competitor Reference**: UniTime — constraint relaxation preview before committing changes.

---

## 13. Student Load Balancing Validator

**Category**: Student Welfare / Compliance

**Justification**: Student groups should not have back-to-back high-cognitive-load subjects (e.g., Mathematics → Physics → Chemistry in three consecutive periods). Most national curriculum frameworks mandate minimum gaps between core subjects. The current entry assignment is subject-agnostic.

**Implementation**: Add `async validate_student_load_balance(tenant_id, timetable_id, high_load_subjects: list[str], max_consecutive: int = 2) -> dict`. For each student group, scan the daily schedule sequence and flag runs of `max_consecutive+1` or more high-load subjects in consecutive periods. Return a per-group violation list.

**Competitor Reference**: SSOS (Student Schedule Optimisation System used in Finnish schools) — mandatory cognitive-load sequencing rules.

---

## 14. Webhook / Notification Trigger Registry

**Category**: Integration / Composability

**Justification**: Downstream systems (parent portal, LMS, staff app) need real-time push notifications when a timetable is published, a substitution is assigned, or a conflict is detected. Without a webhook registry, consumers must poll; polling at scale is O(consumers × events) wasted requests.

**Implementation**: Add `async register_webhook(tenant_id, event_type, target_url, secret_token) -> dict` and `async trigger_webhooks(tenant_id, event_type, payload) -> dict`. On each audit event, call matching webhooks with HMAC-signed payloads. Support retry-with-backoff metadata.

**Competitor Reference**: Google Classroom API, Canvas LMS — webhook/push notification registries for schedule change events.

---

## 15. Timetable Version History with Rollback

**Category**: Governance / Auditability

**Justification**: Regulatory and accreditation bodies require institutions to demonstrate what timetable was in effect on any given date. A version history with point-in-time snapshots and one-click rollback satisfies this requirement and reduces the risk of irreversible edits.

**Implementation**: Add `async snapshot_timetable(tenant_id, timetable_id, label) -> dict` that stores a deep-copy of the timetable plus all entries and slots into a versioned snapshot store. Add `async list_snapshots(tenant_id, timetable_id) -> list[dict]` and `async rollback_to_snapshot(tenant_id, timetable_id, snapshot_id, approved_by) -> dict`. Rollback is approval-gated.

**Competitor Reference**: Mimosa — version history with named snapshots and administrative rollback; UniTime — solution comparison and restore from historical runs.
