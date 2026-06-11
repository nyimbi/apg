# Succession Planning (hcm_scp) — World-Class Improvements

## 1. Development Plan Generation
Auto-generate personalised Individual Development Plans (IDPs) for pool members based on gap analysis between current readiness assessment scores and target role competency profiles. IDPs should include recommended courses, stretch assignments, mentoring relationships, and a time-boxed milestone schedule. Integrate with the LMS/training capability when available via composability bridge.

## 2. Flight Risk Scoring
Compute a flight-risk score for each successor using tenure, engagement survey results (if available), recent performance trend, compensation band position, and external market demand signals. Surfaces high-risk successors in the coverage report so HR can prioritise retention actions before a gap materialises.

## 3. Succession Depth Scoring per Role
Replace the binary "covered / uncovered" metric with a numeric succession depth score: (ready_now * 3) + (ready_in_1_year * 1.5) + (developing * 0.5), normalised to a 0–10 scale. Provides nuanced risk segmentation — a role with three "developing" successors is still riskier than one with two "ready_now" successors.

## 4. Competency Gap Heatmap
Cross-tabulate the competency requirements of critical roles against the assessed competency levels of named successors. Render the delta as a structured dict keyed by `(role_id, successor_id, competency_id)` for downstream visualisation. Drives targeted L&D spend instead of boilerplate training assignments.

## 5. Nine-Box Movement Tracking
Record longitudinal nine-box placements and compute quadrant movement vectors across review cycles (e.g., `enigma → high_potential`, `core_employee → star`). Movement velocity and direction are leading indicators of talent health. Expose as `get_nine_box_movement_history(employee_id)`.

## 6. Bench Strength Index
Aggregate talent pool readiness across the org into a single Bench Strength Index (BSI): `(ready_now + 0.5 * ready_in_1_year) / total_members * 100`. Report BSI by department, function, and overall. Enables board-level succession health discussion without individual name exposure.

## 7. Scenario Simulation ("What-If")
Allow planners to run a what-if simulation that temporarily removes an incumbent (e.g., models sudden departure) and re-scores succession coverage and bench strength index without persisting changes. Returns a simulation snapshot dict. Supports tabletop exercises and board-requested stress tests.

## 8. Retention Risk Alerts
Emit structured alert events when: (a) a "ready_now" successor has been in pool > 18 months without role movement, (b) succession depth score for a critical role drops below a configurable threshold, or (c) a nine-box "star" has not been assessed in > 12 months. Plugs into the intel/alerts capability.

## 9. Successor Diversity Metrics
Track and report gender, generation, and geographic diversity of successor slates for critical roles. Exposes `succession_diversity_report()` returning representation breakdown per role tier and overall. Supports DEI commitments without storing sensitive PII beyond what's already in employee records.

## 10. Automated Readiness Recalculation
Provide `recalculate_readiness(tenant_id, employee_id)` that pulls the most recent performance review scores, nine-box placement, IDP completion rate, and 360 feedback summary (when integrated) to produce an updated readiness recommendation. Surfaces discrepancies between manager-assessed and algorithmically-derived readiness for calibration.

## 11. Cross-Pool Visibility (Dual-Pool Members)
Allow an employee to appear in multiple pools with independent readiness levels per pool (since readiness for VP Finance vs. VP Operations may differ). Track pool-specific readiness and development progress independently rather than sharing a single readiness label.

## 12. Succession Plan Expiry & Review Cadence
Attach a `review_due_date` to each succession scenario and critical role. Expose `get_overdue_reviews(tenant_id)` to surface stale plans. Auto-transition scenarios past due date from `active` to `review_required`, preventing organisations from running on outdated plans.

## 13. Role Risk Registry Integration
Compute a composite role risk score combining: `impact_if_vacant` weight, succession depth score, time_to_fill_estimate_days, and flight risk of the incumbent. Rank critical roles by composite risk to produce a prioritised remediation backlog. Designed for export to risk management systems.

## 14. Batch Assessment Import
Provide `bulk_create_readiness_assessments(tenant_id, assessments: list[dict])` that validates, deduplicates, and commits multiple assessments atomically in a single call. Returns a batch result with per-record success/failure. Eliminates N+1 API calls during annual talent calibration cycles.

## 15. External Candidate Pipeline Integration
Model external candidates as named successor entries with `source: external` and `candidate_pool_id` references, allowing organisations to blend internal and external succession pipelines in a single scenario. Exposes `list_external_successors(tenant_id, role_id)` and produces coverage reports that distinguish internal bench from external pipeline depth.
