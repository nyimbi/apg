# hcm_org — World-Class Improvement Opportunities

Fifteen concrete, prioritised improvements to elevate this capability from functional to production-grade.

---

## 1. Successor Identification & Succession Bench

**Gap**: No mechanism to designate or query position successors. Critical roles have `is_critical` flag but succession depth, readiness ratings, and bench strength are untracked.

**Improvement**: Add a `successors` sub-resource on positions — store `{successor_employee_id, readiness: "ready_now"|"1_year"|"2_year", nominated_by, nominated_at}`. Expose `get_succession_bench(tenant_id, position_id)` and `org_succession_risk_report(tenant_id)` that surfaces critical open positions with bench_depth == 0.

---

## 2. Effective-Date Versioning (Bi-temporal Model)

**Gap**: All mutations are last-write-wins with a single `updated_at` timestamp. There is no way to ask "what did the org look like on 2025-01-01?" or schedule future changes.

**Improvement**: Adopt a bi-temporal schema: `valid_from / valid_to` (business time) alongside `created_at / superseded_at` (transaction time). All writes create new versions; reads accept an `as_of: str` parameter. This enables future-dated restructurings, retroactive corrections, and complete audit replay.

---

## 3. Position Budgeting & Salary Band Enforcement

**Gap**: Positions carry `job_grade` but no compensation anchoring. Budget overruns during restructuring go undetected.

**Improvement**: Add `salary_band_min`, `salary_band_mid`, `salary_band_max`, and `budget_currency` to positions. Introduce `validate_position_budget(tenant_id, org_unit_id)` that aggregates FTE-weighted midpoints vs. approved headcount plan budget and returns a variance report. Reject `assign_employee_to_position` calls where the employee's compensation (cross-capability lookup) falls outside the band.

---

## 4. Org Chart Depth & Span-of-Control Policy Engine

**Gap**: `compute_span_of_control` uses hard-coded thresholds (< 4 = narrow, > 8 = wide). There is no tenant-configurable policy, and no cross-org anomaly scan.

**Improvement**: Store `SpanPolicy(min_direct, max_direct, max_hierarchy_depth)` per tenant. Add `audit_org_structure(tenant_id)` that returns all policy violations — over-spanned managers, under-spanned managers, units exceeding max depth — as a structured finding list with severity ratings.

---

## 5. Circular-Dependency Detection for Reporting Lines

**Gap**: `create_reporting_line` has no cycle guard. An employee can be made their own indirect manager through a chain of `direct` lines.

**Improvement**: Before inserting a new reporting line, traverse the graph upward from `manager_employee_id` to confirm `employee_id` is not already an ancestor. Use iterative DFS with a `seen` set; raise `ValueError("reporting_cycle_detected")` on violation.

---

## 6. Bulk Import / Export (CSV / JSON-LD)

**Gap**: No batch-load path. Standing up a full org chart requires N+1 sequential API calls, making migrations from legacy HRIS systems impractical.

**Improvement**: Add `bulk_import_org_units(tenant_id, records: list[dict])` and `bulk_import_positions(tenant_id, records: list[dict])` that validate, deduplicate, and upsert in a single atomic operation. Return a `BulkImportResult` with per-row status. Add `export_org_chart(tenant_id, format: "json"|"csv")` for downstream consumption.

---

## 7. Role-Based Access Control (RBAC) Integration

**Gap**: The service accepts `tenant_id` but performs no caller identity or permission check. Any caller can delete org units or approve restructurings.

**Improvement**: Introduce a `permission_checker: Callable[[str, str, str], Awaitable[bool]]` injection point (actor_id, resource_type, action). Gate destructive operations (`delete_org_unit`, `approve_headcount_plan`, `update_restructuring`) behind permission assertions. Return `PermissionError` with structured codes when denied.

---

## 8. Org Hierarchy Caching with Invalidation

**Gap**: `_compute_headcount` and `get_org_chart` perform full recursive scans on every call. At 1 000+ units this becomes O(n²) in the worst case.

**Improvement**: Maintain an in-process `_hierarchy_cache: dict[str, HierarchyNode]` keyed by `(tenant_id, unit_id)`. Invalidate affected paths on `create_org_unit`, `move_org_unit`, `assign_employee_to_position`, and `vacate_position`. Expose `invalidate_hierarchy_cache(tenant_id)` for explicit reset. Add cache hit/miss metrics to `health_check`.

---

## 9. Event Bus Integration (Outbox Pattern)

**Gap**: `_emit` appends to an in-memory list. Events are lost on restart and cannot drive downstream capabilities (e.g., payroll, talent) in real-time.

**Improvement**: Replace the in-memory list with an outbox table pattern. Persist events to a `hcm_org_events` table via `AsyncSession`. Introduce a background relay task that polls the outbox and publishes to a configurable message broker (Kafka topic or HTTP webhook). Guarantee at-least-once delivery with idempotency keys.

---

## 10. Position Genealogy & Reclassification History

**Gap**: When a position is reclassified (job grade change, FTE change, title rename), only the current state is retained. Historical comparisons for pay-equity audits are impossible.

**Improvement**: Track `position_history: list[PositionSnapshot]` keyed by `(position_id, effective_date)`. Each `update_position` that modifies grade, FTE, or title creates a new snapshot instead of overwriting. Expose `get_position_history(tenant_id, position_id)` returning the full timeline.

---

## 11. Ghost / Duplicate Position Detection

**Gap**: The API allows creating positions with duplicate `(org_unit_id, title, job_grade)` combinations, which can inflate headcount counts and distort analytics.

**Improvement**: Add a uniqueness check in `create_position` against active positions in the same unit with matching title+grade. Return a `409 Conflict` with the existing record ID. Provide `find_duplicate_positions(tenant_id)` for bulk remediation.

---

## 12. Org Unit Merge Operation

**Gap**: Restructurings can record which units are affected, but there is no atomic "merge unit A into unit B" operation. Callers must manually reassign positions and update parent references.

**Improvement**: Add `merge_org_units(tenant_id, source_unit_id, target_unit_id, effective_date)` that atomically: (a) re-parents all child units of source to target, (b) moves all positions from source to target, (c) terminates source unit. Wrap in a transaction-like rollback list so failures leave state clean.

---

## 13. Org Chart Diff for Restructuring Simulations

**Gap**: Restructurings describe affected entities by ID lists but provide no before/after visual diff. Approvers cannot see the structural impact before signing off.

**Improvement**: Add `simulate_restructuring(tenant_id, restructuring_id)` that returns `{before: OrgChartSnapshot, after: OrgChartSnapshot, added_units, removed_units, moved_units, headcount_delta}`. This turns the approval step from a leap of faith into an informed decision.

---

## 14. Cross-Capability Composition Hooks

**Gap**: The capability operates in isolation. Position fills, vacations, and restructurings have no outbound hooks to trigger downstream HCM workflows (onboarding, offboarding, payroll recalculation).

**Improvement**: Define a `CapabilityHookRegistry` with `register_hook(event_type, async_handler)`. Core events (`position_filled`, `position_vacated`, `restructuring_completed`) call all registered handlers with the event payload. Ship default no-op handlers and document the hook contract for composability with `hcm_recruit`, `hcm_payroll`, and `hcm_talent`.

---

## 15. Comprehensive Test Coverage (CI-Ready)

**Gap**: No tests exist in `tests/`. The `_compute_headcount` recursion, cycle detection in `move_org_unit`, and restructuring state machine are all untested, creating silent regression risk.

**Improvement**: Create `tests/ci/test_hcm_org_service.py` with pytest fixtures for: hierarchy construction, headcount roll-up, move-with-cycle detection, position lifecycle (open → filled → vacated → abolished), reporting line cycle guard, bulk import idempotency, and restructuring state machine transitions. Target ≥ 90% line coverage. Add `tests/ci/test_hcm_org_api.py` for Flask blueprint integration tests using `pytest-httpserver`.
