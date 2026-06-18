# Organizational Management (hcm_org)

Capability for managing organisational structure: org chart, positions, reporting lines, span of control, headcount planning, and restructuring.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/org/health | Health check |
| GET | /api/hcm/org/describe | Capability contract |
| GET | /api/hcm/org/units | List org units |
| GET | /api/hcm/org/units/{id} | Get org unit |
| POST | /api/hcm/org/units | Create org unit |
| PUT | /api/hcm/org/units/{id} | Update org unit |
| PUT | /api/hcm/org/units/{id}/move | Move org unit |
| DELETE | /api/hcm/org/units/{id} | Delete org unit |
| GET | /api/hcm/org/chart | Full org chart |
| GET | /api/hcm/org/positions | List positions |
| GET | /api/hcm/org/positions/{id} | Get position |
| POST | /api/hcm/org/positions | Create position |
| PUT | /api/hcm/org/positions/{id} | Update position |
| PUT | /api/hcm/org/positions/{id}/assign | Assign employee |
| DELETE | /api/hcm/org/positions/{id} | Delete position |
| GET | /api/hcm/org/reporting-lines | List reporting lines |
| POST | /api/hcm/org/reporting-lines | Create reporting line |
| GET | /api/hcm/org/restructurings | List restructurings |
| POST | /api/hcm/org/restructurings | Create restructuring |
| PUT | /api/hcm/org/restructurings/{id} | Update restructuring |
| DELETE | /api/hcm/org/restructurings/{id} | Delete restructuring |
| GET | /api/hcm/org/analytics | Org analytics |
| GET | /api/hcm/org/dashboard | Dashboard |
| GET | /api/hcm/org/audit-events | Audit trail |

## World-Class Enhancements (v2.0)

**I1. Succession Bench** — `get_succession_bench(tenant_id, position_id)` + `org_succession_risk_report` with readiness ratings and bench-depth scoring [Talent Risk]

**I2. Bi-temporal Versioning** — `valid_from/valid_to` + `created_at/superseded_at` on all entities; reads accept `as_of: str` for point-in-time queries and future-dated restructurings [Audit / History]

**I3. Position Budget & Salary Band Enforcement** — `salary_band_min/mid/max` on positions; `validate_position_budget` returns FTE-weighted variance vs. approved plan; assignment rejects out-of-band compensation [Compensation Governance]

**I4. Span-of-Control Policy Engine** — tenant-scoped `SpanPolicy(min_direct, max_direct, max_hierarchy_depth)`; `audit_org_structure` returns structured findings with severity ratings [Governance]

**I5. Circular-Dependency Guard** — iterative DFS ancestor check in `create_reporting_line`; raises `ValueError("reporting_cycle_detected")` before any write [Data Integrity]

**I6. Bulk Import / Export** — `bulk_import_org_units`, `bulk_import_positions` with atomic upsert and per-row `BulkImportResult`; `export_org_chart(format="json"|"csv")` [Migration / Integration]

**I7. RBAC Integration** — injectable `permission_checker: Callable[[actor_id, resource_type, action], Awaitable[bool]]`; destructive operations gated with structured `PermissionError` codes [Security]

**I8. Hierarchy Cache with Invalidation** — in-process `_hierarchy_cache` keyed by `(tenant_id, unit_id)`, invalidated on mutations; `invalidate_hierarchy_cache(tenant_id)` for explicit reset; cache metrics in `health_check` [Performance]

**I9. Outbox Event Bus** — replace in-memory `_emit` list with `hcm_org_events` outbox table; background relay to Bytewax topic or HTTP webhook with at-least-once delivery and idempotency keys [Reliability / Composability]

**I10. Position Reclassification History** — `update_position` on grade/FTE/title creates a `PositionSnapshot` instead of overwriting; `get_position_history(tenant_id, position_id)` returns full timeline [Pay Equity Audit]

**I11. Duplicate Position Detection** — uniqueness check on `(org_unit_id, title, job_grade)` at creation returning `409 Conflict`; `find_duplicate_positions(tenant_id)` for bulk remediation [Data Quality]

**I12. Atomic Org Unit Merge** — `merge_org_units(tenant_id, source_unit_id, target_unit_id, effective_date)` re-parents children, moves positions, terminates source in a single rolled-back-on-failure transaction [Restructuring]

**I13. Restructuring Diff Simulation** — `simulate_restructuring(tenant_id, restructuring_id)` returns `{before, after, added_units, removed_units, moved_units, headcount_delta}` before approval [Change Management]

**I14. Cross-Capability Composition Hooks** — `CapabilityHookRegistry.register_hook(event_type, async_handler)`; fires on `position_filled`, `position_vacated`, `restructuring_completed` for downstream `hcm_recruit`, `hcm_payroll`, `hcm_talent` [Composability]

**I15. CI Test Suite** — `tests/ci/test_hcm_org_service.py` (hierarchy, headcount roll-up, cycle detection, position lifecycle, bulk import idempotency, restructuring state machine, ≥90% line coverage) + `tests/ci/test_hcm_org_api.py` (Flask blueprint integration via pytest-httpserver) [Quality]

## New Methods

### `simulate_restructuring` — preview structural impact before approval

```python
svc = OrgManagementService(tenant_id="acme")

diff = await svc.simulate_restructuring(
    tenant_id="acme",
    restructuring_id="rst_01j..."
)
# diff = {
#   "before": OrgChartSnapshot,
#   "after": OrgChartSnapshot,
#   "added_units": [...],
#   "removed_units": [...],
#   "moved_units": [...],
#   "headcount_delta": -12,
# }
print(f"Net headcount change: {diff['headcount_delta']}")
```

### `merge_org_units` — atomically collapse two units during a restructure

```python
result = await svc.merge_org_units(
    tenant_id="acme",
    source_unit_id="unit_engineering_east",
    target_unit_id="unit_engineering",
    effective_date="2026-07-01",
)
# Atomically: re-parents children, migrates positions, terminates source.
# Rolls back cleanly on any failure.
print(f"Merged {result['positions_moved']} positions into {result['target_unit_id']}")
```

### `org_succession_risk_report` — surface critical roles with empty bench

```python
risk = await svc.org_succession_risk_report(tenant_id="acme")
# Returns list of critical positions where bench_depth == 0
for item in risk["critical_gaps"]:
    print(f"{item['position_title']} (unit: {item['org_unit_id']}) — no successor identified")
```
