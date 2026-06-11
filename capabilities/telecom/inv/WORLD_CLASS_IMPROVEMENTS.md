# Network Inventory (telecom_inv) — World-Class Improvements

15 concrete improvements to elevate this capability from functional to production-grade.

---

## 1. Persistent Storage Backend

**Current**: All state lives in in-memory dicts (`self.assets`, `self._ne_registry`, etc.) — lost on restart.
**Fix**: Introduce an async SQLAlchemy repository layer backed by PostgreSQL. The `get_store()` stub already signals intent; wire it up. Use alembic (already present) to manage migrations.

---

## 2. Deprecation-Safe Depreciation Engine

**Current**: No depreciation calculation exists despite "depreciation" being listed in the capability description.
**Fix**: Add `async def calculate_depreciation(asset_id, tenant_id, method="straight_line")` supporting straight-line, declining-balance, and sum-of-years-digits methods. Store annual depreciation schedule in a new `InvDepreciation` model with `book_value`, `accumulated_depreciation`, and `net_book_value` fields.

---

## 3. Spare Parts & Parts Lifecycle Management

**Current**: No spare parts tracking — cannot answer "do we have a spare 100G transceiver for site KLA-01?"
**Fix**: Add `InvSparePart` model and service methods: `receive_spare_part()`, `issue_spare_part()`, `return_spare_part()`, `spare_parts_stock_report()`. Integrate with `decommission_asset()` to auto-move parts to spares pool.

---

## 4. Real IPAM with CIDR Subnet Arithmetic

**Current**: IP allocation uses `hash(pool_id)` to synthesise addresses — incorrect subnet maths, collisions possible.
**Fix**: Use Python's `ipaddress` module to maintain actual subnet pools. Track allocated/free as host sets within a `ipaddress.IPv4Network` or `IPv6Network`. Prevent double-allocation via a proper bitmap or sorted free-list.

---

## 5. Structured Audit Trail with Event Schema

**Current**: `_audit()` appends bare dicts to a list with no schema enforcement, no timestamps, and `processor="bytewax"` hardcoded.
**Fix**: Define `AuditEvent(Pydantic BaseModel)` with `event_id`, `tenant_id`, `event_type`, `reference_id`, `actor`, `timestamp`, `metadata`. Stream to the `audl` capability via its contract. Add `get_audit_trail(tenant_id, start, end, event_type)`.

---

## 6. Multi-Vendor Device Configuration Fingerprinting

**Current**: `software_version` is a free-form string with no validation or change detection.
**Fix**: Add `async def snapshot_device_config(ne_id, config_text, tenant_id)` that stores a SHA-256 fingerprint and diff against the prior snapshot. Supports detecting unauthorised config drift — feeds `telecom_sec` when integrated.

---

## 7. Network Graph Analytics

**Current**: Topology links are stored in a flat list with no graph operations.
**Fix**: Add `async def shortest_path(source_ne_id, dest_ne_id, tenant_id)` and `async def critical_path_analysis(tenant_id)` using a simple BFS/Dijkstra over `_topology_links`. Identify single points of failure (cut vertices) and report them in `dashboard_summary`.

---

## 8. Asset Lifecycle State Machine

**Current**: `update_asset_status()` accepts any transition without validating legal state transitions (e.g., jumping from "decommissioned" back to "active" is never valid).
**Fix**: Define an explicit FSM: `planning → ordered → received → tested → commissioned → active → maintenance → decommissioned`. Raise `ValueError` on illegal transitions. Expose `get_valid_next_statuses(asset_id, tenant_id)`.

---

## 9. Geographic Proximity Search

**Current**: `InvSite` stores lat/lon but there is no spatial query capability.
**Fix**: Add `async def find_sites_within_radius(lat, lon, radius_km, tenant_id)` using the Haversine formula. Add `async def nearest_spare_depot(site_id, tenant_id)` to find the closest site with available spare parts. Feeds field technician dispatch workflows.

---

## 10. Automated Reconciliation Scheduling

**Current**: `inventory_reconciliation()` is called ad-hoc with caller-supplied `discovered_data`. No scheduling.
**Fix**: Add a `ReconciliationSchedule` model and `async def schedule_reconciliation(site_id, cron_expr, tenant_id)`. On execution, invoke a pluggable `DiscoveryAdapter` (SNMP, Netconf, gNMI) and feed results into `inventory_reconciliation()`. Post results to the `mqeb` event stream.

---

## 11. Contract-Enforced Tenant Isolation at the Repository Layer

**Current**: Tenant isolation is enforced by `_enforce()` at the service layer, but the store dicts use `(tenant_id, item_id)` tuple keys. A coding error could pass the wrong `tenant_id` to `_asset_or_raise` and return another tenant's data.
**Fix**: Wrap the in-memory dicts in a `TenantScopedStore` class that binds a tenant at construction time. Each service method receives a store scoped to the calling tenant — cross-tenant leakage becomes impossible by construction.

---

## 12. Bulk Import Validation & Idempotency

**Current**: `bulk_import_assets()` silently coerces unknown `asset_type` and `status` values to defaults. Duplicate imports create duplicate keys.
**Fix**: Add a dry-run mode (`validate_only=True`) that returns a validation report without committing. Use `asset_id` as an idempotency key — re-importing the same row updates rather than duplicates. Surface per-row validation errors with row numbers and field names.

---

## 13. Deprecation & EoL Cross-Reference with Vendor Advisories

**Current**: `end_of_life_tracking()` requires manual date entry with no cross-reference to vendor advisories.
**Fix**: Add a `VendorAdvisoryAdapter` interface with a stub HTTP implementation. `async def sync_vendor_eol_dates(vendor, tenant_id)` fetches published EoL dates and auto-populates `_eol_records` for matching NEs. Add `urgency="vendor_confirmed"` level when sourced from advisory vs. manual entry.

---

## 14. Export to Industry-Standard Formats

**Current**: `export_inventory()` supports JSON and CSV only.
**Fix**: Add YANG/JSON (RFC 7951 `ietf-network` model), OpenConfig-compatible JSON, and a NetBox-compatible REST payload as export formats. These enable zero-touch import into third-party NMS/OSS platforms and satisfy operator interoperability requirements.

---

## 15. Capacity Planning with Real Utilisation Metrics

**Current**: `capacity_planning()` simulates utilisation as `len(active_circuits) * 8.0 + 20.0` — not tied to actual interface counters.
**Fix**: Define a `UtilisationAdapter` interface that pulls real SNMP/gNMI interface counter data. When `OLLAMA_BASE_URL` is set, run the utilisation time-series through a local LSTM model for demand forecasting. Fall back to the existing formula when the adapter is unavailable. Expose `confidence_interval` in the forecast output.
