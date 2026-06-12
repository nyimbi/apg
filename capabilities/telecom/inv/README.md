# Network Inventory

## Overview
Physical and logical network inventory management covering asset commissioning and decommissioning, circuit provisioning, IP address management (IPAM), network topology documentation, spare parts lifecycle, asset depreciation, device configuration fingerprinting, and automated reconciliation with field audit results. Provides the single source of truth for all network resources.

## Capability ID
`telecom_inv`

## Provides
- asset_inventory_workflow: Commission, track, and decommission physical assets with FSM-enforced lifecycle transitions
- circuit_management_workflow: Logical circuit provisioning and lifecycle (activation, decommission)
- ipam_workflow: IP block allocation, host-level IP pool management, VRF tracking, and release
- topology_documentation_workflow: Network topology capture and graph-based critical-path analysis
- inventory_reconciliation_workflow: Discrepancy detection, approval, and scheduled reconciliation
- spare_parts_workflow: Receive, issue, and stock-report spare hardware at sites
- depreciation_workflow: Straight-line, declining-balance, and sum-of-years-digits schedules
- config_fingerprinting_workflow: SHA-256 config snapshots and drift detection per NE
- vendor_eol_sync_workflow: Auto-populate EoL records from vendor advisory feeds
- network_resource_query: Point-in-time resource availability queries and geographic proximity search
- inv_agent_workflow: Inventory automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Asset change audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| nlpc | Inventory search |
| moni | Asset health monitoring |
| mqeb | Event streaming |
| geos | Geographic site coordinates |

## Configuration
| Key | Description |
|-----|-------------|
| assets.serial_number_required | Serial mandatory for all assets |
| assets.location_required | Location mandatory |
| ipam.vrf_required | VRF must be specified for all allocations |
| reconciliation.auto_discovery_enabled | Network discovery integration |
| governance.unauthorised_decommission_denied | Approval always required |
| depreciation.default_useful_life_years | Default asset useful life (default: 10) |
| spare_parts.low_stock_threshold | Alert when stock ≤ this value (default: 2) |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-inv/assets | GET/POST | Asset console | telecom_inv:assets |
| /telecom-inv/circuits | GET/POST | Circuit management | telecom_inv:circuits |
| /telecom-inv/ipam | GET/POST | IP address management | telecom_inv:ipam |
| /telecom-inv/topology | GET/POST | Topology viewer | telecom_inv:topology |
| /telecom-inv/sites | GET/POST | Site registry | telecom_inv:assets |
| /telecom-inv/reconciliation | GET/POST | Reconciliation console | telecom_inv:reconciliation |
| /telecom-inv/spare-parts | GET/POST | Spare parts stock | telecom_inv:assets |
| /telecom-inv/depreciation | GET | Depreciation schedules | telecom_inv:assets |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | missing context | deny |
| asset_type_not_supported | unknown asset type | deny |
| serial_number_required | missing serial | deny |
| decommission_requires_approval | no approval reference | deny |
| ip_block_vrf_required | no VRF set | deny |
| unauthorised_decommission_denied | agent decommissions | deny |
| cross_tenant_inventory_denied | cross-tenant scope | deny |
| asset_lifecycle_fsm | illegal state transition | deny |

## Asset Lifecycle FSM
```
planning → ordered → received → tested → commissioned → active → maintenance → decommissioned
```
Transitions not in this graph are rejected. Decommission always requires `approval_reference`.

## Data Models
- InvAsset: id, tenant_id, asset_type, serial_number, vendor, model, location, status, commissioned_at
- InvCircuit: id, tenant_id, circuit_type, a_end, z_end, capacity, status, provisioned_at
- InvIpBlock: id, tenant_id, ip_version, prefix, prefix_length, block_type, vrf, allocated_to, allocated_at
- InvTopology: id, tenant_id, topology_type, domain, name, nodes, edges, recorded_at
- InvSite: id, tenant_id, site_name, site_type, latitude, longitude, address, region
- InvReconciliation: id, tenant_id, asset_id, discrepancy_description, approval_reference, status
- InvAgent: id, tenant_id, name, runtime, role, scope

## Key Service Methods
| Method | Description |
|--------|-------------|
| `commission_asset()` | Commission a physical asset |
| `update_asset_status()` | Raw status update (no FSM check) |
| `asset_lifecycle_transition()` | FSM-enforced lifecycle transition |
| `decommission_asset()` | Decommission with approval |
| `calculate_depreciation()` | Straight-line / declining-balance / SYD schedule |
| `receive_spare_part()` | Add spare parts to stock |
| `issue_spare_part()` | Issue spares to an NE for a work order |
| `spare_parts_stock_report()` | Stock levels with low-stock alerts |
| `provision_circuit()` | Provision logical circuit |
| `circuit_activation()` | Activate a provisioned circuit |
| `allocate_ip_block()` | Allocate IPAM block |
| `ip_address_allocation()` | Allocate host IP from pool |
| `register_ne()` | Register network element |
| `snapshot_device_config()` | SHA-256 config snapshot & drift detection |
| `find_sites_within_radius()` | Haversine proximity search |
| `network_graph_critical_paths()` | Articulation-point / SPOF analysis |
| `vendor_eol_sync()` | Sync EoL dates from vendor advisory feed |
| `capacity_planning()` | Monthly utilisation forecast |
| `inventory_reconciliation()` | Discovered vs. inventory diff |
| `inventory_report()` | Filtered inventory report |
| `dashboard_summary()` | Tenant-scoped KPI dashboard |
| `export_inventory()` | JSON or CSV export |

## Streaming Events
- asset_commissioned, asset_decommissioned, asset_lifecycle_{status}
- circuit_provisioned, circuit_activated, circuit_decommissioned
- ip_block_allocated, ip_block_released, ip_address_allocated, ip_address_released
- topology_updated, topology_link_updated, topology_discovered
- discrepancy_detected, reconciliation_approved, inventory_reconciliation_run
- spare_part_received, spare_part_issued
- depreciation_calculated
- config_snapshot_stored, config_drift_detected
- vendor_eol_synced, eol_tracked, eol_critical_alert
- ne_registered, ne_software_updated
- critical_path_analysis_run, capacity_planning_run
- inv_agent_registered

## Edge Cases Handled
- Dual-stack assets stored as ip_version=dual_stack, not two separate records
- Decommissioned assets remain in inventory with status=decommissioned (not deleted)
- IP block release sets allocated_to=None but preserves block definition for reuse
- Circuit a_end and z_end are free-form references to allow cross-domain endpoints
- Topology nodes/edges are stored as JSON strings for schema flexibility
- Asset lifecycle FSM prevents illegal state transitions including resurrection of decommissioned assets
- Config drift detection uses set-based line diffing — order-insensitive for identical configs
- Haversine proximity search handles sites at the poles and antimeridian correctly

## Composability Notes
Provides resource availability data to telecom_pro (provisioning resource reservation) and telecom_ord (order feasibility checking). Site coordinates feed geos for geographic network planning. Circuit data feeds telecom_net for topology-aware alarm correlation. Depreciation data feeds finance capability for asset accounting. Config snapshots feed telecom_sec for security compliance checks.

## World-Class Enhancements (v2.0)

1. **Persistent Storage** — async SQLAlchemy + PostgreSQL repository layer; alembic migrations
2. **Depreciation Engine** — straight-line, declining-balance, and SYD schedules; full annual schedule output
3. **Spare Parts Lifecycle** — receive/issue/return flows; auto-pool from decommissioned assets; low-stock alerts
4. **Real IPAM** — `ipaddress` module subnet arithmetic; bitmap free-list; collision-safe host allocation
5. **Structured Audit Trail** — `AuditEvent` Pydantic schema; tenant/actor/timestamp fields; streams to `audl`
6. **Config Fingerprinting** — SHA-256 snapshot per NE; set-based line diff; drift events to `telecom_sec`
7. **Network Graph Analytics** — BFS/Dijkstra shortest path; Tarjan articulation-point SPOF detection
8. **Asset Lifecycle FSM** — explicit 8-state graph; illegal transitions rejected; `get_valid_next_statuses()`
9. **Geographic Proximity Search** — Haversine `find_sites_within_radius()`; nearest spare depot lookup
10. **Reconciliation Scheduling** — `ReconciliationSchedule` model; cron-driven; pluggable SNMP/Netconf/gNMI adapters
11. **Tenant-Scoped Store** — `TenantScopedStore` wrapper; cross-tenant data leakage impossible by construction
12. **Bulk Import Idempotency** — dry-run `validate_only` mode; asset_id as upsert key; per-row error reporting
13. **Vendor EoL Cross-Reference** — `VendorAdvisoryAdapter`; `vendor_eol_sync()` auto-populates EoL records
14. **Industry-Standard Export** — YANG/RFC 7951, OpenConfig JSON, and NetBox-compatible REST payload formats
15. **Capacity Planning with Real Metrics** — `UtilisationAdapter` for SNMP/gNMI counters; local LSTM forecast via Ollama

## New Methods

### `calculate_depreciation` — Asset depreciation schedule

```python
svc = NetworkInventoryService()
result = await svc.calculate_depreciation(
    asset_id="asset-uuid",
    tenant_id="acme",
    method="declining_balance",   # straight_line | declining_balance | sum_of_years_digits
    useful_life_years=7,
    salvage_value=5000.0,
)
# result["schedule"] → list of {year, annual_depreciation, accumulated_depreciation, net_book_value}
```

### `find_sites_within_radius` — Haversine proximity search

```python
result = await svc.find_sites_within_radius(
    lat=-1.2921,
    lon=36.8219,
    radius_km=50.0,
    tenant_id="acme",
)
# result["sites"] → sorted by distance_km ascending; each entry includes site metadata + distance_km
```

### `network_graph_critical_paths` — Single point of failure detection

```python
result = await svc.network_graph_critical_paths(tenant_id="acme")
# result["cut_vertices"] → list of NE IDs whose removal partitions the network
# result["component_count"] → number of disconnected graph components
# Feed into telecom_net for proactive resilience planning
```
