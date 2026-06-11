# Network Inventory — User Guide

**Capability ID**: `telecom_inv` | **Domain**: `telecom` | **Version**: `1.1.0`

## Description

Physical and logical network inventory management covering asset commissioning, circuit provisioning, IP address management (IPAM), network topology documentation, spare parts lifecycle, asset depreciation, device configuration drift detection, and automated reconciliation. Provides the single source of truth for all network resources.

---

## Installation

```bash
pip install apg-telecom-inv
```

---

## Quick Start

```python
import asyncio
from capabilities.telecom.inv.service import NetworkInventoryService

svc = NetworkInventoryService()

async def main():
    # Commission a router
    asset = svc.commission_asset(
        asset_id="rtr-001",
        tenant_id="acme",
        asset_type="router",
        serial_number="SN-CISCO-001",
        vendor="cisco",
        model="ASR-9001",
        location="NAI-DC-01",
        commissioned_at="2025-01-15T08:00:00Z",
    )
    print(asset)

asyncio.run(main())
```

---

## Core Workflows

### 1. Asset Commissioning & Lifecycle

Commission assets and move them through the FSM-enforced lifecycle:

```
planning → ordered → received → tested → commissioned → active → maintenance → decommissioned
```

```python
# Commission
svc.commission_asset(asset_id="sw-001", tenant_id="acme", asset_type="switch",
                     serial_number="SN-JNP-007", vendor="juniper", model="EX4300",
                     location="MOM-DC-02", commissioned_at="2025-03-01T09:00:00Z")

# Move through lifecycle (FSM-validated)
await svc.asset_lifecycle_transition(
    asset_id="sw-001",
    tenant_id="acme",
    target_status="active",
    actor="ops-engineer@acme.com",
)

# Decommission — requires approval
await svc.asset_lifecycle_transition(
    asset_id="sw-001",
    tenant_id="acme",
    target_status="decommissioned",
    actor="ops-engineer@acme.com",
    approval_reference="CHG-2025-0042",
)
```

Illegal transitions (e.g. `active → ordered`) raise `ValueError` with valid next states listed.

---

### 2. Asset Depreciation

Three methods are supported:

| Method | Flag |
|--------|------|
| Straight-line | `straight_line` (default) |
| Double-declining balance | `declining_balance` |
| Sum-of-years digits | `sum_of_years_digits` |

```python
schedule = await svc.calculate_depreciation(
    asset_id="rtr-001",
    tenant_id="acme",
    method="declining_balance",
    useful_life_years=7,
    salvage_value=5000.0,
)
for year in schedule["schedule"]:
    print(year["year"], year["net_book_value"])
```

---

### 3. Spare Parts Management

Track hardware spares at sites and issue them to network elements.

```python
# Receive stock
await svc.receive_spare_part(
    part_id="trx-100g-lc-001",
    part_type="transceiver",
    vendor="finisar",
    model="FTLC1151RDPL",
    serial_number="FSN-TRX-9912",
    site_id="NAI-DC-01",
    tenant_id="acme",
    quantity=4,
)

# Issue to a NE for a work order
await svc.issue_spare_part(
    part_id="trx-100g-lc-001",
    issued_to_ne_id="rtr-001",
    work_order="WO-2025-1188",
    tenant_id="acme",
    quantity=1,
)

# Stock report
report = await svc.spare_parts_stock_report(tenant_id="acme", site_id="NAI-DC-01")
print(report["low_stock_parts"])   # parts with qty <= 2
```

---

### 4. Circuit Provisioning & Activation

```python
# Create circuit
circuit = await svc.circuit_create(
    circuit_id="ckt-001",
    endpoints=["rtr-001", "rtr-002"],
    bandwidth="10Gbps",
    service_type="ethernet",
    tenant_id="acme",
    protection="1+1",
)

# Activate after end-to-end test
activated = await svc.circuit_activation(
    circuit_id="ckt-001",
    activated_by="noc@acme.com",
    tenant_id="acme",
)
```

---

### 5. IP Address Management (IPAM)

```python
# Seed a pool with known free addresses (optional — service can synthesise)
svc._ip_pool_free["pool-mgmt"] = ["10.10.1.10", "10.10.1.11", "10.10.1.12"]

# Allocate
alloc = await svc.ip_address_allocation(
    pool_id="pool-mgmt",
    host_name="rtr-001.mgmt",
    purpose="management",
    tenant_id="acme",
)
print(alloc["ip_address"])  # 10.10.1.10

# Release
await svc.ip_release(ip_address=alloc["ip_address"], tenant_id="acme")

# Utilisation report
report = await svc.ip_utilisation_report(tenant_id="acme")
print(report["utilisation_pct"])
```

---

### 6. Network Element Registration & Topology

```python
# Register NEs
await svc.register_ne("rtr-001", "router", "cisco", "ASR-9001", "NAI-DC-01",
                       "10.0.0.1", tenant_id="acme")
await svc.register_ne("rtr-002", "router", "juniper", "MX-204", "MOM-DC-02",
                       "10.0.0.2", tenant_id="acme")

# Add adjacency link
await svc.topology_update("rtr-001", "rtr-002", "100GE", tenant_id="acme", bandwidth_mbps=100_000)

# Critical-path / SPOF analysis
graph = await svc.network_graph_critical_paths(tenant_id="acme")
print(graph["cut_vertices"])  # NE IDs whose removal partitions the network
```

---

### 7. Device Configuration Drift Detection

```python
config_v1 = "interface GE0/0\n ip address 10.1.1.1/24\n no shutdown"
snap1 = await svc.snapshot_device_config("rtr-001", config_v1, tenant_id="acme")

# Later — config has changed
config_v2 = "interface GE0/0\n ip address 10.1.1.2/24\n no shutdown"
snap2 = await svc.snapshot_device_config("rtr-001", config_v2, tenant_id="acme")

if snap2["drift_detected"]:
    print(f"Drift on rtr-001: +{snap2['lines_added']} / -{snap2['lines_removed']} lines")
```

---

### 8. Geographic Proximity Search

```python
# Nairobi CBD approx. coordinates
nearby = await svc.find_sites_within_radius(
    lat=-1.2921, lon=36.8219, radius_km=50.0, tenant_id="acme"
)
for site in nearby["sites"]:
    print(site["site_name"], site["distance_km"])
```

---

### 9. End-of-Life Tracking & Vendor Advisory Sync

```python
# Manual EoL entry
await svc.end_of_life_tracking(
    ne_id="rtr-001",
    eol_date="2027-12-31",
    tenant_id="acme",
    replacement_plan="Migrate to ASR-9002 by Q3 2027",
)

# Sync from vendor advisory URL (JSON: [{model, eol_date, replacement_plan}])
sync_result = await svc.vendor_eol_sync(
    vendor="cisco",
    tenant_id="acme",
    advisory_url="https://advisories.example.com/cisco/eol.json",
)
print(sync_result["matched_ne_count"])
```

---

### 10. Inventory Reconciliation

```python
# Data from discovery scan (SNMP / Netconf)
discovered = [
    {"ne_id": "rtr-001", "status": "active"},
    {"ne_id": "rtr-099", "status": "active"},   # unknown to inventory
]
result = await svc.inventory_reconciliation(discovered, tenant_id="acme")
print(result["missing_from_inventory"])   # ["rtr-099"]
print(result["unknown_in_network"])       # NEs in inventory not found on-net
```

---

### 11. Capacity Planning

```python
forecast = await svc.capacity_planning(
    ne_id="rtr-001",
    forecast_months=24,
    tenant_id="acme",
    growth_rate_pct=12.0,
)
print(f"Warning threshold at month: {forecast['warning_threshold_month']}")
print(f"Critical threshold at month: {forecast['critical_threshold_month']}")
```

---

### 12. Reports & Export

```python
# Inventory report
report = await svc.inventory_report(ne_type="router", location="NAI", tenant_id="acme")

# Dashboard summary
dashboard = svc.dashboard_summary(tenant_id="acme")

# Export to CSV
export = await svc.export_inventory(tenant_id="acme", format="csv")

# Compliance check
compliance = await svc.inventory_compliance_check(tenant_id="acme")
print(compliance["compliance_rate_pct"])
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-inv/dashboard` | `telecom_inv:view` | Overview |
| `/telecom-inv/assets` | `telecom_inv:assets` | Physical |
| `/telecom-inv/assets/<id>` | `telecom_inv:assets` | Physical |
| `/telecom-inv/circuits` | `telecom_inv:circuits` | Logical |
| `/telecom-inv/ipam` | `telecom_inv:ipam` | Logical |
| `/telecom-inv/topology` | `telecom_inv:topology` | Topology |
| `/telecom-inv/sites` | `telecom_inv:assets` | Physical |
| `/telecom-inv/reconciliation` | `telecom_inv:reconciliation` | Operations |
| `/telecom-inv/spare-parts` | `telecom_inv:assets` | Physical |
| `/telecom-inv/depreciation` | `telecom_inv:assets` | Physical |

---

## Provides

- `asset_inventory_workflow`
- `asset_lifecycle_fsm_workflow`
- `circuit_management_workflow`
- `ipam_workflow`
- `topology_documentation_workflow`
- `inventory_reconciliation_workflow`
- `spare_parts_workflow`
- `depreciation_workflow`
- `config_fingerprinting_workflow`
- `vendor_eol_sync_workflow`
- `geographic_proximity_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `nlpc`
- `moni`
- `mqeb`
- `geos`

---

## Interoperability

```apg
use telecom_inv;
```

`telecom_inv` feeds resource availability to `telecom_pro` (provisioning resource reservation) and `telecom_ord` (order feasibility). Site coordinates feed `geos`. Circuit data feeds `telecom_net` for topology-aware alarm correlation. Depreciation schedules feed the finance capability for asset accounting. Config snapshots feed `telecom_sec` for compliance audits.

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_INV_`.

| Key | Default | Description |
|-----|---------|-------------|
| `assets.serial_number_required` | `true` | Serial mandatory for all assets |
| `assets.location_required` | `true` | Location mandatory |
| `ipam.vrf_required` | `true` | VRF must be specified |
| `reconciliation.auto_discovery_enabled` | `false` | Network discovery integration |
| `governance.unauthorised_decommission_denied` | `true` | Approval always required |
| `depreciation.default_useful_life_years` | `10` | Default asset useful life |
| `spare_parts.low_stock_threshold` | `2` | Alert when on-hand qty ≤ this value |

---

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Dataclass data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and supported enum values
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 production-grade enhancements
- `README.md` — Quick reference
