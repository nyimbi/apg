# Network Inventory

**Capability ID**: `telecom_inv` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Physical and logical network inventory management covering asset commissioning and decommissioning, circuit provisioning, IP address management (IPAM), network topology documentation, and automated reconciliation with field audit results. Provides the single source of truth for all network resources.

## Installation

```bash
pip install apg-telecom-inv
```

## Provides

- `asset_inventory_workflow`
- `circuit_management_workflow`
- `ipam_workflow`
- `topology_documentation_workflow`
- `inventory_reconciliation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `nlpc`

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

## Key Service Methods

- `describe()`
- `evaluate()`
- `commission_asset()`
- `update_asset_status()`
- `decommission_asset()`
- `provision_circuit()`
- `update_circuit_status()`
- `allocate_ip_block()`
- `release_ip_block()`
- `record_topology()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_inv` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_inv;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_INV_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
