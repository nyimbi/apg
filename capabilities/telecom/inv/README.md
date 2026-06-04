# Network Inventory

## Overview
Physical and logical network inventory management covering asset commissioning and decommissioning, circuit provisioning, IP address management (IPAM), network topology documentation, and automated reconciliation with field audit results. Provides the single source of truth for all network resources.

## Capability ID
`telecom_inv`

## Provides
- asset_inventory_workflow: Commission, track, and decommission physical assets
- circuit_management_workflow: Logical circuit provisioning and lifecycle
- ipam_workflow: IP block allocation, VRF management, and release
- topology_documentation_workflow: Network topology capture per domain
- inventory_reconciliation_workflow: Discrepancy detection and approval
- network_resource_query: Point-in-time resource availability queries
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

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-inv/assets | GET/POST | Asset console | telecom_inv:assets |
| /telecom-inv/circuits | GET/POST | Circuit management | telecom_inv:circuits |
| /telecom-inv/ipam | GET/POST | IP address management | telecom_inv:ipam |
| /telecom-inv/topology | GET/POST | Topology viewer | telecom_inv:topology |
| /telecom-inv/sites | GET/POST | Site registry | telecom_inv:assets |
| /telecom-inv/reconciliation | GET/POST | Reconciliation console | telecom_inv:reconciliation |

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

## Data Models
- InvAsset: id, tenant_id, asset_type, serial_number, vendor, model, location, status
- InvCircuit: id, tenant_id, circuit_type, a_end, z_end, capacity, status
- InvIpBlock: id, tenant_id, ip_version, prefix, prefix_length, block_type, vrf, allocated_to
- InvTopology: id, tenant_id, topology_type, domain, name, nodes, edges
- InvSite: id, tenant_id, site_name, site_type, latitude, longitude, address, region
- InvReconciliation: id, tenant_id, asset_id, discrepancy_description, approval_reference, status
- InvAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- asset_commissioned, asset_decommissioned, circuit_provisioned, circuit_decommissioned
- ip_block_allocated, ip_block_released, topology_updated, discrepancy_detected
- reconciliation_approved, inv_agent_registered

## Edge Cases Handled
- Dual-stack assets stored as ip_version=dual_stack, not two separate records
- Decommissioned assets remain in inventory with status=decommissioned (not deleted)
- IP block release sets allocated_to=None but preserves block definition for reuse
- Circuit a_end and z_end are free-form references to allow cross-domain endpoints
- Topology nodes/edges are stored as JSON strings for schema flexibility

## Composability Notes
Provides resource availability data to telecom_pro (provisioning resource reservation) and telecom_ord (order feasibility checking). Site coordinates feed geos for geographic network planning. Circuit data feeds telecom_net for topology-aware alarm correlation.
