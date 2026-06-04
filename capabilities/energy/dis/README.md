# Distribution Network

## Overview
Distribution Network manages the complete operational lifecycle of electricity distribution infrastructure. It provides network topology management for feeders and equipment, real-time fault detection and isolation, switching order workflows with live-network safety controls, outage recording with SAIDI/SAIFI reliability tracking, SCADA telemetry ingestion across multiple protocols, and automated load balancing with voltage constraint enforcement.

## Capability ID
`energy_dis`

## Provides
| Service | Description |
|---|---|
| `network_topology_management` | Register and maintain feeders and network elements |
| `fault_detection_and_isolation` | Detect, classify, isolate and track network faults |
| `outage_restoration` | Record outages, track restoration, compute SAIDI/SAIFI |
| `switching_order_management` | Create, approve and execute switching orders safely |
| `scada_integration` | Ingest real-time SCADA readings via DNP3, IEC 61850, Modbus etc. |
| `load_balancing` | Apply and record load balancing actions with voltage validation |
| `reliability_kpis` | Compute and report SAIDI, SAIFI, CAIDI |
| `distribution_reporting` | Outage reports, reliability summaries, fault history |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and role-based access |
| `audl` | Audit trail for switching, fault, and outage records |
| `mten` | Multi-tenant network data isolation |
| `conf` | Runtime configuration for voltage limits and protocols |
| `ntfy` | Fault and outage notifications to field crews |
| `wflo` | Switching order and crew dispatch approval workflows |
| `moni` | SCADA real-time monitoring integration |
| `schd` | Scheduled reliability report generation |
| `mqeb` | Event streaming for fault and outage lifecycle |
| `geos` | Geospatial location of network elements and faults |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `scada.polling_interval_seconds` | int | 30 | SCADA polling frequency |
| `switching.approval_required` | bool | true | All switching orders require approval |
| `load_balancing.voltage_limits.min_pu` | float | 0.95 | Minimum acceptable voltage in pu |
| `load_balancing.voltage_limits.max_pu` | float | 1.05 | Maximum acceptable voltage in pu |
| `faults.auto_detect` | bool | true | Auto-raise fault from SCADA anomaly |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-dis/api/v1/dashboard` | GET | Dashboard summary | `energy_dis:view` |
| `/energy-dis/api/v1/feeders` | GET | List feeders | `energy_dis:topology` |
| `/energy-dis/api/v1/feeders` | POST | Register feeder | `energy_dis:topology` |
| `/energy-dis/api/v1/elements` | POST | Register element | `energy_dis:topology` |
| `/energy-dis/api/v1/faults` | GET | List faults | `energy_dis:faults` |
| `/energy-dis/api/v1/faults` | POST | Report fault | `energy_dis:faults` |
| `/energy-dis/api/v1/faults/<id>/isolate` | PUT | Isolate fault | `energy_dis:faults` |
| `/energy-dis/api/v1/faults/<id>/dispatch-crew` | PUT | Dispatch crew | `energy_dis:faults` |
| `/energy-dis/api/v1/switching` | POST | Create switching order | `energy_dis:switching` |
| `/energy-dis/api/v1/switching/<id>/approve` | PUT | Approve switching | `energy_dis:switching` |
| `/energy-dis/api/v1/switching/<id>/execute` | PUT | Execute switching | `energy_dis:switching` |
| `/energy-dis/api/v1/outages` | POST | Record outage | `energy_dis:outages` |
| `/energy-dis/api/v1/scada/readings` | POST | Ingest SCADA reading | `energy_dis:scada` |
| `/energy-dis/api/v1/load-balancing` | POST | Apply load balance | `energy_dis:load_balancing` |
| `/energy-dis/api/v1/reliability` | GET | Reliability KPIs | `energy_dis:reports` |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `fault_type_supported` | fault_type not in supported list | deny |
| `switching_approval_required` | execute without approval | deny |
| `live_network_switching_requires_approval` | live network, no approval | deny |
| `fault_isolation_before_repair` | crew dispatch before isolation | deny |
| `voltage_within_limits` | voltage outside min/max pu | deny |
| `scada_heartbeat_required` | heartbeat_valid=False | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `outage_cause_supported` | cause not in supported list | deny |
| `privileged_dis_agent_requires_human_approval` | agent switching without human approval | deny |

## Data Models
| Model | Key Fields |
|---|---|
| `Feeder` | id, tenant_id, name, substation_id, voltage_level, status, loading_pct |
| `NetworkElement` | id, element_type, feeder_id, voltage_level, status, location_reference |
| `FaultRecord` | id, element_id, fault_type, status, detected_at, isolated_at, affected_customers |
| `SwitchingOrder` | id, element_id, operation, status, approved_by, executed_at |
| `OutageRecord` | id, feeder_id, cause, started_at, restored_at, saidi_minutes, affected_customers |
| `ScadaReading` | id, element_id, protocol, parameter, value, quality, timestamp |
| `LoadBalanceAction` | id, feeder_id, mode, load_transferred_mw, voltage_improvement_pu |

## Streaming Events
- `network_element_registered` / `topology_updated`
- `fault_detected` / `fault_isolated`
- `switching_order_created` / `switching_order_approved` / `switching_operation_executed`
- `outage_started` / `outage_restored`
- `scada_reading_received`
- `load_balance_adjusted`
- `reliability_kpi_calculated`

## Edge Cases Handled
- Crew dispatch blocked until fault is isolated (safety sequencing)
- Live network switching requires explicit approval separate from order creation
- Switching execution checks that the order was approved (not just created)
- SCADA readings rejected when head-end heartbeat has expired
- Voltage constraint validation runs before any load balancing action is recorded
- SAIDI accumulated at outage-restore time, not at report generation time

## Composability Notes
- Pairs with `energy_gen` for coordinated generation dispatch during network events
- Pairs with `energy_met` for consumer-side demand data fed from AMI to SCADA
- Pairs with `energy_grd` for transmission-level topology and contingency handoff
- Geospatial element locations feed `geos` for GIS map overlays
- SAIDI/SAIFI feeds regulatory compliance reporting in `comp`
