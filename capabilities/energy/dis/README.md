# Distribution Network

## Overview
Distribution Network manages the complete operational lifecycle of electricity distribution infrastructure. It provides network topology management for feeders and equipment, real-time fault detection and isolation, switching order workflows with live-network safety controls, outage recording with SAIDI/SAIFI reliability tracking, SCADA telemetry ingestion across multiple protocols, automated load balancing with voltage constraint enforcement, and advanced analytics including ML-driven fault location, self-healing network planning, ENS computation, Volt/VAR optimization, demand response dispatch, and regulatory report generation.

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
| `ml_fault_localization` | Impedance-based ML fault location from SCADA waveform samples |
| `self_healing_planning` | Graph-based optimal switching plan generation for supply restoration |
| `ens_computation` | Energy Not Supplied (ENS) calculation with financial impact |
| `volt_var_optimization` | Capacitor bank and OLTC set-point optimization to reduce losses |
| `regulatory_reporting` | ERA Kenya / Ofgem / NERC / ERC Uganda compliance report generation |
| `emergency_load_shedding` | Priority-ranked automated load shedding plan for supply emergencies |
| `cim_exchange` | IEC CIM XML export (IEC 61968/61970 DL/EQ/TP profiles) |
| `audit_integrity` | Cryptographic hash-chain verification of the audit event trail |
| `demand_response_dispatch` | NATS-driven demand response instruction dispatch to flexible loads |

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
| `mqeb` | Event streaming for fault and outage lifecycle (NATS/bytewax) |
| `geos` | Geospatial location of network elements and faults |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `scada.polling_interval_seconds` | int | 30 | SCADA polling frequency |
| `switching.approval_required` | bool | true | All switching orders require approval |
| `load_balancing.voltage_limits.min_pu` | float | 0.95 | Minimum acceptable voltage in pu |
| `load_balancing.voltage_limits.max_pu` | float | 1.05 | Maximum acceptable voltage in pu |
| `faults.auto_detect` | bool | true | Auto-raise fault from SCADA anomaly |
| `vvo.capacitor_step_voltage_pu` | float | 0.01 | Per-step voltage change from capacitor switching |
| `dr.escalation_gap_pct` | float | 20.0 | Gap % above which DR escalates to load shedding |
| `cim.profile` | str | DL | Default CIM export profile |
| `ens.penalty_rate_per_mwh` | float | 150.0 | Regulatory penalty rate USD/MWh for ENS |
| `self_healing.auto_execute` | bool | false | Auto-execute self-healing plan without confirmation |

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
| `/energy-dis/api/v1/faults/<id>/locate` | POST | ML fault location from waveform | `energy_dis:faults` |
| `/energy-dis/api/v1/faults/<id>/self-healing-plan` | POST | Generate self-healing switching plan | `energy_dis:switching` |
| `/energy-dis/api/v1/switching` | POST | Create switching order | `energy_dis:switching` |
| `/energy-dis/api/v1/switching/<id>/approve` | PUT | Approve switching | `energy_dis:switching` |
| `/energy-dis/api/v1/switching/<id>/execute` | PUT | Execute switching | `energy_dis:switching` |
| `/energy-dis/api/v1/outages` | POST | Record outage | `energy_dis:outages` |
| `/energy-dis/api/v1/outages/<id>/ens` | POST | Compute ENS and financial impact | `energy_dis:outages` |
| `/energy-dis/api/v1/scada/readings` | POST | Ingest SCADA reading | `energy_dis:scada` |
| `/energy-dis/api/v1/load-balancing` | POST | Apply load balance | `energy_dis:load_balancing` |
| `/energy-dis/api/v1/vvo` | POST | Volt/VAR optimization | `energy_dis:load_balancing` |
| `/energy-dis/api/v1/reliability` | GET | Reliability KPIs | `energy_dis:reports` |
| `/energy-dis/api/v1/reports/regulatory` | POST | Generate regulatory report | `energy_dis:reports` |
| `/energy-dis/api/v1/emergency/load-shed` | POST | Emergency load shedding plan | `energy_dis:emergency` |
| `/energy-dis/api/v1/demand-response/dispatch` | POST | Dispatch DR instruction | `energy_dis:demand_response` |
| `/energy-dis/api/v1/cim/export` | POST | Export CIM XML | `energy_dis:topology` |
| `/energy-dis/api/v1/audit/verify` | GET | Verify audit chain integrity | `energy_dis:admin` |

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
| `critical_load_protected` | protect_critical=True blocks critical infrastructure feeders in load shed | deny |
| `dr_escalation_threshold` | gap_pct > 20% triggers load shed escalation | escalate |

## Data Models
| Model | Key Fields |
|---|---|
| `Feeder` | id, tenant_id, name, substation_id, voltage_level, status, loading_pct, priority_class |
| `NetworkElement` | id, element_type, feeder_id, voltage_level, status, location_reference |
| `FaultRecord` | id, element_id, fault_type, status, detected_at, isolated_at, affected_customers |
| `SwitchingOrder` | id, element_id, operation, status, approved_by, executed_at |
| `OutageRecord` | id, feeder_id, cause, started_at, restored_at, saidi_minutes, affected_customers |
| `ScadaReading` | id, element_id, protocol, parameter, value, quality, timestamp |
| `LoadBalanceAction` | id, feeder_id, mode, load_transferred_mw, voltage_improvement_pu |
| `DisAgent` | id, tenant_id, name, runtime, role, scope, registered_at |
| `AuditEvent` | id, tenant_id, event_type, entity_id, occurred_at, payload, event_hash |

## Streaming Architecture (NATS + Bytewax)
- SCADA waveforms arrive on `scada.waveform.<element_id>` → Bytewax enrichment pipeline → `scada.enriched.*`
- Fault events: `faults.detected.*` / `faults.isolated.*` / `faults.located.*`
- Switching events: `switching.order.*` / `switching.autoplan.*` / `switching.shed_override.*`
- Outage events: `outage.started.*` / `outage.restored.*` / `outage.predicted.*`
- Demand response: `dr.dispatch.<feeder_id>` / `dr.ack.<feeder_id>`
- Reporting: `reporting.ens.*` / `reporting.regulatory.*`
- Crew safety: `crew.location.<crew_id>` / `crew.safety_alert.*`

## Streaming Events (Audit Trail)
- `network_element_registered` / `topology_updated` / `network_topology_updated`
- `fault_detected` / `fault_isolated` / `fault_restored` / `fault_location_predicted`
- `switching_order_created` / `switching_order_approved` / `switching_operation_executed`
- `self_healing_plan_generated`
- `outage_started` / `outage_restored`
- `ens_computed`
- `scada_reading_received` / `scada_configured`
- `load_balance_adjusted` / `volt_var_optimized`
- `reliability_indices_calculated` / `reliability_compliance_report_generated`
- `regulatory_report_generated`
- `emergency_load_shed_planned`
- `demand_response_dispatched`
- `cim_xml_exported`
- `audit_chain_verified`

## Edge Cases Handled
- Crew dispatch blocked until fault is isolated (safety sequencing)
- Live network switching requires explicit approval separate from order creation
- Switching execution checks that the order was approved (not just created)
- SCADA readings rejected when head-end heartbeat has expired
- Voltage constraint validation runs before any load balancing action is recorded
- SAIDI accumulated at outage-restore time, not at report generation time
- Self-healing plan requires human confirmation by default; unattended mode is opt-in per feeder
- Emergency load shedding protects critical infrastructure feeders unless explicitly overridden
- CIM XML export validates element_type to CIM class mapping; unknown types fall back to ConductingEquipment
- ENS computation falls back to 1.0 hour duration if timestamps cannot be parsed
- Regulatory compliance check applies regulator-specific SAIDI/SAIFI thresholds
- Audit chain verification detects any tampered record by re-computing SHA-256 from genesis

## Composability Notes
- Pairs with `energy_gen` for coordinated generation dispatch during network events
- Pairs with `energy_met` for consumer-side demand data fed from AMI to SCADA and DR verification
- Pairs with `energy_grd` for transmission-level topology and contingency handoff
- Geospatial element locations feed `geos` for GIS map overlays and crew geofencing
- SAIDI/SAIFI feeds regulatory compliance reporting in `comp`
- ENS records feed financial settlement in `finm`
- CIM exports feed GIS/ADMS systems in `geos` and asset management in `aset`
- DR dispatch integrates with `energy_met` AMI for real-time load verification

## World-Class Improvements
See `WORLD_CLASS_IMPROVEMENTS.md` for 15 detailed enhancement proposals covering:
ML fault localization, autonomous self-healing, predictive outage probability, NATS/Bytewax streaming,
digital twin simulation, IEC CIM alignment, Volt/VAR optimization, crew safety geofencing,
federated multi-utility state sharing, adaptive load shedding, protection coordination modeling,
NATS demand response dispatch, immutable audit ledger, ENS economic computation, and automated regulatory reporting.
