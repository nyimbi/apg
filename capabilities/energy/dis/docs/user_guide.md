# Distribution Network — User Guide

**Capability ID**: `energy_dis` | **Domain**: `energy` | **Version**: `1.1.0`

## Description

Distribution Network (`energy_dis`) manages the complete operational lifecycle of electricity distribution infrastructure. Core operations cover network topology management, real-time fault detection and isolation, switching order workflows with live-network safety controls, outage recording with SAIDI/SAIFI reliability tracking, and SCADA telemetry ingestion. Advanced features include ML-driven fault localization, autonomous self-healing switching plan generation, Energy Not Supplied (ENS) computation, Volt/VAR optimization, demand response dispatch via NATS, emergency load shedding, IEC CIM XML exchange, and regulator-ready compliance reports.

---

## Installation

```bash
pip install apg-energy-dis
```

---

## Quick Start

```python
from apg_energy_dis.service import DistributionNetworkService
import asyncio

svc = DistributionNetworkService(tenant_id="utility_ke", actor_id="operator_001")

# Register a feeder
svc.register_feeder(
    feeder_id="FDR-001",
    tenant_id="utility_ke",
    name="Nairobi CBD Feeder 1",
    substation_id="SUB-NRBI-01",
    voltage_level="11kV",
    normal_capacity_mw=15.0,
    emergency_capacity_mw=18.0,
)

# Report and isolate a fault
async def demo():
    fault = await svc.fault_report(
        location="Kenyatta Ave junction",
        fault_type="phase_to_ground",
        voltage=11.0,
        customers_affected=320,
        reported_by="SCADA_AUTO",
    )
    fault_id = fault["id"]

    await svc.fault_isolation(fault_id, isolation_points=["CB-101", "CB-102"])
    health = await svc.distribution_health_check()
    print(health)

asyncio.run(demo())
```

---

## Core Operations

### Network Topology

#### Register a Feeder
```python
svc.register_feeder(
    feeder_id="FDR-002",
    tenant_id="utility_ke",
    name="Industrial Park Feeder",
    substation_id="SUB-INDPK",
    voltage_level="33kV",
    normal_capacity_mw=40.0,
    emergency_capacity_mw=48.0,
)
```

#### Register a Network Element
```python
svc.register_element(
    element_id="TX-001",
    tenant_id="utility_ke",
    element_type="transformer",   # transformer | switch | cable | busbar | capacitor
    name="11/0.4kV DTR 001",
    feeder_id="FDR-001",
    voltage_level="11kV",
    location_reference="Grid ref: 37M 256780 9863240",
)
```

#### Export Network as IEC CIM XML
```python
result = await svc.export_cim_xml(profile="DL")
# result["cim_xml"] contains IEC 61968-13 conformant XML
# profile options: DL (Distribution Level) | EQ (Equipment) | TP (Topology) | SSH
```

---

### Fault Management

#### Report a Fault
```python
fault = await svc.fault_report(
    location="Mombasa Rd km 12",
    fault_type="phase_to_ground",   # phase_to_ground | phase_to_phase | three_phase | open_circuit | high_impedance
    voltage=11.0,
    customers_affected=850,
    reported_by="field_crew_03",
    cause="tree_contact",
)
fault_id = fault["id"]
```

#### ML Fault Localization from SCADA Waveforms
```python
# waveform_samples arrive via NATS subject scada.waveform.<element_id> in production
waveform_samples = [
    {"voltage_kv": 10.8, "current_ka": 0.32, "timestamp": "2026-06-11T08:00:00Z"},
    {"voltage_kv": 10.2, "current_ka": 0.35, "timestamp": "2026-06-11T08:00:01Z"},
    {"voltage_kv": 9.5,  "current_ka": 0.41, "timestamp": "2026-06-11T08:00:02Z"},
]
location = await svc.predict_fault_location(
    fault_id=fault_id,
    waveform_samples=waveform_samples,
    confidence_threshold=0.75,
)
# Returns: estimated_distance_km, avg_impedance_ohm, confidence, recommended_action
```

#### Isolate a Fault
```python
await svc.fault_isolation(fault_id, isolation_points=["CB-101", "CB-102", "CB-103"])
```

#### Generate Self-Healing Switching Plan
```python
plan = await svc.compute_self_healing_plan(
    fault_id=fault_id,
    available_tie_points=["TIE-A", "TIE-B", "TIE-C"],
    max_switching_operations=6,
    auto_execute=False,  # True requires unattended policy
)
# plan["best_plan"] contains ranked restoration path with estimated customers restored
```

#### Dispatch Crew and Restore
```python
svc.dispatch_crew(fault_id=fault_id, tenant_id="utility_ke", crew_id="CREW-07")
await svc.fault_restoration(
    fault_id=fault_id,
    restoration_time="2026-06-11T10:30:00Z",
    restored_customers=850,
    restoration_method="repair",
)
```

---

### Switching Orders

```python
# Create → Approve → Execute workflow (live-network safe)
order_id = "SW-2026-001"
svc.create_switching_order(
    order_id=order_id,
    tenant_id="utility_ke",
    element_id="CB-205",
    operation="open",       # open | close | trip | reclose | lock_out | normalise
    requested_by="op_001",
    purpose="Planned maintenance isolation",
)
svc.approve_switching_order(order_id, tenant_id="utility_ke", approved_by="supervisor_01")
svc.execute_switching_order(order_id, tenant_id="utility_ke", network_live=True)
```

For ad-hoc operations (creates and executes in one call):
```python
result = await svc.switching_operation(
    switch_id="CB-205",
    action="close",
    authorised_by="supervisor_01",
    reason="Post-fault restoration close",
    work_order_id="WO-2026-445",
)
```

---

### Outage Management

#### Record an Outage
```python
svc.record_outage(
    outage_id="OUT-001",
    tenant_id="utility_ke",
    feeder_id="FDR-001",
    cause="cable_fault",    # cable_fault | weather | equipment_failure | third_party | planned
    started_at="2026-06-11T08:05:00Z",
    restoration_strategy="repair",
    affected_customers=320,
)
```

#### Restore and Compute ENS
```python
svc.restore_outage(outage_id="OUT-001", tenant_id="utility_ke", saidi_minutes=85.0)

ens = await svc.compute_ens(
    outage_id="OUT-001",
    avg_load_mw=2.4,
    tariff_schedule={"residential": 80.0, "commercial": 120.0, "industrial": 100.0},
    penalty_rate_per_mwh=150.0,
)
# Returns: ens_mwh, estimated_revenue_loss_usd, regulatory_penalty_usd, total_financial_impact_usd
```

---

### SCADA Integration

```python
# Ingest a real-time reading (DNP3, IEC 61850, Modbus, etc.)
svc.process_scada_reading(
    reading_id="RDG-001",
    tenant_id="utility_ke",
    element_id="TX-001",
    protocol="IEC_61850",
    parameter="voltage_kv",
    value=10.95,
    unit="kV",
    quality="good",
    timestamp="2026-06-11T08:00:00Z",
    heartbeat_valid=True,
)
```

In production, SCADA adapters publish raw readings to `scada.raw.<protocol>.<element_id>` on NATS.
A Bytewax pipeline applies dead-band filtering, linear interpolation, and unit normalization,
then republishes to `scada.enriched.*`.

---

### Load Balancing and Volt/VAR Optimization

#### Apply Load Balance
```python
svc.apply_load_balance(
    action_id="LB-001",
    tenant_id="utility_ke",
    feeder_id="FDR-001",
    mode="automatic",
    action_type="load_transfer",
    load_transferred_mw=2.5,
    voltage_improvement_pu=0.02,
    voltage_within_limits=True,
)
```

#### Volt/VAR Optimization
```python
vvo = await svc.optimize_volt_var(
    feeder_id="FDR-001",
    voltage_readings=[
        {"element_id": "TX-001", "voltage_pu": 0.93, "reactive_kvar": 120.0},
        {"element_id": "TX-002", "voltage_pu": 0.96, "reactive_kvar": 80.0},
    ],
    target_voltage_pu=1.0,
    max_capacitor_steps=4,
)
# vvo["recommended_setpoints"] gives per-element capacitor step instructions
# vvo["estimated_loss_reduction_pct"] gives expected technical loss reduction
```

---

### Demand Response Dispatch

```python
dr_result = await svc.dispatch_demand_response(
    feeder_id="FDR-001",
    target_reduction_mw=3.0,
    window_minutes=30,
    participant_ids=["PART-001", "PART-002", "PART-003"],
)
# Publishes to NATS dr.dispatch.FDR-001
# If gap_pct > 20%, escalation_needed=True → call emergency_load_shed
```

---

### Emergency Load Shedding

```python
plan = await svc.emergency_load_shed(
    deficit_mw=12.0,
    protect_critical=True,  # never shed critical_infrastructure feeders
)
# Returns shed_feeders ranked by priority, switching_order_ids ready for execution
# After operator confirmation, execute each switching order via execute_switching_order()
```

---

### Reliability KPIs and Reporting

#### SAIDI/SAIFI Calculation
```python
indices = await svc.saidi_saifi_calculation(period="2026-06")
# Returns: saidi_minutes, saifi_interruptions, caidi_minutes
```

#### Regulatory Compliance Report
```python
report = await svc.generate_regulatory_report(
    period="2026-06",
    regulator="ERA_Kenya",   # ERA_Kenya | ERC_Uganda | Ofgem | NERC | ESCOM_Malawi | ZESCO_Zambia
    output_format="json",    # json | csv | xlsx
)
# report["compliant"] = True/False vs regulator-specific SAIDI/SAIFI limits
```

#### Network Analytics Dashboard
```python
analytics = await svc.network_analytics(period="2026-06")
# Returns: reliability_indices, outage_statistics, topology_changes, switching_operations,
#          total_feeders, total_elements
```

---

### Audit Trail Integrity

```python
integrity = await svc.verify_audit_chain()
# integrity["verified"] = True if no tampering detected
# integrity["first_broken_link"] = None if chain is intact, or index of broken event
```

---

## Reliability Indices Reference

| Index | Formula | Good Benchmark |
|---|---|---|
| SAIDI | Σ(customer·interruption·minutes) / total customers served | < 60 min/yr (Ofgem), < 1440 min/yr (ERA Kenya) |
| SAIFI | Σ(customer·interruptions) / total customers served | < 1.0 (Ofgem), < 24.0 (ERA Kenya) |
| CAIDI | SAIDI / SAIFI | Average duration per interruption |
| ENS | avg_load_MW × duration_hours | Minimize; drives regulatory penalty |

---

## Supported Protocols and Standards

| Area | Standards |
|---|---|
| SCADA protocols | DNP3, IEC 61850, Modbus, IEC 60870-5-101/104 |
| Data model exchange | IEC CIM (61968-13 DL, 61970 EQ/TP profiles) |
| Reliability indices | IEEE 1366, IEC 61511, ERA Kenya Grid Code |
| Protection coordination | IEC 60255 inverse-time overcurrent |
| Streaming | NATS JetStream + Bytewax stateful pipelines |

---

## Interoperability

`energy_dis` integrates with other APG capabilities through the composition engine:

```apg
use energy_dis;
```

Key cross-capability integrations:
- `energy_gen` — coordinated generation dispatch during network events
- `energy_met` — AMI demand data fed to SCADA; real-time DR verification
- `energy_grd` — transmission topology handoff for contingency coordination
- `geos` — GIS overlays for element locations, fault sites, and crew geofencing
- `comp` — regulatory SAIDI/SAIFI filing from reliability KPIs
- `finm` — ENS records fed to financial settlement
- `aset` — CIM exports for asset lifecycle management

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables
prefixed with `ENERGY_DIS_`.

| Environment Variable | Default | Description |
|---|---|---|
| `ENERGY_DIS_SCADA_POLLING_INTERVAL` | `30` | SCADA polling interval (seconds) |
| `ENERGY_DIS_SWITCHING_APPROVAL_REQUIRED` | `true` | Require approval before execution |
| `ENERGY_DIS_VOLTAGE_MIN_PU` | `0.95` | Minimum acceptable voltage (pu) |
| `ENERGY_DIS_VOLTAGE_MAX_PU` | `1.05` | Maximum acceptable voltage (pu) |
| `ENERGY_DIS_ENS_PENALTY_RATE` | `150.0` | Regulatory penalty rate (USD/MWh) |
| `ENERGY_DIS_SELF_HEALING_AUTO` | `false` | Auto-execute self-healing plans |
| `ENERGY_DIS_DR_ESCALATION_GAP_PCT` | `20.0` | DR gap % threshold for load-shed escalation |

---

## Further Reading

- `service.py` — Complete business logic implementation
- `models.py` — Data models (`Feeder`, `FaultRecord`, `SwitchingOrder`, etc.)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic request/response schemas
- `capability_contract.py` — Business rule enforcement and supported value enumerations
- `README.md` — Quick reference and composability map
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed world-class enhancement proposals
- `tests/` — Unit, integration, and composition test suites
