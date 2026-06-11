# Network Management — User Guide

**Capability ID**: `telecom_net` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

Network operations centre capability providing fault management with alarm correlation, performance monitoring with threshold alerting, configuration change management with freeze period enforcement, SLA monitoring, NOC shift handover management, post-incident review generation, capacity trend forecasting, and SLA penalty calculation. Designed for 24×7 NOC operations with a dark-themed UI.

## Installation

```bash
pip install apg-telecom-net
```

## Provides

- `fault_management_workflow`
- `performance_management_workflow`
- `configuration_management_workflow`
- `sla_monitoring_workflow`
- `noc_operations_workflow`
- `alarm_correlation_workflow`
- `change_management_workflow`
- `net_agent_workflow`
- `pir_workflow`
- `ne_health_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `wflo`
- `moni`
- `mqeb`
- `schd`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-net/dashboard` | `telecom_net:view` | Overview |
| `/telecom-net/alarms` | `telecom_net:faults` | Fault Management |
| `/telecom-net/faults` | `telecom_net:faults` | Fault Management |
| `/telecom-net/performance` | `telecom_net:performance` | Performance |
| `/telecom-net/config-changes` | `telecom_net:config` | Configuration |
| `/telecom-net/sla` | `telecom_net:sla` | SLA |
| `/telecom-net/noc` | `telecom_net:noc` | NOC |
| `/telecom-net/topology` | `telecom_net:view` | Overview |
| `/telecom-net/correlations` | `telecom_net:faults` | Fault Management |
| `/telecom-net/escalations` | `telecom_net:escalations` | Operations |
| `/telecom-net/agents` | `telecom_net:admin` | Automation |
| `/telecom-net/settings` | `telecom_net:admin` | Administration |

---

## Usage Examples

### Raise an Alarm and Auto-Open a Fault Ticket

```python
import asyncio
from capabilities.telecom.net.service import NetworkManagementService

svc = NetworkManagementService()

result = asyncio.run(svc.fault_alert(
    ne_id="RAN-NODE-001",
    fault_type="hardware_failure",
    severity="critical",
    description="Radio unit power failure detected",
    tenant_id="acme",
))
# result["auto_ticketed"] == True for critical/major faults
print(result["alarm"]["id"], result["ticket"]["id"])
```

### Record a Performance Threshold Crossing

```python
result = asyncio.run(svc.performance_threshold_crossing(
    ne_id="CORE-SW-007",
    metric="utilisation",
    value=95.0,
    threshold=80.0,
    tenant_id="acme",
))
# result["alarm"] is set because value > threshold
print(result["severity"], result["excess_pct"])
```

### Correlate Alarms Across Domains

```python
result = asyncio.run(svc.cross_domain_correlation(
    alarm_ids=["alarm-001", "alarm-002", "alarm-003"],
    tenant_id="acme",
))
print(result["is_cross_domain_incident"], result["domain_groups"])
```

### Detect Configuration Drift

```python
result = asyncio.run(svc.detect_configuration_drift(
    ne_id="CORE-RTR-002",
    current_config="interface GigabitEthernet0/0\n ip address 10.0.0.1 255.255.255.0\n...",
    tenant_id="acme",
))
if result["drift_detected"]:
    print("Drift alarm raised:", result["alarm"]["id"])
```

### Forecast Capacity Breach

```python
result = asyncio.run(svc.capacity_trend_forecast(
    ne_id="CORE-SW-007",
    metric="utilisation",
    tenant_id="acme",
    window=20,
))
print(f"Days to breach: {result['days_to_breach']}, trend: {result['trend']}")
```

### Generate a Post-Incident Review

```python
# Resolve the ticket first
svc.resolve_fault_ticket("ticket-001", "acme", resolved_at="2026-06-11T14:30:00Z")

pir = asyncio.run(svc.generate_pir(
    ticket_id="ticket-001",
    tenant_id="acme",
    author="john.doe",
))
print(f"MTTR: {pir['mttr_minutes']} min, root cause: {pir['root_cause']}")
```

### Compute NE Health Score

```python
score = asyncio.run(svc.ne_health_score(ne_id="RAN-NODE-001", tenant_id="acme"))
print(score["health_score"], score["status"], score["colour"])
```

### SLA Penalty Calculation

```python
credit = asyncio.run(svc.sla_penalty_calculation(
    sla_id="sla-001",
    breach_duration_minutes=45.0,
    tenant_id="acme",
    penalty_rate_per_minute=2.50,
    currency="USD",
))
print(f"Penalty: {credit['penalty_amount']} {credit['currency']}")
```

### NOC Workload Analysis

```python
report = asyncio.run(svc.noc_workload_analysis(tenant_id="acme"))
for shift in report["shift_recommendations"]:
    print(shift["shift"], shift["recommended_min_headcount"])
```

### Multi-Tenant SLA Benchmark (Admin)

```python
benchmark = asyncio.run(svc.multi_tenant_sla_benchmark(admin_tenant_id="admin"))
for entry in benchmark["benchmark"]:
    print(entry["tenant_id"], entry["compliance_rate"], entry["percentile_rank"])
```

---

## Key Service Methods Reference

| Method | Type | Description |
|--------|------|-------------|
| `raise_alarm` | sync | Raise a network alarm from an NE |
| `update_alarm_status` | sync | Acknowledge, clear, or update alarm status |
| `suppress_alarm` | sync | Suppress alarm — requires approval reference |
| `fault_alert` | async | Raise alarm and auto-open ticket (critical/major) |
| `fault_correlation` | async | Correlate batch of alerts to find root events |
| `correlate_alarms` | async | Correlate specific alarm IDs |
| `cross_domain_correlation` | async | Multi-domain alarm correlation |
| `open_fault_ticket` | sync | Open fault ticket from alarm |
| `resolve_fault_ticket` | sync | Resolve and close fault ticket |
| `escalate_fault` | sync | Escalate ticket to higher tier |
| `trouble_ticket_create` | async | Create trouble ticket with priority mapping |
| `trouble_ticket_update` | async | Update ticket with work note |
| `root_cause_analysis` | async | Record RCA findings |
| `generate_pir` | async | Generate Post-Incident Review report |
| `record_performance` | sync | Record KPI metric from NE |
| `performance_threshold_crossing` | async | Process threshold crossing; auto-raise alarm |
| `performance_analytics` | async | KPI aggregation for a period |
| `capacity_trend_forecast` | async | Forecast days-to-breach via linear trend |
| `ne_health_score` | async | Composite NE health score 0–100 |
| `submit_config_change` | sync | Submit change request |
| `complete_config_change` | sync | Mark config change completed |
| `backup_config` | async | Back up NE running configuration |
| `detect_configuration_drift` | async | Detect drift from approved baseline |
| `planned_maintenance` | async | Schedule maintenance window |
| `create_maintenance_window` | async | Create alarm-suppression window |
| `close_maintenance_window` | async | Close maintenance window |
| `firmware_upgrade` | async | Schedule firmware upgrade |
| `record_sla` | sync | Record SLA measurement; flag breaches |
| `sla_penalty_calculation` | async | Compute contractual SLA penalty |
| `multi_tenant_sla_benchmark` | async | Cross-tenant SLA ranking (admin) |
| `record_noc_handover` | sync | Record NOC shift handover |
| `noc_shift_report` | async | Generate NOC shift summary |
| `noc_workload_analysis` | async | Staffing recommendations by shift |
| `network_health_dashboard` | async | Comprehensive NOC health snapshot |
| `network_compliance_report` | async | Compliance report against a standard |
| `export_network_data` | async | Export alarms and tickets (JSON/CSV) |
| `register_agent` | sync | Register automation agent |
| `validate_agent_action` | sync | Enforce agent scope/approval policies |
| `ml_network_fault_predict` | async | Ollama ML fault prediction (optional) |

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_NET_`.

| Key | Default | Description |
|-----|---------|-------------|
| `TELECOM_NET_FAULT_TTL_HOURS` | 72 | Hours before inactive alarms are archived |
| `TELECOM_NET_PERF_INTERVAL_SEC` | 300 | Performance collection interval in seconds |
| `TELECOM_NET_CHANGE_FREEZE_ENABLED` | true | Enable change freeze periods |
| `TELECOM_NET_SLA_BREACH_ALERT` | true | Emit alert on SLA breach |
| `TELECOM_NET_ROLLBACK_ENABLED` | true | Allow change rollback operations |
| `OLLAMA_BASE_URL` | unset | Enable ML fault prediction (optional) |

## Interoperability

`telecom_net` integrates with other APG capabilities through the composition engine:

```apg
use telecom_net;
```

- Alarm and performance data flows to `telecom_ana` (analytics) and `telecom_per` (KPI management)
- Config changes consume resource data from `telecom_inv`
- SLA breach events and penalty credit notes target `telecom_bil` (billing)
- PIR reports feed `comp` (regulatory SLA reporting)
- NE health scores drive the topology view component

## Further Reading

- `service.py` — Business logic (33 async + 25 sync methods)
- `models.py` — Data models (NetAlarm, NetFaultTicket, NetPerformanceRecord, NetConfigChange, NetSlaRecord, NetNocHandover, NetAgent)
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Deterministic policy engine and streaming configuration
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 enhancement proposals
