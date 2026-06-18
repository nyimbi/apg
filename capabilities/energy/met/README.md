# Smart Metering & AMI

## Overview

Smart Metering & AMI (`energy_met`) manages the full lifecycle of advanced metering infrastructure: meter registration, interval data collection, tamper detection with evidence workflows, remote connect/disconnect with approval controls, demand response coordination, data quality flagging, and AMI head-end connectivity monitoring. v2.0 adds load disaggregation, fraud scoring, bi-directional flow tracking, predictive health, carbon tracking, and standards-based interoperability.

## Capability ID
`energy_met`

## Provides
| Service | Description |
|---|---|
| `meter_registry` | Register and maintain smart meter inventory by type, technology and customer |
| `ami_head_end_management` | Monitor AMI head-end connectivity ratio and protocol health |
| `interval_data_collection` | Collect, batch-validate, and store interval readings with quality flags |
| `tamper_detection` | Detect, classify, and investigate tamper events with auto-disconnect support |
| `remote_connect_disconnect` | Issue and track remote commands with approval controls |
| `demand_response_coordination` | Manage DR events, track opt-outs, record actual reductions, OpenADR 2.0b |
| `data_quality_management` | Flag and resolve reading quality issues |
| `meter_data_export` | Export interval data for billing and market settlement (IEC 61968-9, CSV/JSON) |
| `fraud_scoring` | ML-driven non-technical loss detection with configurable alert thresholds |
| `meter_health_scoring` | Predictive health index from comm failures, firmware age, and tamper history |
| `carbon_tracking` | Per-meter CO₂e footprint from grid emission factors (IFRS S2 / GHG Protocol) |
| `bidirectional_metering` | Import/export register tracking and net metering reconciliation for DER |
| `tou_tariff_engine` | TOU/CPP interval bucketing for direct billing feed to `energy_bil` |
| `outage_detection` | Meter-cluster communication loss → outage boundary inference → `energy_dis` |
| `streaming_bridge` | MQTT/Bytewax CloudEvent publication for real-time analytics pipelines |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and permission checks |
| `audl` | Audit trail for tamper events and remote commands |
| `mten` | Multi-tenant meter data isolation |
| `conf` | Head-end protocol and interval configuration |
| `ntfy` | Tamper alerts, prepayment credit alerts, and DR notifications |
| `wflo` | Disconnect approval and tamper investigation workflows |
| `moni` | AMI head-end health monitoring |
| `mqeb` | Event streaming for tamper and command lifecycle |
| `schd` | Scheduled DR events and batch read jobs |
| `intel` | Threat detection feed from tamper events and security logs |
| `energy_dis` | Outage event publication from meter cluster loss detection |
| `energy_bil` | Interval data and TOU buckets for consumption billing |
| `energy_grd` | DR coordination for system-level demand management |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `readings.retention_days` | int | 730 | Interval data retention period |
| `commands.retry_limit` | int | 3 | Max command retry attempts |
| `commands.approval_required_for_disconnect` | bool | true | Disconnect requires approval |
| `demand_response.opt_out_allowed` | bool | true | Customers can opt out of DR |
| `demand_response.notification_required` | bool | true | Notify customers before DR event |
| `fraud.risk_score_threshold` | float | 0.7 | Alert threshold for fraud risk score |
| `streaming.broker_type` | str | — | `mqtt` or `bytewax` when bridge enabled |
| `carbon.default_region_id` | str | — | Fallback grid region for emission factor lookup |

## Quick Start

```python
from capabilities.energy.met.service import SmartMeteringService

svc = SmartMeteringService(tenant_id="acme", actor_id="ops-api")

# Register a net-metering capable smart meter
meter = await svc.register_meter(
    meter_serial="HXE3100042",
    customer_id="cust-881",
    location="Zone 3 / Feeder 12",
    meter_type="net_metering",
    communication_protocol="DLMS",
    multiplier=1.0,
)

# Submit a batch of 15-min interval readings with auto-validation
result = await svc.process_interval_data(
    meter_id=meter["id"],
    interval_readings=[
        {"timestamp": "2026-06-01T00:00:00Z", "value": 1.24, "quality": "valid"},
        {"timestamp": "2026-06-01T00:15:00Z", "value": 1.31, "quality": "valid"},
    ],
    interval_length="15min",
    quality_check=True,
)
# -> {"data_completeness_pct": 100.0, "spikes_detected": 0, ...}

# Tamper evaluation from head-end signals
tamper = await svc.tamper_detection(
    meter_id=meter["id"],
    tamper_indicators={"cover_open": True, "magnetic_field": False, "reverse_energy": False},
    auto_disconnect=False,
)

# Broadcast DR signal to all active meters
dr = await svc.demand_response_signal(
    customer_segment="commercial",
    reduction_kw=500.0,
    duration=2.0,
    event_type="direct_load_control",
    incentive_rate=0.12,
)

# Fleet analytics for a period
analytics = await svc.meter_analytics(period="2026-06")
```

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-met/api/v1/dashboard` | GET | Dashboard summary | `energy_met:view` |
| `/energy-met/api/v1/meters` | GET | List meters | `energy_met:meters` |
| `/energy-met/api/v1/meters` | POST | Register meter | `energy_met:meters` |
| `/energy-met/api/v1/meters/<id>` | GET | Meter detail | `energy_met:meters` |
| `/energy-met/api/v1/readings` | POST | Submit interval reading | `energy_met:readings` |
| `/energy-met/api/v1/tamper` | GET | List tamper events | `energy_met:tamper` |
| `/energy-met/api/v1/tamper` | POST | Report tamper | `energy_met:tamper` |
| `/energy-met/api/v1/tamper/<id>/resolve` | PUT | Resolve tamper | `energy_met:tamper` |
| `/energy-met/api/v1/commands` | POST | Issue remote command | `energy_met:commands` |
| `/energy-met/api/v1/commands/<id>/acknowledge` | PUT | Acknowledge command | `energy_met:commands` |
| `/energy-met/api/v1/demand-response` | POST | Create DR event | `energy_met:demand_response` |
| `/energy-met/api/v1/demand-response/<id>/opt-out` | POST | Meter opt-out | `energy_met:demand_response` |
| `/energy-met/api/v1/data-quality` | POST | Set quality flag | `energy_met:data_quality` |
| `/energy-met/api/v1/head-end` | POST | Update head-end status | `energy_met:admin` |
| `/energy-met/api/v1/meters/<id>/carbon` | GET | Carbon footprint report | `energy_met:view` |
| `/energy-met/api/v1/meters/<id>/health` | GET | Meter health score | `energy_met:view` |
| `/energy-met/api/v1/meters/<id>/fraud-score` | GET | Fraud risk score | `energy_met:admin` |

## World-Class Enhancements (v2.0)

1. **Load Profile Disaggregation** — NILM (non-intrusive load monitoring) over 15-min intervals via edge-hosted Ollama inference. Returns per-appliance energy estimates and confidence scores. (`disaggregate_load_profile`)

2. **Voltage Quality Monitoring** — EN 50160 / ANSI C84.1 sag/swell/interruption tracking. Stores `VoltageQualityEvent` records; summarises violations by phase per period. (`submit_voltage_event`, `voltage_quality_report`)

3. **Net Metering & Bi-Directional Flow Tracking** — Separate import/export registers conforming to ANSI C12.19 and ISO 15118. Net metering reconciliation with configurable feed-in tariff. (`submit_bidirectional_reading`, `net_metering_reconciliation`)

4. **Real-Time Fraud Scoring Pipeline** — 0–1 NTL risk score from peer-group z-score, tamper history, command anomalies, and payment gaps. Persists to `MeterFraudScore`; alerts at configurable threshold. (`score_meter_fraud_risk`)

5. **Predictive Meter Health Scoring** — 0–100 health index from comm success rate, days since last read, firmware age, and open tamper events. Fleet-level report included. (`compute_meter_health_score`, `meter_health_fleet_report`)

6. **Dynamic Load Limiting & Prepayment Credit Management** — STS/IEC 62055-41 token-based prepayment with credit-balance load limiting and low-credit push/SMS alerts. (`set_load_limit`, `update_prepayment_credit`, `prepayment_credit_alert`)

7. **OpenADR 2.0b Demand Response Protocol** — Native VTN/VEN exchange: serialises `oadrDistributeEvent` XML and handles `optIn/optOut` responses. Enables ISO/RTO DR market participation. (`publish_open_adr_event`, `receive_oadr_opt`)

8. **Edge Firmware OTA Orchestration** — Canary deployments (configurable %) with cryptographic manifest verification, per-meter status tracking, and one-command rollback. (`initiate_firmware_campaign`, `track_firmware_campaign`, `rollback_firmware_campaign`)

9. **MDMS Integration via IEC 61968-9 Adapter** — Serialises interval readings to IEC 61968-9 MeterReading XML; bulk HTTP POST with retry and delivery ack. Zero-ETL integration with Oracle MDM, Itron MDM, OSIsoft PI. (`export_mdm_reading_xml`, `push_readings_to_mdms`)

10. **Loss Calculations & Technical Loss Attribution** — Feeder-level comparison of substation injection vs. sum of meter reads. Returns technical loss, NTL estimate, and per-section breakdown stored in `FeederLossRecord`. (`compute_feeder_losses`)

11. **Meter Data Streaming via MQTT/Bytewax Bridge** — Publishes `IntervalReading` as JSON CloudEvents via aiomqtt/bytewax stream processor at read time. Configurable per-tenant topic prefix and broker type. (`publish_reading_to_stream`, `configure_streaming_bridge`)

12. **Outage Detection & FLISR Event Correlation** — Clusters meters with communication loss exceeding a threshold, infers outage boundary from GIS adjacency in < 2 min, emits `OutageEvent` to `energy_dis`. (`detect_outage_cluster`, `correlate_restoration`)

13. **Time-of-Use & Critical Peak Pricing Tariff Engine** — Maps interval readings to tariff buckets (on-peak, off-peak, CPP) using `TouTariffSchedule`. Returns per-bucket kWh totals and pre-bill summary for direct `energy_bil` feed. (`apply_tou_tariff`)

14. **Cybersecurity Event Log (NERC CIP / IEC 62351)** — SHA-256 hash-chained security event records for auth failures, firmware changes, and remote commands. NERC CIP-007 CSV/JSON export and SIEM integration via `intel`. (`log_security_event`, `export_security_log`)

15. **Carbon & Emissions Tracking per Meter** — Joins interval readings with time-matched grid emission factors (kg CO₂e/kWh). Returns total kg CO₂e, average intensity, and hourly emissions time-series. Supports SEC Climate Disclosure, IFRS S2, GHG Protocol Scope 2. (`submit_grid_emission_factor`, `compute_meter_carbon_footprint`)

## New Methods

### `process_interval_data` — Batch validation with quality scoring

```python
result = await svc.process_interval_data(
    meter_id="m-001",
    interval_readings=[
        {"timestamp": "2026-06-01T00:00:00Z", "value": 1.24, "quality": "valid"},
        {"timestamp": "2026-06-01T00:15:00Z", "value": 1.31, "quality": "valid"},
        {"timestamp": "2026-06-01T00:30:00Z", "value": 9999.0, "quality": "valid"},  # spike
    ],
    interval_length="15min",
    quality_check=True,
)
# {"valid_intervals": 2, "spikes_detected": 1, "data_completeness_pct": 66.67, ...}
```

### `tamper_detection` — Auto-classify from head-end signals

```python
result = await svc.tamper_detection(
    meter_id="m-001",
    tamper_indicators={
        "cover_open": True,
        "magnetic_field": False,
        "reverse_energy": True,
        "load_side_voltage": False,
        "meter_tilt": False,
    },
    auto_disconnect=True,  # issues remote_disconnect if tamper confirmed
)
# {"tamper_detected": True, "indicators_detected": ["cover_open", "reverse_energy"],
#  "auto_disconnect_issued": True, "disconnect_command_id": "cmd-uuid", ...}
```

### `demand_response_signal` — Segment-level DR broadcast

```python
signal = await svc.demand_response_signal(
    customer_segment="industrial",
    reduction_kw=2000.0,
    duration=3.0,
    event_type="direct_load_control",
    incentive_rate=0.18,
    currency="KES",
)
# {"meters_targeted": 47, "estimated_energy_reduction_kwh": 6000.0, ...}
```

### `ami_head_end_sync` — Head-end batch telemetry

```python
sync = await svc.ami_head_end_sync(
    batch_id="sync-2026-06-01-0300",
    meters_polled=5000,
    reads_received=4930,
    failures=70,
    protocol="DLMS",
)
# {"success_rate_pct": 98.6, "head_end_status": {"status": "healthy"}, ...}
```

### `meter_analytics` — Period-level operational KPIs

```python
kpis = await svc.meter_analytics(period="2026-06")
# {"active_meters": 4800, "tamper_events": 3, "read_rate_pct": 97.2,
#  "command_success_rate_pct": 99.1, "dr_events": 2, ...}
```

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `meter_type_supported` | meter_type not in supported list | deny |
| `meter_serial_required` | serial_present=False | deny |
| `reading_meter_active` | meter.status != active | deny |
| `disconnect_approval_required` | disconnect command without approval | deny |
| `firmware_update_approval_required` | firmware update without approval | deny |
| `tamper_evidence_required` | evidence_present=False on tamper report | deny |
| `dr_opt_out_respected` | customer_opted_out=True on DR activation | deny |
| `dr_notification_required` | notification_sent=False on DR creation | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `privileged_met_agent_requires_human_approval` | agent disconnect without human approval | deny |

## Data Models
| Model | Key Fields |
|---|---|
| `SmartMeter` | id, serial_number, meter_type, communication_technology, status, customer_id, load_limit_kw |
| `IntervalReading` | id, meter_id, reading_type, interval_length, value, unit, quality_flag |
| `TamperEvent` | id, meter_id, tamper_type, detected_at, evidence_reference, status |
| `RemoteCommand` | id, meter_id, command_type, status, issued_by, approved_by, retry_count |
| `DemandResponseEvent` | id, event_type, target_reduction_kw, actual_reduction_kw, meter_ids, opt_out_meter_ids |
| `DataQualityFlag` | id, reading_id, quality_flag, reason, substitute_value |
| `AmiHeadEndStatus` | id, head_end_name, protocol, connected_meters, total_meters, communication_ratio |
| `MeterFraudScore` | id, meter_id, risk_score, peer_z_score, computed_at |
| `MeterHealthScore` | id, meter_id, health_score, comm_success_rate, firmware_age_days, computed_at |
| `GridEmissionFactor` | id, region_id, timestamp, kg_co2e_per_kwh, source |
| `SecurityEvent` | id, meter_id, event_type, severity, sha256_chain_hash, occurred_at |
| `FirmwareCampaign` | id, tenant_id, firmware_version, canary_pct, status, per_meter_results |

## Streaming Events
- `meter_registered` / `meter_status_changed`
- `interval_reading_received` / `interval_data_processed`
- `tamper_event_detected` / `tamper_event_resolved`
- `remote_command_sent` / `remote_command_executed`
- `demand_response_event_created` / `demand_response_event_completed`
- `data_quality_flag_set`
- `ami_head_end_heartbeat` / `ami_sync_completed`
- `outage_event_detected` / `outage_restoration_confirmed`
- `fraud_risk_alert` / `security_event_logged`
- `firmware_campaign_started` / `firmware_campaign_completed`

## Edge Cases Handled
- Readings rejected for inactive, tampered, or disconnected meters
- Disconnect requires explicit approval; `on_demand_read` does not
- Firmware update treated as privileged command with separate approval rule
- DR opt-out list checked per meter before activating DR event
- Head-end marked "degraded" when communication ratio drops below 90%
- Quality flag substitution value stored separately from original reading
- Batch interval processing: spike detection flags > 500 kWh delta per interval
- Auto-disconnect on tamper requires operator approval in `wflo` unless `tamper_system` actor

## Composability Notes
- Interval data and TOU buckets feed `energy_bil` for consumption billing
- Tamper events and security logs escalate to `intel` threat detection
- DR events coordinate with `energy_grd` for system-level demand management
- Outage clusters emit `OutageEvent` to `energy_dis` for SAIDI/SAIFI tracking
- AMI head-end health feeds `moni` operational dashboards
- Carbon footprint data feeds ESG reporting pipelines (IFRS S2, SEC Climate)
- MQTT/Bytewax streaming feeds real-time SCADA and V/VAR optimisation systems
