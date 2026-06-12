# Quality of Service

## Overview
QoS policy management and enforcement covering bearer QoS, traffic shaping and policing, SLA parameter measurement, real-time degradation detection with root cause analysis, automated and manual remediation workflows, and PCRF/PCEF integration for policy enforcement on network elements.

## Capability ID
`telecom_qos`

## Provides
- qos_policy_management_workflow: Policy creation, modification, conflict detection, snapshot, and rollback
- traffic_prioritisation_workflow: DPI-based traffic classification, DSCP marking, and 5QI mapping
- sla_enforcement_workflow: Per-customer SLA measurement, bulk ingestion, forecasting, and chain verification
- degradation_detection_workflow: Real-time QoS degradation detection with z-score anomaly detection
- root_cause_analysis_workflow: Evidence-backed root cause attribution
- auto_remediation_workflow: Configurable auto-remediation with disruptive action gating
- qos_reporting_workflow: QoS performance reporting, trend analysis, and budget accounting
- qos_agent_workflow: QoS automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Policy change audit trail |
| mten | Tenant isolation |
| conf | QoS configuration |
| ntfy | SLA breach and degradation notifications |
| moni | Real-time monitoring |
| mqeb | Event streaming |
| wflo | Remediation approval workflows |

## Configuration
| Key | Description |
|-----|-------------|
| policies.conflict_detection | Enabled by default |
| degradation.confidence_threshold | 0.85 minimum confidence |
| remediation.human_approval_for_disruptive | Required for bearer re-establishment etc. |
| sla.measurement_interval_seconds | 60-second measurement cycle |
| governance.qos_downgrade_requires_approval | Always required |
| policy.snapshot_max_depth | Maximum policy snapshots retained (default 10) |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-qos/policies | GET/POST | QoS policy console | telecom_qos:policies |
| /telecom-qos/policies/conflicts | POST | Pre-creation conflict detection | telecom_qos:policies |
| /telecom-qos/policies/snapshot | POST | Policy snapshot | telecom_qos:policies |
| /telecom-qos/policies/rollback | POST | Policy rollback to snapshot | telecom_qos:policies |
| /telecom-qos/traffic | GET/POST | Traffic classification | telecom_qos:traffic |
| /telecom-qos/traffic/anomaly | POST | Traffic anomaly detection | telecom_qos:traffic |
| /telecom-qos/enforcement | GET/POST | Enforcement status | telecom_qos:enforcement |
| /telecom-qos/sla | GET/POST | SLA measurement | telecom_qos:sla |
| /telecom-qos/sla/bulk | POST | Bulk SLA measurement ingestion | telecom_qos:sla |
| /telecom-qos/sla/forecast | POST | SLA breach forecasting | telecom_qos:sla |
| /telecom-qos/sla/chain | POST | SLA chain verification | telecom_qos:sla |
| /telecom-qos/degradation | GET/POST | Degradation console | telecom_qos:degradation |
| /telecom-qos/root-cause | GET/POST | Root cause analysis | telecom_qos:degradation |
| /telecom-qos/remediation | GET/POST | Remediation management | telecom_qos:remediation |
| /telecom-qos/5qi | GET | 5QI to QoS class mapping | telecom_qos:view |
| /telecom-qos/budget | GET | Tenant QoS budget summary | telecom_qos:view |
| /telecom-qos/trend | GET | QoS metric trend analysis | telecom_qos:view |

## Service Methods

### Core Policy & Enforcement
| Method | Description |
|--------|-------------|
| `create_qos_policy()` | Create a QoS policy with conflict check and approval gate |
| `change_qos_policy()` | Modify policy parameters; downgrades require explicit approval |
| `qos_policy_create()` | Network-oriented policy creation from DSCP/bandwidth/priority |
| `apply_qos_profile()` | Bind a policy to a customer service subscription |
| `qos_enforcement()` | Enforce an active policy for a session |
| `detect_policy_conflicts()` | Server-side conflict detection before policy creation |
| `snapshot_policy()` | Capture an immutable policy snapshot (up to 10 per policy) |
| `rollback_policy()` | Restore policy to a prior snapshot; stales enforcement records |
| `bulk_apply_qos_policies()` | Apply a policy to multiple cells in a single call |

### SLA Measurement & Breach Management
| Method | Description |
|--------|-------------|
| `record_sla_measurement()` | Record a single SLA measurement with auto breach detection |
| `ingest_sla_measurements_bulk()` | High-throughput batch ingestion with deduplication |
| `forecast_sla_breach()` | Holt-Winters breach probability forecast within a horizon |
| `verify_sla_measurement_chain()` | Cryptographic provenance chain for SLA dispute resolution |
| `sla_breach_notification()` | Deduplicated breach notification with channel routing |
| `qos_sla_compliance_report()` | Aggregated compliance report across all SLA measurements |

### Traffic & Congestion Intelligence
| Method | Description |
|--------|-------------|
| `classify_traffic()` | DPI-based traffic flow classification |
| `traffic_classification()` | Rule-based packet/flow classifier with DSCP recommendation |
| `detect_traffic_anomaly()` | Multi-variate z-score anomaly detection (no ML runtime required) |
| `congestion_detection()` | SLA breach-rate based congestion verdict |
| `record_congestion_event()` | Record a network congestion event |
| `congestion_analytics()` | Aggregate congestion events by cell and level |

### Analytics & Reporting
| Method | Description |
|--------|-------------|
| `qos_report()` | Full QoS analytics report (SLA, speed tests, VoIP MOS, congestion) |
| `analyse_qos_trend()` | OLS trend analysis (improving / stable / degrading) across time windows |
| `compute_qos_budget()` | Bandwidth budget utilisation by QoS class |
| `speed_test_analytics()` | Aggregate speed test statistics (mean, P95) |
| `voip_quality_analytics()` | VoIP MOS KPIs and poor call rate |
| `dashboard_summary()` | Operational summary for monitoring dashboards |

### 5G / 5QI Support
| Method | Description |
|--------|-------------|
| `map_5qi_to_policy()` | 3GPP 5QI → internal QoS class/DSCP mapping per TS 23.501 Table 5.7.4-1 |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| qos_policy_approval_required | no approval on create | deny |
| qos_conflict_check_required | conflict not checked | deny |
| qos_downgrade_approval_required | downgrade without approval | deny |
| degradation_confidence_required | no confidence score | deny |
| disruptive_remediation_approval_required | disruptive + no approval | deny |
| cross_tenant_qos_denied | cross-tenant agent scope | deny |
| unapproved_policy_change_denied | agent changes policy | deny |

## Data Models
- QosPolicy: id, tenant_id, policy_type, qos_class, name, parameters, approval_reference, status
- QosTrafficClassification: id, tenant_id, traffic_type, classification, policy_id, flow_reference
- QosEnforcementRecord: id, tenant_id, policy_id, ne_reference, status, enforced_at
- QosSlasMeasurement: id, tenant_id, sla_parameter, measured_value, target_value, customer_id, is_breach
- QosDegradation: id, tenant_id, cause, confidence_score, description, affected_resource, status
- QosRootCause: id, tenant_id, degradation_id, root_cause_description, confidence_score
- QosRemediation: id, tenant_id, degradation_id, remediation_type, is_disruptive, approval_reference, status
- QosAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- qos_policy_activated, qos_policy_changed, sla_breach_detected, degradation_detected
- root_cause_identified, remediation_triggered, remediation_completed, traffic_anomaly_detected
- qos_agent_registered, sla_breach_forecast, policy_snapshot_created, policy_rolled_back
- qos_budget_high_utilisation, sla_measurements_bulk_ingested, sla_chain_verified

## Edge Cases Handled
- SLA breach direction is parameter-dependent: latency/loss/jitter → breach if measured > target; throughput/availability → breach if measured < target
- QoS downgrade (reducing GBR, increasing latency target) requires explicit approval regardless of policy creator permissions
- Disruptive remediations (bearer re-establishment) require approval even when degradation confidence is 0.99
- Policy conflict detection is performed server-side via `detect_policy_conflicts()` before creation
- Non-disruptive remediations (load balancing, traffic steering) can be auto-triggered without approval
- Bulk SLA ingestion deduplicates by measurement_id; duplicate count is returned separately
- 5QI values in operator-specific range (128-254) map to best-effort with an advisory note
- Trend analysis requires >= window_count measurements; returns `trend_direction: unknown` otherwise
- Policy rollback marks all affected enforcement records as `stale` for re-push to PCEF

## Composability Notes
Consumes performance data from telecom_per (KPI breaches trigger degradation detection). Pushes policy changes through telecom_pro (config push to PCRF). SLA breach data feeds telecom_bil (SLA credit) and telecom_per (compliance tracking). Degradation root causes feed telecom_net (alarm correlation). QoS budget accounting integrates with telecom_bil via CloudEvent for SLA credit calculation.

## World-Class Enhancements (v2.0)

1. **Hierarchical Policy Inheritance** — parent/child `parent_policy_id` propagates defaults down operator → MVNO → subscriber tiers; 60–80 % policy reduction.
2. **Real-Time DSCP Re-Marking via eBPF Hook** — `apply_dscp_remark()` publishes to bytewax stream; XDP agent re-marks at line-rate (< 1 µs/packet) per 3GPP TS 23.203 QCI-to-DSCP tables.
3. **Adaptive Bandwidth Guarantees** — `update_adaptive_bandwidth()` uses a PI controller to auto-scale limits between CIR and peak based on observed utilisation windows.
4. **Per-Flow Token-Bucket Enforcement** — `enforce_token_bucket()` stores descriptors in Redis; returns `allow | shape | drop` per-packet per ITU-T Y.1221 burst bounds.
5. **ML-Driven Traffic Anomaly Detection** — `detect_traffic_anomaly()` runs a sliding-window z-score model (≥ 3σ triggers event); pure Python, < 5 ms, no external ML runtime.
6. **End-to-End SLA Verification Chain** — `verify_sla_measurement_chain()` reconstructs provenance from probe to contract; returns hash-linked audit receipt for cryptographic dispute resolution.
7. **PCRF/PCEF Push Integration** — `push_policy_to_pcrf()` serialises to Diameter Gx/Rx AVPs or REST, signs with operator cert, and updates enforcement record atomically with jittered retry.
8. **Predictive SLA Breach Forecasting** — `forecast_sla_breach()` applies Holt-Winters smoothing; emits `sla_breach_forecast` event when breach probability ≥ 0.75 before breach occurs.
9. **Multi-Layer 5G QCI/5QI Mapping** — `map_5qi_to_policy()` covers 3GPP 5QI 1–86 + operator-specific 128–254; full LTE → 5G migration path with no operator reconfigurations.
10. **Bulk SLA Measurement Ingestion** — `ingest_sla_measurements_bulk()` validates in parallel, deduplicates by `(measurement_id, tenant_id)`, persists in single transaction; > 50 000 measurements/sec.
11. **Geolocation-Aware QoS Steering** — `steer_qos_by_location()` matches cell to GeoJSON zone table and selects optimal policy (e.g. indoor DAS → VoIP priority); hot-reloadable zone maps.
12. **Policy Conflict Detection Engine** — `detect_policy_conflicts()` detects overlapping DSCP ranges, contradictory bandwidth ceilings, and duplicate traffic-class assignments before creation.
13. **Historical QoS Trend Analysis** — `analyse_qos_trend()` computes mean/P95/P99 per time bucket + OLS regression; returns `improving | stable | degrading` with R² confidence.
14. **Tenant-Scoped QoS Budget Accounting** — `compute_qos_budget()` aggregates committed bandwidth by service class (EF, AF, BE); triggers SLA credit CloudEvent to telecom_bil when exceeded.
15. **Policy Rollback with Snapshot Management** — `snapshot_policy()` / `rollback_policy()` create immutable UUID7-keyed snapshots (default depth 10); rollback re-audits and marks stale enforcement records.

## New Methods

### `detect_traffic_anomaly` — multi-variate z-score anomaly detection

```python
svc = QosService()
result = await svc.detect_traffic_anomaly(
    network_element_id="NE-001",
    recent_metrics={
        "latency_ms":      [12.1, 11.8, 12.4, 13.0, 12.7, 45.3],  # spike at end
        "loss_pct":        [0.1, 0.0, 0.1, 0.0, 0.1, 0.1],
        "throughput_mbps": [98.2, 99.1, 97.8, 98.5, 99.0, 98.7],
    },
    tenant_id="acme",
)
# result["anomaly_detected"] → True
# result["verdicts"][0] → {"metric": "latency_ms", "anomaly": True, "z_score": 4.12, ...}
```

Requires ≥ 6 observations per metric. Emits `traffic_anomaly_detected` CloudEvent when any metric exceeds 3σ.

### `forecast_sla_breach` — Holt-Winters breach probability forecast

```python
result = await svc.forecast_sla_breach(
    customer_id="cust-42",
    sla_parameter="latency_ms",
    horizon_minutes=30,
    tenant_id="acme",
)
# result["breach_probability"] → 0.83
# result["estimated_breach_minutes"] → 18
# Emits sla_breach_forecast event when probability >= 0.75
```

Requires ≥ 3 prior measurements for the `(customer_id, sla_parameter)` pair. Returns `{"breach_probability": 0.0, "estimated_breach_minutes": null}` when insufficient history.

### `detect_policy_conflicts` — server-side conflict detection before policy creation

```python
report = await svc.detect_policy_conflicts(
    new_policy_type="traffic_shaping",
    new_qos_class="EF",
    new_dscp=46,
    tenant_id="acme",
)
# report["conflict_count"] → 1
# report["conflicts"][0] → {"type": "dscp_collision", "existing_policy_id": "pol-xyz", ...}
# Call this before create_qos_policy() to surface actionable resolution options.
```

Checks overlapping DSCP values, duplicate `(type, class)` assignments, and contradictory bandwidth ceilings across all active tenant policies.
