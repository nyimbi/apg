# Quality of Service — User Guide

**Capability ID**: `telecom_qos` | **Domain**: `telecom` | **Version**: `1.1.0`
**Company**: Datacraft | **Author**: Nyimbi Odero | **Updated**: 2026-06-11

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Policy Management](#policy-management)
6. [Traffic Classification](#traffic-classification)
7. [SLA Measurement & Breach Management](#sla-measurement--breach-management)
8. [Degradation Detection & Remediation](#degradation-detection--remediation)
9. [Traffic Anomaly Detection](#traffic-anomaly-detection)
10. [5G 5QI Support](#5g-5qi-support)
11. [Analytics & Reporting](#analytics--reporting)
12. [VoIP & Speed Test Quality](#voip--speed-test-quality)
13. [QoS Budget Accounting](#qos-budget-accounting)
14. [Policy Snapshot & Rollback](#policy-snapshot--rollback)
15. [Automation Agents](#automation-agents)
16. [Composability](#composability)
17. [Configuration Reference](#configuration-reference)
18. [Streaming Events Reference](#streaming-events-reference)

---

## Overview

`telecom_qos` provides end-to-end Quality of Service management for telecom operators:

- **Policy lifecycle** — create, modify, snapshot, rollback, and conflict-detect QoS policies with mandatory approval gates
- **Traffic prioritisation** — rule-based and DPI-assisted classification with DSCP marking and 3GPP 5QI mapping
- **SLA enforcement** — per-customer SLA measurement, bulk ingestion, Holt-Winters breach forecasting, and cryptographic chain verification
- **Degradation intelligence** — real-time detection, multi-variate z-score anomaly detection, root cause analysis, and auto-remediation
- **Analytics** — trend analysis, QoS budget accounting, VoIP MOS, speed test grading, congestion tracking

The service layer is fully async, tenant-scoped, and emits CloudEvents for all state transitions.

---

## Installation

```bash
pip install apg-telecom-qos
```

Or within an APG workspace:

```apg
use telecom_qos;
```

---

## Quick Start

```python
import asyncio
from apg_telecom_qos import QualityOfServiceService

svc = QualityOfServiceService()

async def main():
    # Create a policy for VoIP traffic
    policy = await svc.qos_policy_create(
        name="VoIP-Priority",
        traffic_class="voip",
        dscp_marking=46,          # EF — Expedited Forwarding
        bandwidth_limit=5120,     # 5 Mbps
        priority=1,
        tenant_id="acme",
        approval_reference="CR-2026-001",
    )
    print(policy["policy_id"])

    # Bind it to a customer subscription
    await svc.apply_qos_profile(
        customer_id="cust-123",
        service_id="svc-voice-001",
        policy_id=policy["policy_id"],
        tenant_id="acme",
    )

    # Record an SLA measurement
    svc.record_sla_measurement(
        measurement_id="meas-001",
        tenant_id="acme",
        sla_parameter="latency",
        measured_value=120.0,
        target_value=100.0,
        customer_id="cust-123",
        measured_at="2026-06-11T10:00:00Z",
    )

    # Check health
    health = await svc.health_check(tenant_id="acme")
    print(health["status"])

asyncio.run(main())
```

---

## Core Concepts

### Tenant Isolation

Every method accepts `tenant_id` (default: `"default"`).  All stores are keyed by `(tenant_id, item_id)` — no cross-tenant data leakage is possible.

### Approval Gates

The following operations require a non-empty `approval_reference`:

| Operation | Gate |
|-----------|------|
| Policy creation | Always |
| Policy downgrade | Always |
| Disruptive remediation | Always |

### DSCP Marking

Standard DSCP values used across the capability:

| Service | DSCP | Name |
|---------|------|------|
| VoIP / EF | 46 | Expedited Forwarding |
| Video (AF41) | 34 | Assured Forwarding 4.1 |
| Web (AF21) | 18 | Assured Forwarding 2.1 |
| Bulk (AF11) | 10 | Assured Forwarding 1.1 |
| Best Effort | 0 | CS0 |

---

## Policy Management

### Create a Policy

```python
# Network-oriented creation (recommended)
policy = await svc.qos_policy_create(
    name="Gaming-QoS",
    traffic_class="gaming",
    dscp_marking=34,
    bandwidth_limit=20480,   # 20 Mbps
    priority=2,
    tenant_id="acme",
    approval_reference="CR-2026-002",
)

# Raw creation (full control)
policy = svc.create_qos_policy(
    policy_id="pol-001",
    tenant_id="acme",
    policy_type="traffic_shaping",
    qos_class="af41",
    name="Gaming-QoS",
    parameters="dscp=34,bw_limit=20480kbps,priority=2",
    approval_reference="CR-2026-002",
    created_by="ops@acme.com",
)
```

### Detect Conflicts Before Creating

```python
report = await svc.detect_policy_conflicts(
    new_policy_type="traffic_shaping",
    new_qos_class="af41",
    new_dscp=34,
    tenant_id="acme",
)
if not report["safe_to_create"]:
    for conflict in report["conflicts"]:
        print(conflict["conflict_type"], conflict["resolution"])
```

### Modify a Policy

```python
svc.change_qos_policy(
    policy_id="pol-001",
    tenant_id="acme",
    new_parameters="dscp=34,bw_limit=10240kbps,priority=2",
    is_downgrade=True,
    approval_reference="CR-2026-003",
)
```

---

## Traffic Classification

### Classify a Packet Flow

```python
result = await svc.traffic_classification(
    packet_metadata={
        "dst_port": 5060,
        "protocol": "sip",
        "app_id": "voip_client",
        "payload_size_bytes": 200,
    },
    tenant_id="acme",
)
# result["traffic_type"] == "voip"
# result["suggested_dscp"] == 46
# result["recommended_policy_id"] == "pol-001"  (if a matching active policy exists)
```

### Register a Classification

```python
svc.classify_traffic(
    classification_id="cls-001",
    tenant_id="acme",
    traffic_type="voip",
    classification="high_priority",
    policy_id="pol-001",
    flow_reference="flow-xyz",
    classified_at="2026-06-11T10:05:00Z",
)
```

---

## SLA Measurement & Breach Management

### Record a Single Measurement

```python
svc.record_sla_measurement(
    measurement_id="meas-002",
    tenant_id="acme",
    sla_parameter="latency",
    measured_value=85.0,
    target_value=100.0,
    customer_id="cust-123",
    measured_at="2026-06-11T10:10:00Z",
)
```

### Bulk Ingestion

For high-volume probe feeds (> 50 000 measurements/second on commodity hardware):

```python
result = await svc.ingest_sla_measurements_bulk(
    measurements=[
        {
            "measurement_id": "m-001",
            "sla_parameter": "latency",
            "measured_value": 95.0,
            "target_value": 100.0,
            "customer_id": "cust-123",
        },
        {
            "measurement_id": "m-002",
            "sla_parameter": "packet_loss",
            "measured_value": 0.5,
            "target_value": 1.0,
        },
    ],
    tenant_id="acme",
)
print(result["accepted_count"], result["duplicate_count"])
```

### Forecast SLA Breach (Holt-Winters)

```python
forecast = await svc.forecast_sla_breach(
    customer_id="cust-123",
    sla_parameter="latency",
    horizon_minutes=30,
    tenant_id="acme",
)
if forecast["breach_probability"] and forecast["breach_probability"] >= 0.75:
    print(f"Breach likely in {forecast['estimated_breach_minutes']} min")
```

Requires >= 3 measurements for the customer/parameter combination.  A `sla_breach_forecast` CloudEvent is emitted when probability >= 0.75.

### Chain Verification for SLA Disputes

```python
chain = await svc.verify_sla_measurement_chain(
    customer_id="cust-123",
    measurement_ids=["meas-001", "meas-002"],
    tenant_id="acme",
)
print(chain["chain_valid"], chain["chain_receipt"])
```

Each measurement must appear in the audit trail.  Returns a SHA-256 receipt (first 16 hex chars) suitable for dispute resolution.

### SLA Breach Notification

```python
notif = await svc.sla_breach_notification(
    customer_id="cust-123",
    service_id="svc-voice-001",
    breach_type="latency",
    tenant_id="acme",
    channel="email",
)
# notif["duplicate_suppressed"] == True if already sent today for same cust/service/type
```

---

## Degradation Detection & Remediation

### Record a Degradation Event

```python
svc.record_degradation(
    degradation_id="deg-001",
    tenant_id="acme",
    cause="radio_interference",
    confidence_score=0.92,
    description="Elevated latency on sector 3",
    affected_resource="cell-BTS-042",
    evidence_reference="pm-data-2026-06-11-10:00",
    detected_at="2026-06-11T10:15:00Z",
)
```

`confidence_score` must be >= 0.85 (configurable via `degradation.confidence_threshold`).

### Root Cause Analysis

```python
svc.record_root_cause(
    rca_id="rca-001",
    tenant_id="acme",
    degradation_id="deg-001",
    root_cause_description="Adjacent channel interference from BTS-039",
    confidence_score=0.95,
    evidence_reference="spectrum-scan-2026-06-11",
    identified_at="2026-06-11T10:20:00Z",
)
```

### Trigger Remediation

```python
# Non-disruptive — no approval required
svc.trigger_remediation(
    remediation_id="rem-001",
    tenant_id="acme",
    degradation_id="deg-001",
    remediation_type="traffic_steering",
    is_disruptive=False,
    approval_reference=None,
    triggered_at="2026-06-11T10:21:00Z",
)

# Disruptive — approval mandatory
svc.trigger_remediation(
    remediation_id="rem-002",
    tenant_id="acme",
    degradation_id="deg-001",
    remediation_type="bearer_reestablishment",
    is_disruptive=True,
    approval_reference="APPR-2026-007",
    triggered_at="2026-06-11T10:22:00Z",
)
```

### Service Degradation Alert

```python
alert = await svc.service_degradation_alert(
    customer_id="cust-123",
    service_id="svc-voice-001",
    current_quality={
        "latency_ms": 180.0,
        "packet_loss_pct": 2.5,
        "jitter_ms": 35.0,
        "download_mbps": 8.0,
    },
    tenant_id="acme",
)
print(alert["severity"])  # "critical" (3 violations)
```

---

## Traffic Anomaly Detection

Uses a sliding-window z-score model — no external ML runtime required, inference < 5 ms.

```python
result = await svc.detect_traffic_anomaly(
    network_element_id="BTS-042",
    recent_metrics={
        "latency_ms":    [50, 52, 51, 53, 54, 180],   # last value is anomalous
        "loss_pct":      [0.1, 0.1, 0.2, 0.1, 0.1, 0.15],
        "throughput_mbps": [100, 98, 102, 99, 101, 95],
    },
    tenant_id="acme",
)
print(result["anomaly_detected"])  # True
for v in result["metric_verdicts"]:
    if v["anomaly"]:
        print(f"{v['metric']}: z={v['z_score']}")
```

An anomaly is flagged when `abs(z-score) >= 3.0` on any metric.  Requires >= 6 observations per metric.

---

## 5G 5QI Support

Map 3GPP 5QI values (TS 23.501 Table 5.7.4-1) to internal QoS descriptors:

```python
descriptor = await svc.map_5qi_to_policy(five_qi=1, tenant_id="acme")
# {
#   "five_qi": 1,
#   "resource_type": "GBR",
#   "priority": 20,
#   "pdb_ms": 100,
#   "per": "1e-2",
#   "dscp": 46,
#   "qos_class": "ef",
#   "service": "Conversational Voice",
#   "matched_policy_id": "pol-001",  # if an active EF policy exists
#   ...
# }
```

Standardised 5QI values covered: 1-9, 65-66, 69-70, 80, 82-86.
Operator-specific range 128-254 maps to best-effort with an advisory note.

---

## Analytics & Reporting

### Full QoS Report

```python
report = await svc.qos_report(
    period="2026-06",
    service_type="mobile",
    tenant_id="acme",
)
```

Fields: `sla_compliance_rate`, `sla_breach_count`, `speed_test_grade_distribution`, `voip_avg_mos`, `congestion_events`, `degradation_count`.

### Trend Analysis (OLS)

```python
trend = await svc.analyse_qos_trend(
    metric="latency",
    window_count=6,
    window_size_minutes=60,
    tenant_id="acme",
)
print(trend["trend_direction"])   # "degrading" | "stable" | "improving"
print(trend["slope"], trend["r_squared"])
```

`window_count` must be 2-24.  Returns `trend_direction: unknown` when insufficient measurements exist.

### SLA Compliance Report

```python
compliance = await svc.qos_sla_compliance_report(
    tenant_id="acme",
    period="monthly",
)
print(compliance["compliance_rate_pct"])
```

---

## VoIP & Speed Test Quality

### VoIP MOS Calculation (ITU-T G.107 E-model)

```python
mos = await svc.voip_mos_calculation(
    call_id="call-abc-001",
    metrics={
        "packet_loss_pct": 0.5,
        "latency_ms": 80.0,
        "jitter_ms": 8.0,
        "codec_efficiency": 0.92,
    },
    tenant_id="acme",
)
print(mos["mos"], mos["quality"])   # 4.12, "good"
```

MOS >= 4.3 = excellent | 4.0-4.2 = good | 3.5-3.9 = fair | < 3.5 = poor.

### Speed Test Assessment

```python
result = await svc.speed_test_result(
    customer_id="cust-123",
    download_mbps=45.0,
    upload_mbps=12.0,
    latency_ms=20.0,
    tenant_id="acme",
    server_location="Nairobi-1",
)
print(result["grade"])  # "A"
```

Benchmarks: download >= 10 Mbps, upload >= 5 Mbps, latency <= 100 ms.

---

## QoS Budget Accounting

Track committed bandwidth utilisation across all active policies:

```python
budget = await svc.compute_qos_budget(
    tenant_id="acme",
    period="monthly",
)
# {
#   "total_committed_kbps": 204800.0,
#   "enforcement_utilisation_pct": 78.5,
#   "budget_by_class": {
#     "EF":   {"committed_kbps": 51200.0, "policy_count": 1},
#     "AF41": {"committed_kbps": 102400.0, "policy_count": 2},
#     "BE":   {"committed_kbps": 51200.0, "policy_count": 1},
#   },
#   ...
# }
```

A `qos_budget_high_utilisation` CloudEvent is emitted when utilisation >= 90 %.

---

## Policy Snapshot & Rollback

Capture a point-in-time snapshot before any modification, then restore if needed.

```python
# Snapshot before change
snap = await svc.snapshot_policy(policy_id="pol-001", tenant_id="acme")
snapshot_id = snap["snapshot_id"]

# Make the change
svc.change_qos_policy(
    policy_id="pol-001",
    tenant_id="acme",
    new_parameters="dscp=34,bw_limit=5120kbps,priority=3",
    is_downgrade=True,
    approval_reference="CR-2026-009",
)

# Roll back if the change caused issues
result = await svc.rollback_policy(
    policy_id="pol-001",
    snapshot_id=snapshot_id,
    tenant_id="acme",
)
print(result["stale_enforcement_records"])  # records that need re-push to PCEF
```

Up to 10 snapshots are retained per policy; the oldest are pruned automatically.

---

## Automation Agents

Register and govern QoS automation agents (e.g. ML-driven traffic steering daemons):

```python
agent = svc.register_agent(
    agent_id="agent-ts-001",
    tenant_id="acme",
    name="TrafficSteering-v2",
    runtime="python",
    role="remediation",
    scope="cell:BTS-*",
)

# Validate a privileged action before executing
svc.validate_agent_action(
    tenant_id="acme",
    privileged_scope=True,
    human_approval_recorded=True,
    cross_tenant_qos_scope=False,
    unapproved_policy_change_scope=False,
)
```

Cross-tenant scope and unapproved policy changes are always denied by capability rules.

---

## Composability

```
telecom_per  ──KPI breaches──►  telecom_qos  ──SLA breaches──►  telecom_bil (SLA credit)
                                     │
                                     ├──policy push──►  telecom_pro (PCRF config)
                                     ├──degradation──►  telecom_net (alarm correlation)
                                     └──compliance──►  telecom_per (compliance tracking)
```

Trigger keywords for APG composition engine: `qos_policy`, `traffic_shaping`, `sla_enforcement`, `bearer_qos`, `dscp_marking`, `5qi_mapping`, `congestion_detection`.

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `policies.conflict_detection` | `true` | Enable server-side conflict detection |
| `degradation.confidence_threshold` | `0.85` | Minimum confidence for degradation recording |
| `remediation.human_approval_for_disruptive` | `true` | Require approval for disruptive remediations |
| `sla.measurement_interval_seconds` | `60` | Probe measurement cycle |
| `governance.qos_downgrade_requires_approval` | `true` | Require approval for downgrades |
| `policy.snapshot_max_depth` | `10` | Max snapshots retained per policy |
| `anomaly.z_score_threshold` | `3.0` | Z-score threshold for anomaly flagging |
| `budget.high_utilisation_threshold_pct` | `90` | Utilisation % for budget event emission |
| `forecast.breach_alert_probability` | `0.75` | Probability threshold for breach forecast event |

---

## Streaming Events Reference

| Event | Trigger |
|-------|---------|
| `qos_policy_activated` | Policy created |
| `qos_policy_changed` | Policy parameters modified |
| `policy_snapshot_created` | Snapshot captured |
| `policy_rolled_back` | Policy restored from snapshot |
| `policy_conflict_detected` | Conflict found in `detect_policy_conflicts()` |
| `traffic_classified` | Traffic flow classified |
| `traffic_anomaly_detected` | Z-score anomaly detected |
| `enforcement_status_updated` | Enforcement record updated |
| `sla_breach_detected` | SLA measurement is a breach |
| `sla_breach_forecast` | Breach probability >= 0.75 |
| `sla_measurements_bulk_ingested` | Bulk ingestion completed |
| `sla_chain_verified` | Measurement chain verification completed |
| `degradation_detected` | Degradation event recorded |
| `root_cause_identified` | Root cause analysis recorded |
| `remediation_triggered` | Remediation action initiated |
| `remediation_completed` | Remediation action completed |
| `service_degradation_alert` | Customer service alert raised |
| `qos_agent_registered` | Automation agent registered |
| `qos_profile_applied` | Policy bound to customer subscription |
| `qos_session_enforced` | Session-level enforcement applied |
| `qos_budget_high_utilisation` | Tenant bandwidth utilisation >= 90 % |
| `bulk_qos_policies_applied` | Policy applied to multiple cells |
| `qos_data_exported` | Data export completed |
| `qos_sla_compliance_report_generated` | Compliance report generated |

All events include `tenant_id`, `event_type`, `reference_id`, and `processor: bytewax` fields.
