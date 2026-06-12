# IOTD IoT Device Integration Capability

IOTD provides APG applications with a tenant-scoped device-operations runtime:
device identity, certificate ownership, fleet grouping, encrypted telemetry
ingestion, governed command dispatch, command acknowledgement, signed firmware
registration, firmware deployment, stale-device review, health reporting,
device-operation agents, UI metadata, theme tokens, audit evidence, and
Bytewax-backed lifecycle events.

The package stays dependency-light. Production device brokers, edge runtimes,
certificate authorities, audit sinks, digital twins, observability systems, and
Bytewax workers are represented as APG adapters in the executable contract and
are bound by the host application.

## What It Provides

- Device registry with tenant, owner, certificate, fleet, status, last-seen,
  and metadata fields.
- Telemetry ingestion with encryption, schema validation, and Bytewax event
  stream enforcement.
- Command center with dangerous-command approval and command acknowledgement.
- Firmware lifecycle with signed artifacts, target-device validation, and
  deployment records.
- Health reporting for online/offline devices, stale devices, pending commands,
  and unsigned firmware risk.
- First-class AI device-operation agents with runtime, role, scope,
  registration, and contribution-disclosure guardrails.
- Audit events for device, telemetry, command, firmware, health, and agent
  lifecycle changes.
- Device heartbeat tracking with low-battery auto-alerts.
- Metric threshold alerting with configurable min/val bands and alert levels.
- Bulk telemetry ingestion with per-event success/failure reporting.
- Fleet-level health dashboard with per-fleet breakdown and threshold breach counts.
- Per-device analytics (telemetry counts, command counts, heartbeat history).
- Device commissioning records with geolocation support.
- Device-to-device digital twin sync with delta reporting.
- Protocol bridge configuration (MQTT, CoAP, Modbus, OPC-UA, HTTP, AMQP, WebSocket).
- Offline event buffering with flush-on-reconnect policy.
- Device group management for logical fleet segmentation.
- Scheduled OTA firmware updates with approval gating.
- Full device decommissioning with pending-firmware cleanup.
- Real-time metric stream subscriptions.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped device, telemetry, command, firmware,
  health, audit, and agent records.
- `device_runtime.py` contains dependency-light schema, freshness, and health
  helpers.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe function helpers.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.iotd import IotdService

service = IotdService()
service.register_device(
    device_id="device-1",
    tenant_id="tenant-demo",
    device_key="device-key-1",
    owner_id="ops-owner",
    certificate_id="cert-1",
    fleet_id="line-a",
)
service.ingest_telemetry(
    event_id="event-1",
    tenant_id="tenant-demo",
    device_id="device-1",
    schema_name="temperature",
    payload={"timestamp": "now", "temperature": 42.5},
)
service.dispatch_command(
    command_id="command-1",
    tenant_id="tenant-demo",
    device_id="device-1",
    command="restart",
    dangerous=True,
    approval_id="approval-1",
)
```

## AI Device-Operation Agents

Register AI agents before they assist with fleet operations:

```python
agent = service.register_iotd_agent(
    tenant_id="tenant-demo",
    name="Fleet reviewer",
    runtime="codex",
    role="fleet_operator",
    scope="Review telemetry anomalies and command risk before shift handoff",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles cover fleet operations, telemetry review, command review, firmware
review, and security review.

## New Methods

### Heartbeat with auto-alert

Record a device heartbeat. Battery level below 10% automatically emits a
`low_battery_alert` audit event:

```python
service.device_heartbeat(
    device_id="device-1",
    tenant_id="tenant-demo",
    timestamp="2026-06-12T08:00:00Z",
    status="online",
    signal_strength=0.85,
    battery_level=0.08,  # triggers low_battery_alert audit event
)
```

### Metric threshold alerting

Define per-device, per-metric alert bands. Thresholds are evaluated
automatically on every `ingest_telemetry` call for numeric payload fields:

```python
service.set_threshold(
    device_id="device-1",
    tenant_id="tenant-demo",
    metric="temperature",
    min_val=0.0,
    max_val=85.0,
    alert_level="critical",
)
# Breach is detected and audited automatically when telemetry arrives.
# Explicit evaluation against a reading:
alert = service.threshold_alert(
    device_id="device-1",
    tenant_id="tenant-demo",
    metric="temperature",
    value=92.3,
)
# alert["breached"] == True, alert["direction"] == "above_max"
```

### Bulk telemetry ingestion

Ingest readings from many devices in one call. Returns a summary with
per-device `ok`/`error` outcomes — failed entries do not abort the batch:

```python
summary = service.bulk_telemetry_ingest(
    tenant_id="tenant-demo",
    device_readings=[
        {"event_id": "e-1", "device_id": "device-1", "schema_name": "temperature", "payload": {"temperature": 42.1}},
        {"event_id": "e-2", "device_id": "device-2", "schema_name": "pressure",    "payload": {"pressure": 1013.2}},
    ],
)
# summary["succeeded"], summary["failed"], summary["failures"]
```

### Fleet health dashboard

Single call for a tenant-wide operational snapshot:

```python
dashboard = service.fleet_health_dashboard(tenant_id="tenant-demo")
# Returns: total_devices, online_devices, stale_devices, fleet_breakdown,
#          threshold_breach_count, pending_firmware_schedules, ...
```

### Device commissioning with geolocation

Marks a device fully commissioned and records its physical installation context:

```python
record = service.device_commissioning(
    device_id="device-1",
    tenant_id="tenant-demo",
    installation_site="Warehouse-B-Rack-12",
    commissioned_by="field-engineer-42",
    notes="Mounted at 3m elevation, clear line of sight",
    geolocation={"lat": -1.286389, "lon": 36.817223, "altitude": 1661.0},
)
```

## API Reference

| Method | Description |
|---|---|
| `register_device(...)` | Provision a device with tenant, owner, certificate, fleet, and status |
| `ingest_telemetry(...)` | Ingest an encrypted, schema-validated telemetry event |
| `dispatch_command(...)` | Send a command; dangerous commands require explicit approval |
| `acknowledge_command(...)` | Record device acknowledgement for a dispatched command |
| `register_firmware(...)` | Register a signed firmware artifact |
| `deploy_firmware(...)` | Deploy firmware to a set of validated devices |
| `health_report(...)` | Generate a tenant health snapshot |
| `register_iotd_agent(...)` | Register an AI agent with runtime, role, and scope guardrails |
| `device_heartbeat(...)` | Record a heartbeat; auto-alerts on battery < 10% |
| `set_threshold(...)` | Configure a min/max alert band for a device metric |
| `alert_threshold(...)` | Alias for `set_threshold` |
| `threshold_alert(...)` | Evaluate a reading against a threshold explicitly |
| `bulk_telemetry_ingest(...)` | Batch ingest telemetry for multiple devices |
| `iot_analytics(...)` | Descriptive analytics for a fleet group over a period |
| `fleet_health_dashboard(...)` | Aggregated fleet health view for a tenant |
| `device_analytics(...)` | Per-device telemetry, command, and heartbeat analytics |
| `device_commissioning(...)` | Record physical commissioning with site and geolocation |
| `twin_sync(...)` | Sync desired state to a device digital twin; returns delta |
| `protocol_bridge(...)` | Configure a protocol bridge between two IoT protocols |
| `offline_buffer(...)` | Configure offline buffering with flush-on-reconnect |
| `device_group(...)` | Create a named logical group of devices |
| `device_firmware_update(...)` | Schedule a per-device OTA firmware update |
| `firmware_ota(...)` | Alias for `device_firmware_update` |
| `device_command(...)` | Alias for `dispatch_command` |
| `data_subscribe(...)` | Register a subscription for real-time device metric streams |
| `decommission(...)` | Decommission a device; clears pending firmware schedules |
| `stale_device_queue(...)` | List devices that exceed the stale-review threshold |
| `dashboard_summary(...)` | Compact summary counts for a tenant dashboard |
| `list_devices(...)` | List all registered devices, optionally filtered by tenant |
| `list_telemetry(...)` | List all telemetry events |
| `list_commands(...)` | List all commands |
| `list_firmware(...)` | List all firmware artifacts |
| `list_deployments(...)` | List all firmware deployments |
| `list_audit_events(...)` | List all audit events |
| `list_health_reports(...)` | List all health reports |
| `list_iotd_agents(...)` | List all registered AI agents |
| `list_thresholds(...)` | List all metric thresholds for a tenant |
| `list_heartbeats(...)` | List heartbeat history for a specific device |
| `list_commissioning(...)` | List all commissioning records for a tenant |
| `describe(...)` | Return the full capability contract for a tenant |
| `evaluate(...)` | Evaluate an arbitrary context against capability rules |
| `validate_batch_iot_mutation(...)` | Validate that a batch operation uses the required event bus |

## Composition

IOTD composes with:

- `auth` for identity, permissions, and device RBAC.
- `encr` for telemetry and device credential protection.
- `audl` for durable audit events.
- `conf` for tenant device and telemetry policy.
- `edge` for edge runtime deployment.
- `dtwn` for digital-twin synchronization.
- `logt` and `moni` for logs and monitoring.

Batch device mutation and telemetry pipelines must use the `bytewax`
event-stream adapter.

## World-Class Enhancements (v2.0)

The following 15 improvements are planned. Items 1, 2, 7, and 12 are
foundational and should be sequenced first.

1. **Full Async Service Layer** — Convert all I/O-bound methods to `async def`
   coroutines. Throughput multiplier of 10-100x under concurrent device load.

2. **Persistent Storage via Repository Pattern** — Introduce `IotdRepository`
   ABC with `PostgresIotdRepository` and `InMemoryIotdRepository`. Service
   becomes storage-agnostic; survives restarts; supports horizontal scaling.

3. **Streaming Telemetry Pipeline with Backpressure** — Replace synchronous
   batch loop with an async generator pipeline enforcing per-tenant rate limits
   and signalling backpressure to MQTT/CoAP producers via Bytewax.

4. **Device Shadow / Digital Twin Versioning** — Add optimistic concurrency
   control (`shadow_version` integer) to `twin_sync`. Concurrent updates raise
   `ConflictError`. Ring buffer of last-N states enables audit rollback.

5. **Protocol Adapter Plugin Architecture** — Replace hardcoded protocol set
   with a `IotdProtocolAdapter` ABC registry. MQTT, CoAP, Modbus, OPC-UA, and
   custom adapters registered at runtime with independent reconnection logic.

6. **Anomaly Detection Engine** — `IotdAnomalyDetector` maintains per-device
   rolling windows (mean, stddev, EMA) per metric. Flags N-sigma outliers as
   typed `AnomalyEvent` objects, self-calibrating to sensor drift.

7. **Event-Sourced Audit Log with CQRS Projection** — Replace mutable `_audit_events`
   dict with an append-only immutable event store. Projections derived from
   event replay enable time-travel debugging and compliance evidence chains.

8. **Certificate Lifecycle Management** — `CertificateLifecycle` subsystem
   tracks expiry, rotation schedules, and revocation status. Emits
   `cert_expiring_soon` alerts at 30/7/1-day lead times via configurable CA adapter.

9. **Edge-Aware Firmware Rollout with Canary Strategy** — `RolloutStrategy`
   (canary, blue-green, ring-based) controls the percentage of devices receiving
   a new firmware version per stage. Failed devices pause the rollout and trigger
   configurable auto-rollback.

10. **Geospatial Device Registry** — Formalise `geolocation` as a `GeoPoint`
    model (lat, lon, altitude, accuracy). Add `devices_within_radius`,
    `devices_in_polygon`, and `nearest_devices` query methods.

11. **Structured Telemetry Schema Registry** — `TelemetrySchemaRegistry` backed
    by JSON Schema with compatibility modes (BACKWARD / FORWARD / FULL). Rejects
    schema-breaking firmware changes before they reach the storage layer.

12. **Multi-Tenant Rate Limiting and Quota Enforcement** — `IotdQuotaEngine`
    enforces per-tenant limits on devices, telemetry events/s, commands/min, and
    firmware deployments/hr. Returns `429 QuotaExceeded` with retry-after metadata
    before any storage write.

13. **Reactive Health Watchdog with Auto-Remediation** — Async `HealthWatchdog`
    runs a continuous monitoring loop, escalates silence from `warning` to
    `critical`, and triggers configurable remediation actions (reboot command,
    alert escalation, `audl` ticket creation).

14. **Federated Fleet Management** — `FederationGateway` aggregates health,
    telemetry summaries, and command queues from remote `IotdService` instances
    via pluggable HTTP/gRPC transport. Cross-fleet commands require federated
    approval.

15. **OpenTelemetry Instrumentation** — Every service method instrumented with
    OTel spans (trace ID, device ID, tenant ID as attributes) and `iotd.*`
    meters (ingest rate, command latency, firmware deploy success rate) exported
    to OTLP with exemplar linkage.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/iotd/__init__.py capabilities/common/iotd/capability_contract.py capabilities/common/iotd/models.py capabilities/common/iotd/device_runtime.py capabilities/common/iotd/service.py capabilities/common/iotd/api.py capabilities/common/iotd/views.py capabilities/common/iotd/app.py capabilities/common/iotd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/iotd/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/iotd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/iotd --json
```

Live device brokers, edge gateways, certificate authorities, durable audit
stores, rendered UI, and Bytewax workers are integration concerns outside the
package proof.

---

© 2025 Datacraft — www.datacraft.co.ke
