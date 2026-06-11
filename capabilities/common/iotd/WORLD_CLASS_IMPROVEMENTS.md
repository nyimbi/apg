# IOTD World-Class Improvements

**Capability**: IoT Device Integration (`iotd`)
**Domain**: `common`
**Author**: Nyimbi Odero
**Date**: 2026-06-11

---

## 1. Full Async Service Layer

The entire service is synchronous. All I/O-bound operations (telemetry ingestion, command dispatch, firmware downloads, health checks) should be `async def` coroutines using `asyncio`. This enables non-blocking integration with FastAPI, async message brokers, and edge gateways without thread-pool overhead.

**Impact**: Throughput multiplier of 10-100x under concurrent device load.

---

## 2. Persistent Storage Backend via Repository Pattern

`_devices`, `_telemetry`, etc. are plain in-memory dicts. Introduce a `IotdRepository` abstract base with `PostgresIotdRepository` and `InMemoryIotdRepository` implementations. The service becomes storage-agnostic, enabling horizontal scaling and crash recovery without refactoring callers.

**Impact**: Production viability; survives restarts; enables read replicas.

---

## 3. Streaming Telemetry Pipeline with Backpressure

`bulk_telemetry_ingest` loops synchronously. Replace with an async generator pipeline that yields processed events, applies per-tenant rate limits, and signals backpressure to upstream producers (MQTT brokers, CoAP endpoints). Integrate with Bytewax's native async push interface.

**Impact**: Eliminates memory spikes on high-frequency sensor bursts; enables real-time downstream consumers.

---

## 4. Device Shadow / Digital Twin Versioning

`twin_sync` overwrites state without version tracking. Implement optimistic concurrency control using a monotonic `shadow_version` integer. Concurrent updates raise `ConflictError` with the expected/actual versions. Keep a `shadow_history` ring buffer of the last N states for rollback.

**Impact**: Eliminates lost-update bugs in multi-operator environments; enables state audit trails.

---

## 5. Protocol Adapter Plugin Architecture

`protocol_bridge` hardcodes a set of supported protocols. Replace with a plugin registry (`IotdProtocolAdapter` ABC) that allows MQTT, CoAP, Modbus, OPC-UA, and custom adapters to be registered at runtime. Each adapter handles frame parsing, QoS mapping, and reconnection logic independently.

**Impact**: Zero-code integration of new field protocols; enables third-party protocol extensions.

---

## 6. Anomaly Detection Engine

Add a lightweight statistical anomaly detector (`IotdAnomalyDetector`) that maintains per-device rolling windows (mean, stddev, exponential moving average) for each telemetry metric. Flag readings outside N-sigma bands as anomalies and emit typed `AnomalyEvent` objects rather than raw audit strings.

**Impact**: Replaces manual threshold configuration with self-calibrating baselines; catches drift, spikes, and drop-outs automatically.

---

## 7. Event-Sourced Audit Log with CQRS Projection

`_audit_events` is a flat dict mutated in-place. Replace with an append-only event store where every state change emits an immutable `IotdDomainEvent`. Projections (current device state, fleet health view, alert history) are derived from event replay. This enables time-travel debugging and compliance evidence without extra instrumentation.

**Impact**: Full audit immutability; replayable state; compliance-ready evidence chain.

---

## 8. Certificate Lifecycle Management

Certificates are stored as opaque IDs. Add a `CertificateLifecycle` subsystem tracking expiry dates, rotation schedules, and revocation status against a configurable CA adapter. Emit `cert_expiring_soon` alerts at configurable lead times (e.g., 30/7/1 days).

**Impact**: Prevents silent device authentication failures from expired certificates in production fleets.

---

## 9. Edge-Aware Firmware Rollout with Canary Strategy

`deploy_firmware` pushes to all devices at once. Introduce `RolloutStrategy` (canary, blue-green, ring-based) where the service controls the percentage of devices receiving a new firmware version at each stage. Failed devices pause the rollout and trigger a configurable auto-rollback policy.

**Impact**: Eliminates fleet-wide outages from bad firmware; matches industry OTA best practices.

---

## 10. Geospatial Device Registry

Device commissioning records `geolocation` as a free dict. Formalize it with a `GeoPoint` model (lat, lon, altitude, accuracy) and add geospatial query methods: `devices_within_radius`, `devices_in_polygon`, `nearest_devices`. Enable region-based fleet operations and proximity-based alert routing.

**Impact**: Enables location-aware fleet management and site-based dashboards without external GIS tools.

---

## 11. Structured Telemetry Schema Registry

`schema_name` is a bare string with no enforcement beyond field presence. Build a `TelemetrySchemaRegistry` backed by JSON Schema documents. Validate payloads against the registered schema version, reject schema-breaking changes with a compatibility mode (BACKWARD / FORWARD / FULL), and store schema evolution history.

**Impact**: Eliminates silent payload drift; enforces contract between firmware teams and backend processors.

---

## 12. Multi-Tenant Rate Limiting and Quota Enforcement

No per-tenant limits exist. Add `IotdQuotaEngine` with configurable limits for: devices per tenant, telemetry events per second, commands per minute, firmware deployments per hour. Return `429 QuotaExceeded` with retry-after metadata before any storage write.

**Impact**: Prevents noisy-neighbour saturation in multi-tenant deployments; provides SLA enforcement primitives.

---

## 13. Reactive Health Watchdog with Auto-Remediation

`health_report` is a passive snapshot. Add an async `HealthWatchdog` that runs a continuous monitoring loop: polls device heartbeats, escalates from `warning` to `critical` after configurable silence intervals, and triggers configurable remediation actions (reboot command, alert escalation, ticket creation via `audl` adapter).

**Impact**: Turns passive health snapshots into active reliability infrastructure; reduces mean time to detect.

---

## 14. Federated Fleet Management

All state is scoped to a single service instance. Add `FederationGateway` supporting multi-region and multi-cluster fleet views by aggregating health, telemetry summaries, and command queues from remote `IotdService` instances via a pluggable transport (HTTP, gRPC). Cross-fleet commands require federated approval.

**Impact**: Enables global device operations from a single control plane without centralising all telemetry data.

---

## 15. OpenTelemetry Instrumentation

No distributed tracing or metrics exist. Instrument every service method with OpenTelemetry spans (trace ID, device ID, tenant ID as attributes), emit `iotd.*` meters (telemetry ingest rate, command latency, firmware deploy success rate), and export to OTLP. Add exemplar linkage between metrics and traces.

**Impact**: Full observability without code changes in consuming applications; enables SLO alerting and performance regression detection.

---

*Each improvement is independent and can be delivered as a discrete sprint. Items 1, 2, 7, and 12 are foundational and should be sequenced first.*
