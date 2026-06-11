# DTWN World-Class Improvements

15 high-impact improvements to elevate the Digital Twin Framework to production-grade quality.

---

## 1. Async-First Service Layer

**Current**: All methods are synchronous, blocking the event loop when composed with async APG capabilities.
**Improvement**: Rewrite all public methods as `async def`. Keep thin synchronous shims (`_sync_*`) for legacy callers. This unblocks concurrent telemetry ingestion, parallel simulation runs, and fan-out topology traversal without thread-pool overhead.

---

## 2. Persistent State Backend Abstraction

**Current**: All state lives in in-process Python dicts; restarts lose all data.
**Improvement**: Introduce a `StateBackend` protocol with `get`, `set`, `delete`, `scan` methods. Ship two implementations: `InMemoryBackend` (current, for tests) and `PostgresBackend` (asyncpg + JSONB, for production). The service constructor accepts `backend: StateBackend | None = None` and defaults to in-memory. This makes persistence a drop-in swap with zero service-layer changes.

---

## 3. Event Bus Integration

**Current**: Audit events are stored internally but never published outward; downstream capabilities are blind to twin state changes.
**Improvement**: Add an `EventBus` protocol with a single `publish(topic: str, payload: dict) -> None` method. Wire it into every state-mutating method. Default to `NoOpEventBus`. Production deployments inject a Kafka/Redis Streams/CloudEvents bus. This enables real-time push to `anom`, `edge`, `mchn`, and `audl` without polling.

---

## 4. Time-Series Telemetry Store

**Current**: Only the latest fused state is kept; historical telemetry measurements are lost after fusion.
**Improvement**: Maintain a time-ordered `list[TelemetrySample]` per twin (capped ring buffer, configurable `max_samples_per_twin`). Expose `get_telemetry_history(twin_id, start, end)` and `compute_rolling_stats(twin_id, metric, window)`. This powers trend analysis, drift detection, and calibration without an external TSDB.

---

## 5. Deterministic Simulation Reproducibility

**Current**: `simulation_outputs` in `twin_engine.py` uses implicit entropy; replaying the same inputs does not guarantee identical outputs.
**Improvement**: Require a `random_seed: int` parameter for all simulation runs. Store it in `SimulationRun`. Rerunning with the same seed + same state must produce bit-identical results. Enables regression testing, audit reproduction, and what-if scenario diffing.

---

## 6. Streaming Telemetry Ingestion via AsyncGenerator

**Current**: `bulk_ingest_telemetry` is a blocking list comprehension with no backpressure.
**Improvement**: Add `async def stream_telemetry(tenant_id, stream: AsyncIterator[dict]) -> AsyncIterator[dict]` that yields fused results as they arrive. Internally uses an asyncio queue with configurable `batch_size` and `flush_interval_ms`. This matches IoT gateway throughput patterns (10k+ msgs/s) without memory blowup.

---

## 7. Graph Topology Traversal

**Current**: `TopologyLink` records exist but there is no graph query layer — callers must filter manually.
**Improvement**: Add `get_topology_graph(tenant_id) -> dict[str, list[str]]` (adjacency list), `find_path(tenant_id, source_id, target_id) -> list[str]` (BFS), and `get_connected_component(tenant_id, twin_id) -> list[str]`. This unlocks impact analysis ("if this pump fails, which downstream twins are affected?") as a first-class service operation.

---

## 8. Federated What-If Analysis (Multi-Twin)

**Current**: `what_if_analysis` operates on a single twin in isolation.
**Improvement**: Add `federated_what_if(tenant_id, changes_by_twin: dict[str, dict], propagation_depth: int)` that applies changes to a root twin, traverses topology links up to `propagation_depth` hops, and propagates derived state changes to each connected twin. Returns a per-twin risk delta map. Enables fleet-level scenario planning.

---

## 9. Model Versioning and Rollback

**Current**: `model_update` overwrites confidence in place; old confidence values are gone.
**Improvement**: Store a `list[ModelVersion]` per model (id, confidence, calibration_evidence, updated_at, updated_by). Add `rollback_model(model_id, version_id)` that restores a prior version. Gate simulation runs against model versions so historical simulations remain reproducible. This is mandatory for ISO 55000 audit trails.

---

## 10. Structured Anomaly Alerting with Severity Routing

**Current**: `anomaly_detect` returns a dict but takes no action — downstream systems must poll.
**Improvement**: Add an `AlertRouter` protocol with `route(severity: str, twin_id: str, anomalies: list[dict]) -> None`. Wire it into `anomaly_detect`. Default to `LoggingAlertRouter`. Production deployments inject PagerDuty, Slack, or a webhook router. This closes the detect→alert loop without polling.

---

## 11. Digital Twin Health Score (Composite KPI)

**Current**: Health is surfaced only as a dashboard dict; there is no single queryable scalar.
**Improvement**: Add `compute_health_score(tenant_id, twin_id) -> float` returning 0.0–1.0, derived from: telemetry freshness, anomaly rate, prediction risk, model confidence, and maintenance urgency. Expose it on the dashboard dict. This gives operators a single number to threshold alerts on.

---

## 12. Shadow Mode (Read-Only Sync without State Mutation)

**Current**: All telemetry ingestion mutates twin state. There is no way to observe incoming data without affecting the live twin.
**Improvement**: Add a `shadow_mode: bool` flag to `ingest_telemetry` and `twin_sync`. In shadow mode, measurements are recorded and anomaly-checked but twin state is not mutated. Useful for new sensor validation, model testing against live feeds, and canary deployments.

---

## 13. Twin Snapshot and Restore

**Current**: No mechanism to capture a point-in-time twin state and restore it later.
**Improvement**: Add `snapshot_twin(tenant_id, twin_id, label: str) -> dict` (stores full twin state + state_version + timestamp under a named label) and `restore_twin_snapshot(tenant_id, twin_id, snapshot_id: str) -> dict`. Required for what-if rollback, maintenance window simulation, and disaster recovery testing.

---

## 14. Capability Metrics and Telemetry Instrumentation

**Current**: No internal instrumentation; operators cannot observe method call rates, latency, or error rates.
**Improvement**: Add a `MetricsCollector` protocol with `increment(metric, labels)` and `timing(metric, duration_ms, labels)`. Wrap all public methods with a lightweight decorator that records call count, duration, and success/failure. Default to `NoOpMetricsCollector`. Prometheus/OpenTelemetry implementations slot in without touching service logic.

---

## 15. Policy-as-Data Rule Engine Extension

**Current**: `evaluate_capability_rules` is a hand-coded Python function; adding new rules requires code changes and redeployment.
**Improvement**: Load rules from a JSON/YAML policy file at service init time. Each rule is `{condition_key, operator, threshold, action, reason}`. The engine evaluates rules against the context dict using the operator and threshold. New compliance requirements (GDPR data residency, sector-specific safety standards) are rule additions, not code changes. This decouples governance policy from service release cycles.
