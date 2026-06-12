# DTWN - Digital Twin Framework

DTWN is the APG capability for governed digital twins, simulation models,
authenticated telemetry fusion, topology mapping, prediction review, AI
twin-agent governance, audit, and lifecycle stream metadata. It gives generated
APG applications a tenant-aware digital-twin lifecycle that can be composed
with prediction, IoT, geospatial, computer-vision, anomaly, edge, machine, and
audit capabilities.

The implementation is dependency-light and side-effect free. It records twin
state, model evidence, telemetry samples, topology links, simulation outputs,
prediction review state, AI twin-agent records, and audit events without
calling a live IoT broker, geospatial service, computer-vision pipeline,
machine controller, simulator, time-series database, or prediction service.

## What DTWN Provides

- Digital twins with tenant, asset identity, owner, type, location, topology
  references, state, state version, and lifecycle status.
- Simulation models with owner, type, version, calibration evidence, approval
  metadata, confidence threshold, and status.
- Authenticated telemetry ingestion that fuses measurements into twin state and
  advances deterministic state versions.
- Topology links between twins for dependency and asset graph views.
- Simulation runs that use current twin state and approved model confidence to
  produce deterministic risk and recommendation outputs.
- Prediction records with risk score, confidence, horizon, recommendation, and
  high-risk review state.
- First-class AI twin agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Anomaly detection against configurable threshold bounds per metric.
- Failure prediction with risk scoring and maintenance recommendations.
- What-if analysis comparing hypothetical vs. current twin state and risk.
- Parameter optimisation returning optimal values within declared bounds.
- Sensor calibration records with offset/scale tracking.
- Energy optimisation recommendations from an asset energy profile.
- Lifecycle stage tracking (design through dispose) with full history.
- Maintenance prediction with RUL estimation and urgency classification.
- Event replay to reconstruct twin state from a sequence of historical events.
- Performance comparison of simulated metrics vs. actual telemetry.
- Twin comparison (state diff) across two assets.
- Bulk twin creation and bulk telemetry ingestion.
- JSON and CSV export for twins and predictions.
- ISO 55000 compliance reporting.
- Deterministic rules for tenant context, twin ownership, asset identity,
  model calibration, model confidence, simulation model approval, telemetry
  source authentication, telemetry measurements, production simulation
  approval, high-risk prediction review, AI twin agents, state-change audit,
  cross-tenant isolation, and Bytewax batch mutation streams.
- View models for dashboards, twins, topology, simulation lab, agents, audit,
  analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## Quick Start

```python
from capabilities.common.dtwn.service import DtwnService

service = DtwnService()
tenant_id = "tenant-dtwn"

twin = service.create_twin(
    twin_id="twin-pump-1",
    tenant_id=tenant_id,
    asset_id="asset-pump-1",
    name="Pump 1",
    owner="operations",
    twin_type="pump",
    location={"site": "plant-a", "lat": 1.2, "lon": 36.8},
    initial_state={"temperature": 42, "vibration": 18},
)

model = service.register_simulation_model(
    model_id="model-pump-risk",
    tenant_id=tenant_id,
    name="Pump risk model",
    version="1.0.0",
    owner="model-risk",
    model_type="physics_ml_hybrid",
    calibration_evidence="calibration-report-001",
    approved_by="chief-engineer",
    confidence=0.91,
)

agent = service.register_twin_agent(
    tenant_id=tenant_id,
    agent_id="codex-twin-reviewer",
    name="Codex Twin Reviewer",
    runtime="codex",
    role="prediction_reviewer",
    scope="Review high-risk twin predictions and simulation evidence.",
    contribution_disclosed=True,
    policy_ref="policy:dtwn:agents:v1",
)

telemetry = service.ingest_telemetry(
    sample_id="tel-1",
    tenant_id=tenant_id,
    twin_id=twin["id"],
    source_id="iot-gateway-1",
    source_type="iot",
    authenticated=True,
    measurements={"temperature": 88, "vibration": 64},
    geospatial_context={"site": "plant-a"},
)

simulation = service.run_simulation(
    run_id="sim-1",
    tenant_id=tenant_id,
    twin_id=twin["id"],
    model_id=model["id"],
    scenario="high load",
    environment="production",
    approved_by="shift-lead",
)
```

Use `api.py` when composing generated application handlers, and use `views.py`
for framework-neutral screen state:

```python
from capabilities.common.dtwn.views import dashboard_model, twin_agents_model

dashboard = dashboard_model(service, tenant_id)
agents = twin_agents_model(service, tenant_id)
```

## Public API

| Method | Signature summary | Purpose |
|--------|-------------------|---------|
| `create_twin` | `(twin_id, tenant_id, asset_id, name, owner, twin_type, ...)` | Register a new digital twin |
| `register_simulation_model` | `(model_id, tenant_id, name, version, owner, ...)` | Register a simulation model with calibration evidence |
| `ingest_telemetry` | `(sample_id, tenant_id, twin_id, source_id, ..., measurements)` | Fuse sensor measurements into twin state |
| `twin_sync` | `(tenant_id, twin_id, sensor_data, source_id)` | Real-time physical-to-digital sync shorthand |
| `state_update` | `(tenant_id, twin_id, updates, updated_by)` | Partial state update without sensor auth requirement |
| `link_topology` | `(link_id, tenant_id, source_twin_id, target_twin_id, relationship)` | Record a directed dependency between twins |
| `run_simulation` | `(run_id, tenant_id, twin_id, model_id, scenario, environment, approved_by)` | Execute a simulation run |
| `simulate_scenario` | `(tenant_id, twin_id, scenario_parameters, duration, ...)` | Parameterised scenario simulation |
| `record_prediction` | `(prediction_id, tenant_id, twin_id, model_id, risk_score, ...)` | Store a prediction with review gating |
| `review_prediction` | `(prediction_id, tenant_id, reviewer)` | Mark a high-risk prediction as reviewed |
| `anomaly_detect` | `(tenant_id, twin_id, threshold_config)` | Threshold-based anomaly scan on twin state |
| `predict_failure` | `(tenant_id, twin_id, horizon_days, model_id)` | Probability-of-failure within horizon |
| `what_if_analysis` | `(tenant_id, twin_id, changes, analysis_id)` | Hypothetical state change + risk delta |
| `federated_what_if` | `(tenant_id, changes_by_twin, propagation_depth)` | Multi-twin what-if with topology propagation |
| `optimise_parameters` | `(tenant_id, twin_id, objective, parameter_bounds, iterations)` | Parameter optimisation within declared bounds |
| `sensor_calibrate` | `(tenant_id, twin_id, sensor_id, calibration_data, calibrated_by)` | Record sensor offset/scale calibration |
| `model_update` | `(tenant_id, model_id, calibration_data, calibrated_by)` | Update model confidence from MAE evidence |
| `event_replay` | `(tenant_id, twin_id, events, replayed_by)` | Reconstruct state from historical event list |
| `maintenance_predict` | `(tenant_id, twin_id, model_type, horizon_days)` | RUL and maintenance urgency estimate |
| `energy_optimise` | `(tenant_id, twin_id, energy_profile, target_reduction_pct)` | Energy consumption optimisation recommendations |
| `lifecycle_track` | `(tenant_id, twin_id, lifecycle_stage, metadata, tracked_by)` | Record lifecycle stage transition |
| `twin_compare` | `(tenant_id, twin_id_a, twin_id_b)` | State diff between two twins |
| `twin_dashboard` | `(tenant_id, twin_id)` | Per-twin health and activity summary |
| `performance_comparison` | `(tenant_id, twin_id, period)` | Simulated vs. actual metric coverage |
| `twin_event_log` | `(tenant_id, twin_id, event_type, limit)` | Filtered audit event log for a twin |
| `register_twin_agent` | `(tenant_id, agent_id, name, runtime, role, scope, ...)` | Register an AI twin agent |
| `change_twin_status` | `(tenant_id, twin_id, status, reason, actor)` | Lifecycle status transition with audit |
| `validate_batch_twin_mutation` | `(tenant_id, event_stream, actor)` | Validate Bytewax batch mutation stream |
| `bulk_create_twins` | `(tenant_id, twins, owner)` | Create multiple twins in one call |
| `bulk_ingest_telemetry` | `(tenant_id, samples)` | Ingest multiple telemetry samples in one call |
| `export_twins` | `(tenant_id, fmt)` | Export twin records as JSON or CSV |
| `export_predictions` | `(tenant_id, fmt)` | Export predictions as JSON or CSV |
| `compliance_report` | `(tenant_id, standard)` | ISO 55000 lifecycle/maintenance coverage report |
| `health_check` | `(tenant_id)` | Service health and record counts |
| `dashboard_summary` | `(tenant_id)` | Tenant-level aggregate dashboard |
| `list_twins / list_models / list_telemetry / ...` | `(tenant_id)` | Filtered list accessors for all record types |
| `describe` | `(tenant_id)` | Capability contract |
| `evaluate` | `(context)` | Rule engine evaluation |

## New Methods

### anomaly_detect — threshold-based anomaly scanning

```python
result = service.anomaly_detect(
    tenant_id=tenant_id,
    twin_id="twin-pump-1",
    threshold_config={
        "temperature": {"min": 0, "max": 85, "severity": "critical"},
        "vibration":   {"min": 0, "max": 60, "severity": "warning"},
    },
)
# result["anomalies"] -> list of {metric, current_value, direction, severity}
# result["anomaly_count"] -> int
```

### what_if_analysis — risk delta for hypothetical changes

```python
analysis = service.what_if_analysis(
    tenant_id=tenant_id,
    twin_id="twin-pump-1",
    changes={"temperature": 95, "vibration": 72},
)
# analysis["risk_delta"]           -> float (positive = worse)
# analysis["projected_risk_score"] -> float 0-1
# analysis["recommendation"]       -> "proceed" | "review"
```

### maintenance_predict — remaining useful life estimate

```python
maintenance = service.maintenance_predict(
    tenant_id=tenant_id,
    twin_id="twin-pump-1",
    model_type="rul",       # "rul" | "condition_based" | "time_based" | "failure_probability"
    horizon_days=30,
)
# maintenance["estimated_rul_days"]  -> int
# maintenance["maintenance_urgency"] -> "immediate" | "soon" | "scheduled"
# maintenance["recommended_actions"] -> list[str]
```

### lifecycle_track — ISO 55000 asset lifecycle

```python
stage = service.lifecycle_track(
    tenant_id=tenant_id,
    twin_id="twin-pump-1",
    lifecycle_stage="operate",   # design | manufacture | commission | operate | maintain | decommission | dispose
    metadata={"commissioning_date": "2025-01-15"},
    tracked_by="asset-manager",
)
# stage["history"] -> list of {stage, timestamp}
```

### compliance_report — ISO 55000 coverage check

```python
report = service.compliance_report(tenant_id=tenant_id, standard="iso55000")
# report["compliant"]                    -> bool
# report["lifecycle_coverage_pct"]       -> float
# report["maintenance_coverage_pct"]     -> float
# report["twins_with_lifecycle_tracking"] -> int
```

## World-Class Enhancements (v2.0)

The following 15 improvements are specified in `WORLD_CLASS_IMPROVEMENTS.md`
and define the production-grade evolution path for this capability.

| # | Enhancement | Summary |
|---|-------------|---------|
| 1 | **Async-First Service Layer** | All public methods become `async def`; thin sync shims for legacy callers. Enables concurrent telemetry ingestion and parallel simulation without thread-pool overhead. |
| 2 | **Persistent State Backend Abstraction** | `StateBackend` protocol with `InMemoryBackend` (tests) and `PostgresBackend` (asyncpg + JSONB, production). Zero service-layer changes to swap persistence. |
| 3 | **Event Bus Integration** | `EventBus` protocol wired into every state-mutating method. Enables real-time push to `anom`, `edge`, `mchn`, `audl` without polling. Default: `NoOpEventBus`. |
| 4 | **Time-Series Telemetry Store** | Ring-buffer of `TelemetrySample` per twin. Exposes `get_telemetry_history(twin_id, start, end)` and `compute_rolling_stats(twin_id, metric, window)` for drift detection and trend analysis. |
| 5 | **Deterministic Simulation Reproducibility** | `random_seed: int` required for all simulation runs; stored in `SimulationRun`. Same seed + same state = bit-identical output. Enables regression testing and scenario diffing. |
| 6 | **Streaming Telemetry Ingestion** | `async def stream_telemetry(tenant_id, stream: AsyncIterator[dict])` yields fused results as they arrive using an asyncio queue with configurable `batch_size` and `flush_interval_ms`. |
| 7 | **Graph Topology Traversal** | `get_topology_graph`, `find_path` (BFS), `get_connected_component` as first-class service operations. Enables impact analysis: "if pump fails, which downstream twins are affected?" |
| 8 | **Federated What-If Analysis** | `federated_what_if(tenant_id, changes_by_twin, propagation_depth)` applies changes to a root twin, propagates derived state across topology up to N hops, returns per-twin risk delta map. |
| 9 | **Model Versioning and Rollback** | Full `list[ModelVersion]` per model. `rollback_model(model_id, version_id)` restores a prior confidence version. Historical simulations remain reproducible. Required for ISO 55000 audit trails. |
| 10 | **Structured Anomaly Alerting** | `AlertRouter` protocol wired into `anomaly_detect`. Default: `LoggingAlertRouter`. Production: PagerDuty/Slack/webhook. Closes the detect-alert loop without polling. |
| 11 | **Digital Twin Health Score** | `compute_health_score(tenant_id, twin_id) -> float` (0.0–1.0) derived from telemetry freshness, anomaly rate, prediction risk, model confidence, and maintenance urgency. Single KPI for threshold alerts. |
| 12 | **Shadow Mode** | `shadow_mode: bool` flag on `ingest_telemetry` and `twin_sync`. Anomalies are checked and recorded but twin state is not mutated. Safe for new sensor validation and canary deployments. |
| 13 | **Twin Snapshot and Restore** | `snapshot_twin(tenant_id, twin_id, label)` and `restore_twin_snapshot(tenant_id, twin_id, snapshot_id)` for what-if rollback, maintenance window simulation, and disaster recovery testing. |
| 14 | **Capability Metrics Instrumentation** | `MetricsCollector` protocol wrapping all public methods with call count, duration, and success/failure. Default: `NoOpMetricsCollector`. Prometheus/OpenTelemetry implementations slot in without touching service logic. |
| 15 | **Policy-as-Data Rule Engine** | Rules loaded from JSON/YAML at service init: `{condition_key, operator, threshold, action, reason}`. New compliance requirements (GDPR, sector safety standards) are config changes, not code changes. Decouples governance from release cycles. |

## Contract And Composition

`get_capability_contract()` publishes:

- configuration for twins, telemetry, simulation, twin agents, governance,
  observability, adapters, UI, and theme;
- JSON-schema-style configuration requirements;
- deterministic rule engine;
- UI routes under `/dtwn`;
- theme tokens under `dtwn_digital_twin_ops`;
- Bytewax lifecycle-stream metadata.

DTWN depends on `pred`, `iotd`, `geos`, and `cvsn`. Optional adapter boundaries
are `aicr`, `anom`, `edge`, `mchn`, `bytewax`, and `audl`.

## Guardrail Summary

DTWN denies or requires review when:

- tenant context is missing;
- a twin lacks owner or asset identity;
- a simulation model lacks calibration evidence or minimum confidence;
- telemetry lacks authenticated source identity or measurements;
- a simulation lacks an approved model;
- a production simulation lacks approval evidence;
- a high-risk prediction lacks review evidence;
- an AI twin agent is unregistered, uses an unsupported runtime or role, lacks
  explicit scope, or has undisclosed contributions;
- a twin state change lacks reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch twin mutation does not declare Bytewax.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/dtwn/__init__.py capabilities/common/dtwn/models.py capabilities/common/dtwn/twin_engine.py capabilities/common/dtwn/service.py capabilities/common/dtwn/api.py capabilities/common/dtwn/views.py capabilities/common/dtwn/capability_contract.py capabilities/common/dtwn/app.py capabilities/common/dtwn/test_capability_contract.py capabilities/common/dtwn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dtwn/test_capability_contract.py capabilities/common/dtwn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dtwn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dtwn --json
```
