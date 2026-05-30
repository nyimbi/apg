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
- Deterministic rules for tenant context, twin ownership, asset identity,
  model calibration, model confidence, simulation model approval, telemetry
  source authentication, telemetry measurements, production simulation
  approval, high-risk prediction review, AI twin agents, state-change audit,
  cross-tenant isolation, and Bytewax batch mutation streams.
- View models for dashboards, twins, topology, simulation lab, agents, audit,
  analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## How To Use It

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

Battery-conscious DTWN checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/dtwn/__init__.py capabilities/common/dtwn/models.py capabilities/common/dtwn/twin_engine.py capabilities/common/dtwn/service.py capabilities/common/dtwn/api.py capabilities/common/dtwn/views.py capabilities/common/dtwn/capability_contract.py capabilities/common/dtwn/app.py capabilities/common/dtwn/test_capability_contract.py capabilities/common/dtwn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/dtwn/test_capability_contract.py capabilities/common/dtwn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dtwn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dtwn --json
```
