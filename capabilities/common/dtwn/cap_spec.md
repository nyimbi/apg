# Digital Twin Framework Capability Specification

- **Capability Name**: Digital Twin Framework
- **Capability ID**: `dtwn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`dtwn` provides tenant-scoped digital-twin registration, model governance,
authenticated telemetry fusion, topology mapping, simulation execution,
prediction review, and audit evidence for APG applications. It turns the
executable capability contract into a dependency-light Python runtime that
generated applications can compose without requiring a live IoT broker,
geospatial service, computer-vision pipeline, machine controller, simulator,
time-series database, or prediction service.

## Current Runtime Behavior

The package currently provides:

- Digital twins with asset identity, owner, type, location, topology references,
  state, state version, and lifecycle status.
- Simulation model registration with calibration evidence, approval metadata,
  confidence thresholds, model type, and version.
- Authenticated telemetry ingestion that fuses measurements into twin state and
  advances deterministic state versions.
- Topology links between twins for dependency and asset-graph views.
- Simulation runs that use current twin state and approved model confidence to
  produce deterministic risk and recommendation outputs.
- Prediction records with high-risk review gates and review completion.
- Dashboard, topology, simulation-lab, prediction, telemetry, and audit view
  models.
- Append-only audit events with stable digests.

Runtime state is in-memory and dependency-light. Durable storage, IoT brokers,
edge gateways, geospatial services, computer-vision signals, machine control,
simulation engines, time-series databases, anomaly detection, and external
prediction services are integration boundaries.

## Provided Services

- `twin_registry`
- `simulation_models`
- `telemetry_fusion`
- `prediction_workflows`
- `asset_topology`

## Required Services

- `pred` for production prediction workflows.
- `iotd` for IoT device and telemetry ingestion integration.
- `geos` for geospatial context enrichment.
- `cvsn` for computer-vision signal enrichment.

Optional composition points are `aicr`, `anom`, `edge`, and `mchn`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The active configuration includes twin ownership
and asset-identity constraints, telemetry source authentication, simulation
model and approval constraints, prediction review thresholds, UI route metadata,
and theme metadata.

Tenant context is required for executable operations.

## Rules

- `tenant_context_required`
- `twin_requires_owner`
- `simulation_requires_model`
- `telemetry_requires_authenticated_source`
- `simulation_requires_approval`
- `high_risk_prediction_requires_review`

The service evaluates these rules before state changes that create twins,
ingest telemetry, run simulations, or record predictions.

## UI

The package exposes 8 APG Python UI route contracts through `views.py` and the
package semantic model:

- dashboard
- twins
- models
- telemetry
- simulations
- predictions
- topology
- settings

## Theme

The package uses the `dtwn_digital_twin_ops` APG theme contract, including
twin-card, topology-view, simulation-lab, and telemetry-panel component tokens.

## Verification

Focused package verification should include:

```bash
./.venv/bin/python -m py_compile capabilities/common/dtwn/*.py capabilities/common/dtwn/tests/*.py
./.venv/bin/pytest -q capabilities/common/dtwn/test_capability_contract.py capabilities/common/dtwn/tests
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dtwn --json
```
