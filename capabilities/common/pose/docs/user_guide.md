# Pose Estimation

**Capability ID**: `pose` | **Domain**: `common` | **Version**: `1.0.0`

## Description

POSE is APG's governed human pose-estimation capability. It provides tenant-scoped model registration, tracking sessions, frame capture, pose estimates, biomechanical analysis, 3D reconstruction records, AI pose-agent

## Installation

```bash
pip install apg-common-pose
```

## Provides

- `pose_estimation`
- `multi_person_tracking`
- `biomechanical_analysis`
- `pose_3d_reconstruction`
- `edge_pose_inference`

## Requires

- `cvsn`
- `aicr`
- `mlcm`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pose/dashboard` | `pose:view` | Overview |
| `/pose/estimate` | `pose:estimate` | Runtime |
| `/pose/tracking` | `pose:track` | Runtime |
| `/pose/analysis` | `pose:analyze` | Analysis |
| `/pose/reconstruction` | `pose:analyze` | Analysis |
| `/pose/sessions` | `pose:view` | Analysis |
| `/pose/models` | `pose:manage_models` | Models |
| `/pose/quality` | `pose:view` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_model()`
- `start_session()`
- `record_frame()`
- `estimate_pose()`
- `analyze_pose()`
- `reconstruct_3d()`
- `register_pose_agent()`
- `change_session_state()`

_(See `service.py` for complete API.)_

## Interoperability

`pose` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pose;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `POSE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
