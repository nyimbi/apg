# AI Model Lifecycle Management

**Capability ID**: `mlcm` | **Domain**: `common` | **Version**: `1.0.0`

## Description

MLCM is the APG capability for governed AI model operations. It gives generated applications a tenant-scoped model registry, version lineage, evaluation gates, promotion approvals, deployment controls, drift response, rollback, retirement,

## Installation

```bash
pip install apg-common-mlcm
```

## Provides

- `model_lifecycle`
- `model_governance`
- `model_lifecycle_agent_composition`

## Requires

- `aicr`
- `moni`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mlcm/dashboard` | `mlcm:view` | Overview |
| `/mlcm/models` | `mlcm:view_models` | Registry |
| `/mlcm/versions` | `mlcm:manage_models` | Registry |
| `/mlcm/model-cards` | `mlcm:view_models` | Registry |
| `/mlcm/evaluation` | `mlcm:evaluate` | Quality |
| `/mlcm/baselines` | `mlcm:evaluate` | Quality |
| `/mlcm/promotion` | `mlcm:promote` | Release |
| `/mlcm/deployments` | `mlcm:deploy` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_model()`
- `create_version()`
- `record_evaluation()`
- `request_promotion()`
- `create_target()`
- `deploy_model()`
- `record_drift()`
- `record_drift_review()`

_(See `service.py` for complete API.)_

## Interoperability

`mlcm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mlcm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MLCM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
