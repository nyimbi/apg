# Intelligence Analytics

**Capability ID**: `intel_analytics` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_analytics` is an executable APG capability for governed, evidence-backed intelligence analytics. It can be composed into generated APG applications that need threat analytics, fraud analytics, public-safety

## Installation

```bash
pip install apg-intel-analytics
```

## Provides

- `analytics_authority_workflow`
- `analytics_workspace_workflow`
- `analytics_dataset_workflow`
- `analytics_feature_workflow`
- `analytics_model_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-analytics/dashboard` | `intel_analytics:view` | Overview |
| `/intel-analytics/authorities` | `intel_analytics:authorities` | Governance |
| `/intel-analytics/workspaces` | `intel_analytics:workspaces` | Planning |
| `/intel-analytics/datasets` | `intel_analytics:datasets` | Data |
| `/intel-analytics/features` | `intel_analytics:features` | Data |
| `/intel-analytics/models` | `intel_analytics:models` | Analysis |
| `/intel-analytics/runs` | `intel_analytics:runs` | Analysis |
| `/intel-analytics/insights` | `intel_analytics:insights` | Analysis |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `register_dataset()`
- `record_feature_set()`
- `record_model()`
- `record_run()`
- `record_insight()`
- `record_dashboard()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_analytics` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_analytics;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_ANALYTICS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
