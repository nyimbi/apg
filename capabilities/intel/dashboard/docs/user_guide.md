# Intelligence Dashboard

**Capability ID**: `intel_dashboard` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_dashboard` is an executable APG capability package for building governed intelligence-dashboard applications. It gives generated APG apps a concrete runtime for lawful authority, dashboard workspaces, dashboards, data sources,

## Installation

```bash
pip install apg-intel-dashboard
```

## Provides

- `dashboard_authority_workflow`
- `dashboard_workspace_workflow`
- `dashboard_composition_workflow`
- `dashboard_source_workflow`
- `dashboard_metric_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-dashboard/dashboard` | `intel_dashboard:view` | Overview |
| `/intel-dashboard/authorities` | `intel_dashboard:authorities` | Governance |
| `/intel-dashboard/workspaces` | `intel_dashboard:workspaces` | Planning |
| `/intel-dashboard/dashboards` | `intel_dashboard:dashboards` | Composition |
| `/intel-dashboard/sources` | `intel_dashboard:sources` | Data |
| `/intel-dashboard/metrics` | `intel_dashboard:metrics` | Data |
| `/intel-dashboard/widgets` | `intel_dashboard:widgets` | Composition |
| `/intel-dashboard/filters` | `intel_dashboard:filters` | Composition |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `record_dashboard()`
- `record_source()`
- `record_metric()`
- `record_widget()`
- `record_filter()`
- `record_view()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_dashboard` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_dashboard;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_DASHBOARD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
