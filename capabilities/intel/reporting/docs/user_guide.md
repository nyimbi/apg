# Intelligence Reporting

**Capability ID**: `intel_reporting` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_reporting` is an executable APG capability package for building governed intelligence-reporting applications. It gives generated APG apps a concrete runtime for lawful authority, reporting workspaces, templates, products,

## Installation

```bash
pip install apg-intel-reporting
```

## Provides

- `reporting_authority_workflow`
- `reporting_workspace_workflow`
- `reporting_template_workflow`
- `reporting_product_workflow`
- `reporting_section_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-reporting/dashboard` | `intel_reporting:view` | Overview |
| `/intel-reporting/authorities` | `intel_reporting:authorities` | Governance |
| `/intel-reporting/workspaces` | `intel_reporting:workspaces` | Planning |
| `/intel-reporting/templates` | `intel_reporting:templates` | Products |
| `/intel-reporting/products` | `intel_reporting:products` | Products |
| `/intel-reporting/sections` | `intel_reporting:sections` | Products |
| `/intel-reporting/citations` | `intel_reporting:citations` | Evidence |
| `/intel-reporting/approvals` | `intel_reporting:approvals` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `record_workspace()`
- `record_template()`
- `record_product()`
- `record_section()`
- `record_citation()`
- `record_approval()`
- `record_distribution()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_reporting` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_reporting;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_REPORTING_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
