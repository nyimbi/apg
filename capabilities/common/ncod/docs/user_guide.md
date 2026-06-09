# No-Code/Low-Code Builder

**Capability ID**: `ncod` | **Domain**: `common` | **Version**: `1.0.0`

## Description

NCOD is APG's governed no-code and low-code application composition capability. It gives tenants a deterministic app library, screen composer, component catalog, data modeler, workflow binding surface, script and connector extension

## Installation

```bash
pip install apg-common-ncod
```

## Provides

- `app_builder`
- `page_composer`
- `data_modeler`
- `workflow_binding`
- `script_extensions`

## Requires

- `wflo`
- `scpt`
- `auth`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ncod/dashboard` | `ncod:view` | Overview |
| `/ncod/apps` | `ncod:manage_apps` | Apps |
| `/ncod/builder` | `ncod:build` | Build |
| `/ncod/pages` | `ncod:build` | Build |
| `/ncod/data-models` | `ncod:build` | Build |
| `/ncod/components` | `ncod:build` | Build |
| `/ncod/workflows` | `ncod:build` | Automation |
| `/ncod/publishing` | `ncod:publish` | Release |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_app()`
- `add_page()`
- `add_component()`
- `define_data_model()`
- `bind_data_source()`
- `attach_workflow()`
- `create_theme_variant()`
- `add_script_extension()`

_(See `service.py` for complete API.)_

## Interoperability

`ncod` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ncod;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `NCOD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
