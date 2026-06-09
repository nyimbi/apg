# Plugin/Extension Framework

**Capability ID**: `plgn` | **Domain**: `common` | **Version**: `1.0.0`

## Description

PLGN gives APG applications a tenant-scoped extension system: plugin manifests, curated marketplace listings, permission review, sandbox policy, release gates, installation, activation, plugin-governance agents, UI metadata, theme

## Installation

```bash
pip install apg-common-plgn
```

## Provides

- `plugin_registry`
- `extension_marketplace`
- `permission_review`
- `sandbox_policy`
- `plugin_release_lifecycle`

## Requires

- `auth`
- `secu`
- `conf`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/plgn/dashboard` | `plgn:view` | Overview |
| `/plgn/marketplace` | `plgn:install` | Marketplace |
| `/plgn/plugins` | `plgn:view` | Plugins |
| `/plgn/manifests` | `plgn:publish` | Plugins |
| `/plgn/permissions` | `plgn:review` | Security |
| `/plgn/sandbox` | `plgn:review` | Security |
| `/plgn/releases` | `plgn:publish` | Release |
| `/plgn/agents` | `plgn:admin` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_plugin()`
- `install_plugin()`
- `uninstall_plugin()`
- `plugin_health_check()`
- `plugin_event_hook()`
- `plugin_sandboxed_execution()`
- `plugin_permission_check()`
- `plugin_marketplace_listing()`

_(See `service.py` for complete API.)_

## Interoperability

`plgn` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use plgn;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PLGN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
