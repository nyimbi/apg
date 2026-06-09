# Multi-Channel Output

**Capability ID**: `mchn` | **Domain**: `common` | **Version**: `1.0.0`

## Description

MCHN provides APG applications with a tenant-scoped output runtime: output channels, approved templates, delivery policies, delivery routes, rendered messages and documents, delivery batches, provider receipts, output agents, UI

## Installation

```bash
pip install apg-common-mchn
```

## Provides

- `channel_routing`
- `format_rendering`
- `output_templates`
- `delivery_policy`
- `delivery_receipts`

## Requires

- `ntfy`
- `auth`
- `conf`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mchn/dashboard` | `mchn:view` | Overview |
| `/mchn/render` | `mchn:render` | Rendering |
| `/mchn/templates` | `mchn:manage_templates` | Rendering |
| `/mchn/routes` | `mchn:route` | Routing |
| `/mchn/channels` | `mchn:admin` | Channels |
| `/mchn/agents` | `mchn:admin` | Operations |
| `/mchn/analytics` | `mchn:view` | Operations |
| `/mchn/policies` | `mchn:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_channel()`
- `publish_template()`
- `create_delivery_policy()`
- `create_route()`
- `render_output()`
- `deliver_batch()`
- `record_receipt()`
- `create_record()`

_(See `service.py` for complete API.)_

## Interoperability

`mchn` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mchn;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MCHN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
