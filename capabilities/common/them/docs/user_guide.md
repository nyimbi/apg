# UI/UX Theming and Branding

**Capability ID**: `them` | **Domain**: `common` | **Version**: `1.0.0`

## Description

THEM is the APG capability for governed visual systems. It gives generated applications a composable runtime for tenant theme records, design tokens, brand assets, preview evidence, accessibility contrast gates, publication approvals,

## Installation

```bash
pip install apg-common-them
```

## Provides

- `theme_tokens`
- `brand_governance`
- `asset_libraries`
- `preview_workflows`
- `theme_publication_governance`

## Requires

- `conf`
- `auth`
- `i18n`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/them/dashboard` | `them:view` | Overview |
| `/them/themes` | `them:design` | Design |
| `/them/tokens` | `them:design` | Design |
| `/them/branding` | `them:manage_brand` | Brand |
| `/them/assets` | `them:manage_brand` | Brand |
| `/them/preview` | `them:view` | Review |
| `/them/agents` | `them:admin` | Automation |
| `/them/policies` | `them:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_theme()`
- `update_tokens()`
- `add_brand_asset()`
- `create_preview()`
- `publish_theme()`
- `register_them_agent()`
- `validate_agent_theme_action()`
- `validate_batch_theme_rollout()`

_(See `service.py` for complete API.)_

## Interoperability

`them` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use them;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `THEM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
