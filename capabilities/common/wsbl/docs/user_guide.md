# Website Builder

**Capability ID**: `wsbl` | **Domain**: `common` | **Version**: `1.0.0`

## Description

WSBL is the APG capability for governed website and page composition. It gives generated applications a composable runtime for tenant sites, domains, pages, components, public-site controls, publishing, rollback, accessibility,

## Installation

```bash
pip install apg-common-wsbl
```

## Provides

- `site_management`
- `page_composition`
- `component_library`
- `publishing_workflows`
- `site_theming`

## Requires

- `them`
- `auth`
- `ncod`
- `accs`
- `cons`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/wsbl/dashboard` | `wsbl:view` | Overview |
| `/wsbl/sites` | `wsbl:manage_sites` | Sites |
| `/wsbl/pages` | `wsbl:build` | Pages |
| `/wsbl/editor` | `wsbl:build` | Build |
| `/wsbl/components` | `wsbl:build` | Build |
| `/wsbl/publishing` | `wsbl:publish` | Release |
| `/wsbl/analytics` | `wsbl:view` | Operations |
| `/wsbl/agents` | `wsbl:admin` | Automation |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_site()`
- `register_domain()`
- `validate_domain()`
- `create_component()`
- `review_component()`
- `create_page()`
- `add_page_section()`
- `create_publish_request()`

_(See `service.py` for complete API.)_

## Interoperability

`wsbl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use wsbl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `WSBL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
