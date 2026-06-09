# Self-Service BI

**Capability ID**: `bia_sbi` | **Domain**: `bia` | **Version**: `1.0.0`

## Description

The Self-Service BI capability (bia_sbi) provides a drag-and-drop visual chart builder, natural-language query (NLQ) processing, a governed data catalogue with tiered access control, user sandboxes with row limits and auto-expiry, and a template gallery — giving business users governed self-service analytics without requiring SQL expertise.

## Installation

```bash
pip install apg-bia-sbi
```

## Provides

- `drag_drop_visual_builder`
- `natural_language_queries`
- `governed_data_catalogue`
- `user_sandboxes`
- `template_gallery`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `nlpc`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bia/sbi/` | `bia_sbi:view` | Overview |
| `/bia/sbi/builder` | `bia_sbi:build` | Builder |
| `/bia/sbi/workspaces/<id>` | `bia_sbi:build` | Builder |
| `/bia/sbi/ask` | `bia_sbi:query` | Query |
| `/bia/sbi/catalogue` | `bia_sbi:catalogue` | Catalogue |
| `/bia/sbi/catalogue/<id>` | `bia_sbi:catalogue` | Catalogue |
| `/bia/sbi/sandboxes` | `bia_sbi:sandbox` | Sandboxes |
| `/bia/sbi/sandboxes/<id>` | `bia_sbi:sandbox` | Sandboxes |

## Key Service Methods

- `describe()`
- `natural_language_query()`
- `submit_nlq()`
- `list_nlq_history()`
- `suggested_insights()`
- `drag_and_drop_report_create()`
- `data_catalogue_search()`
- `dataset_preview()`
- `bookmark_report()`
- `list_bookmarks()`

_(See `service.py` for complete API.)_

## Interoperability

`bia_sbi` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bia_sbi;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BIA_SBI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
