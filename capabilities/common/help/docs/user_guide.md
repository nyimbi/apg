# Help and Knowledge Base

**Capability ID**: `help` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`help` provides APG's common capability for tenant-scoped help centers and governed support knowledge. It composes source registration, article authoring, publication approval, localization, cited answer generation, feedback curation, first-class provider-neutral help agents, audit events, UI routes, visual theming, and Bytewax lifecycle guardrails into a generated-application packet that runs without live search or RAG infrastructure.

## Installation

```bash
pip install apg-common-help
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/help/dashboard` | `help:view` | Overview |
| `/help/home` | `help:view` | Help |
| `/help/articles` | `help:view` | Help |
| `/help/editor` | `help:edit_articles` | Authoring |
| `/help/sources` | `help:publish` | Authoring |
| `/help/answers` | `help:ask` | Assistant |
| `/help/localization` | `help:edit_articles` | Authoring |
| `/help/curation` | `help:publish` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_source()`
- `approve_source()`
- `create_article()`
- `publish_article()`
- `search_articles()`
- `generate_answer()`
- `localize_article()`
- `record_feedback()`

_(See `service.py` for complete API.)_

## Interoperability

`help` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use help;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HELP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
