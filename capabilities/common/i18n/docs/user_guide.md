# Internationalization

**Capability ID**: `i18n` | **Domain**: `common` | **Version**: `1.0.0`

## Description

I18N provides APG applications with tenant-scoped localization services: locale registration, fallback policy, regional formats, glossary terms, translation memory, reviewed translation publication, coverage reporting,

## Installation

```bash
pip install apg-common-i18n
```

## Provides

- `locale_management`
- `translation_memory`
- `content_localization`
- `language_fallbacks`
- `regional_formatting`

## Requires

- `conf`
- `nlpc`
- `auth`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/i18n/dashboard` | `i18n:view` | Overview |
| `/i18n/locales` | `i18n:manage_locales` | Locales |
| `/i18n/translations` | `i18n:translate` | Translations |
| `/i18n/glossaries` | `i18n:translate` | Translations |
| `/i18n/coverage` | `i18n:view` | Quality |
| `/i18n/publishing` | `i18n:publish` | Release |
| `/i18n/agents` | `i18n:admin` | Governance |
| `/i18n/audit` | `i18n:admin` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_locale()`
- `add_glossary_term()`
- `upsert_translation()`
- `reuse_translation_memory()`
- `publish_translations()`
- `resolve_text()`
- `coverage_report()`
- `register_i18n_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`i18n` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use i18n;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `I18N_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
