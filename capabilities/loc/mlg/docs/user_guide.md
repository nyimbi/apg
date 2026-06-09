# Multi-Language & Localisation

**Capability ID**: `loc_mlg` | **Domain**: `loc` | **Version**: `1.0.0`

## Description

Multi-Language & Localisation (MLG) manages translation workflows, locale configuration, RTL language support, date/number formatting rules, content localisation, and terminology management. It enforces reviewer independence, approval-gated publishing, RTL direction consistency, and tenant-scoped translation memory across all supported languages and locales.

## Installation

```bash
pip install apg-loc-mlg
```

## Provides

- `locale_configuration`
- `translation_management`
- `rtl_support`
- `date_number_formatting`
- `content_localisation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/loc-mlg/dashboard` | `loc_mlg:view` | Overview |
| `/loc-mlg/locales` | `loc_mlg:locales` | Setup |
| `/loc-mlg/locales/create` | `loc_mlg:locales_write` | Setup |
| `/loc-mlg/translations` | `loc_mlg:translations` | Translations |
| `/loc-mlg/translations/create` | `loc_mlg:translations_write` | Translations |
| `/loc-mlg/translations/review` | `loc_mlg:translations_review` | Translations |
| `/loc-mlg/formatting` | `loc_mlg:formatting` | Configuration |
| `/loc-mlg/formatting/create` | `loc_mlg:formatting_write` | Configuration |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `describe()`
- `evaluate()`
- `configure_locale()`
- `get_locale()`
- `get_locale_by_code()`
- `list_locales()`
- `update_locale()`
- `get_default_locale()`

_(See `service.py` for complete API.)_

## Interoperability

`loc_mlg` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use loc_mlg;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `LOC_MLG_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
