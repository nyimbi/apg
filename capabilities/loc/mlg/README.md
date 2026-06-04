# Multi-Language & Localisation

## Overview

Multi-Language & Localisation (MLG) manages translation workflows, locale configuration, RTL language support, date/number formatting rules, content localisation, and terminology management. It enforces reviewer independence, approval-gated publishing, RTL direction consistency, and tenant-scoped translation memory across all supported languages and locales.

## Capability ID

`loc_mlg`

## Provides

| Service | Description |
|---------|-------------|
| `locale_configuration` | Configure locales with language, script, direction, date/number formats |
| `translation_management` | Full translation lifecycle: draft → review → approve → publish → deprecate |
| `rtl_support` | Right-to-left language enforcement with Unicode BiDi compliance |
| `date_number_formatting` | Per-locale formatting rules for dates, numbers, and currency display |
| `content_localisation_workflow` | Localise UI strings, documents, email templates, legal text, and notifications |
| `locale_registry` | Tenant-scoped registry of active locales with fallback chain |
| `terminology_management` | Domain-specific glossary with preferred translations and forbidden terms |
| `translation_memory` | Key/language/namespace lookup for published translations |
| `locale_aware_rendering` | Provide locale metadata for downstream renderers |

## Requires

| Capability | Reason |
|-----------|--------|
| `auth` | Permission enforcement |
| `audl` | Immutable audit trail |
| `mten` | Tenant context isolation |
| `conf` | Configuration management |
| `ntfy` | Alerts for pending reviews and expiring translations |
| `wflo` | Translation approval workflow state machine |
| `nlpc` | NLP-assisted translation suggestions and terminology extraction |
| `moni` | Translation coverage and SLA monitoring |
| `mqeb` | bytewax event streaming for translation lifecycle events |

## Configuration

| Key | Type | Description |
|-----|------|-------------|
| `tenant_id` | string | Tenant identifier |
| `locales.default_locale` | string | Default locale code (e.g. `en_KE`) |
| `locales.fallback_locale` | string | Fallback when locale not found |
| `translations.reviewer_required_for_approval` | bool | Require independent reviewer |
| `content_localisation.publish_requires_approved_status` | bool | Block publishing unapproved translations |
| `rtl.auto_detect_direction` | bool | Auto-set direction for known RTL languages |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/loc-mlg/api/v1/locales` | GET | List locales | `loc_mlg:locales` |
| `/loc-mlg/api/v1/locales` | POST | Configure locale | `loc_mlg:locales_write` |
| `/loc-mlg/api/v1/locales/<id>` | GET | Get locale | `loc_mlg:locales` |
| `/loc-mlg/api/v1/locales/<id>` | PUT | Update locale | `loc_mlg:locales_write` |
| `/loc-mlg/api/v1/translations` | GET | List translations | `loc_mlg:translations` |
| `/loc-mlg/api/v1/translations` | POST | Create translation | `loc_mlg:translations_write` |
| `/loc-mlg/api/v1/translations/<id>` | GET | Get translation | `loc_mlg:translations` |
| `/loc-mlg/api/v1/translations/<id>/submit` | POST | Submit for review | `loc_mlg:translations_write` |
| `/loc-mlg/api/v1/translations/<id>/approve` | POST | Approve translation | `loc_mlg:translations_review` |
| `/loc-mlg/api/v1/translations/<id>/publish` | POST | Publish translation | `loc_mlg:translations_write` |
| `/loc-mlg/api/v1/translations/lookup` | GET | Look up published key | `loc_mlg:translations` |
| `/loc-mlg/api/v1/formatting` | GET | List formatting rules | `loc_mlg:formatting` |
| `/loc-mlg/api/v1/formatting` | POST | Configure formatting | `loc_mlg:formatting_write` |
| `/loc-mlg/api/v1/formatting/<id>` | GET | Get formatting rule | `loc_mlg:formatting` |
| `/loc-mlg/api/v1/terminology` | GET | List terminology | `loc_mlg:terminology` |
| `/loc-mlg/api/v1/terminology` | POST | Add term | `loc_mlg:terminology` |
| `/loc-mlg/api/v1/terminology/search` | GET | Search by term text | `loc_mlg:terminology` |
| `/loc-mlg/api/v1/agents` | GET | List agents | `loc_mlg:admin` |
| `/loc-mlg/api/v1/agents` | POST | Register agent | `loc_mlg:admin` |
| `/loc-mlg/api/v1/dashboard` | GET | Dashboard summary | `loc_mlg:view` |
| `/loc-mlg/api/v1/audit-events` | GET | Audit log | `loc_mlg:admin` |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `tenant_context_required` | no tenant context | deny |
| `locale_code_supported` | unsupported locale code | deny |
| `locale_direction_supported` | unsupported text direction | deny |
| `rtl_bypass_denied` | RTL language + non-RTL direction | deny |
| `translation_translator_required` | create without translator | deny |
| `self_review_denied` | reviewer = translator | deny |
| `unapproved_publish_denied` | publish without approved status | deny |
| `untranslated_legal_text_blocked` | legal_text content type + not approved | deny |
| `translation_key_required` | missing translation key | deny |
| `privileged_agent_action_requires_human_approval` | privileged + no approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| `LocaleConfigResponse` | id, tenant_id, locale_code, language, script, text_direction, date_format, number_format, is_rtl, is_default |
| `TranslationResponse` | id, tenant_id, translation_key, source_language, target_language, content_type, translated_text, status, namespace, version |
| `FormattingRuleResponse` | id, tenant_id, locale_id, date_format, number_format, currency_display, thousand_separator, decimal_separator |
| `TerminologyResponse` | id, tenant_id, term, language, definition, domain, preferred_translation, forbidden_terms |
| `MlgAgentResponse` | id, tenant_id, name, runtime, role, scope |
| `MlgAuditEvent` | id, tenant_id, event_type, reference_id, actor_id, processor, stream |

## Streaming Events

| Event | Trigger |
|-------|---------|
| `locale_configured` | New locale configured |
| `locale_updated` | Locale updated |
| `translation_created` | Translation entry created |
| `translation_submitted_for_review` | Submitted for review |
| `translation_approved` | Approved by reviewer |
| `translation_published` | Published |
| `translation_deprecated` | Deprecated |
| `formatting_rule_configured` | Formatting rule added |
| `rtl_locale_activated` | RTL locale configured |
| `terminology_added` | Term added to glossary |
| `agent_registered` | Agent registered |

## Edge Cases Handled

- Arabic, Hebrew, Farsi, and other RTL languages are rejected at model validation if `text_direction` is not set to `rtl`
- Only one locale can be the tenant default at a time — setting a new default automatically demotes the previous default
- A translator cannot review their own translation — the `reviewer_id` must differ from `translator_id`
- Publishing a translation requires `status == "approved"` — direct publish from `draft` or `pending_review` is blocked
- Legal text content type requires an approved translation before any publish operation
- Translation lookup is namespace-scoped: `default` namespace is used if not specified, allowing multi-namespace coexistence for the same key
- Formatting rules reference `locale_id` — the locale must exist in the tenant registry before formatting can be configured

## Composability Notes

- `mco` entity country assignments drive default locale selection per entity
- `mcy` currency codes are combined with MLG currency display modes for locale-aware money formatting
- Rendered UIs consume MLG translation lookups by key/language/namespace at render time
- `nlpc` can be fed MLG terminology glossaries for domain-aware NLP pipeline grounding
- MLG emits all lifecycle events to `apg.loc.mlg.lifecycle` for downstream bytewax consumers
