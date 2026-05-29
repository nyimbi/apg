# Internationalization Capability Specification

- **Capability Name**: Internationalization
- **Capability ID**: `i18n`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`i18n` provides an executable, tenant-aware localization runtime for
APG-generated Python applications. It supports locale ownership, regional
formatting metadata, fallback chains, glossary terms, translation memory,
human and reviewed machine translations, restricted-content filtering,
coverage reports, publication batches, UI route metadata, visual theme
metadata, and publishable package evidence.

The package is dependency-light and deterministic in process. Production NLP,
machine-translation, terminology, RBAC, audit, theme, and configuration
integrations should be attached behind APG adapters rather than embedded in
the package facade.

## Runtime Surfaces

- `LocaleDefinition`: tenant-scoped locale, owner, fallback locale, timezone,
  regional format, and enablement state.
- `GlossaryTerm`: governed source terminology with localized variants.
- `TranslationEntry`: localized string with lifecycle state, source, reviewer,
  restricted-content flag, version, and publication timestamp.
- `CoverageReport`: required-key coverage result with missing keys and review
  requirement.
- `PublishBatch`: approved publication set for a locale.
- `localization_runtime.py`: I18N-specific fallback resolution, translation
  memory matching, and coverage calculation.
- `I18nService`: executable facade for locale creation, glossary management,
  translation upsert, translation-memory reuse, publication, fallback
  resolution, coverage reporting, compatibility records, dashboard summaries,
  and contract rule evaluation.
- `api.py`: dependency-light helpers for generated apps and package probes.
- `views.py`: locale console, translation workbench, glossary manager,
  coverage dashboard, publish queue, dashboard, routes, rules, and theme view
  models.

## Provided Services

- `locale_management`
- `translation_memory`
- `content_localization`
- `language_fallbacks`
- `regional_formatting`

## Required Services

- `tenant_context`

Optional production adapters may use `conf`, `nlpc`, `auth`, `audl`, `mchn`,
`help`, and `them` as described by package registration metadata.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Locale configuration controls ownership, default locale, fallback
requirements, and regional formatting. Translation configuration controls
translation memory, glossary expectations, machine-translation review, and
minimum coverage.

## Rules

The package enforces the contract rule engine through `I18nService`:

- `tenant_context_required`
- `locale_requires_owner`
- `machine_translation_requires_review`
- `publish_requires_approval`
- `restricted_content_requires_filtering`
- `low_coverage_requires_review`

Additional service guardrails validate same-tenant locale and translation
references, reviewed-before-publish state, locale/translation consistency,
translation-memory hits, missing translation resolution, restricted-content
RBAC filtering, and generated-package compatibility records.

## UI

The package exposes APG Python UI route contracts and dependency-light view
models for:

- dashboard
- locale console
- translation workbench
- glossary manager
- coverage dashboard
- publish queue
- language policies
- settings

## Theme

The package uses the `i18n_localization_workbench` APG theme contract with
locale matrix, translation editor, coverage dashboard, and publish queue
component tokens.

## Verification

Use focused package verification first:

```bash
./.venv/bin/python -m py_compile capabilities/common/i18n/__init__.py capabilities/common/i18n/models.py capabilities/common/i18n/localization_runtime.py capabilities/common/i18n/service.py capabilities/common/i18n/api.py capabilities/common/i18n/views.py capabilities/common/i18n/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/i18n/test_capability_contract.py capabilities/common/i18n/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/i18n --json
./.venv/bin/apg capabilities publish-plan capabilities/common/i18n --json
```
