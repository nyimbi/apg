# I18N Internationalisation Capability

I18N provides APG applications with tenant-scoped localisation services:
locale registration, fallback policy, regional formats, glossary terms,
translation memory, reviewed translation publication, coverage reporting,
language governance, localisation-agent registration, UI metadata, theme
tokens, and Bytewax-backed lifecycle events.

The capability is intentionally dependency-light inside the package. External
identity, configuration, audit sinks, machine translation, natural-language
services, help-content stores, and theme systems are represented as APG
adapters in the executable contract and can be bound by the host application.

## What It Provides

- Locale management with owner, fallback locale, regional format, timezone,
  and enabled-state metadata.
- Explicit supported language-code policy, including more than 40 African
  language codes for African-market ERP, public-sector, education, health,
  commerce, and field-service applications.
- Translation workbench behaviour for human, machine, and memory-reused
  translations with full version history.
- Glossary management for domain terms and localised variants with fuzzy
  lookup.
- Reviewed publication batches with approval guardrails.
- Coverage reports that identify missing keys and require review when coverage
  drops below the configured threshold.
- Runtime text resolution through tenant-local fallback chains with audit trail.
- Bulk import/export in JSON, PO, and CSV formats.
- Machine-translation job submission backed by locally-hosted Ollama models.
- Per-locale analytics: translation counts, coverage percentage, last activity.
- Script-direction detection (RTL/LTR) and font-stack recommendations.
- Date, number, and currency localisation using locale regional formats.
- Plural-rules registry per locale (CLDR categories: zero/one/two/few/many/other).
- First-class AI localisation agents with runtime, role, scope, registration,
  and contribution-disclosure rules.
- Audit events for locale, glossary, translation, publication, coverage, and
  agent lifecycle changes.
- UI route, view-model, theme, API, package-manifest, semantic-model, and
  release-report evidence.
- Bytewax event-stream metadata for batch localisation mutation and lifecycle
  telemetry.

## Main Files

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | Normative capability behaviour |
| `PLAN.md` | Implementation packet plan |
| `capability_contract.py` | Executable configuration, rules, routes, theme, deps, Bytewax metadata |
| `models.py` | Tenant-scoped localisation records (Pydantic v2) |
| `service.py` | Dependency-light runtime (37 public methods) |
| `localization_runtime.py` | Fallback resolver, TM matcher, coverage calculator |
| `api.py` | Simple function helpers for generated applications |
| `views.py` | UI view models and composition metadata |
| `tests/` | Unit, integration, and composition tests |
| `app.py`, `semantic_model.json`, `package_manifest.json`, `release_report.json` | Generated package evidence |

## Quick Start

```python
from capabilities.common.i18n import I18nService

service = I18nService()

# 1. Register a locale
service.create_locale(
    locale_id="locale-sw",
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    display_name="Swahili Kenya",
    owner_id="language-owner",
)

# 2. Add a translation
service.upsert_translation(
    translation_id="tr-welcome-sw",
    tenant_id="tenant-demo",
    key="app.welcome",
    locale_code="sw-KE",
    source_text="Welcome",
    translated_text="Karibu",
    reviewer_id="reviewer-1",
)

# 3. Publish
service.publish_translations(
    batch_id="pub-sw-1",
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    translation_ids=["tr-welcome-sw"],
    approver_id="publisher-1",
    approval_recorded=True,
)

# 4. Resolve at runtime
print(service.resolve_text("tenant-demo", "app.welcome", "sw-KE")["text"])
# -> "Karibu"
```

## API Reference

### Core (synchronous)

| Method | Description |
|--------|-------------|
| `create_locale(...)` | Register a locale with owner, fallback, format, timezone |
| `add_glossary_term(...)` | Add a domain term with localised variants |
| `upsert_translation(...)` | Create or update a translation entry |
| `reuse_translation_memory(...)` | Re-use a matching entry from TM store |
| `publish_translations(...)` | Publish a reviewed batch with approval guard |
| `resolve_text(tenant, key, locale)` | Runtime lookup with fallback chain |
| `coverage_report(...)` | Generate a coverage snapshot for a locale |
| `register_i18n_agent(...)` | Register an AI localisation agent |
| `validate_batch_i18n_mutation(stream)` | Check batch mutation against policy |
| `dashboard_summary(tenant_id)` | Aggregate counts across all stores |
| `list_locales / list_translations / list_glossary_terms / ...` | Collection accessors |

### Extended (async)

| Method | Description |
|--------|-------------|
| `locale_create(...)` | Async alias for `create_locale`; preferred for new callers |
| `translation_import(...)` | Bulk import from PO/JSON payload |
| `translation_export(...)` | Export to JSON, PO, or CSV |
| `translation_review(...)` | Human approve/reject a single entry |
| `batch_approve_translations(...)` | Bulk approve a list of entries |
| `translation_version(...)` | Full version history for an entry |
| `translation_search(...)` | Full-text search across source and translated text |
| `machine_translate(...)` | Submit MT job via Ollama; creates DRAFT entry |
| `plural_rules(...)` | Register or retrieve CLDR plural rules for a locale |
| `date_localise(...)` | Format ISO-8601 datetime per locale regional format |
| `number_localise(...)` | Format number per locale conventions |
| `currency_localise(...)` | Format monetary amount with ISO 4217 currency code |
| `rtl_check(...)` | Detect right-to-left script direction |
| `font_detect(...)` | Recommend font stack for locale script |
| `missing_keys_report(...)` | Keys in reference locale absent from target locale |
| `locale_fallback(...)` | Safe fallback resolve — never raises, returns resolved flag |
| `locale_clone(...)` | Clone locale definition and optionally all translations |
| `locale_analytics(...)` | Per-locale counts, coverage %, last activity |
| `locale_timezone_list(...)` | Distinct timezones across tenant locales |
| `glossary_lookup(...)` | Find glossary entries matching a source term |

## World-Class Enhancements (v2.0)

These 15 improvements bring the capability to enterprise i18n tool-chain parity.
Implementation status and target PRs are tracked in `WORLD_CLASS_IMPROVEMENTS.md`.

1. **CLDR Plural Rules Engine** — Full Unicode CLDR plural categories (zero/one/two/few/many/other) replacing the naive `n != 1` default. Fixes Arabic (6 forms), Russian, Polish, and most African languages.

2. **ICU MessageFormat 2.0 Templates** — Pure-Python MF2 parser supporting variable interpolation, select expressions, and plural selectors inside a single translation key.

3. **Babel/CLDR Date, Number, Currency Formatting** — Replace heuristic separator swapping with Babel `format_date`, `format_number`, `format_currency` calls for correct output in all CLDR-covered locales.

4. **Streaming Translation Export** — Convert `translation_export` to an async generator yielding NDJSON/PO/CSV chunks, eliminating in-memory accumulation for 100k+ key catalogs.

5. **Locale-Aware Collation** — `locale_sort(tenant_id, locale_code, items)` backed by `PyICU`/`pyuca` returning CLDR-correct alphabetical order for Swahili, Yoruba, Czech, CJK, and others.

6. **TM Fuzzy-Match Score Threshold** — Translation memory returns the best-scoring match (Levenshtein/Jaro-Winkler) above a configurable `min_score` threshold, matching CAT-tool TM leverage behaviour.

7. **Differential Coverage Alerts** — `coverage_report` computes a delta against the previous snapshot (keys gained/lost, severity band) and publishes it as a CloudEvent to the Bytewax stream.

8. **Pseudo-Localisation Mode** — `pseudo_localise(tenant_id, locale_code, key_pattern)` inflates strings 30-40% and wraps them in `[!!!...!!!]` markers for pre-translation UI layout QA.

9. **Namespace-Scoped Keys** — `<namespace>.<key>` two-part schema with per-namespace ownership, coverage reports, and RBAC ("marketing team owns `marketing.*`").

10. **AI Glossary Consistency Checker** — `glossary_consistency_check(tenant_id, locale_code)` scans published translations for terms whose glossary localisations are missing from the translated text.

11. **Locale Lifecycle State Machine** — Explicit `draft → active → deprecated → archived` transitions via `locale_activate`, `locale_deprecate`, `locale_archive` with guard rules preventing publication to deprecated/archived locales.

12. **Bulk Translation Diff/Merge** — `translation_diff(...)` produces a per-key action plan (keep/update/conflict/add/delete); `translation_merge_apply(plan_id, approved_keys)` commits only approved changes.

13. **Fallback Chain Audit Trail** — `resolve_text` records which locale in the fallback chain served the text and emits it as a structured event to identify hot fallback locales driving translation priorities.

14. **DST-Aware Timezone Formatting** — `date_localise` `aware=True` flag expands the stored IANA zone to a `ZoneInfo`-backed DST-correct conversion before formatting, fixing wrong wall-clock times for non-UTC users.

15. **Translation Linting Pipeline** — `translation_lint(tenant_id, locale_code, rules)` checks `no_untranslated`, `html_tag_parity`, `placeholder_parity`, `max_length(n)`, `no_double_spaces`, and `required_punctuation_end`, returning per-entry violations with severity and suggested fixes.

## New Methods: Usage Examples

### Bulk Import

```python
result = await service.translation_import(
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    entries=[
        {
            "translation_id": "tr-save-sw",
            "key": "app.save",
            "source_text": "Save",
            "translated_text": "Hifadhi",
            "translation_review_recorded": True,
            "reviewer_id": "reviewer-1",
        },
    ],
    importer_id="ops-pipeline",
    overwrite_existing=False,
)
# {"locale_code": "sw-KE", "imported": 1, "skipped": 0, "failed": []}
```

### Plural Rules Registration

```python
await service.plural_rules(
    tenant_id="tenant-demo",
    locale_code="ar-SA",
    rules={
        "zero":  "n == 0",
        "one":   "n == 1",
        "two":   "n == 2",
        "few":   "n % 100 in 3..10",
        "many":  "n % 100 in 11..99",
        "other": "",
    },
    actor="locale-admin",
)
```

### Machine Translation

```python
result = await service.machine_translate(
    translation_id="tr-cancel-sw",
    tenant_id="tenant-demo",
    key="app.cancel",
    locale_code="sw-KE",
    source_text="Cancel",
    engine="ollama",
    model="qwen3",
    reviewer_id="reviewer-1",
)
# Produces a DRAFT entry; reviewer must call translation_review(..., approved=True)
```

### Missing Keys Report

```python
report = await service.missing_keys_report(
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    reference_locale="en-US",
)
# {"missing_key_count": 12, "missing_keys": ["app.delete", ...], "extra_keys": [...]}
```

### Locale Analytics

```python
analytics = await service.locale_analytics(tenant_id="tenant-demo")
for row in analytics:
    print(f"{row['locale_code']}: {row['coverage_percent']}% published")
```

### Locale Clone

```python
result = await service.locale_clone(
    tenant_id="tenant-demo",
    source_locale_code="sw-KE",
    new_locale_id="locale-sw-TZ",
    new_locale_code="sw-TZ",
    new_display_name="Swahili Tanzania",
    owner_id="locale-admin",
    clone_translations=True,   # copies all entries as DRAFT for review
)
```

## AI Localisation Agents

Register AI agents before they assist localisation work:

```python
agent = service.register_i18n_agent(
    tenant_id="tenant-demo",
    name="Swahili reviewer",
    runtime="codex",
    role="translation_reviewer",
    scope="Review Swahili UI translations and flag glossary drift",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: locale planning, translation, translation review, glossary
stewardship, coverage review, publication review. Unsupported runtimes/roles,
missing scope, or undisclosed AI contribution are blocked by the rule engine.

## Composition

I18N composes with:

| Capability | Purpose |
|-----------|---------|
| `conf` | Tenant configuration and feature flags |
| `auth` | Identity, permissions, RBAC filtering |
| `audl` | Durable audit events |
| `nlpc` | Text analysis, terminology, locale-aware language support |
| `mchn` | Optional machine-translation providers |
| `help` | Localised help and documentation content |
| `them` | Tenant visual theme integration |

Batch localisation mutation must use the `bytewax` event-stream adapter.
The package does not bind live workers directly.

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/i18n/__init__.py \
    capabilities/common/i18n/capability_contract.py \
    capabilities/common/i18n/models.py \
    capabilities/common/i18n/localization_runtime.py \
    capabilities/common/i18n/service.py \
    capabilities/common/i18n/api.py \
    capabilities/common/i18n/views.py \
    capabilities/common/i18n/app.py

./.venv/bin/pytest -q capabilities/common/i18n/tests/
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/i18n --json
./.venv/bin/apg capabilities publish-plan capabilities/common/i18n --json
```

Live identity, audit store, machine-translation providers, natural-language
providers, help-content stores, rendered UI, and Bytewax workers are
integration concerns outside the package proof.

---

© 2025 Datacraft — www.datacraft.co.ke
