# Multi-Language & Localisation — User Guide

**Capability ID**: `loc_mlg` | **Domain**: `loc` | **Version**: `1.1.0`

---

## Overview

Multi-Language & Localisation (MLG) manages the full translation lifecycle — from locale configuration and RTL enforcement through draft → review → approve → publish → deprecate — with terminology management, locale-aware number/date formatting, AI quality scoring, SLA tracking, and cross-tenant locale sync.

---

## Installation

```bash
pip install apg-loc-mlg
```

---

## Quick Start

```python
import asyncio
from capabilities.loc.mlg.service import MultiLanguageLocalisationService
from capabilities.loc.mlg.models import LocaleConfigCreate, TranslationCreate

svc = MultiLanguageLocalisationService()

async def main():
    # 1. Configure a locale
    locale = await svc.configure_locale(LocaleConfigCreate(
        tenant_id="acme",
        locale_code="sw_KE",
        language="sw",
        script="Latin",
        text_direction="ltr",
        date_format="DD/MM/YYYY",
        number_format="1,234.56",
        is_default=True,
        is_rtl=False,
    ))

    # 2. Create a translation
    tr = await svc.create_translation(TranslationCreate(
        tenant_id="acme",
        translation_key="welcome_message",
        source_language="en",
        target_language="sw",
        content_type="ui_string",
        source_text="Welcome to our platform",
        translated_text="Karibu kwenye jukwaa letu",
        translator_id="translator-001",
    ))

    # 3. Standard workflow: submit → approve → publish
    await svc.submit_translation_for_review("acme", tr.id)
    await svc.approve_translation("acme", tr.id, reviewer_id="reviewer-002")
    await svc.publish_translation("acme", tr.id)

    # 4. Look up the published translation at render time
    result = await svc.lookup_translation("acme", "welcome_message", "sw")
    print(result.translated_text)  # "Karibu kwenye jukwaa letu"

asyncio.run(main())
```

---

## Locale Configuration

### Configure a locale

```python
locale = await svc.configure_locale(LocaleConfigCreate(
    tenant_id="acme",
    locale_code="ar_SA",
    language="ar",
    script="Arabic",
    text_direction="rtl",   # required for RTL languages
    date_format="DD/MM/YYYY",
    number_format="1,234.56",
    is_rtl=True,
))
```

RTL languages (Arabic, Hebrew, Farsi, Urdu, etc.) require both `text_direction="rtl"` and `is_rtl=True`. The model validator enforces this at creation time.

### List locales

```python
# All active locales
locales = await svc.list_locales("acme")

# Filter to RTL only
rtl_locales = await svc.list_locales("acme", is_rtl=True)

# Filter by language
sw_locales = await svc.list_locales("acme", language="sw")
```

### Get default locale

```python
default = await svc.get_default_locale("acme")
```

Only one locale can be default per tenant. Configuring a new default automatically demotes the previous one.

### Fallback chain

```python
chain = await svc.locale_fallback_chain("acme", "sw_KE")
# ["sw_KE", "sw", "en"]
```

---

## Translation Lifecycle

```
draft → pending_review → approved → published
                      ↘ deprecated (any state)
```

| Transition | Method |
|-----------|--------|
| draft → pending_review | `submit_translation_for_review` |
| pending_review → approved | `approve_translation` |
| approved → published | `publish_translation` |
| any → deprecated | `deprecate_translation` |
| published → rolled back | `rollback_translation` |

### Create a translation

```python
tr = await svc.create_translation(TranslationCreate(
    tenant_id="acme",
    translation_key="invoice.total",
    source_language="en",
    target_language="fr",
    content_type="ui_string",
    source_text="Total",
    translated_text="Total",
    translator_id="translator-001",
    namespace="finance",
))
```

### Reviewer independence

The reviewer must be a different user from the translator:

```python
# This raises PermissionError — reviewer_is_translator rule
await svc.approve_translation("acme", tr.id, reviewer_id="translator-001")

# Correct
await svc.approve_translation("acme", tr.id, reviewer_id="reviewer-002")
```

### Bulk workflow

```python
ids = [t.id for t in pending_translations]

# Approve all
result = await svc.batch_approve_translations("acme", ids, reviewer_id="reviewer-002")
print(result["approved"], "approved,", result["errors"], "errors")

# Publish all approved
result = await svc.batch_publish_translations("acme", ids)
```

### Rollback

```python
# Restore version 3 of a translation key
rolled_back = await svc.rollback_translation(
    "acme", translation_id, target_version=3
)

# View full version history
history = await svc.get_translation_history("acme", "welcome_message", "sw")
```

---

## Machine Translation

```python
result = await svc.machine_translate_batch(
    tenant_id="acme",
    texts=["Hello", "Thank you", "Invoice"],
    target_language="sw",
)
for t in result["translations"]:
    print(t["source"], "→", t["translated"])
```

Production: delegates to Ollama. Stub responses returned in in-memory mode.

---

## Plural Rules

```python
rule = await svc.plural_rule_define(
    "acme", "sw",
    rule_expression="n == 1 ? 'one' : 'other'"
)
```

---

## Formatting Rules

### Configure

```python
from capabilities.loc.mlg.models import FormattingRuleCreate

rule = await svc.configure_formatting(FormattingRuleCreate(
    tenant_id="acme",
    locale_id=locale.id,
    date_format="DD/MM/YYYY",
    number_format="1,234.56",
    thousand_separator=",",
    decimal_separator=".",
    time_format_24h=True,
    first_day_of_week=1,   # Monday
))
```

### Apply formatting at runtime

```python
formatted = await svc.format_number("acme", locale.id, 1234567.89)
# "1,234,567.89"
```

---

## Terminology / Glossary

### Add a term

```python
from capabilities.loc.mlg.models import TerminologyCreate

term = await svc.add_terminology(TerminologyCreate(
    tenant_id="acme",
    term="Invoice",
    language="en",
    definition="A document requesting payment",
    domain="finance",
    preferred_translation="Ankara",
    forbidden_terms=["Bill", "Receipt"],
))
```

### Search

```python
matches = await svc.search_terminology("acme", "inv", language="en")
```

### Validate a translation against the glossary

```python
violations = await svc.validate_against_glossary(
    "acme",
    translated_text="Here is your Bill for the month",
    target_language="en",
    domain="finance",
)
# [{"forbidden_term": "Bill", "suggested_replacement": "Ankara", "position": 15, ...}]
```

---

## Coverage & Analytics

### Coverage matrix

```python
matrix = await svc.coverage_matrix("acme")
# {"sw": {"default": 73.4, "finance": 61.0}, "fr": {"default": 88.2}}
```

### Missing translations report

```python
report = await svc.missing_translations_report("acme", locale="sw", namespace="finance")
print(f"{report['completion_pct']}% complete, {len(report['missing_keys'])} keys missing")
```

### Locale analytics

```python
stats = await svc.locale_analytics("acme", period="2026-Q2")
```

---

## SLA and Workload Monitoring

### Translator workload

```python
workloads = await svc.translator_workload("acme")
for w in workloads:
    print(w["translator_id"], "pending_review:", w["pending_review"])

# Single translator
wl = await svc.translator_workload("acme", translator_id="translator-001")
```

### SLA violations

```python
report = await svc.sla_violations_report("acme", max_days_in_review=3)
for v in report["violations"]:
    print(v["translation_key"], "waiting", v["days_waiting"], "days")
```

Integrate with `ntfy` to alert managers when `violation_count > 0`.

---

## AI Quality Scoring

```python
scores = await svc.score_translation_quality("acme", translation_id)
print(scores["overall"])          # e.g. 0.866
print(scores["scores"]["fluency"]) # e.g. 0.91
```

Dimensions scored: `accuracy`, `fluency`, `terminology_adherence`, `style_consistency`, `cultural_appropriateness`.

Production: delegates to Ollama with a structured QA prompt including source text, translated text, and domain glossary context.

---

## Super-Admin: Cross-Tenant Locale Sync

```python
result = await svc.sync_locale_baseline(
    source_tenant_id="template-tenant",
    target_tenant_ids=["acme", "beta-corp", "gamma-inc"],
    actor_id="superadmin",
)
for entry in result["report"]:
    print(entry["target_tenant_id"], "—", entry["locales_synced"], "locales synced")
```

Idempotent: existing locale codes in target tenants are skipped. Safe to run repeatedly.

---

## Import / Export

### Bulk import

```python
result = await svc.translation_import(
    "acme", "sw",
    {"welcome_message": "Karibu", "logout": "Toka"},
    namespace="default",
)
print(result["imported"], "imported,", result["errors"], "errors")
```

### Export for download

```python
export = await svc.locale_export("acme", "sw", format="json")
print(export["download_ref"])  # "/exports/acme/loc-export-sw-42.json"
```

### Clone locale

```python
result = await svc.locale_clone("acme", source_locale="en", target_locale="sw")
print(result["cloned_keys"], "keys cloned as drafts")
```

---

## Audit Events

```python
events = await svc.list_audit_events("acme", limit=20)
for e in events:
    print(e["event_type"], e["reference_id"], e["occurred_at"])
```

All events stream to `apg.loc.mlg.lifecycle` via bytewax.

---

## Agents

```python
from capabilities.loc.mlg.models import MlgAgentCreate

agent = await svc.register_agent(MlgAgentCreate(
    tenant_id="acme",
    name="Auto-Translator",
    runtime="ollama",
    role="translator",
    scope="ui_strings",
))
```

Supported runtimes: see `capability_contract.SUPPORTED_AGENT_RUNTIMES`.
Supported roles: see `capability_contract.SUPPORTED_AGENT_ROLES`.

---

## Dashboard

```python
summary = await svc.dashboard_summary("acme")
# {
#   "locale_count": 5,
#   "rtl_locale_count": 2,
#   "translation_count": 1420,
#   "pending_review_count": 38,
#   "published_count": 1201,
#   ...
# }
```

---

## Flask Blueprint Routes

All routes are prefixed `/loc-mlg`. Authentication uses header `X-Tenant-ID` and `X-Permissions`.

| Path | Method | Permission |
|------|--------|-----------|
| `/loc-mlg/dashboard` | GET | `loc_mlg:view` |
| `/loc-mlg/locales` | GET | `loc_mlg:locales` |
| `/loc-mlg/locales` | POST | `loc_mlg:locales_write` |
| `/loc-mlg/locales/<id>` | GET | `loc_mlg:locales` |
| `/loc-mlg/locales/<id>` | PUT | `loc_mlg:locales_write` |
| `/loc-mlg/translations` | GET | `loc_mlg:translations` |
| `/loc-mlg/translations` | POST | `loc_mlg:translations_write` |
| `/loc-mlg/translations/<id>` | GET | `loc_mlg:translations` |
| `/loc-mlg/translations/<id>/submit` | POST | `loc_mlg:translations_write` |
| `/loc-mlg/translations/<id>/approve` | POST | `loc_mlg:translations_review` |
| `/loc-mlg/translations/<id>/publish` | POST | `loc_mlg:translations_write` |
| `/loc-mlg/translations/lookup` | GET | `loc_mlg:translations` |
| `/loc-mlg/formatting` | GET | `loc_mlg:formatting` |
| `/loc-mlg/formatting` | POST | `loc_mlg:formatting_write` |
| `/loc-mlg/formatting/<id>` | GET | `loc_mlg:formatting` |
| `/loc-mlg/terminology` | GET | `loc_mlg:terminology` |
| `/loc-mlg/terminology` | POST | `loc_mlg:terminology` |
| `/loc-mlg/terminology/search` | GET | `loc_mlg:terminology` |
| `/loc-mlg/agents` | GET | `loc_mlg:admin` |
| `/loc-mlg/agents` | POST | `loc_mlg:admin` |
| `/loc-mlg/audit-events` | GET | `loc_mlg:admin` |

---

## Business Rules Reference

| Rule | Trigger | Effect |
|------|---------|--------|
| `tenant_context_required` | Missing or empty tenant_id | deny |
| `rtl_bypass_denied` | RTL language + non-RTL direction | deny |
| `self_review_denied` | reviewer_id == translator_id | deny |
| `unapproved_publish_denied` | Publish from draft/pending_review | deny |
| `translation_key_required` | Missing translation key | deny |
| `privileged_agent_action_requires_human_approval` | Privileged agent + no approval | deny |

---

## Composability

```apg
use loc_mlg;
```

- `mco` country assignments → default locale selection per entity
- `mcy` currency codes + MLG currency display modes → locale-aware money formatting
- `nlpc` + MLG terminology glossaries → domain-aware NLP pipeline grounding
- `moni` + `coverage_matrix` → translation coverage dashboards
- `ntfy` + `sla_violations_report` → SLA breach alerts
- All lifecycle events → `apg.loc.mlg.lifecycle` bytewax stream

---

## Further Reading

- `service.py` — Business logic and all service methods
- `models.py` — Pydantic v2 models
- `api.py` — REST API endpoints
- `views.py` — Flask blueprint views
- `capability_contract.py` — Supported values and policy rules
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement specifications
- `README.md` — Quick reference
