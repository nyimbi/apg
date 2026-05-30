# I18N Internationalization Capability

I18N provides APG applications with tenant-scoped localization services:
locale registration, fallback policy, regional formats, glossary terms,
translation memory, reviewed translation publication, coverage reporting,
language governance, localization-agent registration, UI metadata, theme
tokens, and Bytewax-backed lifecycle events.

The capability is intentionally dependency-light inside the package. External
identity, configuration, audit sinks, machine translation, natural-language
services, help-content stores, and theme systems are represented as APG
adapters in the executable contract and can be bound by the host application.

## What It Provides

- Locale management with owner, fallback locale, regional format, timezone, and
  enabled-state metadata.
- Explicit supported language-code policy, including more than 40 African
  language codes for African-market ERP, public-sector, education, health,
  commerce, and field-service applications.
- Translation workbench behavior for human, machine, and memory-reused
  translations.
- Glossary management for domain terms and localized variants.
- Reviewed publication batches with approval guardrails.
- Coverage reports that identify missing keys and require review when coverage
  drops below the configured threshold.
- Runtime text resolution through tenant-local fallback chains.
- First-class AI localization agents with runtime, role, scope, registration,
  and contribution-disclosure rules.
- Audit events for locale, glossary, translation, publication, coverage, and
  agent lifecycle changes.
- UI route, view-model, theme, API, package-manifest, semantic-model, and
  release-report evidence.
- Bytewax event-stream metadata for batch localization mutation and lifecycle
  telemetry.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, dependencies, provides/requires, and Bytewax stream metadata.
- `models.py` defines the tenant-scoped localization records.
- `service.py` implements the dependency-light runtime.
- `api.py` exposes simple function helpers for generated applications.
- `views.py` exposes UI view models and composition metadata.
- `test_capability_contract.py` verifies the focused lifecycle, guardrails, and
  generated evidence.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` are generated package evidence.

## Basic Usage

```python
from capabilities.common.i18n import I18nService

service = I18nService()
service.create_locale(
    locale_id="locale-sw",
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    display_name="Swahili Kenya",
    owner_id="language-owner",
)
service.upsert_translation(
    translation_id="tr-welcome-sw",
    tenant_id="tenant-demo",
    key="app.welcome",
    locale_code="sw-KE",
    source_text="Welcome",
    translated_text="Karibu",
    reviewer_id="reviewer-1",
)
service.publish_translations(
    batch_id="pub-sw-1",
    tenant_id="tenant-demo",
    locale_code="sw-KE",
    translation_ids=["tr-welcome-sw"],
    approver_id="publisher-1",
    approval_recorded=True,
)
print(service.resolve_text("tenant-demo", "app.welcome", "sw-KE")["text"])
```

## AI Localization Agents

Register AI agents before they assist localization work:

```python
agent = service.register_i18n_agent(
    tenant_id="tenant-demo",
    name="Swahili reviewer",
    runtime="codex",
    role="translation_reviewer",
    scope="Review Swahili UI translations and flag glossary drift",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are locale planning, translation, translation review, glossary
stewardship, coverage review, and publication review. Unsupported runtimes,
unsupported roles, missing scope, or undisclosed AI contribution are blocked by
the rule engine.

## Composition

I18N composes with:

- `conf` for tenant configuration and feature flags.
- `auth` for identity, permissions, and RBAC filtering.
- `audl` for durable audit events.
- `nlpc` for text analysis, terminology, and locale-aware language support.
- `mchn` for optional machine-translation providers.
- `help` for localized help and documentation content.
- `them` for tenant visual theme integration.

Batch localization mutation must use the `bytewax` event-stream adapter. The
package does not bind live workers directly.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/i18n/__init__.py capabilities/common/i18n/capability_contract.py capabilities/common/i18n/models.py capabilities/common/i18n/localization_runtime.py capabilities/common/i18n/service.py capabilities/common/i18n/api.py capabilities/common/i18n/views.py capabilities/common/i18n/app.py capabilities/common/i18n/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/i18n/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/i18n --json
./.venv/bin/apg capabilities publish-plan capabilities/common/i18n --json
```

Live identity, audit store, machine-translation providers, natural-language
providers, help-content stores, rendered UI, and Bytewax workers are integration
concerns outside the package proof.
