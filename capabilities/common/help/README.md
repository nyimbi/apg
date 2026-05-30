# Help and Knowledge Base Capability

`help` provides APG's common capability for tenant-scoped help centers and governed support knowledge. It composes source registration, article authoring, publication approval, localization, cited answer generation, feedback curation, audit events, UI routes, visual theming, and Bytewax event-stream guardrails into a generated-application packet that runs without live search or RAG infrastructure.

## What It Provides

- Governed source registry with owners, source URIs, visibility, approval, and audit events.
- Knowledge articles with owner, title, body, topic, locale, source approval, publication approval, restricted-content filtering, and freshness review.
- Help search and cited answers backed by the dependency-light in-memory search and answer composer.
- Localization records with supported locale, translator, source locale, and fallback locale controls.
- Feedback capture with rating bounds and curation review for low support ratings.
- Curation items with reviewer and evidence requirements.
- Audit events for source, article, answer, localization, feedback, and curation state changes.
- Bytewax enforcement for batch help mutations.
- Dependency-light API helpers, UI view models, package manifest, semantic model, and release evidence.

## Runtime Shape

The generated runtime is `service.HelpService`. It is deterministic and in-memory so generated applications can exercise the help lifecycle without external search indexes, RAG providers, databases, notification systems, or audit services.

Primary methods:

- `register_source(...)`
- `approve_source(...)`
- `create_article(...)`
- `publish_article(...)`
- `search_articles(...)`
- `generate_answer(...)`
- `localize_article(...)`
- `record_feedback(...)`
- `close_curation_item(...)`
- `dashboard_summary(...)`

## Configuration And Rules

`capability_contract.py` is the source of truth for:

- configuration defaults
- configuration schema
- deterministic rules
- UI route contracts
- theme tokens
- APG adapter map

The rule engine returns `allow`, `require_review`, or `deny` decisions with matched rules and required actions.

## UI Surfaces

The package exposes route contracts for:

- dashboard
- home
- articles
- editor
- sources
- answers
- localization
- curation
- audit
- analytics
- settings

`views.py` provides dependency-light models for these screens.

## How To Use

```python
from capabilities.common.help.service import HelpService

service = HelpService()
source = service.register_source(
    "source-1",
    "tenant-1",
    "Account Runbook",
    "kb://runbooks/account",
    "owner-support",
)
service.approve_source(source["id"], "tenant-1", "publisher-1")
article = service.create_article(
    "article-1",
    "tenant-1",
    "Reset a password",
    "Users reset passwords from account settings after confirming email.",
    "owner-support",
    topics=["account", "password"],
    source_ids=[source["id"]],
)
service.publish_article(article["id"], "tenant-1", "publisher-1", True)
answer = service.generate_answer("answer-1", "tenant-1", "How do I reset my password?")
```

Use `register_capability()` to expose the full APG registration payload to the composition engine.

## Verification

Focused verification for this packet should use:

```bash
./.venv/bin/python -m py_compile capabilities/common/help/__init__.py capabilities/common/help/capability_contract.py capabilities/common/help/models.py capabilities/common/help/help_runtime.py capabilities/common/help/service.py capabilities/common/help/api.py capabilities/common/help/views.py capabilities/common/help/app.py capabilities/common/help/test_capability_contract.py capabilities/common/help/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/help/test_capability_contract.py capabilities/common/help/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/help --json
./.venv/bin/apg capabilities publish-plan capabilities/common/help --json
```
