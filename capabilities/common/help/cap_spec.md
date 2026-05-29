# Help and Knowledge Base Capability Specification

- **Capability Name**: Help and Knowledge Base
- **Capability ID**: `help`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`help` provides an executable, tenant-aware help center and knowledge-base
runtime for APG-generated Python applications. It supports article authoring,
approval, publication, restricted-content filtering, deterministic search,
cited answer composition, user feedback, curation queues, support analytics,
UI route metadata, visual theme metadata, and publishable package evidence.

The package remains dependency-light and runs in process. Production RAG,
semantic search, RBAC, audit logging, notification, chat, and documentation
index integrations should be attached behind adapters rather than embedded in
the package facade.

## Runtime Surfaces

- `HelpArticle`: owned knowledge article with lifecycle state, visibility,
  locale, topics, sources, review timestamps, and publication timestamps.
- `HelpAnswer`: cited generated or curated answer with confidence and blocking
  metadata.
- `HelpFeedback`: user feedback linked to articles or answers and routed into
  curation when review is required.
- `HelpCurationItem`: approval, freshness, and support-feedback review task.
- `help_runtime.py`: HELP-specific search indexing, cited answer composition,
  and freshness inspection.
- `HelpService`: executable facade for article authoring, publication, search,
  answer generation, feedback, curation, compatibility records, dashboard
  summaries, and contract rule evaluation.
- `api.py`: dependency-light helpers for generated apps and package probes.
- `views.py`: help center, article editor, answer console, curation queue,
  support analytics, dashboard, routes, rules, and theme view models.

## Provided Services

- `help_center`
- `knowledge_articles`
- `assisted_answers`
- `content_curation`
- `support_analytics`

## Required Services

- `tenant_context`

Optional production adapters may use `ragn`, `srch`, `nlpc`, `auth`, `audl`,
`chat`, and `ntfy` as described by package registration metadata.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. The content configuration controls ownership, approval,
freshness-review age, and localization. The answers configuration controls
citations, RAG readiness, answer confidence, and unsafe answer blocking.

## Rules

The package enforces the contract rule engine through `HelpService`:

- `tenant_context_required`
- `article_requires_owner`
- `publication_requires_approval`
- `answer_requires_citations`
- `restricted_content_requires_filtering`
- `stale_article_requires_review`

Additional service guardrails validate same-tenant article and answer
references, rating ranges, publication state, restricted-content filtering,
answer citation availability, curation deduplication, and generated-package
compatibility records.

## UI

The package exposes APG Python UI route contracts and dependency-light view
models for:

- dashboard
- help center
- article library
- article editor
- answer console
- curation queue
- support analytics
- settings

## Theme

The package uses the `help_support_knowledge` APG theme contract with article
library, answer panel, curation queue, and feedback table component tokens.

## Verification

Use focused package verification first:

```bash
./.venv/bin/python -m py_compile capabilities/common/help/__init__.py capabilities/common/help/models.py capabilities/common/help/help_runtime.py capabilities/common/help/service.py capabilities/common/help/api.py capabilities/common/help/views.py capabilities/common/help/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/help/test_capability_contract.py capabilities/common/help/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/help --json
./.venv/bin/apg capabilities publish-plan capabilities/common/help --json
```
