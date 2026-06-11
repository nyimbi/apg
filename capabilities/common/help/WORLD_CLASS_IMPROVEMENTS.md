# Help & Knowledge Base — World-Class Improvements

**Capability**: `help` | **Path**: `capabilities/common/help`
**Date**: 2026-06-11 | **Author**: Nyimbi Odero © Datacraft 2025

---

## 1. Async-Native Service Layer

The current `HelpService` is synchronous. All I/O-bound operations (database reads, Ollama inference, vector index lookups) block the event loop. Every public method should be `async def`, enabling true concurrent article fetching, answer generation, and feedback aggregation under asyncio without thread-pool overhead.

---

## 2. Pluggable Vector Search Backend

`HelpSearchIndex` uses token overlap scoring. Replace with a pluggable `SearchBackend` protocol that accepts `BM25InMemoryBackend`, `ChromaBackend`, `WeaviateBackend`, or `OllamaEmbeddingBackend` at construction time. This lets the service run locally with BM25 and graduate to semantic search (Ollama `nomic-embed-text`) without changing call sites.

---

## 3. Ollama-Backed RAG Answer Generation

`HelpAnswerComposer` generates deterministic stub answers. Wire it to a local Ollama instance via `httpx.AsyncClient`: embed the query, retrieve top-k chunks from the vector backend, and stream the completion from `llama3` or `mistral`. Surface a streaming async generator `stream_answer(...)` alongside the blocking `generate_answer(...)` for real-time chat UX.

---

## 4. Persistent Storage via SQLAlchemy + Alembic

In-memory dicts are already replaced by stubs in `database/store.py` and `alembic/versions/0001_initial.py`, but `HelpService` still reads from `self._articles` etc. Complete the wiring: inject a `StoreBackend` protocol, provide an `AsyncSQLAlchemyBackend` (PostgreSQL via `asyncpg`) and keep the current `InMemoryBackend` for tests. Every list/get/put maps to a repository method rather than dict lookups.

---

## 5. Event-Driven Audit Bus

`_record_event` appends to `self._audit_events`. Replace with an async `EventBus` that fans out to pluggable sinks: in-memory (tests), PostgreSQL `audit_events` table (default), and optionally a Kafka/Redpanda topic. This decouples the service from audit persistence and enables real-time audit dashboards via Server-Sent Events.

---

## 6. Structured Feedback Analytics Pipeline

`feedback_aggregate` and `feedback_analysis` do ad-hoc list comprehensions. Build a `FeedbackAnalyticsPipeline` that computes sliding-window averages, sentiment trend detection (via Ollama `phi3` classification), and per-topic CSAT scores. Expose `async get_feedback_trends(tenant_id, window_days)` returning time-series data consumable by the analytics UI.

---

## 7. Article Freshness Scoring & Proactive Notifications

`HelpFreshnessInspector` uses a fixed age threshold. Extend with a staleness score (0–1) combining: days since review, feedback velocity, edit recency, and topic popularity. Expose `async score_article_freshness(article_id)` and hook it into a scheduled job that sends email/webhook notifications to article owners before content becomes stale.

---

## 8. Multi-Tenant RBAC with Attribute-Based Access Control

Visibility filtering uses a boolean `rbac_filter_applied` flag passed by callers. Replace with a `RBACContext` dataclass carrying `user_id`, `roles`, `tenant_id`, and `custom_attributes`. Implement `async authorize(context, resource, action)` that evaluates ABACpolicies stored in the capability contract, eliminating the possibility of callers bypassing access checks.

---

## 9. Bulk Article Import with Validation Pipeline

`faq_bulk_create` exists but is FAQ-only. Implement `async bulk_import_articles(tenant_id, payload, format)` supporting `json`, `markdown`, `csv`, and `docx` input. Run each item through a validation pipeline: schema check → duplicate detection → source approval check → content safety scan (Ollama `llava` for images embedded in markdown). Return a detailed `ImportReport` with per-row results.

---

## 10. AI-Assisted Article Drafting

Add `async draft_article(tenant_id, title, context_hints, locale)` that calls a local Ollama model to produce an initial article body from a title and bullet-point hints. The draft is saved with `status=draft` and flagged `ai_assisted=True`. Human review is required before publication — enforced by policy rule `ai_draft_requires_human_review`.

---

## 11. Contextual Help Overlays (In-App Help)

The current capability serves articles as full pages. Add `async get_contextual_help(tenant_id, context_key, locale)` that returns a short tooltip-sized snippet (≤ 280 chars) extracted from a matched article. Enable product teams to tag articles with `context_keys` (e.g. `"payments.refund_button"`) and surface just-in-time help inside Flask-AppBuilder views via a JS injection hook.

---

## 12. Semantic Duplicate Detection

Before `create_article` completes, run `async detect_duplicates(tenant_id, title, body)` using cosine similarity against the existing article corpus (via the vector backend). Return a `DuplicateWarning` with the top-3 similar articles and similarity scores. Callers receive the warning in the response; governance policy can escalate it to a curation item automatically.

---

## 13. Article Lifecycle Webhooks

Integrate a `WebhookDispatcher` that fires HTTP POST events on key transitions: `article.published`, `article.unpublished`, `feedback.low_rating`, `curation.opened`. Tenants register webhook endpoints via `async register_webhook(tenant_id, event_type, url, secret)`. Delivery uses exponential back-off with 3 retries and stores delivery receipts in the audit store.

---

## 14. Internationalization Quality Gates

`localize_article` currently accepts any string as translated body. Add `async validate_translation(localization_id)` that: (a) checks the locale is in `supported_locales`, (b) runs a back-translation check via Ollama (translate back to source and compare BLEU-score proxy), and (c) flags severe divergence as a curation item. This prevents mistranslations from reaching end users.

---

## 15. Capability-Level Health & SLO Dashboard

Extend `dashboard_summary` to emit structured SLO metrics: `answer_confidence_p50`, `answer_confidence_p95`, `feedback_csat_7d`, `stale_article_pct`, `open_curation_age_p90`. Expose `async get_slo_report(tenant_id, period)` returning a `SLOReport` model. Wire into the APG observability bus so operators can alert on SLO breaches without bespoke monitoring code.
