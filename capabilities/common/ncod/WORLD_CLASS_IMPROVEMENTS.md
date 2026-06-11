# NCOD - World-Class Improvement Proposals

**Capability**: No-Code Builder (`ncod`)
**Author**: Nyimbi Odero
**Copyright**: © 2025 Datacraft

---

## 1. Async-First Service Layer

**Problem**: All `NcodService` methods are synchronous, blocking the event loop when I/O-bound persistence, webhook delivery, or remote validation is wired in.

**Solution**: Convert the service to `async def` throughout. Caller sites using `asyncio.run()` remain simple; FastAPI / Bytewax adapters gain zero-overhead coroutine composition. Internal helpers stay sync where no I/O occurs.

**Impact**: Unlocks concurrent app builds, parallel connector probing, and non-blocking publish pipelines without thread-pool workarounds.

---

## 2. Persistent Backend via Async SQLAlchemy

**Problem**: All state is held in dicts in process memory — lost on restart, not query-able, not concurrent-safe.

**Solution**: Wire `async_sessionmaker` (SQLAlchemy 2 + asyncpg) behind a `NcodStore` interface with the same method signatures. The in-process dict store remains as a `MemoryNcodStore` for tests. Alembic migrations already exist at `alembic/versions/0001_initial.py`.

**Impact**: Production-grade durability, multi-process horizontal scaling, and SQL-level reporting on app/release/deployment state.

---

## 3. Event Streaming via Domain Events

**Problem**: Mutations emit only internal audit records — no external subscriber can react to `app_published` or `component_added` without polling.

**Solution**: Publish typed `NcodDomainEvent` instances to a `asyncio.Queue` (in-process) or a Bytewax / Kafka adapter on every state-change. The existing `domain/events.py` skeleton provides the entry point.

**Impact**: Real-time dashboards, downstream capability triggers (WFLO automation on publish), and complete event-sourcing replay.

---

## 4. AI-Assisted Component Generation

**Problem**: The `BuilderAgent` model registers agents but never actually invokes them — generation is left entirely to external tooling.

**Solution**: Add `async generate_component_from_prompt(...)` that delegates to a locally-hosted Ollama model (LLaVA / Mistral) via the `ai` capability adapter. The agent produces a structured `BuilderComponent` proposal that the service validates before persisting.

**Impact**: True no-code UX — describe a component in natural language, get a validated, policy-checked record back.

---

## 5. Form Schema Inference from Data Model

**Problem**: Users who define a `DataModelDefinition` must manually recreate the same fields as form `BuilderComponent` props, causing drift.

**Solution**: `async infer_form_from_data_model(...)` inspects a `DataModelDefinition` and emits a complete set of typed `BuilderComponent` records (input, select, checkbox) wired to the model's fields via `DataBinding`, enforcing accessibility labels automatically.

**Impact**: Eliminates one of the most common no-code builder pain points — form scaffolding from an existing schema.

---

## 6. Multi-Tenant App Cloning / Templating Engine

**Problem**: `app_template` supports only four hard-coded archetypes and cannot clone a live app across tenants.

**Solution**: `async clone_app(...)` deep-copies an app's pages, components, data models, workflow bindings, theme variants, and connector bindings to a new tenant namespace, rewriting all internal IDs deterministically and emitting a full audit trail.

**Impact**: White-labelling, tenant onboarding acceleration, and A/B deployment of app variants.

---

## 7. Incremental Validation with Per-Check Caching

**Problem**: `validate_app` reruns every check on every call, even when nothing has changed since the last validation.

**Solution**: Hash each sub-domain (pages, components, data models, etc.) and cache check outcomes keyed by content-hash. Only invalidated domains are re-evaluated. The `ValidationResult` carries a `cache_key` and `stale_checks` list.

**Impact**: Sub-millisecond validation on unchanged apps; critical for interactive builder UIs that validate on every drag-drop.

---

## 8. Visual Diff / Change-Set Between Versions

**Problem**: `version_control_app` records a commit message but cannot show what changed between two versions.

**Solution**: `async app_diff(tenant_id, app_id, version_a, version_b)` produces a structured change-set (`added`, `removed`, `modified`) across pages, components, data models, and bindings — similar to a git diff at the logical layer.

**Impact**: Auditors and reviewers can inspect what changed before approving a production publish without reading raw audit event streams.

---

## 9. Role-Based Builder Permissions per Page

**Problem**: RBAC policy is a single opaque string on the `BuilderApp` — there is no per-page or per-component permission model.

**Solution**: `async set_page_permissions(...)` attaches a structured `PagePermission` record (`roles_allowed`, `roles_denied`, `conditions`) to each `BuilderPage`. `add_component` and `preview_deploy` enforce page-level RBAC before mutating or serving.

**Impact**: Fine-grained access control for multi-team apps where different groups own different screens.

---

## 10. Webhook / Notification Dispatch on Lifecycle Events

**Problem**: External systems (CI/CD pipelines, Slack, PagerDuty) have no automated signal when an app is published or a deployment fails.

**Solution**: `async register_webhook(...)` stores `WebhookConfig` records per event type. After each `_audit(...)` call, the service fires `async _dispatch_webhooks(event_type, payload)` via `aiohttp` with retry and circuit-breaker semantics.

**Impact**: Zero-integration-cost observability — developers and operators receive real-time notifications without polling the audit stream.

---

## 11. Snapshot / Rollback of App State

**Problem**: Once a component is added or modified there is no way to revert to a known-good state short of replaying audit events manually.

**Solution**: `async snapshot_app(...)` serializes the full app graph (pages, components, models, bindings) to a JSON blob stored in the `NcodSnapshot` table. `async restore_snapshot(...)` atomically replaces all in-scope records.

**Impact**: Safe experimentation — builders can try destructive restructuring with a one-call undo, enabling a proper Ctrl-Z mental model.

---

## 12. Data Pipeline Preview / Sample Data Injection

**Problem**: `DataBinding` records exist but there is no way to test them with real or synthetic data before publishing.

**Solution**: `async preview_data_binding(binding_id, tenant_id, sample_rows)` validates `sample_rows` against the binding's `schema.fields`, runs any attached `ScriptExtension` transforms, and returns a preview result set with per-field type conformance scores.

**Impact**: Catches schema mismatches and transform errors at design time, not at runtime in production.

---

## 13. Accessibility Audit Report

**Problem**: `accessibility_checked` is a boolean flag set by the caller — there is no automated verification of actual accessibility compliance.

**Solution**: `async accessibility_audit(tenant_id, app_id)` iterates all interactive `BuilderComponent` records and applies WCAG 2.1 Level AA heuristics: missing `accessibility_label`, low-contrast theme tokens, missing form field associations, keyboard-navigation ordering gaps. Returns a structured `AccessibilityReport` with per-component findings and a WCAG compliance score.

**Impact**: Removes the honor-system boolean; gives tenants an actionable checklist before enabling `accessibility_checked`.

---

## 14. App Performance Budget Enforcement

**Problem**: No-code builders are notorious for producing bloated apps — unlimited components, unbounded data bindings, and unvalidated chart queries.

**Solution**: `async enforce_performance_budget(tenant_id, app_id, budget_ref)` loads a `PerformanceBudget` policy (max components per page, max data binding result-set rows, max workflow steps) and emits `budget_violation` audit events with severity `warning` or `error` when thresholds are breached. `validate_app` includes budget checks.

**Impact**: Prevents runaway resource consumption before it reaches the deployment target, especially critical on `edge_worker` runtimes.

---

## 15. Composable Micro-App Federation

**Problem**: Large organizations need to split apps across teams but compose them at runtime — NCOD has no sub-app or module composition model.

**Solution**: `async federate_app(host_app_id, remote_app_id, mount_route, tenant_id)` creates a `FederatedMount` record that embeds a remote app's route tree under a mount point in the host app, similar to Webpack Module Federation. The publish gate verifies both apps are in `validated` status and that their policy refs are compatible.

**Impact**: Enterprise-scale composability — platform teams publish shared modules (auth screens, reporting dashboards) that product teams embed without copy-paste, with independent lifecycle management.

---

*15 improvements identified. Priorities 1–3 and 5–6 are immediately actionable without external dependencies. Priorities 10–15 require new models and store schema additions.*
