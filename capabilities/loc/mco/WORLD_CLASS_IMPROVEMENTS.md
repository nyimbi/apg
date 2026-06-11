# World-Class Improvements: loc_mco (Multi-Country Operations)

**Capability**: `loc_mco` | **Version**: 1.0.0 → 2.0.0

---

## Improvement 1 — Replace in-memory dicts with async PostgreSQL store

**Current**: All state lives in instance-level Python dicts. Restarts lose data; no horizontal scaling.
**Improvement**: Inject an async SQLAlchemy `AsyncSession` (or `asyncpg` connection pool) into the service constructor. Swap every `self._countries[...]` read/write for `await store.get/upsert`. Zero-downtime migrations via Alembic (already scaffolded in `alembic/`).
**Impact**: Production-grade durability, horizontal scale, MVCC isolation between tenants.

---

## Improvement 2 — Fix broken attribute references in 7 "new" methods

**Current**: Methods added in lines 613–785 reference `self._intercompany_transactions`, `self._compliance_mappings`, and `entity.entity_id`/`entity.entity_name` — none of which exist. They silently return empty or raise `AttributeError` at runtime.
**Improvement**: Align all attribute names with the actual store dicts (`self._intercompany`, `self._compliance`) and model fields (`entity.id`, `entity.name`). Add integration tests that exercise every method path.
**Impact**: Eliminates silent runtime failures; turns dead code into working code.

---

## Improvement 3 — Multi-tenancy via constructor injection, not per-call enforcement

**Current**: `_enforce_tenant_context` is called at the top of every method, creating a repetitive, error-prone guard that a future developer will inevitably forget.
**Improvement**: Require `tenant_id: str` at construction time (or via a `BoundedCache`-backed factory). Every internal method operates on `self._tenant_id` without callers having to pass it, eliminating an entire class of cross-tenant leakage bugs.
**Impact**: Impossible-by-construction tenant isolation; less boilerplate per method.

---

## Improvement 4 — Async batch operations for bulk entity and country onboarding

**Current**: No batch API. Registering 50 subsidiaries requires 50 sequential round-trips.
**Improvement**: Add `register_entities_bulk(payloads: list[EntityCreate])` and `register_countries_bulk(payloads: list[CountryCreate])` that fan out with `asyncio.gather` and return a `BatchResult` (successes, failures with per-item errors). Single audit event with item count.
**Impact**: 10-50x throughput improvement for group onboarding workflows; partial failure reporting.

---

## Improvement 5 — Compliance deadline alerting via `ntfy` integration

**Current**: `list_compliance_mappings` returns records but nothing surfaces items overdue for review.
**Improvement**: Add `compliance_review_alerts(tenant_id, lookahead_days=14)` that scans `next_review_date`, compares to `datetime.utcnow()`, and emits structured `compliance_review_due` events (with entity_id, domain, days_remaining) to the `ntfy` capability. Scheduled invocation via APG cron hook.
**Impact**: Proactive compliance posture; zero overdue reviews slipping through silently.

---

## Improvement 6 — Entity hierarchy traversal (parent → subsidiary graph)

**Current**: `parent_entity_id` is stored but never leveraged. No way to query the full ownership graph.
**Improvement**: Add `get_entity_hierarchy(tenant_id, root_entity_id)` returning a nested tree dict built via iterative BFS over `self._entities`. Include depth, direct children count, and aggregate `is_active` flag per subtree.
**Impact**: Enables group consolidation dashboards, audit scope calculation, and CBCR (Country-by-Country Reporting) scoping.

---

## Improvement 7 — Idempotent transaction submission (deduplication key)

**Current**: Duplicate `create_intercompany_transaction` calls with identical data create multiple records. No idempotency guard.
**Improvement**: Accept an optional `idempotency_key: str` in `IntercompanyTransactionCreate`. Hash `(tenant_id, idempotency_key)` and cache the result for 24 hours in `BoundedCache`. Return the existing record on replay without re-emitting events.
**Impact**: Safe retry semantics for finance integrations; prevents double-booking in ERP sync jobs.

---

## Improvement 8 — Structured transfer pricing risk scoring

**Current**: `transfer_pricing_check` uses a hard-coded 100k baseline proxy — meaningless in production.
**Improvement**: Replace heuristic with a configurable `MarketRateProvider` protocol. Implementations: `ConfigMarketRateProvider` (reads from `conf` capability), `OECDComparableProvider` (fetches OECD CbCR data async). Risk score = weighted deviation × entity size × jurisdiction tax-haven flag. Return `risk_tier: low|medium|high|critical`.
**Impact**: Actual OECD BEPS compliance posture; auditable risk scoring instead of magic numbers.

---

## Improvement 9 — Statutory report overdue escalation workflow

**Current**: Reports are flagged `overdue` in status but there is no escalation path or SLA tracking.
**Improvement**: Add `escalate_overdue_reports(tenant_id, escalation_owner_id)` that finds all `overdue` reports, creates an `McoEscalationEvent`, updates each report with `escalated_to` and `escalated_at`, and emits `statutory_report_escalated` events. Supports configurable escalation chains (owner → manager → compliance officer).
**Impact**: Closes the compliance loop; overdue reports cannot be indefinitely ignored.

---

## Improvement 10 — Currency-normalised intercompany exposure summary

**Current**: `dashboard_summary` shows transaction counts but not financial exposure. Amounts live in mixed currencies with no normalisation.
**Improvement**: Add `intercompany_exposure_summary(tenant_id, reporting_currency)` that fetches live FX rates via an injected `FxRateProvider`, converts all outstanding ICT amounts to `reporting_currency`, and returns `gross_exposure`, `net_exposure`, and `currency_breakdown` dict.
**Impact**: CFO-grade treasury view; risk management cannot function without normalised exposure.

---

## Improvement 11 — Event sourcing for compliance mapping state machine

**Current**: Compliance mapping status is mutated in-place. Historical state (who changed what, when) is in audit events but the mapping record itself shows only current state.
**Improvement**: Implement a `ComplianceMappingStateTransition` event model. Store each transition (old_status → new_status, actor, timestamp, justification) in a separate `_compliance_history` dict. Add `get_compliance_mapping_history(tenant_id, mapping_id)` returning the ordered transition log.
**Impact**: Full audit trail required by ISO 27001, SOC 2, and most AML regulations.

---

## Improvement 12 — Typed capability contract enforcement via Pydantic

**Current**: `_enforce` takes an untyped `dict[str, Any]` and calls `evaluate_capability_rules`. Typos in context keys silently fail enforcement.
**Improvement**: Define `CapabilityContext` Pydantic models per operation (e.g. `WriteOperationContext`, `ApprovalContext`). `_enforce` accepts a typed context model, calls `.model_dump()` internally. Missing required fields raise `ValidationError` at call site, not deep in the rule engine.
**Impact**: Shifts enforcement failures from runtime surprises to development-time type errors.

---

## Improvement 13 — Consolidation elimination entries for intercompany transactions

**Current**: `holding_consolidation` sums subsidiary revenue/liabilities naively — it does not eliminate intercompany balances, producing inflated consolidated figures.
**Improvement**: `holding_consolidation` calls `intercompany_reconcile` for every pair of subsidiaries, collects `net_balance` values, and subtracts them from consolidated totals. Returns `eliminated_amount`, `gross_consolidated`, and `net_consolidated` separately.
**Impact**: IFRS 10 / GAAP-correct consolidation; material misstatement risk eliminated.

---

## Improvement 14 — Paginated list operations with cursor-based pagination

**Current**: All `list_*` methods return full unbounded lists. At scale (10k+ entities, 100k+ transactions) this causes OOM and multi-second response times.
**Improvement**: Add `cursor: str | None = None` and `limit: int = 100` to all `list_*` signatures. Cursor encodes `(created_at, id)` as a base64 token. Return `ListPage[T]` with `items`, `next_cursor`, `total_count`. Compatible with UUID7 monotonic ordering.
**Impact**: Sub-100ms list responses at any data volume; required for production use.

---

## Improvement 15 — CBCR (Country-by-Country Reporting) aggregate export

**Current**: No OECD BEPS Action 13 CbCR support despite having all required data (entities per jurisdiction, intercompany flows, registration numbers).
**Improvement**: Add `generate_cbcr_report(tenant_id, fiscal_year)` that aggregates entity count, revenue, tax paid, and intercompany flows per jurisdiction. Output conforms to the OECD XML schema (Table I, II, III) and can be serialised to JSON or XML. Emit `cbcr_report_generated` event with hash of the output for audit.
**Impact**: Directly addresses OECD BEPS Action 13 obligations for MNEs; differentiates the capability from generic MCO tools.
