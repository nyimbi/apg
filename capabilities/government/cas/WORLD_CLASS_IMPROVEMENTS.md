# Case Management — World-Class Improvements

**Capability**: `government_cas` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Real Async Throughout

**Problem**: All service methods are synchronous, blocking the event loop under load and blocking DB calls.
**Fix**: Convert every public method to `async def`, replace `datetime.utcnow()` with `asyncio`-safe clock, and use `asyncasyncpg` / `SQLAlchemy asyncio` for persistence. The single existing `async` method (`ml_case_priority_score`) shows the intent but is inconsistent.

---

## 2. Persistent PostgreSQL Backing Store

**Problem**: In-memory dicts (`self.cases`, `self.assignments`, …) lose all data on process restart. No real-world government system can accept this.
**Fix**: Wire `database/store.py` (already partially present) to the service via a `CasStore` async repository. Use `asyncpg` connection pools keyed by `tenant_id`. Alembic migrations already exist — they just need to be executed.

---

## 3. Proper SLA met / breached Computation

**Problem**: `SlaRecord.met` and `SlaRecord.breached` are set to `False` on construction and never updated. `sla_monitoring()` therefore always returns 0% compliance.
**Fix**: Compute breach status at query time by comparing `due_date` against `datetime.utcnow()`. Add an async `async_check_sla_breaches()` background task that updates all records and emits `case_sla_breached` events.

---

## 4. Event-Driven Architecture via Message Bus

**Problem**: `_audit()` writes to an in-memory list with no real event emission. Referenced `bytewax` stream (`apg.government.cas.lifecycle`) is never populated.
**Fix**: Introduce an async `EventBus` abstraction (backed by Redis Streams / Kafka / bytewax depending on configuration) that emits CloudEvents on every lifecycle transition. Downstream capabilities (`intel`, `ntfy`) subscribe rather than poll.

---

## 5. Inter-Agency Routing Engine

**Problem**: `inter_agency_referral()` returns a dict but does nothing — no routing logic, no delivery confirmation, no acknowledgement tracking.
**Fix**: Build an async `AgencyRouter` that maintains a registry of agency endpoints (REST/gRPC), dispatches referrals with retry, records acknowledgement tokens, and surfaces routing status via a new `referral_status()` method.

---

## 6. ML-Assisted Auto-Triage

**Problem**: `ml_case_priority_score()` is a stub that returns `{"ml_enhanced": False}` unless `OLLAMA_BASE_URL` is set and an unspecified `MLCapability` succeeds.
**Fix**: Implement a real async inference path using `httpx` against Ollama (`llama3:instruct` or `mistral`). Extract structured priority (urgent/high/medium/low), recommended officer skill set, and predicted resolution time from case subject + description. Feed output directly into `create_case()` when `auto_triage=True`.

---

## 7. Duplicate Case Detection

**Problem**: `bulk_case_import()` and `create_case()` silently create duplicates for the same `citizen_id` + `case_type`. The README mentions duplicate detection as an edge case but it is not implemented.
**Fix**: Add an async `find_duplicate_cases(citizen_id, case_type, lookback_days=30)` method. In `create_case()`, call this first and return a `"possible_duplicate"` flag with the existing case IDs so the caller can decide whether to proceed.

---

## 8. Structured Case Lifecycle State Machine

**Problem**: `case.status` is a free string mutated in-place with no transition guard. A closed case can be re-closed, an appealed case can be assigned — no invariant is enforced.
**Fix**: Define a `CaseStatus` `StrEnum` and a transition adjacency map. Wrap all status mutations in `_transition_status(case, new_status)` that raises `InvalidTransitionError` on illegal moves and emits a `case_status_changed` event.

---

## 9. Full-Text Case Search with PostgreSQL `tsvector`

**Problem**: `case_search()` does Python-level substring matching over all in-memory cases — O(n) and case-sensitive.
**Fix**: Add a `tsvector` column to `cas_case` and a GIN index. Implement `async_case_search()` using `to_tsquery` / `websearch_to_tsquery` via asyncpg. Return ranked results with snippet highlighting.

---

## 10. Citizen-Facing Case Portal JWT Claims

**Problem**: There is no mechanism for citizens to authenticate and retrieve only their own cases. The `citizen_case_portal_workflow` capability is advertised but unimplemented.
**Fix**: Add `async get_citizen_cases(citizen_id, jwt_token)` that validates a JWT signed by the `auth` capability, extracts `citizen_id` from claims, and returns only that citizen's cases in the current tenant. Never expose other citizens' data.

---

## 11. SLA Breach Auto-Escalation

**Problem**: The business rule `sla_breach_triggers_escalation` is referenced in README configuration but `set_sla()` / `sla_monitoring()` never read it.
**Fix**: In `async_check_sla_breaches()`, after marking a record breached, read the tenant config key `governance.sla_breach_triggers_escalation`. If `True`, automatically call `case_escalate()` with reason `sla_breach` and supervisor from the officer's chain-of-command.

---

## 12. Immutable Audit Log to append-only table

**Problem**: `audit_events` is a mutable Python list — events can be silently deleted or overwritten. Regulatory compliance (GDPR audit, government accountability) requires tamper-evident logs.
**Fix**: Write audit events to a dedicated `cas_audit_log` PostgreSQL table with a `GENERATED ALWAYS AS IDENTITY` PK, no `UPDATE`/`DELETE` grants for the application user, and a periodic Merkle-hash checkpointing job. Expose `audit_trail_verified(case_id)` that recomputes and validates the hash chain.

---

## 13. Batch Processing via Bytewax Stream

**Problem**: `validate_batch()` returns a static dict acknowledging the stream name without actually enqueuing anything.
**Fix**: Implement `async process_case_batch(cases)` that serialises each case as a CloudEvent and publishes to `apg.government.cas.lifecycle` via a bytewax source connector. Include back-pressure handling and a dead-letter queue for failed records.

---

## 14. Case Age Accurate Computation

**Problem**: `case_age_report()` increments `0-7d` for every open case regardless of actual age because `CitizenCase` has no `created_at` timestamp.
**Fix**: Add `created_at: str = field(default_factory=...)` to `CitizenCase`. In `case_age_report()`, compute `(datetime.utcnow() - datetime.fromisoformat(case.created_at)).days` and bucket correctly. This unblocks meaningful SLA reporting.

---

## 15. Role-Based Access Control on Service Methods

**Problem**: Every method calls `_enforce()` but the context always sets `policy_attached=True` and `authenticated=True` statically. No real RBAC check occurs — a read-only viewer can call `close_case()`.
**Fix**: Inject an `AuthContext` dataclass (role, permissions, citizen_id) into the service constructor. In `_enforce()`, map the `operation` to a required permission string (e.g. `government_cas:close`) and verify `AuthContext.permissions` contains it. Return HTTP 403 with a structured error payload from the Flask-AppBuilder API layer.
