# Document Control (grc_doc) — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

Fifteen high-leverage improvements to elevate this capability from a solid APG
capability packet to an enterprise-grade, cloud-native document-control system.

---

## 1. Async-First Service Layer

**Problem**: The service is synchronous, blocking the event loop when called
from async Python applications.

**Fix**: Convert all mutating methods to `async def`. Use `asyncio.gather` for
bulk operations (e.g. `bulk_archive`). Keep synchronous shim wrappers for
CLI/legacy callers so backward compatibility is preserved.

**Impact**: Unlocks non-blocking I/O paths to PostgreSQL (asyncpg), S3/object
store, and the Bytewax event bus without context switching overhead.

---

## 2. Persistent Document Store via asyncpg + PostgreSQL

**Problem**: In-memory dicts evaporate on restart; no durable state.

**Fix**: Introduce a `DocumentStore` abstraction (adapter pattern) with an
`AsyncPGDocumentStore` implementation. Wire it into the service constructor
with a default in-memory store so tests remain dependency-free.

**Impact**: Production tenants get ACID guarantees, indexes, and cross-replica
consistency. Tests stay fast without any mock patching.

---

## 3. Structured Pydantic v2 Event Payloads

**Problem**: Audit events are raw `dict[str, Any]`, losing schema enforcement
and downstream type safety.

**Fix**: Model every lifecycle event (`DocumentCreatedEvent`,
`DocumentApprovedEvent`, …) as Pydantic v2 `BaseModel` with `model_validate`.
Serialize with `.model_dump(mode="json")` before emitting to Bytewax.

**Impact**: Events become self-documenting, schema-versioned, and directly
deserializable by downstream consumers.

---

## 4. Approval Workflow Engine with Multi-Stage Support

**Problem**: Approval is a single-step operation with no routing, escalation,
or delegation.

**Fix**: Introduce `WorkflowDefinition` (stages, required approvers, quorum
thresholds, escalation rules) and a `WorkflowExecution` runtime stored as a
first-class entity. `approve_document` becomes a stage-advance operation.

**Impact**: SOX, ISO 27001, and NIST document-control requirements map
directly to workflow stage configurations without custom code.

---

## 5. Digital Signature Verification

**Problem**: No cryptographic assurance that an approved document has not been
tampered with post-approval.

**Fix**: Add `sign_document` and `verify_document_signature` methods. Use
`cryptography` (PyPI) to compute PKCS#7 / detached Ed25519 signatures. Store
the signature blob and verification certificate reference on the document
record.

**Impact**: Closes the chain-of-custody gap required by 21 CFR Part 11, EU
eIDAS, and NIST SP 800-57.

---

## 6. Attribute-Based Encryption for Restricted Content

**Problem**: Classification="restricted" today is a label only; the content
itself is not protected at rest.

**Fix**: Add `encrypt_document_content` and `decrypt_document_content` async
methods backed by envelope encryption (AES-256-GCM data key + KMS-wrapped
master key). Store cipher-text and key-ref on the record; never store plaintext
for restricted docs.

**Impact**: Satisfies GDPR Article 32, ISO 27001 A.10, and common data-at-rest
requirements without coupling to any specific cloud KMS.

---

## 7. Full-Text Search via Tantivy / Meilisearch Adapter

**Problem**: `document_search` does substring matching over in-memory dicts,
O(N) per query.

**Fix**: Define a `SearchAdapter` interface. Ship a `TantivySearchAdapter` for
local deployments and a `MeilisearchSearchAdapter` for cloud. The service calls
`search_adapter.index(doc)` on every create/update and `search_adapter.query`
on search.

**Impact**: Sub-10 ms full-text search across millions of documents with
ranking, faceting, and highlighted snippets.

---

## 8. Document Lineage Graph

**Problem**: `document_link` creates ad-hoc relationships with no queryable
lineage model.

**Fix**: Introduce a `DocumentLineage` entity storing directed edges
(`source_id`, `target_id`, `relationship_type`, `created_at`, `created_by`).
Add `get_document_lineage(doc_id)` returning an upstream + downstream walk.
Store in a PostgreSQL recursive CTE-friendly schema or a lightweight adjacency
list.

**Impact**: Enables impact analysis ("which policies reference this control
standard?") required for ISO 27001 A.5.1 control mapping.

---

## 9. Legal-Hold Notification and Escalation

**Problem**: Legal holds are set silently with no notification, expiry, or
automatic escalation.

**Fix**: Add `escalate_legal_hold` that emits a structured `LegalHoldEscalated`
event with SLA deadline, responsible party, and hold reason. Integrate with the
Bytewax topology to drive downstream notification (email, Slack, webhook).

**Impact**: Closes the notification gap in eDiscovery and litigation-hold
workflows required by FRCP Rule 37(e) and common enterprise legal-ops processes.

---

## 10. Retention-Class Enforcement with Automated Disposition Queue

**Problem**: `retention_enforce` flags expired documents but takes no action
and relies on manual follow-through.

**Fix**: Add `queue_disposition_review` that moves expired, non-held documents
into a `DispositionQueue` with a configurable review SLA. Add
`auto_dispose_approved_queue` for fully-automated disposition of low-risk
documents when policy allows it.

**Impact**: Eliminates the retention-gap risk that causes organizations to
retain data indefinitely, creating regulatory exposure.

---

## 11. OCR + NLP Pipeline Integration (Ollama-backed)

**Problem**: `full_text_index` rebuilds a count of documents but performs no
actual text extraction or semantic indexing.

**Fix**: Add `async extract_document_text` and `async classify_document_content`
that call a local Ollama endpoint (OLLAMA_BASE_URL env var). Gate on env var
presence so CI tests skip gracefully. Store extracted text, entity spans, and
classification labels on the document record.

**Impact**: Closes the unstructured-document gap — scanned PDFs, images, and
legacy Word docs become searchable and correctly classified without cloud
vendor lock-in.

---

## 12. Watermarking and Redaction Engine

**Problem**: There is no mechanism to produce redacted copies or watermarked
distributions for external parties.

**Fix**: Add `redact_document` (mask PII/sensitive spans by XPath or regex
rules) and `watermark_document` (embed tenant + recipient + timestamp
watermark). Output is a new derivative document linked via lineage to the
source.

**Impact**: Enables controlled external disclosure for legal discovery,
regulatory submission, and contractor access without exposing source documents.

---

## 13. Webhook and Push Notification Outbox

**Problem**: Audit events are written to `_audit_events` in-memory with no
delivery mechanism to external consumers.

**Fix**: Implement a transactional outbox table (`grc_doc_outbox`) populated
inside the same transaction as every state change. A background worker drains
the outbox to webhook endpoints or an AMQP exchange. Include retry logic with
exponential backoff and a dead-letter queue.

**Impact**: Makes event delivery at-least-once reliable, a hard requirement for
any integration with SIEM, GRC platforms (ServiceNow, Archer), or audit vaults.

---

## 14. Row-Level Tenant Isolation via PostgreSQL RLS

**Problem**: Tenant isolation is enforced in Python code — a single misrouted
query could leak cross-tenant data.

**Fix**: Add PostgreSQL Row Level Security policies on every `ds_*` table
(`CREATE POLICY tenant_isolation ON ds_documents USING (tenant_id =
current_setting('app.tenant_id'))`). Set `app.tenant_id` at the database
session level when opening a connection.

**Impact**: Defense-in-depth: even a SQL injection or application logic bug
cannot exfiltrate another tenant's documents. Required by SOC 2 Type II CC6.6.

---

## 15. Capability Metrics and SLA Monitoring

**Problem**: `dashboard_summary` returns aggregate counts but no latency,
throughput, or SLA compliance metrics.

**Fix**: Add `async record_operation_metric(op, duration_ms, tenant_id)` and
`async get_sla_report(tenant_id, period_days)` that compute P50/P95/P99
operation latencies, approval cycle times, and retention-compliance rates.
Expose a Prometheus-compatible `/metrics` endpoint via the Flask-AppBuilder
blueprint.

**Impact**: Gives GRC ops teams measurable SLAs on document cycle time,
approval throughput, and retention compliance — the three metrics auditors
consistently ask for.
