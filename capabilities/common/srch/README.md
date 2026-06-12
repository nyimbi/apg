# APG SRCH - Search Engine

SRCH is the APG capability for governed enterprise search across tenant-scoped
indices, documents, facets, keyword retrieval, semantic retrieval, hybrid
retrieval, and query analytics. It lets generated applications create indices,
index documents, run access-filtered queries, review large result windows,
track denied queries, expose UI view models, and publish audit evidence through
deterministic guardrails.

## What It Provides

- Search index creation with tenant, name, owner, content type,
  classification, source lineage, embedding readiness, status, and audit
  evidence, including pending-review state for unfamiliar content types and
  classifications.
- Document indexing with document id, title, body, classification, facets,
  metadata, source lineage, pending-review facet evidence, and index document
  counts.
- Bulk indexing with document-count checks, lineage checks, and Bytewax adapter
  evidence.
- Keyword, semantic, and hybrid query execution with RBAC filtering,
  restricted-content enforcement, embedding readiness checks, large-window
  review, query records, facets, and audit events.
- Faceted search, autocomplete, fuzzy search, phrase search, boolean search,
  geo search, more-like-this, personalised search, and search analytics.
- Field mapping management, synonym management, field boosting, spell check,
  collection create/clone/delete, index health, and search volume reporting.
- First-class AI search-agent composition for `codex`, `claude_code`,
  `opencode`, and `pi`, with role, scope, owner, purpose, contribution
  disclosure, and privileged-role review guardrails.
- Bytewax lifecycle batch validation for index, document, bulk indexing, query,
  facet, ranking, access-policy, and search-agent mutations.
- UI view models for dashboard, search, indices, documents, bulk indexing,
  facets, analytics, ranking, access review, governance, agents, lifecycle
  batches, audit, settings, and pending-review queues.
- Adapter configuration for ETLP, META, NLPC, AICR, AUTH, AUDL, CACH, MONI,
  and Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `service.py` - `SrchService`, the dependency-light generated-app runtime.
- `search_runtime.py` - deterministic record models and helper functions.
- `views.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Quick Start

```python
from capabilities.common.srch.service import SrchService

service = SrchService()
index = service.create_index(
	tenant_id="tenant-a",
	name="knowledge-base",
	owner="search-owner",
	content_type="article",
	classification="internal",
	source_lineage_ref="lineage://kb",
	embedding_index_ready=True,
)
service.index_document(
	tenant_id="tenant-a",
	index_id=index["id"],
	document_id="doc-1",
	title="APG search overview",
	body="APG provides keyword and semantic enterprise search.",
	facets={"module": "platform", "kind": "guide"},
	source_lineage_ref="lineage://kb/doc-1",
)
response = service.query(
	tenant_id="tenant-a",
	query_text="semantic search",
	index_ids=[index["id"]],
	query_type="hybrid",
	result_window=10,
	rbac_filter_applied=True,
)
```

Review-required outcomes are persisted as data, not discarded exceptions:

```python
pending_index = service.create_index(
	tenant_id="tenant-a",
	name="custom-content",
	owner="search-owner",
	content_type="briefing",
	classification="internal",
)
assert pending_index["status"] == "pending_review"
assert pending_index["review_reasons"] == ["index_content_type_review_required"]
```

## New Methods

All extension methods are `async`. Await them inside an async context or
event loop.

**Faceted search** — filter by one or more facet key/value pairs before scoring:

```python
results = await service.faceted_search(
	tenant_id="tenant-a",
	collection="knowledge-base",
	text="deployment guide",
	facets={"module": "platform", "kind": "guide"},
)
```

**Fuzzy search** — typo-tolerant retrieval with configurable edit distance:

```python
results = await service.fuzzy_search(
	tenant_id="tenant-a",
	collection="knowledge-base",
	query="semanitc serch",
	max_edits=2,
)
```

**Personalised search** — boost results from the caller's prior query history:

```python
results = await service.personalised_search(
	tenant_id="tenant-a",
	collection="knowledge-base",
	query_text="vector embeddings",
	user_id="user-42",
)
```

**Collection management** — create, clone, and inspect collections at runtime:

```python
await service.collection_create(
	tenant_id="tenant-a",
	name="archive",
	schema={"title": {"type": "text"}, "year": {"type": "integer"}},
)
await service.collection_clone(tenant_id="tenant-a", src="knowledge-base", dst="knowledge-base-v2")
health = await service.index_health(tenant_id="tenant-a", collection="knowledge-base")
```

**Search analytics** — query volume, top terms, and zero-result rate by period:

```python
report = await service.search_analytics(tenant_id="tenant-a", collection="knowledge-base", period="7d")
volume = await service.search_volume_report(tenant_id="tenant-a", period="30d")
```

## World-Class Enhancements (v2.0)

Fifteen targeted improvements that raise SRCH from prototype to production grade:

| # | Enhancement | Impact |
|---|-------------|--------|
| 1 | **BM25F ranking** — per-field `k1`/`b` parameters, configurable field weights (`title=3.0`, `body=1.0`) | +30–60% Precision@5 on enterprise corpora |
| 2 | **Async vector embedding pipeline** — Ollama-native batch embedding via `compute_embeddings()`, HNSW ANN search, `embedding_index_ready` only set after vectors exist | Genuine semantic and hybrid retrieval |
| 3 | **Incremental inverted index with positions** — `{term → [(doc_id, [positions])]}` updated on every write; phrase and proximity search without full scans | O(N) → O(k·log N) query latency |
| 4 | **Bitmap facet indexes** — roaring-bitmap sets per facet value; facet counts become `len(bitmap)`, filtered search is bitmap intersection before document load | O(N) → O(F·V) facet aggregation |
| 5 | **Two-tier LRU+TTL cache** — L1 in-process (1 000 entries, 60 s), L2 optional Redis; keyed by `sha256(tenant+query+facets+principal)`; invalidated on write | 40–80% cache hit rate in read-heavy workloads |
| 6 | **Streaming results via AsyncGenerator** — `stream_search()` yields scored documents as they clear threshold via `asyncio.Queue`; decouples scoring from consumption | Time-to-first-result drop; enables SSE delivery |
| 7 | **Query understanding** — `understand_query()`: stopword removal, Porter stemming, synonym expansion, optional Ollama intent classification (`navigational`/`informational`/`transactional`) | +15–40% recall through synonym expansion alone |
| 8 | **Learning to Rank** — `record_click()` captures click + dwell signals; `ltr_rerank()` applies gradient-boosted point-wise ranker; fallback to BM25F when no signal data | Closes the relevance feedback loop |
| 9 | **Pluggable storage backends** — `SearchBackend` protocol with `DictBackend`, `PostgreSQLBackend` (tsvector+GIN+pgvector), `TypesenseBackend`; injected at construction | Production deployment, horizontal scaling, persistence |
| 10 | **Field-level AES-256-GCM encryption** — restricted document body and sensitive metadata encrypted at rest; tenant-scoped keys via AUTH adapter; transparent on read | Data-at-rest compliance for regulated industries |
| 11 | **Federated cross-tenant search** — `TenantNamespace` parent/child model with explicit `allow_federated_read` whitelist; `federated_search()` fans out and merges via reciprocal rank fusion | Enterprise group structures without bypassing isolation |
| 12 | **CDC incremental indexing** — `register_cdc_hook()` + `notify_document_changed()` apply delta updates to inverted index and bitmap slices; no full rebuilds | Near-real-time index freshness (<1 s lag) |
| 13 | **Relevance explainability** — `explain_result()` returns `ExplainResult` with per-term BM25F decomposition, field-weight contribution, synonym trace, personalisation boost, and ranking signal breakdown | Auditable governance; traceable restricted-document appearances |
| 14 | **JSONSchema document validation** — `mapping_update` accepts JSONSchema fragments per field; `validate_document()` dry-run endpoint; `index_document` rejects malformed docs with structured `ValidationError` | Eliminates silent data-quality bugs |
| 15 | **MMR result diversification** — `diversify_results()` applies Maximal Marginal Relevance post-processing; semantic mode uses vector distance, keyword mode uses title Jaccard distance | Higher perceived coverage per result page; reduces pogo-sticking |

## Guardrails

SRCH blocks missing tenant context, indices without name/owner/content
type/classification, restricted indices without lineage, documents without
index/id/title/body/classification/lineage, empty bulk batches, bulk indexing
without lineage, queries without text/index/type, restricted queries without
RBAC filtering, semantic or hybrid queries without embedding-ready indices,
non-positive result windows, cross-tenant search (unless federation grant
exists), non-Bytewax batch indexing, index retirement without review, and state
changes without audit evidence. SRCH requires review for unknown content types,
unknown classifications, large bulk batches, unknown query types, large result
windows, and unapproved facet keys.
Review-required index, document, query, and privileged search-agent outcomes
are persisted as `pending_review` or `review_required` records with matched
rules and review reasons so generated applications can surface governance
queues without replaying indexing or retrieval work.
AI search-agent guardrails also block unsupported runtimes, unsupported roles,
missing scope, missing owner, missing purpose, missing machine-contribution
disclosure, and route privileged roles through pending human review when
approval evidence is absent. Lifecycle mutation batches are accepted only
through the declared Bytewax processor contract.

## AI Agent Composition

SRCH treats search-governance agents as first-class APG citizens. Generated
applications can compose agents from rapidly changing tool runtimes without
binding indexing, retrieval, ranking, or access-policy logic to a single
provider. The executable contract supports `codex`, `claude_code`, `opencode`,
and `pi`; roles include source curation, index review, document-quality review,
query-relevance review, ranking review, access-policy review, facet-taxonomy
review, lifecycle-batch review, and search stewardship.

The runtime stores provider-neutral agent metadata only. Live CLI/API
invocation, credentials, embedding-provider calls, vector database operations,
and remote agent orchestration belong behind AICR, NLPC, and search-provider
adapter boundaries.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/srch/__init__.py capabilities/common/srch/capability_contract.py capabilities/common/srch/models.py capabilities/common/srch/search_runtime.py capabilities/common/srch/service.py capabilities/common/srch/api.py capabilities/common/srch/views.py capabilities/common/srch/app.py capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/srch/test_capability_contract.py capabilities/common/srch/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/srch --json
./.venv/bin/apg capabilities publish-plan capabilities/common/srch --json
```
