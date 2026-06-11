# SRCH – 15 World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Real BM25F Ranking with Field-Weighted Term Frequency

**Problem**: The current scorer is a raw hit-count over a concatenated `title+body` blob, which over-weights long documents and ignores that a match in the title is far more informative than one in the body.

**Fix**: Implement BM25F (Best Match 25 Fusion). Each field gets its own `k1` / `b` saturation parameter and a configurable field-weight (`title=3.0`, `body=1.0`, metadata fields as configured). The final score is a weighted sum of per-field BM25 sub-scores. This produces dramatically better ranking at no external dependency cost.

**Impact**: Precision@5 gains of 30–60% on typical enterprise corpora, especially for short queries.

---

## 2. Asynchronous Vector Embedding Pipeline (Ollama-Native)

**Problem**: `embedding_index_ready` is a boolean flag with no actual embedding computation behind it. Semantic and hybrid queries silently degrade to keyword-only.

**Fix**: Add `async def compute_embeddings(tenant_id, collection, model="nomic-embed-text")` that calls the locally-hosted Ollama embedding endpoint in batches, stores dense vectors per document, and marks `embedding_index_ready=True` only after vectors exist. HNSW approximate nearest-neighbour search replaces the placeholder.

**Impact**: Enables genuine semantic retrieval and hybrid search as originally specified.

---

## 3. Incremental Inverted Index with Positional Information

**Problem**: Every query scans all documents linearly — O(N) per query regardless of corpus size.

**Fix**: Maintain an in-memory inverted index keyed `{term → [(doc_id, [positions])}` updated on every `index_document` / `delete_document` call. Lookups become O(hits) rather than O(corpus). Positional lists enable exact phrase search and proximity scoring without full scans.

**Impact**: Query latency drops from O(N) to O(k·log N) where k is the result set size. Enables span-based phrase and proximity queries.

---

## 4. Faceted Aggregation via Bitmap Indexes

**Problem**: `facets()` iterates all documents to count facet values — another O(N) scan that blocks the event loop in high-volume deployments.

**Fix**: Maintain per-facet roaring-bitmap-style sets (`{collection → {facet_key → {facet_value → set[doc_pk]}}}`). Facet counts become `len(bitmap)` lookups; facet-filtered searches become bitmap intersections computed before document retrieval.

**Impact**: Facet aggregation goes from O(N) to O(F·V) where F = active facet keys, V = values per key. Intersection-first retrieval also eliminates irrelevant document loads.

---

## 5. Tiered Caching with Cache-Key Hashing (LRU + TTL)

**Problem**: The `BoundedCache` import exists but is never used for query results. Identical queries re-execute full scans.

**Fix**: Integrate a two-tier cache: L1 (in-process LRU, 1 000 entries, 60 s TTL) keyed by `sha256(tenant_id + canonical_query + facet_filters + rbac_principal)`. L2 can be Redis via an optional adapter. Cache invalidation fires on `index_document`, `delete_document`, and `reindex_collection` for the affected collection.

**Impact**: Cache hit rates of 40–80% are typical in read-heavy search workloads; eliminates redundant computation entirely.

---

## 6. Streaming Search Results via AsyncGenerator

**Problem**: `query()` and all extension search methods return a complete list, forcing full materialisation before the caller receives the first hit.

**Fix**: Add `async def stream_search(...)  -> AsyncIterator[dict]` that yields scored documents one at a time as they clear the relevance threshold, using `asyncio.Queue` to decouple scoring workers from result consumers.

**Impact**: Time-to-first-result drops dramatically for large result windows; enables reactive/server-sent-event delivery to browser clients.

---

## 7. Query Understanding: Intent Classification + Query Rewriting

**Problem**: Queries arrive as raw strings with no pre-processing — no stopword removal, no stemming, no acronym expansion, no intent detection.

**Fix**: Add `async def understand_query(query_text, tenant_id)` that applies: (a) stopword removal using a configurable list, (b) lightweight Porter stemming for English, (c) synonym expansion using `_synonyms` already stored per collection, (d) optional Ollama-based intent classification (`navigational` / `informational` / `transactional`) for routing to specialised retrievers.

**Impact**: Recall improvement of 15–40% through synonym expansion alone; intent classification enables query-type auto-detection.

---

## 8. Learning to Rank (LTR) via Click-Through Feedback

**Problem**: `personalised_search` applies a fixed 0.1-multiplier on query history — not a trained model.

**Fix**: Add `async def record_click(tenant_id, query_id, doc_id, position, dwell_ms)` to capture click and dwell-time signals. Add `async def ltr_rerank(tenant_id, results, query_text)` that applies a gradient-boosted point-wise ranker (trained offline via `lightgbm` or a simple logistic model on the collected signal log). Fallback to BM25F order if no signal data exists.

**Impact**: Directly closes the relevance feedback loop — the core mechanism that separates good search from great search.

---

## 9. Distributed Tenant Sharding with Pluggable Backends

**Problem**: All state lives in a single Python process dict. One heavy tenant starves all others, and there is no persistence across restarts.

**Fix**: Introduce a `SearchBackend` protocol with a `DictBackend` (current), a `PostgreSQLBackend` (using full-text `tsvector` + `gin` index for keyword, `pgvector` for semantic), and a `TypesenseBackend` adapter. `SrchService.__init__` accepts `backend: SearchBackend = DictBackend()`. The service layer delegates all storage I/O to the backend.

**Impact**: Enables production-grade deployment, horizontal scaling, and cross-process data sharing without changing the service API.

---

## 10. Field-Level Encryption for Restricted Documents

**Problem**: Documents classified `restricted` are stored in plaintext in the in-memory dict. Any process with access to the Python heap can read them.

**Fix**: Add `async def encrypt_field(value, classification, tenant_key)` / `decrypt_field(...)` using AES-256-GCM (via `cryptography` package). Restricted document `body` and configured sensitive metadata fields are encrypted at rest. Keys are tenant-scoped and sourced from the AUTH adapter boundary.

**Impact**: Satisfies data-at-rest encryption requirements for regulated industries without changing the retrieval API (decryption happens transparently on read).

---

## 11. Hierarchical Multi-Tenant Namespace + Cross-Tenant Federated Search

**Problem**: Cross-tenant search is explicitly blocked but there is no mechanism for legitimate federated scenarios (e.g., shared knowledge bases, parent/child tenant hierarchies).

**Fix**: Introduce a `TenantNamespace` model with `parent_tenant_id` and an explicit `allow_federated_read: list[str]` whitelist. Add `async def federated_search(calling_tenant_id, target_tenants, query_text, ...)` that validates the federation grant, fans out queries, and merges result lists with per-tenant score normalisation (reciprocal rank fusion).

**Impact**: Enables enterprise group structures (e.g., a holding company searching subsidiaries' knowledge bases) without bypassing tenant isolation.

---

## 12. Continuous Index Freshness via Change Data Capture Hooks

**Problem**: `reindex_collection` is a synchronous full-rebuild that takes O(N) time and blocks queries.

**Fix**: Add a CDC hook protocol: `register_cdc_hook(collection, callback: AsyncCallable)`. External systems (ETLP, META adapters) call `notify_document_changed(tenant_id, collection, doc_id, change_type)`. The service applies incremental delta indexing — updating only the affected inverted-index entries and bitmap slices — rather than a full rebuild.

**Impact**: Index freshness moves from batch-refresh cycles to near-real-time (<1 s lag) at a fraction of the CPU cost.

---

## 13. Pluggable Relevance Explainability (LIME / SHAP-style)

**Problem**: Scores are opaque integers. Users and governance reviewers cannot understand why a document ranked where it did.

**Fix**: Add `async def explain_result(tenant_id, query_id, doc_id)` that returns a structured `ExplainResult` containing: per-term contribution, field-weight contribution, synonym expansion trace, personalisation boost amount, and ranking-signal contributions. Implement as a BM25F score decomposition with a JSON explanation tree.

**Impact**: Directly supports SRCH's governance mandate — auditors can trace exactly why a restricted document appeared in a result set.

---

## 14. Schema-Enforced Document Validation with JSONSchema

**Problem**: `index_document` accepts arbitrary dicts for `facets` and `metadata`. Invalid or unexpected shapes silently pass through, leading to inconsistent facet aggregations and broken UI view models.

**Fix**: Extend `mapping_update` to accept JSONSchema fragments for each field. `index_document` runs `jsonschema.validate(doc, collection_schema)` before storage, rejecting malformed documents with structured `ValidationError` detail. Add `async def validate_document(tenant_id, collection, document)` as a dry-run validation endpoint.

**Impact**: Eliminates an entire class of silent data-quality bugs; enables self-service schema governance without code changes.

---

## 15. Adaptive Result Diversification (MMR / DPP)

**Problem**: Top-k results are often near-duplicates (high similarity to each other), wasting result-window slots and degrading perceived search quality.

**Fix**: Implement Maximal Marginal Relevance (MMR) post-processing in `async def diversify_results(results, query_vector, lambda_param=0.5)`. For semantic search, MMR balances relevance to the query against dissimilarity to already-selected results. For keyword-only mode, use title-level Jaccard distance as the diversity signal.

**Impact**: Users perceive significantly higher coverage per result page; reduces pogo-sticking and improves dwell time on result pages.
