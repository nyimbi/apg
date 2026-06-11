# RAGN World-Class Improvements

15 concrete improvements to elevate the RAG Engine to production-grade, research-quality standing.

---

## 1. Adaptive Chunking with Semantic Boundaries

Current fixed-size word-split chunking severs sentences mid-thought. Replace with a
sentence-boundary-aware segmenter (spaCy / NLTK `sent_tokenize`) that respects
paragraph, heading, and list structure. Add a `SemanticChunkStrategy` enum
(`fixed`, `sentence`, `paragraph`, `topic`) configurable per knowledge base so
ingestion quality is tunable without schema changes.

## 2. Hierarchical Index (Parent-Child Chunks)

Store a parent chunk (512 tokens) and child chunks (128 tokens) derived from it.
Retrieval uses child chunks for precision; context assembly fetches the full parent
for coherence. This mirrors the LlamaIndex `ParentDocumentRetriever` pattern and
eliminates the precision-vs-context size tension without sacrificing either.

## 3. Real Cross-Encoder Re-Ranking

The current `rerank_results` uses Jaccard overlap. Swap for a locally hosted
cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2` via Ollama or a local FastAPI
inference server) that scores (query, chunk) pairs jointly. Add
`CrossEncoderRerankConfig` with `model`, `batch_size`, and `score_threshold` fields
and wire it as an optional post-retrieval step in `query_knowledge_base`.

## 4. Hypothetical Document Embeddings (HyDE)

Before embedding the user query, generate a hypothetical answer with a small LLM
(`qwen3:1b`), embed that answer instead of the raw query, and use the resulting
vector for nearest-neighbour lookup. Typically yields 5-15% retrieval recall
improvement on open-domain QA benchmarks at low latency cost.

## 5. Late-Interaction ColBERT Embeddings

Supplement dense single-vector embeddings (bge-m3) with ColBERT-style multi-vector
token embeddings for MaxSim scoring. Add a `VectorStrategy` enum
(`dense`, `sparse`, `colbert`, `hybrid`) and route to the correct index at query
time. ColBERT scores significantly higher on out-of-domain retrieval without
fine-tuning.

## 6. Persistent Disk-Backed Vector Index (hnswlib / Milvus Lite)

`_chunks` and `_embeddings` are pure in-memory dicts that evaporate on restart.
Integrate `hnswlib` as a zero-dependency local vector store with
`save_index` / `load_index` on every mutation. For multi-node deployments, add a
`MilvusLiteVectorBackend` shim behind the existing `VectorService` interface so the
upgrade is transparent to callers.

## 7. Streaming Response Generation

`answer_generate` returns a complete string. Add `answer_generate_stream` returning
an `AsyncGenerator[str, None]` that yields tokens as they arrive from Ollama's
streaming endpoint (`/api/generate` with `stream=True`). Wire a
`StreamingConversationResponse` view model so the Flask-AppBuilder UI can
server-sent-events the response to the browser.

## 8. RAG Fusion Multi-Query Retrieval

Instead of a single query vector, generate N query variants (already scaffolded in
`query_expand`), retrieve top-k chunks per variant in parallel, then merge with
Reciprocal Rank Fusion (RRF). RRF is parameter-free and consistently outperforms
single-query retrieval by 8-12% on BEIR benchmarks. Implement as
`rag_fusion_retrieve(query, kb_id, n_variants, top_k)`.

## 9. Answer Attribution Heat-Map

Extend `citation_extract` to produce a character-span alignment between each
sentence in the generated answer and its source chunk, stored as
`(chunk_id, start_char, end_char, confidence)` tuples. This enables the UI to
highlight exactly which source sentence supports each answer sentence — a strong
trust signal for enterprise compliance use cases.

## 10. Continuous RAGAS-Style Auto-Evaluation

After every `answer_generate` call, asynchronously run a background eval task that
scores faithfulness, answer relevance, and context precision using the RAGAS metric
definitions (currently partially implemented in `rag_evaluate`). Accumulate scores
in a rolling time-series and expose `get_quality_trend(kb_id, window_hours)` for
automated quality regression alerting.

## 11. Knowledge Graph Triple Extraction

After chunking, run a lightweight NER + relation-extraction pass (spaCy +
`rebel-large` or a local Ollama prompt) to extract `(subject, predicate, object)`
triples per chunk. Store as `KGTriple` records linked to chunks. During retrieval,
optionally augment vector search results with graph traversal (entity
neighbourhood expansion) for multi-hop questions that pure vector search misses.

## 12. Incremental Re-Indexing via Change Data Capture

Current `document_refresh` drops all chunks and re-chunks the entire document.
Implement a content-addressed diff: hash each paragraph, identify added/removed
paragraphs, and update only the affected chunks + their embeddings. For large
policy libraries (thousands of pages) this reduces re-indexing time from minutes to
seconds.

## 13. Role-Scoped Retrieval Filters

Add a `RetrievalFilter` model with `allowed_classifications`, `required_tags`, and
`excluded_document_ids` fields. Enforce filters at the vector-search level (pre-
filter by metadata before ANN search) rather than post-filtering results. This
guarantees that a retrieval call for a `PUBLIC` user never touches `CONFIDENTIAL`
chunks even if cosine similarity would rank them first.

## 14. Offline-First Bulk Ingestion Pipeline

Add `bulk_ingest(documents: list[BulkIngestItem]) -> AsyncGenerator[IngestProgress, None]`
that streams progress events (Bytewax-compatible `CloudEvent` payloads) as each
document completes. Supports checkpoint/resume on failure so a 10,000-document
corpus load survives network interruptions without duplicating already-indexed
content.

## 15. Observability via OpenTelemetry Spans

Wrap every public async method with an OTel span using `opentelemetry-sdk` (already
a transitive dep in most Python stacks). Emit `rag.retrieval.latency_ms`,
`rag.generation.latency_ms`, `rag.chunk.count`, and `rag.cache.hit` metrics as OTel
attributes. This plugs natively into Grafana / Prometheus without bespoke
monitoring code and gives SREs sub-operation visibility during incidents.
