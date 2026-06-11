# APG RAG Engine — User Guide

> Enterprise Retrieval-Augmented Generation for the APG platform.
> © 2025 Datacraft · nyimbi@gmail.com · www.datacraft.co.ke

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Knowledge Base Management](#knowledge-base-management)
4. [Document Ingestion](#document-ingestion)
5. [Retrieval Strategies](#retrieval-strategies)
6. [Answer Generation](#answer-generation)
7. [Conversations](#conversations)
8. [Advanced Retrieval Features](#advanced-retrieval-features)
9. [Evaluation and Feedback](#evaluation-and-feedback)
10. [Cache Management](#cache-management)
11. [Administration and Security](#administration-and-security)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The RAGN capability provides tenant-scoped retrieval-augmented generation:
documents are ingested into knowledge bases, chunked and embedded into a vector
index, and then retrieved at query time to ground LLM-generated answers in verifiable
enterprise context.

**Key properties**

- All operations are async and tenant-isolated.
- Local Ollama-served models handle embedding and generation — no external API keys.
- A layered retrieval stack (vector search → re-rank → RAG Fusion → HyDE) is
  configurable per knowledge base.
- Feedback, evaluation, and quality-trend APIs support continuous improvement.

---

## Core Concepts

| Concept | Description |
|---|---|
| **Knowledge Base** | Named container for related documents, with its own embedding model, chunk settings, and similarity threshold. |
| **Document** | An ingested file (PDF, DOCX, TXT, HTML, Markdown, CSV, JSON). |
| **Chunk** | An overlapping text segment extracted from a document, stored with its embedding vector. |
| **Embedding** | A dense float vector representing the semantic content of a chunk. |
| **Retrieval** | Finding the most relevant chunks for a query, using one or more strategies. |
| **Context** | The concatenated text of retrieved chunks fed to the LLM. |
| **Answer** | LLM-generated response grounded in the retrieved context. |
| **Citation** | Sentence-level link between an answer sentence and its source chunk. |
| **Conversation** | A stateful multi-turn dialogue over a knowledge base. |

---

## Knowledge Base Management

### Create a Knowledge Base

```python
from capabilities.common.ragn.service import RAGService, RAGServiceConfig

config = RAGServiceConfig(tenant_id="acme", capability_id="rag")
service = RAGService(config, db_pool=None, ollama_integration=None)

kb = await service.create_knowledge_base(KnowledgeBaseCreate(
    name="HR Policies",
    description="All HR policy documents",
    embedding_model="bge-m3",
    generation_model="qwen3",
    chunk_size=512,
    chunk_overlap=64,
    similarity_threshold=0.7,
    max_retrievals=10,
    user_id="alice",
))
print(kb.id)  # UUID7
```

### Retrieve and List

```python
kb = await service.get_knowledge_base(kb.id)
all_kbs = await service.list_knowledge_bases(user_id="alice", limit=20)
```

### Chunk Size Guidelines

| Content Type | Recommended Chunk Size | Overlap |
|---|---|---|
| FAQs / short policies | 256–512 tokens | 32–64 |
| General narrative text | 512–1024 tokens | 64–128 |
| Long legal or technical docs | 1024–2048 tokens | 128–256 |

---

## Document Ingestion

### Add and Process a Document

```python
with open("travel_policy.pdf", "rb") as f:
    content = f.read()

doc = await service.add_document(
    kb_id=kb.id,
    document_create=DocumentCreate(
        title="Travel Policy 2025",
        filename="travel_policy.pdf",
        file_type="pdf",
        content_hash="sha256:...",
        metadata={"classification": "internal", "version": "2.1"},
        user_id="alice",
    ),
    content=content,
    process_immediately=True,
)
```

When `process_immediately=True` the pipeline runs:
1. Content extraction (DocumentProcessor)
2. Chunking (chunk_document)
3. Embedding (embed_chunk per chunk)
4. Vector index ingestion (VectorService.index_chunks)

### Manual Chunking

```python
result = await service.chunk_document(
    document_id=doc.id,
    text="Full extracted text here...",
    chunk_size=512,
    chunk_overlap=64,
    metadata={"source": "travel_policy.pdf"},
)
print(result["chunk_count"])
```

### Bulk Embedding

After chunking, embed all chunks concurrently:

```python
summary = await service.bulk_chunk_embed(
    chunk_ids=result["chunk_ids"],
    model="bge-m3",
    concurrency=8,  # limits concurrent Ollama calls
)
print(summary["succeeded"], "chunks embedded")
```

### Refresh an Existing Document

**Full re-process** (drops all chunks, re-chunks from scratch):

```python
log = await service.document_refresh(doc.id, new_content=open("travel_policy_v2.pdf","rb").read())
```

**Incremental re-index** (only changed paragraphs re-processed — significantly faster for large docs):

```python
log = await service.incremental_reindex(doc.id, new_content=new_bytes)
print(log["unchanged"], "paragraphs unchanged,", log["added"], "added,", log["removed"], "removed")
```

---

## Retrieval Strategies

RAGN supports four retrieval modes, from basic to advanced.

### 1. Standard Similarity Search

```python
result = await service.similarity_search(
    query="What approval is required for international travel?",
    kb_id=kb.id,
    top_k=5,
    threshold=0.5,
)
```

### 2. Role-Scoped Filtered Search

Enforce access control at retrieval time (not post-filter):

```python
result = await service.retrieval_filter_search(
    query="Budget limits for equipment purchase",
    kb_id=kb.id,
    allowed_classifications=["internal", "public"],
    required_tags=["finance"],
    excluded_document_ids=["doc-draft-123"],
    top_k=5,
)
```

This guarantees `CONFIDENTIAL` chunks are never scored or returned for users
lacking that classification, even if they are the most cosine-similar.

### 3. RAG Fusion (Recommended for complex queries)

Generates N query variants, retrieves per-variant, then merges with Reciprocal Rank
Fusion (RRF):

```python
result = await service.rag_fusion_retrieve(
    query="How does the travel policy handle meal allowances?",
    kb_id=kb.id,
    n_variants=3,
    top_k=5,
)
# result["fusion_method"] == "rrf"
# result["results"] sorted by rrf_score
```

RRF is parameter-free and typically improves recall by 8-12% over single-query
retrieval on enterprise document corpora.

### 4. HyDE (Hypothetical Document Embeddings)

Generates a synthetic answer first, then uses that richer text as the retrieval probe:

```python
result = await service.hyde_query(
    query="What is the company's policy on work-from-home equipment?",
    kb_id=kb.id,
    top_k=5,
)
print(result["hypothetical_answer"])  # the synthetic probe text
```

### Multi-Hop Retrieval

For questions requiring chained reasoning across multiple documents:

```python
result = await service.multi_hop_query(
    hop_id="hop-001",
    initial_query="What is the impact of the travel policy on project budgets?",
    kb_id=kb.id,
    hops=2,
    top_k=3,
)
for hop in result["hop_results"]:
    print(f"Hop {hop['hop']}: {hop['query']}")
```

### Re-Ranking

After any retrieval, re-rank chunk IDs by cross-encoder score:

```python
ranked = await service.rerank_results(
    rerank_id="rr-001",
    query="international travel approval",
    chunk_ids=[r["id"] for r in result["results"]],
)
```

### Query Expansion

Generate alternative phrasings to broaden recall:

```python
expansion = await service.query_expand(
    expansion_id="exp-001",
    query="expense reimbursement process",
    strategy="rephrase",
    n_variants=3,
)
print(expansion["variants"])
```

---

## Answer Generation

### Build Context from Chunks

```python
ctx = await service.context_build(
    query="international travel approval",
    chunk_ids=[r["id"] for r in ranked["ranked"][:5]],
    max_tokens=2048,
)
```

### Generate an Answer

```python
answer = await service.answer_generate(
    answer_id="ans-001",
    query="What approval is required for international travel?",
    context=ctx["context"],
    model="qwen3",
    max_tokens=512,
)
print(answer["answer"])
```

### End-to-End RAG (single call)

```python
response = await service.generate_response(
    kb_id=kb.id,
    query_text="What approval is required for international travel?",
    conversation_id=None,
    generation_model="qwen3",
)
```

---

## Citations

### Extract Citations

```python
cites = await service.citation_extract(
    citation_id="cite-001",
    answer_text=answer["answer"],
    chunk_ids=ctx["included_chunk_ids"],
)
for c in cites["citations"]:
    print(c["chunk_id"], c["matched_sentences"])
```

### Character-Span Attribution (UI Heat-Map)

```python
attribution = await service.chunk_attribution_map(
    answer_text=answer["answer"],
    chunk_ids=ctx["included_chunk_ids"],
)
for a in attribution["alignments"]:
    print(f"'{a['sentence']}' ← {a['chunk_id']} (confidence {a['confidence']})")
```

### Verify a Claim Against a Chunk

```python
verification = await service.source_verify(
    verification_id="ver-001",
    chunk_id="doc-001:chunk:0",
    claim="International travel requires manager approval",
)
print(verification["supported"], verification["support_score"])
```

---

## Conversations

```python
conv = await service.create_conversation(
    kb_id=kb.id,
    conversation_create=ConversationCreate(
        title="Travel policy Q&A",
        generation_model="qwen3",
        temperature=0.7,
        user_id="alice",
    ),
)

response = await service.chat(
    conversation_id=conv.id,
    user_message="What approval is needed for a trip to London?",
    user_context={"role": "engineer", "department": "R&D"},
)
```

---

## Advanced Retrieval Features

### RAG Fusion — When to Use It

| Scenario | Recommended Strategy |
|---|---|
| Short, precise factual lookup | `similarity_search` |
| Complex or ambiguous questions | `rag_fusion_retrieve` |
| Out-of-vocabulary queries | `hyde_query` |
| Multi-document reasoning | `multi_hop_query` |
| Strict access-control required | `retrieval_filter_search` |

### Combining Strategies

```python
# 1. Expand the query
expansion = await service.query_expand("exp-002", query, n_variants=3)

# 2. Fuse retrieval across all variants
fused = await service.rag_fusion_retrieve(query, kb.id, n_variants=3, top_k=8)

# 3. Re-rank
ranked = await service.rerank_results("rr-002", query, [r["id"] for r in fused["results"]])

# 4. Build context and generate
ctx = await service.context_build(query, [r["chunk_id"] for r in ranked["ranked"][:5]])
answer = await service.answer_generate("ans-002", query, ctx["context"])
```

---

## Evaluation and Feedback

### Evaluate an Answer

```python
eval_rec = await service.rag_evaluate(
    eval_id="eval-001",
    query="What approval is required for international travel?",
    answer=answer["answer"],
    ground_truth="Manager and finance approval are required for international travel.",
    retrieved_chunk_ids=ctx["included_chunk_ids"],
)
print(eval_rec["faithfulness"], eval_rec["answer_relevance"], eval_rec["answer_correctness"])
```

Metrics:
- **faithfulness**: answer text supported by retrieved chunks.
- **answer_relevance**: answer aligns with the question.
- **answer_correctness**: answer matches ground truth.

### Quality Trend

```python
trend = await service.quality_trend(kb_id=kb.id, window_hours=24)
print(trend["avg_correctness"])  # last 24 h rolling average
```

### Collect Feedback

```python
feedback = await service.feedback_incorporate(
    feedback_id="fb-001",
    query="What approval is required for international travel?",
    answer_id=answer["id"],
    rating=5,
    comment="Accurate and well-cited.",
    user_id="alice",
)
```

Ratings: 1 (poor) to 5 (excellent).

### Feedback Summary

```python
summary = await service.get_feedback_summary(kb_id=kb.id)
print(summary["avg_rating"], summary["nps_score"])
print(summary["distribution"])   # {"1": 0, "2": 1, "3": 3, "4": 10, "5": 24}
print(summary["sentiment"])      # {"positive": 34, "neutral": 3, "negative": 1}
```

---

## Cache Management

```python
# Cache a result for 5 minutes
await service.cache_query(query, kb.id, result=fused, ttl_seconds=300)

# Look up — returns None if expired
cached = await service.cache_lookup(kb.id, query)

# Invalidate a single query
await service.cache_invalidate(kb.id, query)

# Invalidate all queries for a KB
await service.cache_invalidate(kb.id)
```

---

## Administration and Security

### Health Check

```python
health = await service.health_check()
# health["database_connection"], health["components_healthy"], health["components"]
```

### Service Status

```python
status = await service.service_status()
print(status["chunks_in_memory"], status["cache_entries"], status["feedback_count"])
```

### Analytics

```python
analytics = await service.rag_analytics(tenant_id="acme")
print(analytics["average_feedback_rating"], analytics["average_answer_correctness"])
```

### Role-Based Access Control

Apply `retrieval_filter_search` with `allowed_classifications` derived from the
user's role to enforce data access policy at query time:

| User Role | allowed_classifications |
|---|---|
| Public | `["public"]` |
| Staff | `["public", "internal"]` |
| Manager | `["public", "internal", "restricted"]` |
| Executive | `["public", "internal", "restricted", "confidential"]` |

---

## Troubleshooting

### No Results Returned

1. Lower `threshold` (try 0.3 for development, 0.5-0.6 for production).
2. Verify chunks exist: `await service.list_chunks(document_id=doc.id)`.
3. Confirm embeddings were generated: `await service.bulk_chunk_embed(chunk_ids, model="bge-m3")`.
4. Use `rag_fusion_retrieve` to broaden recall across query variants.

### Slow Retrieval

1. Reduce `top_k`.
2. Pre-filter with `retrieval_filter_search` to reduce the candidate set.
3. Use `cache_query` for repeated queries; `cache_lookup` returns in O(1).
4. Check Ollama server load — `bulk_chunk_embed` has a `concurrency` cap to avoid overload.

### Poor Answer Quality

1. Inspect `rag_evaluate` metrics — low faithfulness indicates context mismatch.
2. Monitor `quality_trend` for regression over time.
3. Switch retrieval strategy to `rag_fusion_retrieve` or `hyde_query`.
4. Increase `chunk_size` for complex contextual queries.
5. Collect user feedback with `feedback_incorporate` and review `get_feedback_summary`.

### Stale Chunks After Document Update

- For small edits use `incremental_reindex` — only changed paragraphs are re-processed.
- For complete rewrites use `document_refresh`.
- Invalidate the query cache after re-indexing: `await service.cache_invalidate(kb_id)`.

### Access Control Violations

- Always use `retrieval_filter_search` with explicit `allowed_classifications` rather
  than post-filtering.
- Never rely on similarity score to suppress confidential content.
- Add `excluded_document_ids` to block draft or embargoed documents.

---

*For API reference see `api.py`. For architectural details see `docs/architecture.md`.
For operational runbooks see `docs/operations_manual.md`.*
