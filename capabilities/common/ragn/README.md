# Retrieval-Augmented Generation (RAGN)

RAGN provides APG's executable retrieval-augmented generation capability:
tenant-scoped knowledge bases, document ingestion, governed context retrieval,
cited answer generation, conversation memory, answer curation, audit evidence,
first-class RAG agents, Bytewax lifecycle batches, UI view models, and package
metadata for generated Python applications.

Use RAGN when an application needs answers grounded in enterprise context:
support assistants, ERP copilots, policy Q&A, operational runbooks, research
workbenches, compliance evidence navigation, or AI-agent context packs. The
dependency-light runtime is intentionally small, while the contract exposes
adapter seams for search, NLP, model lifecycle, AI core, knowledge graph,
metadata, auth, audit, cache, metrics, provider-neutral agent runtimes, and
Bytewax event streams.

## What RAGN Provides

- Knowledge-base records with tenant, owner, source attribution, classification,
  review policy, and audit events.
- Document ingestion records with knowledge-base linkage, source URI, content
  hash, classification, large-ingest review gates, and audit events.
- Retrieval records with query, knowledge base, document evidence, context
  confidence, result window, restricted-source filtering, and review gates.
- Answer generation records with retrieved context, citations, model policy,
  prompt-injection and unsafe-answer guardrails, pending-context review gates,
  and audit events.
- Conversation-turn records with user identity, conversation id, answer linkage,
  turn-count review, and audit events.
- Citation validation for source, document, and chunk evidence.
- Answer curation records with curator, decision, evidence, and audit events.
- First-class RAG agents for Codex, Claude Code, opencode, and Pi style
  runtimes with explicit role, owner, scope, purpose, contribution disclosure,
  and human approval status for privileged roles.
- Bytewax lifecycle batch validation for corpus, document, retrieval, context,
  generation, citation, evaluation, safety, and RAG-agent operations.
- UI route metadata and view models for dashboard, studio, knowledge bases,
  documents, retrieval, generation, conversations, citations, curation,
  governance, agent roster, lifecycle batch monitor, audit, and settings
  screens.
- Durable review evidence on review-required outcomes: `pending_review`
  status, `decision`, `matched_rules`, `review_reasons`, and
  `audit_evidence`. True deny outcomes still fail immediately.

## Runtime Surfaces

- `capability_contract.py` defines configuration, deterministic rules, UI
  routes, adapters, and theme tokens.
- `rag_runtime.py` is the generated-app runtime used by tests, APIs, and package
  probes.
- `api.py` exposes dependency-light API helper functions.
- `views.py` exposes generated-app view models.
- `app.py` exposes the package semantic model and self-test.
- `service.py` remains the production adapter surface for deployments that wire
  the heavier async database and model-runtime dependencies.

## Lifecycle

1. Create a knowledge base with owner and source-attribution policy.
2. Ingest classified documents into the knowledge base.
3. Retrieve context for a user query with confidence and access filtering.
4. Generate an answer from retrieved context with citations and model policy.
5. Record conversation turns when the answer is part of a dialogue.
6. Curate generated answers when confidence, safety, or business impact requires
   review.
7. Register provider-neutral RAG agents with bounded roles and scopes.
8. Validate RAG lifecycle batches through Bytewax-first processor contracts.
9. Inspect dashboard summaries, evidence trails, governance rules, agent review
   queues, lifecycle batches, pending review queues, and audit events.

## Example

```python
from capabilities.common.ragn.rag_runtime import RagnService

service = RagnService()
kb = service.create_knowledge_base(
    knowledge_base_id="kb-policy",
    tenant_id="tenant-a",
    name="Policy knowledge base",
    owner="knowledge-steward",
    source_attribution="policy-library",
)
doc = service.ingest_document(
    document_id="doc-travel-policy",
    tenant_id="tenant-a",
    knowledge_base_id=kb["id"],
    title="Travel policy",
    source_uri="meta://policies/travel",
    content_hash="sha256:travel",
    classification="internal",
)
retrieval = service.retrieve_context(
    retrieval_id="ret-travel",
    tenant_id="tenant-a",
    knowledge_base_id=kb["id"],
    query="What approval is required for international travel?",
    document_ids=[doc["id"]],
    context_confidence=0.91,
)
answer = service.generate_answer(
    answer_id="ans-travel",
    tenant_id="tenant-a",
    retrieval_id=retrieval["id"],
    query="What approval is required for international travel?",
    answer_text="International travel requires manager and finance approval.",
    citations=[{"source_id": "policy-library", "document_id": doc["id"], "chunk_id": "chunk-1"}],
)
agent = service.register_rag_agent(
    agent_id="agent-grounding",
    tenant_id="tenant-a",
    name="Grounding reviewer",
    runtime="codex",
    role="grounding_reviewer",
    scope="kb-policy answers",
    owner="knowledge-steward",
    purpose="Review generated answers for grounded evidence",
    human_approval_required=True,
)
batch = service.validate_ragn_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=3,
    operation="rag_agent_batch",
)
```

## Guardrails

RAGN denies operations without tenant context, knowledge-base identity, owner,
source attribution, document title, content hash, source URI, valid
classification, retrieval query, retrieval knowledge base, restricted-source
access filters, generation query, retrieved context, citations, external model
policy, conversation id, user id, citation evidence, curator, curation decision,
curation evidence, Bytewax batch mutation, tenant isolation, or audit evidence
for state changes. It persists pending-review records, rather than discarding
the operation as a transient exception, for large ingestion batches, large
retrieval windows, low-confidence context, answers generated from pending
context, long conversations, and privileged RAG-agent registrations without
recorded human approval. It rejects unsupported agent runtimes, unsupported
agent roles, missing agent scope, missing owner, missing purpose, missing
machine-contribution disclosure, non-Bytewax lifecycle streams, and unsupported
lifecycle operations.

## Composition

RAGN depends on SRCH, NLPC, AICR, CONF, and AUDL. Optional adapters connect it
to MLCM, AUTH, CACH, KNGR, GRPH, META, MONI, and durable Bytewax topologies.
Generated applications compose RAGN through its semantic model, UI manifest,
agent manifest, streaming manifest, API helpers, service runtime, rule engine,
and theme contract.

---

## World-Class Enhancements (v2.0)

- **I1.** RAGN World-Class Improvements
- **I2.** Adaptive Chunking with Semantic Boundaries
- **I3.** Hierarchical Index (Parent-Child Chunks)
- **I4.** Real Cross-Encoder Re-Ranking
- **I5.** Hypothetical Document Embeddings (HyDE)
- **I6.** Late-Interaction ColBERT Embeddings
- **I7.** Persistent Disk-Backed Vector Index (hnswlib / Milvus Lite)
- **I8.** Streaming Response Generation
- **I9.** RAG Fusion Multi-Query Retrieval
- **I10.** Answer Attribution Heat-Map
- **I11.** Continuous RAGAS-Style Auto-Evaluation
- **I12.** Knowledge Graph Triple Extraction
- **I13.** Incremental Re-Indexing via Change Data Capture
- **I14.** Role-Scoped Retrieval Filters
- **I15.** Offline-First Bulk Ingestion Pipeline

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
