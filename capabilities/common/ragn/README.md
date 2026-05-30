# Retrieval-Augmented Generation (RAGN)

RAGN provides APG's executable retrieval-augmented generation capability:
tenant-scoped knowledge bases, document ingestion, governed context retrieval,
cited answer generation, conversation memory, answer curation, audit evidence,
UI view models, and package metadata for generated Python applications.

Use RAGN when an application needs answers grounded in enterprise context:
support assistants, ERP copilots, policy Q&A, operational runbooks, research
workbenches, compliance evidence navigation, or AI-agent context packs. The
dependency-light runtime is intentionally small, while the contract exposes
adapter seams for search, NLP, model lifecycle, AI core, knowledge graph,
metadata, auth, audit, cache, metrics, and Bytewax event streams.

## What RAGN Provides

- Knowledge-base records with tenant, owner, source attribution, classification,
  review policy, and audit events.
- Document ingestion records with knowledge-base linkage, source URI, content
  hash, classification, large-ingest review gates, and audit events.
- Retrieval records with query, knowledge base, document evidence, context
  confidence, result window, restricted-source filtering, and review gates.
- Answer generation records with retrieved context, citations, model policy,
  prompt-injection and unsafe-answer guardrails, and audit events.
- Conversation-turn records with user identity, conversation id, answer linkage,
  turn-count review, and audit events.
- Citation validation for source, document, and chunk evidence.
- Answer curation records with curator, decision, evidence, and audit events.
- UI route metadata and view models for dashboard, studio, knowledge bases,
  documents, retrieval, generation, conversations, citations, curation,
  governance, audit, and settings screens.
- Bytewax adapter evidence for batch RAG mutation flows.

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
7. Inspect dashboard summaries, evidence trails, governance rules, and audit
   events.

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
```

## Guardrails

RAGN denies operations without tenant context, knowledge-base identity, owner,
source attribution, document title, content hash, source URI, valid
classification, retrieval query, retrieval knowledge base, restricted-source
access filters, generation query, retrieved context, citations, external model
policy, conversation id, user id, citation evidence, curator, curation decision,
curation evidence, Bytewax batch mutation, tenant isolation, or audit evidence
for state changes. It requires review for large ingestion batches, large
retrieval windows, low-confidence context, and long conversations.

## Composition

RAGN depends on SRCH, NLPC, and AICR. Optional adapters connect it to MLCM,
AUTH, AUDL, CACH, KNGR, GRPH, META, MONI, and Bytewax-backed event streams.
Generated applications compose RAGN through its semantic model, UI manifest,
API helpers, service runtime, rule engine, and theme contract.
