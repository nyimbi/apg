# GRAG Capability Specification Pointer

The active GRAG specification is `SPECIFICATION.md`.

The executable contract is `capability_contract.py`; the dependency-light generated-app runtime is `grag_runtime.py`; package evidence is emitted by `app.py`.

This packet defines graph sources, vector sources, hybrid retrieval, reasoning
paths, graph-grounded answers, curation, publication, first-class GraphRAG
agents, Bytewax lifecycle batches, deterministic guardrails, visual theming, and
generated Python package evidence.

Review-required lifecycle outcomes are durable capability data. Graph-source
retirement, low-confidence hybrid retrieval, reasoning from pending retrieval,
deep reasoning paths, low-confidence answers, answers generated from
pending-review graph context, and privileged GraphRAG-agent registrations are
persisted as `pending_review` records with policy decisions, matched rules,
review reasons, and audit evidence. Denied outcomes remain hard-blocking.

Use the old Apache AGE, Ollama, visualization, and production service modules
as adapter implementation references, not as the dependency-light generated-app
surface.
