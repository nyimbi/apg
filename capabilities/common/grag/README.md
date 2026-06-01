# GRAG - Graph-based RAG

GRAG is the APG capability for graph-grounded retrieval augmented generation.
It composes the document and answer workflow from RAGN with the graph
management surfaces from KNGR and GRPH so generated applications can retrieve
context from both vector indexes and knowledge-graph paths, then produce cited
answers with reasoning evidence. It also treats GraphRAG agents as first-class
citizens so Codex, Claude Code, opencode, Pi, and future runtimes can review
retrieval, reasoning, provenance, and generation behind the same APG guardrails.

The generated-app surface is dependency-light and executable. The heavier Apache AGE, Ollama, visualization, and service modules remain available as production adapters, but application composition should start from the contract, runtime, API helpers, and UI metadata in this directory.

## What GRAG Provides

- Governed graph-source registration with owner, graph id, classification, and provenance references.
- Governed vector-source registration with index id, embedding model, source document references, and owner.
- Hybrid retrieval records that bind a query to both graph and vector evidence.
- Multi-hop reasoning paths with start node, hop count, evidence path, explanation, and review gates.
- Graph-grounded answer generation with provenance refs, citations, model policy controls, confidence gates, and unsafe-answer blocking.
- Curation and publication lifecycle for approved answers.
- Durable review evidence for review-required graph-source retirement, hybrid
  retrieval, reasoning, answer, and GraphRAG-agent outcomes.
- First-class GraphRAG agents for provider-neutral runtime composition, scoped
  review roles, accountable ownership, machine-contribution disclosure, and
  privileged-role approval.
- Bytewax lifecycle batch validation for graph source, vector source, hybrid
  query, reasoning path, provenance, generation, curation, publication, and
  GraphRAG-agent operations.
- Deterministic rule evaluation for tenant isolation, evidence, provenance, Bytewax streaming, and audit readiness.
- UI route metadata and view models for dashboards, query console, source management, retrieval, reasoning, provenance, generation, curation, governance, audit, and settings.
- Theme tokens and component-level theme hooks for generated APG applications.

## How To Use It

Import the dependency-light runtime when composing generated Python applications:

```python
from capabilities.common.grag.grag_runtime import GragService

service = GragService()

graph = service.register_graph_source(
	"graph-policy",
	"tenant-a",
	"Policy graph",
	"knowledge-steward",
	"grph-policy",
	["source:policy-library"],
)
vector = service.register_vector_source(
	"vector-policy",
	"tenant-a",
	"idx-policy",
	"text-embedding-3-large",
	["doc-travel"],
	"knowledge-steward",
)
retrieval = service.run_hybrid_query(
	"query-travel",
	"tenant-a",
	"What approval is required for international travel?",
	graph["id"],
	vector["id"],
	retrieval_confidence=0.91,
)
path = service.build_reasoning_path(
	"path-travel",
	"tenant-a",
	retrieval["id"],
	"policy:travel",
	["policy:travel", "approval:manager", "approval:finance"],
	2,
	"Travel policy links international trips to manager and finance approval.",
)
answer = service.generate_answer(
	"answer-travel",
	"tenant-a",
	retrieval["id"],
	path["id"],
	"What approval is required for international travel?",
	"International travel requires manager and finance approval.",
	["source:policy-library", "path:path-travel"],
	[{"source_id": "policy-library", "document_id": "doc-travel", "chunk_id": "chunk-1"}],
)
curation = service.curate_answer(
	"curation-travel",
	"tenant-a",
	answer["id"],
	"knowledge-steward",
	"approved",
	"Reviewed against the policy graph and source document.",
)
publication = service.publish_answer(
	"publication-travel",
	"tenant-a",
	answer["id"],
	curation["id"],
	"knowledge-steward",
)
agent = service.register_grag_agent(
	"agent-reasoning",
	"tenant-a",
	"Reasoning reviewer",
	"codex",
	"reasoning_path_reviewer",
	"policy graph reasoning paths",
	"knowledge-steward",
	"Review multi-hop graph reasoning for grounded answers",
	human_approval_required=True,
)
batch = service.validate_grag_lifecycle_batch(
	"tenant-a",
	"bytewax",
	4,
	"graphrag_agent_batch",
)
```

Use `capabilities.common.grag.api` when a generated app wants simple function-style endpoints. Use `capability_contract.py` when the APG compiler or composition layer needs configuration, rules, routes, adapters, or theme tokens.

## Composition Contract

GRAG depends on:

- `ragn` for RAG concepts and answer composition.
- `kngr` for governed knowledge-graph ownership and provenance.
- `grph` for graph primitives and graph lifecycle composition.

Optional adapters include `srch`, `nlpc`, `aicr`, `onto`, `meta`, `auth`, `audl`, `cach`, and `moni`. Event streaming is explicitly configured for Bytewax.

## Guardrails

The contract exposes more than 45 deterministic rules. The runtime enforces the
important lifecycle rules directly, including tenant context, source
registration, hybrid retrieval readiness, restricted source filtering, low
confidence reviews, reasoning evidence, citations, provenance, external model
policy, unsafe answer blocking, curation evidence, publication approval,
Bytewax streaming, cross-tenant denial, audit evidence, supported GraphRAG
agent runtime and role, explicit agent scope, owner, purpose, machine
contribution disclosure, privileged-role human approval status, and Bytewax-only
lifecycle batches.

Review-required outcomes are persisted as `pending_review` records with
`decision`, `matched_rules`, `review_reasons`, and `audit_evidence`. True deny
outcomes still fail immediately.

## Files

- `SPECIFICATION.md` defines the capability behavior and integration boundaries.
- `PLAN.md` records the implementation plan for the lifecycle packet.
- `capability_contract.py` is the executable APG contract.
- `grag_runtime.py` is the dependency-light generated-app runtime.
- `api.py` exposes import-light API helper functions.
- `views.py` contains legacy model definitions plus generated-app UI metadata helpers.
- `app.py` exposes package metadata, semantic model generation, and self-test.
- `test_capability_contract.py` and `test_package_contract.py` provide focused verification.
