# ONTO - Ontology Management

ONTO is the APG capability for governed ontologies, taxonomies, controlled vocabularies, semantic mappings, validation, publication, ontology exchange, first-class ontology agents, and Bytewax lifecycle batches. It gives generated applications an executable vocabulary workbench that can be composed with Knowledge Graph, Metadata, NLP, Search, Auth, Audit, AICR, Cache, Metrics, and Bytewax-backed event processing.

The generated-app surface is dependency-light. It uses in-process domain records for ontologies, namespaces, terms, taxonomy edges, mappings, reviews, validation reports, publications, exports, ontology agents, lifecycle batches, and audit events. Production systems can attach persistent stores, RDF/OWL/SKOS processors, graph stores, approval workflows, NLP assistants, external AI-agent runtimes, and durable Bytewax topologies through the configured adapters.

## What ONTO Provides

- Tenant-scoped ontology registry with owners, domains, versions, and publication state.
- Namespace governance for ontology prefixes and URIs.
- Controlled vocabulary terms with owners, definitions, statuses, synonyms, external references, and deprecation links.
- Taxonomy edge management with self-relation and cycle prevention.
- Semantic mappings to metadata, graph, search, external ontology, or application concepts.
- Review gates for duplicate terms, breaking changes, low-confidence mappings, external mappings, deprecations, imports, and validation issues.
- Durable pending-review records with policy evidence for review-required lifecycle outcomes.
- Validation reports for publication readiness.
- Publication lifecycle with approval, duplicate checks, draft-term checks, mapping-review checks, and version bumps.
- Export records for RDF, OWL, JSON-LD, SKOS, and CSV style interchange.
- W3C SKOS concept-scheme export with full `skos:broader` / `skos:narrower` / `skos:altLabel` mappings.
- Cross-ontology alignment with lexical and synonym-aware strategies and optional auto-mapping.
- Semantic version diffing — detects added/removed/deprecated terms and recommends patch/minor/major bump.
- Merkle-style audit-chain integrity verification with SHA-256 event hashing.
- LLM-assisted term-definition generation via local Ollama (dependency-free template fallback).
- Jaccard-similarity near-duplicate term detection (drop-in for embedding-based cosine similarity).
- OWL logical axiom assertion (subClassOf, equivalentClass, disjointWith, objectProperty, dataProperty).
- Ontology merge with additive / override / skip_existing strategies.
- Structural consistency checks: duplicate labels, taxonomy cycles, orphan terms.
- SPARQL-like term query with label filtering.
- Graph visualisation payload (nodes + edges) for rendering tooling.
- OWL reasoner simulation (EL/RL/QL/DL profiles).
- OWL/RDF XML import via label extraction.
- Turtle RDF serialisation export.
- Graph capability sync adapter (`sync_to_graph`) for pushing nodes and edges to the `grph` capability.
- Async bulk term creation for high-throughput vocabulary imports.
- Concept shorthand API (`concept_define`, `property_add`, `axiom_assert`).
- First-class ontology-agent registration for Codex, Claude Code, opencode, and Pi style assistants behind provider-neutral AICR adapter contracts.
- Bytewax lifecycle batch validation for ontology, namespace, term, taxonomy, mapping, validation, publication, exchange, and agent changes.
- UI route metadata, view models, theme tokens, and audit evidence for generated APG applications.

## How To Use It

```python
from capabilities.common.onto.service import OntoService

service = OntoService()

ontology = service.register_ontology(
	"customer-ontology",
	"tenant-a",
	"Customer Ontology",
	"data-stewards",
	"crm",
)
namespace = service.register_namespace(
	"ns-cust",
	"tenant-a",
	ontology["id"],
	"cust",
	"https://example.com/ontology/customer#",
	"data-stewards",
)
customer = service.create_term(
	"term-customer",
	"tenant-a",
	ontology["id"],
	"Customer",
	"data-stewards",
	"A party that purchases products or services.",
)
account = service.create_term(
	"term-account",
	"tenant-a",
	ontology["id"],
	"Account",
	"data-stewards",
	"A commercial relationship with a customer.",
)
service.add_synonym("tenant-a", customer["id"], "Client")
service.add_taxonomy_edge(
	"edge-customer-account",
	"tenant-a",
	ontology["id"],
	customer["id"],
	account["id"],
)
service.curate_term("review-customer", "tenant-a", customer["id"], "chief-steward")
service.curate_term("review-account", "tenant-a", account["id"], "chief-steward")
service.create_mapping(
	"map-customer-meta",
	"tenant-a",
	customer["id"],
	"meta:party.customer",
	"exact",
	0.96,
)

pending_mapping = service.create_mapping(
	"map-account-external-review",
	"tenant-a",
	account["id"],
	"external:sales.account",
	"close",
	0.62,
)
assert pending_mapping["status"] == "pending_review"
assert pending_mapping["decision"] == "require_review"
service.review_mapping(
	"review-account-external",
	"tenant-a",
	pending_mapping["id"],
	"chief-steward",
	"Approved after source-system review.",
)

service.validate_ontology("validation-customer", "tenant-a", ontology["id"])
publication = service.publish_ontology(
	"publication-customer",
	"tenant-a",
	ontology["id"],
	approval_recorded=True,
	approval_ref="approval:onto-42",
)
export = service.export_ontology("export-customer", "tenant-a", ontology["id"], "jsonld")

agent = service.register_ontology_agent(
	"agent-taxonomy-review",
	"tenant-a",
	"Taxonomy Reviewer",
	"codex",
	"taxonomy_reviewer",
	"customer ontology taxonomy",
	"data-stewards",
	"Review ontology hierarchy changes before publication",
	contribution_disclosed=True,
)
batch = service.validate_onto_lifecycle_batch(
	"tenant-a",
	"bytewax",
	4,
	"ontology_agent_batch",
	"ontobatch-customer-review",
)
```

Use `capability_contract.py` for compiler and composition metadata. Use `api.py` for generated endpoint-style helpers. Use `views.py` for dashboard, registry, namespace, term, taxonomy, mapping, validation, exchange, publication, governance, ontology-agent roster, lifecycle-batch monitor, audit, and settings view models.

## API Reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_ontology` | `(ontology_id, tenant_id, name, owner, domain, ...)` | Create a new ontology in the registry |
| `register_namespace` | `(namespace_id, tenant_id, ontology_id, prefix, uri, owner)` | Register a namespace prefix/URI pair |
| `create_term` | `(term_id, tenant_id, ontology_id, label, owner, ...)` | Create a controlled vocabulary term |
| `curate_term` | `(review_id, tenant_id, term_id, reviewer, ...)` | Approve/curate a term with review evidence |
| `add_synonym` | `(tenant_id, term_id, synonym)` | Add an altLabel synonym to a term |
| `add_taxonomy_edge` | `(edge_id, tenant_id, ontology_id, parent_term_id, child_term_id, ...)` | Assert a hierarchy relationship |
| `create_mapping` | `(mapping_id, tenant_id, term_id, target_ref, mapping_type, confidence, ...)` | Map a term to an external/internal concept |
| `review_mapping` | `(review_id, tenant_id, mapping_id, reviewer, ...)` | Approve a pending mapping |
| `deprecate_term` | `(review_id, tenant_id, term_id, replacement_term_id, reviewer, ...)` | Deprecate with mandatory replacement |
| `validate_ontology` | `(report_id, tenant_id, ontology_id, ...)` | Run publication-readiness validation |
| `publish_ontology` | `(publication_id, tenant_id, ontology_id, approval_recorded, ...)` | Publish with approval gate and version bump |
| `export_ontology` | `(export_id, tenant_id, ontology_id, export_format)` | Export as rdf/owl/jsonld/skos/csv |
| `register_ontology_agent` | `(agent_id, tenant_id, name, runtime, role, scope, owner, purpose, ...)` | Register an AI agent for ontology work |
| `validate_onto_lifecycle_batch` | `(tenant_id, event_stream, mutation_count, operation, ...)` | Validate a Bytewax lifecycle batch |
| `concept_define` | `(tenant_id, ontology_id, label, owner, definition, synonyms, ...)` | Shorthand: create term with stable auto-ID |
| `property_add` | `(tenant_id, term_id, property_name, property_value)` | Add/update a metadata property on a term |
| `axiom_assert` | `(tenant_id, ontology_id, axiom_id, axiom_type, subject_term_id, predicate, object_ref, asserted_by)` | Assert an OWL logical axiom |
| `ontology_merge` | `(tenant_id, source_ontology_id, target_ontology_id, merge_strategy, ...)` | Merge terms from source into target |
| `consistency_check` | `(tenant_id, ontology_id)` | Check for cycles, duplicates, orphan terms |
| `sparql_query` | `(tenant_id, ontology_id, query, actor)` | SPARQL-like term query with label filtering |
| `ontology_visualise` | `(tenant_id, ontology_id)` | Return nodes/edges for visualisation |
| `reasoner_run` | `(tenant_id, ontology_id, reasoner, actor)` | Run OWL reasoner (EL/RL/QL/DL) |
| `import_owl` | `(tenant_id, ontology_id, owl_content, imported_by)` | Import terms from OWL/RDF XML |
| `export_turtle` | `(tenant_id, ontology_id, exported_by)` | Export as Turtle RDF |
| `async find_similar_terms` | `(tenant_id, ontology_id, candidate_label, top_k, similarity_threshold)` | Near-duplicate detection via Jaccard similarity |
| `async align_ontologies` | `(tenant_id, source_ontology_id, target_ontology_id, strategy, confidence_cutoff, auto_create_above)` | Cross-ontology term alignment |
| `async compute_version_diff` | `(tenant_id, ontology_id)` | Structural diff with recommended semver bump |
| `async export_skos` | `(tenant_id, ontology_id, exported_by)` | Serialize as W3C SKOS Turtle |
| `async verify_audit_chain` | `(tenant_id)` | SHA-256 Merkle integrity check on audit log |
| `async suggest_definition` | `(tenant_id, ontology_id, label, synonyms, domain_hint, model)` | LLM-generated definition candidates via Ollama |
| `async sync_to_graph` | `(tenant_id, ontology_id, graph_adapter)` | Push nodes/edges to `grph` capability adapter |
| `async async_bulk_create_terms` | `(tenant_id, ontology_id, terms, owner, stop_on_error)` | Bulk term creation with per-item error reporting |
| `dashboard_summary` | `(tenant_id)` | Full tenant metrics across all record types |
| `ontology_package` | `(tenant_id)` | Complete export of all tenant ontology data |
| `list_pending_reviews` | `(tenant_id)` | All objects in `pending_review` status across all types |

## World-Class Enhancements (v2.0)

Fifteen architectural improvements are specified in `WORLD_CLASS_IMPROVEMENTS.md`. Implementation status varies; stubs are in-place for all 15 with full implementations for the subset below.

| # | Enhancement | Status |
|---|-------------|--------|
| 1 | **Full SPARQL 1.1 Query Engine** — `rdflib`-backed SELECT/CONSTRUCT/ASK/DESCRIBE against an in-memory ConjunctiveGraph | Stub (regex label filter) |
| 2 | **OWL/RDF Round-Trip Serialization** — `serialize_rdf` / `deserialize_rdf` preserving OWL 2 axioms through Turtle, N-Triples, JSON-LD, RDF/XML | Stub (Turtle delegated to `export_ontology`) |
| 3 | **Pluggable OWL Reasoner Backend** — abstract `ReasonerBackend` protocol; pure-Python EL transitive closure + `owlready2`/HermiT adapter | Stub (heuristic inferred count) |
| 4 | **Async Persistence Adapter** — `StorageBackend` protocol; PostgreSQL (asyncpg JSONB) and SQLite backends; write-through on mutations | Planned |
| 5 | **Change-Delta Event Streaming** — `OntologyChangeDelta` CloudEvents to Bytewax/Kafka/Redis; downstream `grph`/`srch`/`meta` subscribe | Planned |
| 6 | **Semantic Similarity-Guided Duplicate Detection** — Ollama `nomic-embed-text` cosine similarity; configurable threshold gates `create_term` | **Implemented** (`find_similar_terms` via Jaccard; Ollama adapter is drop-in) |
| 7 | **Ontology Alignment / Mapping Discovery** — pairwise embedding alignment; lexical + synonym strategies; auto-create above confidence cutoff | **Implemented** (`align_ontologies`) |
| 8 | **SKOS Hierarchy Import/Export** — W3C SKOS Turtle serialization mapping edges to `skos:broader/narrower/related` and synonyms to `skos:altLabel` | **Implemented** (`export_skos`) |
| 9 | **Governance Workflow Engine** — named-stage state machine (`submitted → under_review → approved/rejected → archived`); `NotificationAdapter` hook | Planned |
| 10 | **Multi-Tenancy Row-Level Security** — PostgreSQL RLS DDL + in-process `TenantIsolatedStore`; `IsolationViolationError` on cross-tenant access | Planned (Python-level filter active) |
| 11 | **Ontology Versioning with Semantic Diff** — reconstruct snapshot from audit events; auto-classify patch/minor/major by OWL change severity | **Implemented** (`compute_version_diff`) |
| 12 | **Full-Text and Vector Term Search** — `tantivy`/SQLite FTS5 inverted index + `faiss`/`annoy` ANN over term embeddings; ranked results | Planned |
| 13 | **Composition Hooks for grph / srch / nlpc** — `sync_to_graph`, `index_for_search`, `extract_nlp_entities` adapters; opt-in per capability contract | **Implemented** (`sync_to_graph`; others planned) |
| 14 | **Audit Trail Integrity Hashing** — SHA-256 Merkle chain with `prev_hash`/`event_hash` per event; HMAC-signed chain tip | **Implemented** (`verify_audit_chain`) |
| 15 | **LLM-Assisted Term Definition Generation** — Ollama (`mistral`/`llama3.2`) three-candidate definitions; provenance block with model + prompt hash | **Implemented** (`suggest_definition`) |

## New Methods

### Near-Duplicate Detection

```python
import asyncio

result = asyncio.run(service.find_similar_terms(
	"tenant-a",
	ontology["id"],
	"Buyer",          # candidate label to check before creating
	top_k=5,
	similarity_threshold=0.8,
))
# result["potential_duplicate_count"] > 0 → review before creating
# result["candidates"] → [{"label": "Customer", "similarity": 0.9, ...}, ...]
```

### Cross-Ontology Alignment

```python
# Align CRM ontology terms against Finance ontology, auto-create high-confidence mappings
result = asyncio.run(service.align_ontologies(
	"tenant-a",
	source_ontology_id="crm-ontology",
	target_ontology_id="finance-ontology",
	strategy="synonym",          # also considers altLabels
	confidence_cutoff=0.6,
	auto_create_above=0.9,       # auto-persist exact matches
))
# result["candidate_count"], result["auto_created_count"]
```

### SKOS Export

```python
result = asyncio.run(service.export_skos("tenant-a", ontology["id"], exported_by="curator"))
print(result["turtle"])
# @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
# onto:term-customer a skos:Concept ;
#     skos:prefLabel "Customer" ;
#     skos:altLabel "Client" ;
#     skos:inScheme onto:scheme .
```

### Audit Chain Verification

```python
# Call after any publication to verify tamper-evidence
result = asyncio.run(service.verify_audit_chain("tenant-a"))
assert result["chain_valid"]
# result["event_count"], result["chain_tip_hash"]
```

### LLM-Assisted Definition Generation

```python
result = asyncio.run(service.suggest_definition(
	"tenant-a",
	ontology["id"],
	label="Account",
	synonyms=["Client Account", "Customer Account"],
	domain_hint="CRM",
	model="llama3.2",            # local Ollama model
))
# result["candidates"] → list of 3 definition strings
# result["source"] → "ollama:llama3.2" or "template_fallback"
# result["provenance"] → {"prompt_hash": "...", "model": "llama3.2"}
```

### Ontology Merge

```python
result = service.ontology_merge(
	"tenant-a",
	source_ontology_id="crm-ontology",
	target_ontology_id="master-ontology",
	merge_strategy="skip_existing",  # also: "additive", "override"
	merged_by="data-steward",
)
# result["terms_imported"], result["terms_skipped"]
```

## Guardrails

ONTO exposes deterministic rules for tenant context, ontology identity, namespace uniqueness, term ownership, term status, duplicate review, deprecation replacement, synonym values, taxonomy integrity, mapping confidence, external mapping review, breaking change review, curation evidence, validation, publication readiness, import/export controls, first-class ontology-agent governance, Bytewax lifecycle batches, tenant isolation, and audit evidence.

Hard deny decisions raise `PermissionError`. Review-required decisions persist the attempted ontology object as `pending_review` with `decision`, `matched_rules`, `review_reasons`, and `audit_evidence`, so generated applications can render approval queues and human review workflows.

## Files

- `SPECIFICATION.md` defines the capability behavior and boundaries.
- `PLAN.md` records the packet implementation plan.
- `WORLD_CLASS_IMPROVEMENTS.md` specifies the 15 v2.0 enhancements.
- `capability_contract.py` is the executable APG contract.
- `models.py` contains dependency-light domain records.
- `ontology_runtime.py` contains deterministic helper functions.
- `service.py` executes the ontology lifecycle.
- `api.py` exposes payload helper functions.
- `views.py` exposes generated-app UI models.
- `app.py` emits semantic package evidence and self-test results.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/onto/__init__.py \
  capabilities/common/onto/capability_contract.py \
  capabilities/common/onto/models.py \
  capabilities/common/onto/ontology_runtime.py \
  capabilities/common/onto/service.py \
  capabilities/common/onto/api.py \
  capabilities/common/onto/views.py \
  capabilities/common/onto/app.py \
  capabilities/common/onto/test_capability_contract.py \
  capabilities/common/onto/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/onto/test_capability_contract.py \
  capabilities/common/onto/tests/test_package_contract.py

./.venv/bin/python -c "from capabilities.common.onto import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/onto --json
./.venv/bin/apg capabilities publish-plan capabilities/common/onto --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/onto --json
```
