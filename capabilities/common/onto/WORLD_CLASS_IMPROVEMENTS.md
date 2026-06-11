# ONTO - World-Class Improvement Proposals

**Capability**: Ontology Management (`onto`)
**Author**: Nyimbi Odero
**Date**: 2026-06-11
**Version**: 1.0.0

---

## 1. Full SPARQL 1.1 Query Engine

**Current**: `sparql_query()` is a regex stub that parses only `rdfs:label` filters from WHERE clauses.

**Improvement**: Integrate `rdflib` as an optional dependency and execute real SPARQL 1.1 SELECT/CONSTRUCT/ASK/DESCRIBE queries against an in-memory `rdflib.ConjunctiveGraph`. Serialize the registered ontology terms/edges/mappings into RDF triples on demand and run the query through `rdflib`'s native SPARQL processor. Return SPARQL-standard variable bindings in the result set.

**Impact**: Enables downstream agents to issue standards-compliant queries — essential for interop with ontology libraries like BioPortal, Wikidata, and Schema.org.

---

## 2. OWL/RDF Round-Trip Serialization

**Current**: `import_owl()` uses regex label extraction; `export_turtle()` delegates to a stub.

**Improvement**: Implement `async serialize_rdf()` and `async deserialize_rdf()` backed by `rdflib`. Support full OWL 2 functional syntax, Turtle, N-Triples, JSON-LD, and RDF/XML. Preserve axiom annotations, data-property restrictions, and class expressions through round-trips.

**Impact**: Zero information loss when exchanging ontologies with Protégé, TopBraid Composer, and W3C-compliant triple stores.

---

## 3. Pluggable OWL Reasoner Backend

**Current**: `reasoner_run()` returns a hard-coded `inferred = len(edges) // 2` estimate.

**Improvement**: Define an abstract `ReasonerBackend` protocol. Ship two concrete implementations: (a) a pure-Python transitive-closure reasoner covering OWL EL profile and (b) an adapter to `owlready2`/`HermiT` via subprocess when available. Cache inferred axioms in `self._inferred_axioms` and invalidate on ontology mutation.

**Impact**: Applications can detect unsatisfiable classes, compute subclass hierarchies, and check disjointness constraints — the minimum bar for production knowledge-management systems.

---

## 4. Async Persistence Adapter

**Current**: All state lives in in-process dicts; no persistence boundary.

**Improvement**: Define an async `StorageBackend` protocol with `get / put / delete / query` methods. Ship a `PostgreSQLBackend` using `asyncpg` (JSONB column per domain record type) and an `SQLiteBackend` for local dev. `OntoService.__init__` accepts an optional `backend: StorageBackend` and lazily loads / write-through on mutations. All public methods become `async`.

**Impact**: Production deployments survive restarts; enables multi-process / multi-pod deployments without data loss.

---

## 5. Change-Delta Event Streaming

**Current**: `_audit()` only writes to the internal `_audit_events` dict.

**Improvement**: Add `async emit_delta()` that publishes structured `OntologyChangeDelta` CloudEvents (JSON-CE) to a configurable transport (Bytewax topic, Kafka, Redis Streams, or local asyncio queue). Deltas carry old/new values, actor, timestamp, and capability provenance. Downstream capabilities (grph, srch, meta) subscribe to apply incremental updates.

**Impact**: Eliminates polling; graph indexes, search indexes, and metadata stores stay consistent without full re-sync.

---

## 6. Semantic Similarity-Guided Duplicate Detection

**Current**: Duplicate detection uses `normalize_label()` equality — purely lexical.

**Improvement**: Add `async find_similar_terms()` that computes cosine similarity between candidate term embeddings (via a locally hosted Ollama embedding model, e.g. `nomic-embed-text`) and existing terms. Return ranked candidates with similarity scores. Gate `create_term()` with a configurable `similarity_threshold` that triggers `require_review` when a near-duplicate is found.

**Impact**: Catches semantic duplicates invisible to string normalization (e.g., "Client" vs "Customer" vs "Buyer") before they pollute the vocabulary.

---

## 7. Ontology Alignment / Mapping Discovery

**Current**: Mappings are manually asserted; no automated alignment.

**Improvement**: Add `async align_ontologies()` that takes two ontology IDs, computes pairwise term embeddings from the Ollama service, and returns candidate `SemanticMapping` records ranked by confidence. Optionally auto-create mappings above a high-confidence threshold. Support Lexical, Structural, and Semantic alignment strategies.

**Impact**: Slashes the manual effort of cross-ontology integration from weeks to minutes.

---

## 8. SKOS Hierarchy Import/Export

**Current**: Export supports only opaque format strings; no SKOS semantics.

**Improvement**: Implement `async export_skos()` that serializes the ontology as W3C SKOS using `rdflib`, mapping APG taxonomy edges to `skos:broader` / `skos:narrower` / `skos:related`, terms to `skos:Concept`, and synonyms to `skos:altLabel`. Add `async import_skos()` as the inverse. Emit a SKOS `ConceptScheme` root node automatically.

**Impact**: Interoperability with controlled-vocabulary systems (Zotero, Dublin Core, library catalogs) without bespoke adapters.

---

## 9. Governance Workflow Engine

**Current**: Pending reviews are recorded but there is no state-machine driving them to resolution.

**Improvement**: Add a `WorkflowEngine` with named stages (`submitted → under_review → approved | rejected → archived`). Each stage transition fires an `_audit()` event and optionally notifies registered reviewers via a pluggable `NotificationAdapter`. `OntoService` delegates all status transitions through the engine.

**Impact**: Replaces ad-hoc status strings with a deterministic, auditable machine that satisfies ISO 19989 governance traceability requirements.

---

## 10. Multi-Tenancy Isolation via Row-Level Security

**Current**: Tenant isolation is enforced by Python-level dict filters in `_list()` and `_require_*()`.

**Improvement**: When using the PostgreSQL backend, emit `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` DDL and enforce tenant policies at the database layer. In-process mode adds a `TenantContext` context-var that is validated on every dict access through a `TenantIsolatedStore` wrapper, raising `IsolationViolationError` on cross-tenant access.

**Impact**: Defense-in-depth; cross-tenant leakage becomes impossible at the storage layer, not just the application layer.

---

## 11. Ontology Versioning with Semantic Diff

**Current**: `publish_ontology()` calls `bump_patch_version()` with no structural diff.

**Improvement**: Add `async compute_version_diff()` that compares two published versions by reconstructing their snapshot from audit events. Return added/removed/changed terms, edges, and mappings. Automatically classify the version bump as patch / minor / major using OWL change-severity rules (breaking = axiom removal or narrowing; non-breaking = additions).

**Impact**: Consumers of the ontology can detect breaking changes programmatically and trigger downstream re-validation workflows automatically.

---

## 12. Full-Text and Vector Term Search

**Current**: `sparql_query()` returns all terms or filters by exact label substring — O(n) scan.

**Improvement**: Maintain a `tantivy`-backed (or `sqlite FTS5`) inverted index for fast full-text term search. Additionally, build a `faiss` / `annoy` approximate nearest-neighbor index over term embeddings for semantic search. Expose `async search_terms()` accepting a free-text or embedding query and returning ranked results with snippet highlights.

**Impact**: UX for ontology curators goes from exact-match lookup to Google-quality retrieval at O(log n) cost.

---

## 13. Composition Hooks for grph / srch / nlpc

**Current**: Composability is documented but no active adapter wiring exists.

**Improvement**: Add `async sync_to_graph()` (emits term nodes and edges to the `grph` capability), `async index_for_search()` (pushes terms to the `srch` capability's indexer), and `async extract_nlp_entities()` (sends definitions to `nlpc` for entity extraction, enriching `external_refs` automatically). Each adapter is opt-in via a capability contract flag.

**Impact**: A single `publish_ontology()` call atomically propagates the vocabulary to graph, search, and NLP pipelines — the correct default behavior for a production data platform.

---

## 14. Audit Trail Integrity Hashing

**Current**: Audit events are plain dict records with no tamper evidence.

**Improvement**: Add Merkle-style chaining: each `OntoAuditEvent` stores a `prev_hash` (SHA-256 of the previous event's serialization) and its own `event_hash`. Add `async verify_audit_chain()` that recomputes and validates the chain from genesis, returning the first broken link if any. Sign the chain tip with an HMAC key stored in tenant configuration.

**Impact**: Meets regulatory requirements (GDPR audit trails, SOC 2 Type II, ISO 27001) for tamper-evident change logs without an external ledger.

---

## 15. LLM-Assisted Term Definition Generation

**Current**: Term definitions are free-text fields with no quality assistance.

**Improvement**: Add `async suggest_definition()` that accepts a term label, synonyms, and domain context and calls the local Ollama LLM (e.g., `mistral`, `llama3.2`) via the APG `aicr` adapter. Return three candidate definitions ranked by precision and coverage. The curation workflow can accept one verbatim or edit before committing. Log the LLM model and prompt version in `metadata` for provenance.

**Impact**: Dramatically reduces the cognitive load on ontology curators; consistent, high-quality definitions from day one instead of placeholder stubs.
