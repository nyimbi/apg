# Knowledge Graph (KNGR) — World-Class Improvement Proposals

### I1. Persistent Graph Storage via PostgreSQL
**Category**: Infrastructure
**Justification**: In-memory dicts disappear on restart. Production graphs (Netflix, LinkedIn) persist billions of nodes to disk with ACID guarantees; APG's current design loses all state on service restart, making it non-production by default.
**Implementation**: Replace `self._entities` etc. with async SQLAlchemy sessions backed by PostgreSQL. Use `asyncpg` + connection pooling. Each store method becomes an `INSERT … ON CONFLICT DO UPDATE` (upsert). `list_*` methods become paginated `SELECT` with cursor-based pagination.
**Competitor**: Neo4j AuraDB, AWS Neptune, TigerGraph — all persist graphs durably with transactional writes.

### I2. Vector-Similarity Semantic Search
**Category**: Query / Intelligence
**Justification**: Substring search (`q in haystack`) misses synonyms, misspellings, and cross-lingual matches. OpenAI / Weaviate ship embedding-powered search that returns semantically related results with sub-50ms p99 latency at billion-entity scale.
**Implementation**: Add `async def semantic_search(query, tenant_id, top_k, threshold)`. Embed entity canonical labels via a locally-hosted Ollama `nomic-embed-text` model. Store vectors in `pgvector`. At query time compute cosine similarity via `<=>` operator. Cache hot vectors with `BoundedCache`.
**Competitor**: Weaviate, Qdrant, Milvus — all provide HNSW-indexed ANN search over entity embeddings.

### I3. Async-First Service Layer
**Category**: Performance / Architecture
**Justification**: Every public method is synchronous, forcing callers to wrap with `asyncio.run()` or `run_in_executor`. A synchronous service serialises I/O: at 1000 concurrent tenants each DB call blocks the thread pool. FastAPI / Starlette natively `await` async service methods.
**Implementation**: Convert all public methods to `async def`. Replace synchronous dict lookups with `await db.get(...)`. Use `asyncio.gather()` in `list_knowledge_graph` to fan-out sub-queries concurrently. Add `async with AsyncSession()` context manager per request.
**Competitor**: LangGraph, Haystack 2.0 — both ship async-native pipelines.

### I4. Transitive Inference Engine
**Category**: Reasoning
**Justification**: Current reasoning is purely explicit (you supply `relationship_ids`). Real KGs derive implicit facts: if A `is_a` B and B `is_a` C then A `is_a` C. Without inference, analysts must manually chain every relationship, which doesn't scale past a few hundred nodes.
**Implementation**: Add `async def apply_inference_rules(tenant_id)`. Load registered inference rules (`inference_rule` records). For each rule pattern, iterate matching relationship pairs and materialise implied relationships with `confidence_score = min(r1.confidence, r2.confidence) * 0.9` decay. Store as `inferred:True` flag on `KnowledgeRelationship`.
**Competitor**: Stardog (SPARQL-based inference), GraphDB (OWL reasoner).

### I5. Multi-Hop SPARQL-Subset Query API
**Category**: Query Language
**Justification**: Consumers currently interact via imperative Python calls. Every ad-hoc analysis requires new code. Graph databases expose a declarative query language (SPARQL, Cypher, GQL) so analysts compose queries without programming.
**Implementation**: Add `async def sparql_query(sparql_text, tenant_id)`. Parse a restricted SPARQL SELECT + WHERE with subject/predicate/object patterns via `rdflib` or a hand-rolled recursive-descent parser for the `SELECT ?x WHERE { ?x <pred> ?y . ?y <pred2> ?z }` subset. Return binding dicts. Use `entity_search` + BFS for variable resolution.
**Competitor**: Stardog, Virtuoso, GraphDB — all ship SPARQL 1.1 engines.

### I6. Confidence Decay Over Time
**Category**: Knowledge Quality
**Justification**: A fact with 0.95 confidence recorded two years ago may now be stale. Without temporal decay, the graph treats a decade-old procurement relationship the same as one recorded today. Financial KGs at Bloomberg/Refinitiv implement half-life decay models.
**Implementation**: Add `async def refresh_confidence(tenant_id, half_life_days)`. For each entity/relationship, compute `new_conf = original_conf * 0.5 ** (age_days / half_life_days)`. Entities below `min_threshold` transition to `pending_review`. Decay parameters stored per-source in `KnowledgeSource.attributes`.
**Competitor**: Refinitiv entity resolution, Bloomberg BLAW — both track fact freshness.

### I7. Change-Data-Capture Event Streaming
**Category**: Integration / Architecture
**Justification**: Currently callers must poll `list_audit_events`. Downstream systems (search indexers, alert engines, data warehouses) need to react to graph mutations in real time. CDC is the industry standard for zero-lag data integration.
**Implementation**: Add `async def stream_mutations(tenant_id, since_cursor)` as an async generator yielding `KngrAuditEvent` dicts. Backed by a PostgreSQL `LISTEN/NOTIFY` channel per tenant. Integrate with Bytewax source operator via the existing `validate_kngr_lifecycle_batch` lifecycle contract.
**Competitor**: Debezium + Kafka CDC for Neo4j, Amazon Neptune Streams, TigerGraph CDC.

### I8. Role-Based Access Control per Entity Type
**Category**: Security / Governance
**Justification**: Current tenant isolation is binary (same `tenant_id` = full access). Enterprise graphs require column/row-level security: a procurement analyst can read `purchase_request` entities but not `personnel` entities. Missing RBAC leaks PII across business units.
**Implementation**: Add `async def check_entity_permission(actor_roles, entity_type, operation, tenant_id) -> bool`. Store per-entity-type ACL in `capability_contract`. Thread `actor_roles` through every read/write path before the existing `evaluate()` rule check. Return 403 on deny rather than `PermissionError`.
**Competitor**: Apache Ranger, Neo4j RBAC, AWS Lake Formation column-level security.

### I9. Ontology-Driven Entity Type Validation
**Category**: Data Quality
**Justification**: `entity_type` is a free-form string today, so `"PurchaseRequest"`, `"purchase_request"`, and `"PR"` coexist as different types, fragmenting the graph. Wikidata and schema.org enforce registered type vocabularies.
**Implementation**: Add `async def validate_entity_type(entity_type, tenant_id) -> bool`. Load allowed types from the tenant's registered ontology sources (`connector == "ontology"`). Reject `resolve_entity` calls with unregistered types unless `strict_ontology=False`. Cache type sets with `BoundedCache` keyed by `(tenant_id, ontology_id)`.
**Competitor**: Wikidata (Q-item type hierarchy), schema.org (type lattice), FIBO ontology for finance.

### I10. Incremental Subgraph Embedding for GraphRAG
**Category**: AI Integration
**Justification**: LLM-powered retrieval-augmented generation (GraphRAG) needs entity/relationship embeddings that stay current as the graph evolves. Batch re-embedding every night is too slow for operational KGs that mutate hundreds of times per hour.
**Implementation**: Add `async def embed_subgraph(root_id, tenant_id, depth, model)`. Calls Ollama `nomic-embed-text` on each entity's concatenated canonical_label + attributes JSON. Stores vectors in `pgvector`. Maintains a dirty-flag queue so only mutated subgraphs are re-embedded on each call.
**Competitor**: Microsoft GraphRAG (incremental graph index), Diffbot KG, Amazon Neptune ML.

### I11. Distributed Graph Sharding for Multi-Tenant Scale
**Category**: Scalability
**Justification**: A single Python dict holding all tenant entities is an O(N) bottleneck. At 10M entities across 500 tenants, list operations scan every record. TigerGraph shards by vertex partition; Dgraph shards by predicate.
**Implementation**: Add `async def shard_graph(tenant_id, shard_count)`. Assign entities to shards by `hash(entity_id) % shard_count`. Store shard assignments in PostgreSQL. Route reads/writes to the correct shard via an async connection pool with per-shard connection strings. Cross-shard queries use `asyncio.gather()` fan-out then merge.
**Competitor**: TigerGraph (multi-machine sharding), Dgraph (Raft-replicated shards), JanusGraph (Cassandra backend).

### I12. Graph Versioning and Temporal Snapshots
**Category**: Audit / Compliance
**Justification**: Regulatory compliance (SOX, GDPR Article 17) requires the ability to reconstruct what the graph looked like at any past point in time. Current `GraphPublication` is a one-shot snapshot with no delta history.
**Implementation**: Add `async def graph_snapshot(tenant_id, label)` and `async def graph_at(tenant_id, timestamp)`. Each mutation writes a delta record `(entity_id, op, payload, timestamp)` to a `kngr_deltas` table. `graph_at` replays deltas up to the target timestamp using event sourcing. Snapshots compress the full state using `orjson` + `zstd`.
**Competitor**: Dolt (Git for databases), Terminus DB (immutable graph with time travel), Dgraph (multi-version concurrency).

### I13. Uncertainty-Aware Probabilistic Reasoning
**Category**: Reasoning / AI
**Justification**: Binary confidence scores ignore epistemic uncertainty (we don't know what we don't know). Probabilistic KGs (ProbLog, BayesDB) attach probability distributions to facts and propagate uncertainty through inference chains, producing calibrated confidence intervals rather than point estimates.
**Implementation**: Add `async def probabilistic_infer(query, tenant_id, samples)`. Model relationship confidence as `Beta(alpha, beta)` distributions. Use Monte Carlo sampling (`samples` draws per relationship) to propagate uncertainty through reasoning paths. Return `{"mean": float, "std": float, "ci_95": [lo, hi]}` alongside the path result.
**Competitor**: ProbLog (probabilistic logic), BayesDB (probabilistic DB), Markov Logic Networks.

### I14. Automated Entity Deduplication via Blocking + Matching
**Category**: Data Quality / ML
**Justification**: `conflict_detect` only finds exact canonical_label matches. Real-world entity duplication involves abbreviations, transliterations, and data entry errors. LinkedIn's entity resolution team estimates 30-40% of enterprise KGs contain duplicates without automated dedup.
**Implementation**: Add `async def auto_deduplicate(tenant_id, threshold, dry_run)`. Use blocking keys (`entity_type` + first 3 chars of label). Within each block, compute Jaro-Winkler similarity on canonical_label and alias overlap. Pairs above `threshold` are presented as candidate merges. With `dry_run=False`, auto-merge via existing `graph_merge`. Emit `dedup_candidate` audit events.
**Competitor**: Zingg (ML-powered dedup), Tamr (human-in-the-loop entity resolution), Dedupe.io.

### I15. Federated Knowledge Graph Query Across Tenants
**Category**: Enterprise Composition
**Justification**: Cross-business-unit analysis (e.g. shared supplier risk across procurement + finance tenants) requires federated queries that respect per-tenant access controls. Today `tenant_id` isolation is absolute — cross-tenant queries are impossible even when explicitly authorised.
**Implementation**: Add `async def federated_query(query, tenant_ids, actor_roles)`. Authenticate actor has `kngr:federate` permission on each listed tenant. Fan-out the query via `asyncio.gather()` across tenant service shards. Merge results using entity deduplication (I14) to collapse cross-tenant duplicates. Emit per-tenant audit events for the federated access. Return a merged result set with per-entity `source_tenant` annotation.
**Competitor**: AWS Lake Formation cross-account queries, Stardog Virtual Graphs (federated SPARQL), Linked Data Fragments.
