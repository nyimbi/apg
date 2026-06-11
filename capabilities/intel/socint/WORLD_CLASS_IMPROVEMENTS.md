# Social Intelligence (intel_socint) — World-Class Improvement Roadmap

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke

---

## Overview

The following 15 improvements elevate `intel_socint` from a strong prototype to a
production-grade SOCINT platform competitive with commercial tools like Brandwatch,
Recorded Future Social, and Babel Street.

---

## 1. Streaming Ingestion via Bytewax + Kafka

**Current state**: Collection is synchronous and simulated via hash-derived counts.

**Improvement**: Replace `collect_posts` and `bulk_post_collection` with a Bytewax
dataflow that consumes from a Kafka topic (`apg.intel.socint.raw`), applies sliding-window
deduplication by `content_fingerprint`, and emits to `apg.intel.socint.processed`. Each
message carries AVRO schema with platform, handle, timestamp, and SHA-256 fingerprint —
no raw PII crosses the boundary. Throughput target: 50 k posts/sec per partition.

**Impact**: Eliminates polling latency, enables real-time alerting, integrates with APG
event bus without code changes at call sites.

---

## 2. LLM-Backed Semantic Sentiment (Ollama/Mistral)

**Current state**: Keyword lexicon scoring with 14 positive + 14 negative tokens.

**Improvement**: Replace the word-counting loop in `sentiment_analysis_batch` with async
calls to a locally-hosted `mistral:7b-instruct` model via the Ollama REST API. Prompt
template elicits structured JSON: `{sentiment, score, reasoning, entities}`. Cache results
in a tenant-scoped Redis set keyed by `content_fingerprint` (TTL 24h). Batch 64 posts per
LLM call to amortise overhead. Fallback to lexicon scorer when Ollama is unavailable.

**Impact**: Sentiment accuracy improves from ~60% (lexicon) to ~88% (instruction-tuned
7B). Named-entity extraction falls out for free, enabling influence attribution without a
separate NER pipeline.

---

## 3. Graph Database for Influence Networks (Neo4j or Memgraph)

**Current state**: Synthetic ego-network built from hash arithmetic; stored in a flat
dict.

**Improvement**: Persist nodes and edges to a Neo4j (or Memgraph) instance. Model:
`(:Account)-[:FOLLOWS|RETWEETS|MENTIONS]->(:Account)` with edge weight = normalised
engagement. Expose `async graph_shortest_path(source, target)` and
`async community_detection(algorithm="louvain")` methods that call the Cypher query
layer. Use Memgraph's in-memory mode for sub-10ms traversals on graphs up to 10M edges.

**Impact**: Enables genuine betweenness centrality, community partitioning, and
attribution paths — currently infeasible with the dict-based adjacency model.

---

## 4. Temporal Anomaly Detection on Posting Cadence

**Current state**: `persona_analysis` checks only static thresholds (posts/day > 50 →
bot indicator).

**Improvement**: Add `async cadence_anomaly_detection(handle, days=30)` that builds an
hourly posting histogram, fits a Poisson baseline, and applies a CUSUM control chart to
flag sub-hourly bursts. Integrate the signal into `threat_actor_social_profile` as
`cadence_anomaly_score`. Use `numpy`/`scipy.stats` (already available in the APG venv) —
no external service dependency.

**Impact**: Catches scripted bots that post at fixed intervals (low stddev) and burst
amplifiers (CUSUM spike) that pass the simple threshold test.

---

## 5. Multi-Language NLP via LangDetect + Ollama

**Current state**: `disinformation_detection` processes content as if it is English.

**Improvement**: Add `async detect_language(content)` using `langdetect` (fast, local).
When non-English content is detected, invoke an Ollama translation prompt before running
disinformation checks, and attach `detected_language` and `translated_content` to the
result. Expose `async multilingual_sentiment(content, target_lang="en")` as a first-class
method.

**Impact**: Enables monitoring of Arabic, Swahili, French, and Russian-language
narratives — critical for East African and pan-African threat landscapes.

---

## 6. Persistent Storage with SQLAlchemy Async + Alembic

**Current state**: All state is held in Python dicts; lost on process restart.

**Improvement**: Replace every `dict[tuple, Model]` with `AsyncSession` queries against
the PostgreSQL tables defined in `database/schema.sql`. Wire into the existing Alembic
migration chain (`alembic/versions/`). Use `async with session.begin()` context managers
to ensure atomicity. Expose `store` as a `DatabaseStore` collaborator injected at
construction, keeping the service layer testable without a live DB.

**Impact**: Durability, multi-process horizontal scaling, and audit log persistence
across restarts — required for any production deployment.

---

## 7. Content-Similarity Clustering (MinHash LSH)

**Current state**: Deduplication in `bulk_post_collection` is exact fingerprint match
only.

**Improvement**: Add `async cluster_similar_content(post_ids, threshold=0.8)` using
`datasketch` MinHash LSH. Build a `MinHashLSHForest` from post content fingerprints,
query for all pairs above the Jaccard threshold, and return cluster labels. Use this to
group near-duplicate posts (slightly rephrased propaganda, astroturfing templates) before
sentiment aggregation to avoid over-counting.

**Impact**: Detects coordinated inauthentic behaviour where accounts post paraphrased
copies of the same narrative — missed by exact-fingerprint dedup.

---

## 8. STIX 2.1 / MISP Export

**Current state**: `export_intelligence` produces a minimal JSON summary with only
signal/post counts.

**Improvement**: Implement `async export_stix(bundle_id)` that serialises
`SocialSignal`, `InfluenceAssessment`, and `NetworkAssessment` objects as STIX 2.1
`Indicator`, `ThreatActor`, and `Relationship` objects using the `stix2` Python library.
Also implement `async export_misp_event(event_id)` targeting MISP feed format. Include
TLP markings derived from the originating `SocialAuthority.classification`.

**Impact**: Enables downstream sharing with CERT/CC, national CERTs, and commercial
threat feeds without manual transformation.

---

## 9. Real-Time Alerting via Webhooks / ntfy

**Current state**: Audit events are appended to a list; no external notification.

**Improvement**: Wire the injected `notify` collaborator to emit webhook payloads (JSON
POST) or ntfy.sh push notifications when: (a) `viral_content_alert` fires, (b)
`cib_detected = True`, (c) `state_sponsored_suspected = True`, or (d)
`radicalization_rate > 0.2`. Add `async configure_alert_channel(channel_type, endpoint,
secret)` for runtime webhook registration. Use `httpx.AsyncClient` for non-blocking HTTP
delivery with exponential-backoff retry (max 3 attempts).

**Impact**: Converts the service from a polling analytics backend into an event-driven
alerting system — the key operational difference for crisis monitoring use cases.

---

## 10. Influence Decay Modelling

**Current state**: `influence_network_map` computes a static `influence_score` that
never changes.

**Improvement**: Add `async compute_influence_decay(handle, half_life_days=30)` that
applies an exponential decay function `score(t) = score_0 * e^(-λt)` where λ =
ln(2)/half_life. Track last-activity timestamp per handle. Return `current_score`,
`peak_score`, `decay_rate`, and `days_since_active`. Integrate into
`threat_actor_social_profile` as `decayed_influence_score`.

**Impact**: Prevents stale threat actor profiles from over-weighting dormant accounts;
focuses analyst attention on active, rising influencers.

---

## 11. Structured Capability Metrics / Prometheus Export

**Current state**: `health_check` returns a static dict; no time-series observability.

**Improvement**: Instrument key paths with `prometheus_client` counters and histograms:
`socint_posts_collected_total{platform, tenant}`,
`socint_sentiment_latency_seconds{model}`,
`socint_disinfo_score_histogram{tenant}`. Expose `async metrics_snapshot()` returning
the current Prometheus text format, and wire a `/metrics` route into the Flask-AppBuilder
blueprint. Add `async export_telemetry(backend)` for OpenTelemetry trace export.

**Impact**: Enables SRE dashboards (Grafana), SLA enforcement, and anomaly detection on
operational throughput without bespoke monitoring code.

---

## 12. Operator-Controlled PII Minimisation

**Current state**: `collect_posts` stores `content_fingerprint` but nothing prevents
callers from attaching raw handles or content to audit events.

**Improvement**: Enforce a `PiiPolicy` Pydantic model injected at construction:
`{store_handles: bool, store_content: bool, retention_days: int}`. Gate every write path
through `_pii_guard(data, policy)` which strips or hashes fields based on policy. Add
`async apply_retention_policy()` that deletes records older than `retention_days` from
all in-memory stores (and the DB store when wired). Log every deletion to the audit trail
with reason `RETENTION_POLICY_APPLIED`.

**Impact**: GDPR/PDPA compliance without application-level changes by consumers of the
capability.

---

## 13. Federated Multi-Tenant Query

**Current state**: All methods are scoped to `self.tenant_id`; cross-tenant aggregation
requires instantiating multiple service objects.

**Improvement**: Add `async federated_query(tenant_ids, method_name, **kwargs)` that
fans out the named method call across tenant service instances (from an injected registry)
using `asyncio.gather`, merges results, and returns a `FederatedResult` with per-tenant
breakdown and merged aggregate. Guard with an explicit `FEDERATED_QUERY` authority type.

**Impact**: Enables multi-organisation threat intelligence sharing (e.g., ISAC use cases)
where a lead analyst needs a unified view across participant tenants.

---

## 14. Explainable AI Scoring Audit Trail

**Current state**: `bot_probability`, `disinfo_score`, and `cib_probability` are numeric
scores with no derivation trace.

**Improvement**: Attach a `scoring_trace: list[ScoringStep]` to every analytical result
where each step carries `{rule_name, weight, value, contribution}`. Implement
`async explain_score(result_id, score_field)` that retrieves the trace and returns a
human-readable explanation string. Store traces in the audit log. Generate a
`feature_importance` dict showing which indicators contributed most.

**Impact**: Required for regulatory defensibility (EU AI Act Art. 13 transparency
obligations) and analyst calibration — prevents black-box over-reliance.

---

## 15. Adaptive Keyword Expansion via Word Embeddings

**Current state**: `monitor_platform` accepts a static keyword list; no semantic
expansion.

**Improvement**: Add `async expand_keywords(seed_terms, top_k=20)` that calls a
locally-hosted embedding model (e.g., `nomic-embed-text` via Ollama) to compute cosine
similarity across a cached vocabulary index. Return the top-k semantically similar terms
with similarity scores. Integrate into `monitor_platform` as an optional
`auto_expand=True` flag that silently widens the keyword set. Rebuild the vocabulary
index nightly from collected post tokens.

**Impact**: Captures emerging slang, euphemisms, and cross-language cognates used to
evade keyword-based monitoring — the primary evasion vector in coordinated influence
operations.

---

## Priority Matrix

| # | Improvement | Effort | Impact | Priority |
|---|-------------|--------|--------|----------|
| 1 | Streaming Ingestion (Bytewax/Kafka) | High | Critical | P0 |
| 6 | Persistent Storage (SQLAlchemy Async) | High | Critical | P0 |
| 2 | LLM-Backed Sentiment (Ollama) | Medium | High | P1 |
| 9 | Real-Time Alerting (Webhooks/ntfy) | Medium | High | P1 |
| 3 | Graph DB for Influence Networks | High | High | P1 |
| 12 | PII Minimisation / Retention Policy | Medium | High | P1 |
| 8 | STIX 2.1 / MISP Export | Medium | High | P2 |
| 14 | Explainable AI Scoring Audit Trail | Low | Medium | P2 |
| 4 | Temporal Anomaly Detection | Medium | Medium | P2 |
| 7 | Content-Similarity Clustering (LSH) | Medium | Medium | P2 |
| 11 | Prometheus / OTEL Metrics | Low | Medium | P2 |
| 5 | Multi-Language NLP | Medium | Medium | P3 |
| 10 | Influence Decay Modelling | Low | Medium | P3 |
| 13 | Federated Multi-Tenant Query | High | Medium | P3 |
| 15 | Adaptive Keyword Expansion | Medium | Medium | P3 |
