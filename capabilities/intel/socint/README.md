# APG Social Media Intelligence

`intel_socint` is an executable APG capability for lawful public or authorized
social-source intelligence. It can be composed into generated APG applications
that need social monitoring, public-safety alerting, fraud and disinformation
review, crisis tracking, brand-risk analysis, or policy monitoring.

## What It Provides

- Authority, topic, source, post, signal, influence, network, referral,
  dissemination, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, evidence,
  terms review, approvals, Bytewax lifecycle routing, and AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/socint/app.py
./.venv/bin/pytest -q capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_socint --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.socint import SocialMediaIntelligenceService

service = SocialMediaIntelligenceService(tenant_id="tenant-a")
authority = service.record_authority(
    "auth-1", "tenant-a", "legal_mandate", "case-scope",
    "confidential", "approver-1", "2026-12-31", "evidence-auth",
)
```

## Key Service Methods

### Sync (CRUD / Governance)

| Method | Description |
|--------|-------------|
| `record_authority()` | Register a collection authority with classification and evidence |
| `record_topic()` | Define a monitoring topic scoped to an authority |
| `register_source()` | Register a social source (platform account, channel, hashtag) |
| `record_post()` | Record a collected post with confidence score and fingerprint |
| `record_signal()` | Tag a post with a signal type and risk level |
| `record_influence()` | Attach an influence-type assessment to a signal |
| `record_network()` | Record a network-topology assessment |
| `record_referral()` | Create an intelligence referral to another team |
| `record_dissemination()` | Log an intelligence dissemination event |
| `record_review()` | Record analyst review status on any object |
| `register_socint_agent()` | Register an AI agent with role and runtime |
| `validate_agent_action()` | Enforce AI-agent guardrails (no harassment/doxxing/evasion) |
| `validate_batch()` | Gate a Bytewax stream processing batch |
| `dashboard_summary()` | Return aggregate counts for all entity types |

### Async (Operational / Analytical)

| Method | Description |
|--------|-------------|
| `monitor_platform()` | Register a keyword/handle monitor on a social platform |
| `collect_posts()` | Collect posts matching a search query (fingerprint-only, no PII) |
| `bulk_post_collection()` | Fan-out collection across multiple queries with deduplication |
| `sentiment_analysis_batch()` | Batch sentiment scoring across post IDs |
| `influence_network_map()` | Build an ego-network graph with centrality scores |
| `disinformation_detection()` | Detect emotional amplification, source laundering, coordination |
| `narrative_tracking()` | Track narrative evolution and detect sentiment pivots |
| `viral_content_alert()` | Alert on posts exceeding an engagement threshold |
| `persona_analysis()` | Analyse a handle for bot indicators and authenticity |
| `social_graph_analysis()` | Compute clustering coefficient and bridge-node detection |
| `coordinated_inauthentic_behaviour()` | Detect CIB across a set of accounts |
| `cross_platform_narrative_analysis()` | Narrative drift and amplification across platforms |
| `influence_operation_detection()` | Detect state-sponsored influence operation markers |
| `threat_actor_social_profile()` | Composite threat profile (persona + influence + graph) |
| `topic_trending_analysis()` | Rank topics by post velocity over a time window |
| `radicalization_indicator_scan()` | Scan posts for violent/extremist content patterns |
| `platform_policy_compliance()` | Check collection compliance against platform ToS |
| `export_intelligence()` | Export intelligence products (JSON / CSV) |
| `socint_report()` | Generate a structured SOCINT report for a topic/period |
| `health_check()` | Service health and operational metrics |
| **`cadence_anomaly_detection()`** | CUSUM/Poisson burst detection on posting cadence |
| **`content_similarity_cluster()`** | Jaccard-based near-duplicate clustering for astroturf detection |
| **`influence_decay_model()`** | Exponential decay model for influence score over time |
| **`pii_retention_audit()`** | Audit and purge records exceeding retention window |
| **`multilingual_content_analysis()`** | Sentiment + disinformation with language detection |
| **`stix_export()`** | Export signals/influence as a STIX 2.1 bundle |
| **`community_detection()`** | Label-propagation community partitioning for handle sets |
| **`keyword_expansion()`** | Semantic keyword expansion (n-gram proxy; Ollama-ready) |
| **`score_explanation()`** | EU AI Act Art. 13 scoring trace and feature importance |

Bold entries are new in v1.2.0.

## Guardrails

The capability is defensive and compliance-first. It does not implement live
scraping, login/cookie collection, evasion, account automation, direct
messaging, takedown actions, harassment, doxxing, or platform-abuse workflows.
AI-agent actions that request those scopes are denied by the rule engine.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/intel/socint/*.py capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/socint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/socint --json
```

## World-Class Improvement Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised improvements covering
streaming ingestion, LLM-backed sentiment, graph-DB influence networks, STIX 2.1
export, Prometheus metrics, PII minimisation, federated multi-tenant query, and
explainable AI scoring.
