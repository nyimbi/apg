# Social Media Intelligence — User Guide

**Capability ID**: `intel_socint` | **Domain**: `intel` | **Version**: `1.2.0`

## Description

`intel_socint` is an executable APG capability for lawful public or authorized
social-source intelligence. It supports social monitoring, public-safety alerting,
fraud and disinformation review, crisis tracking, brand-risk analysis, and policy
monitoring. All operations are tenant-scoped and enforce a lawful authority chain.

---

## Installation

```bash
pip install apg-intel-socint
```

---

## Quick Start

```python
import asyncio
from capabilities.intel.socint import SocialMediaIntelligenceService

svc = SocialMediaIntelligenceService(tenant_id="tenant-a", actor_id="analyst-1")

# 1. Establish lawful authority
auth = svc.record_authority(
    "auth-1", "tenant-a", "legal_mandate", "public-safety-scope",
    "confidential", "approver-1", "2027-01-01", "ref-legal-001",
)

# 2. Define a monitoring topic
topic = svc.record_topic(
    "topic-1", "tenant-a", "keyword", "election_misinformation",
    "high", "auth-1", "ref-topic-001",
)

# 3. Register a source
source = svc.register_source(
    "src-1", "tenant-a", "public_account", "TWITTER",
    "https://twitter.com/example", "owner-1", "auth-1",
    "tos-review-001", "ref-src-001",
)

# 4. Run async operations
async def run():
    monitor = await svc.monitor_platform("TWITTER", ["election", "vote"], ["@example"])
    posts   = await svc.collect_posts("TWITTER", "election misinformation", limit=200)
    senti   = await svc.sentiment_analysis_batch(["post-1", "post-2", "post-3"])
    report  = await svc.socint_report("election_misinformation", "7d")
    print(report)

asyncio.run(run())
```

---

## Governance Workflow

Every SOCINT operation must be authorised through the following chain:

```
SocialAuthority → SocialTopic → SocialSource → SocialPost → SocialSignal
                                                              ↓
                                             InfluenceAssessment | NetworkAssessment
                                                              ↓
                                             SOCINTReferral | SOCINTDissemination
```

### Record an Authority

```python
svc.record_authority(
    authority_id="auth-1",
    tenant_id="tenant-a",
    authority_type="legal_mandate",       # SUPPORTED_AUTHORITY_TYPES
    scope_reference="case-scope-001",
    classification="confidential",        # SUPPORTED_CLASSIFICATIONS
    approver_id="supervisor-1",
    expires_at="2027-12-31",
    evidence_reference="warrant-001",
)
```

### Register a Source

```python
svc.register_source(
    source_id="src-telegram-1",
    tenant_id="tenant-a",
    source_type="public_channel",         # SUPPORTED_SOURCE_TYPES
    platform_type="TELEGRAM",             # SUPPORTED_PLATFORM_TYPES
    source_reference="https://t.me/example",
    owner_id="collection-team",
    authority_id="auth-1",
    terms_review_reference="tos-review-002",
    evidence_reference="ref-src-002",
)
```

---

## Platform Monitoring

```python
monitor = await svc.monitor_platform(
    platform="TELEGRAM",
    keywords=["protest", "rally", "uprising"],
    handles=["@channel_name"],
)
# Returns: monitor_id, estimated_daily_volume, status, platform_weight
```

Supported platforms and their engagement weights:

| Platform | Weight |
|----------|--------|
| TELEGRAM | 1.2 |
| TIKTOK | 1.1 |
| TWITTER / X | 1.0 |
| FACEBOOK | 0.9 |
| YOUTUBE | 0.95 |
| INSTAGRAM | 0.8 |
| REDDIT | 0.85 |
| WEIBO | 0.75 |
| VK | 0.7 |

---

## Sentiment Analysis

```python
result = await svc.sentiment_analysis_batch(post_ids=["p1", "p2", "p3"])
# Returns: batch_id, positive/negative/neutral counts and percentages, mean_score
```

For multilingual content:

```python
result = await svc.multilingual_content_analysis(
    content="Hii ni habari ya uongo kabisa",
    expected_language="en",
)
# Returns: detected_language, language_confidence, sentiment_label,
#          disinfo_score, translation_note
```

Wire an Ollama `mistral:7b-instruct` endpoint via the `notify` collaborator for
production-grade accuracy (~88% vs ~60% for the built-in lexicon).

---

## Influence Mapping

```python
# Ego network up to depth 2
inf_map = await svc.influence_network_map(handle="@actor", depth=2)

# Influence score decay over time
decay = await svc.influence_decay_model(handle="@actor", half_life_days=30)

# Composite threat profile
profile = await svc.threat_actor_social_profile(handle="@actor")
# Returns: bot_probability, influence_score, is_bridge_node, threat_risk_score
```

---

## Disinformation and CIB Detection

```python
# Single content check
check = await svc.disinformation_detection(content="BREAKING: They are coming! Sources say...")
# Returns: indicators, disinfo_score, is_suspected_disinfo, recommended_action

# Coordinated inauthentic behaviour across accounts
cib = await svc.coordinated_inauthentic_behaviour(account_ids=["acct-1", "acct-2", "acct-3"])
# Returns: cib_indicators, cib_probability, cib_detected, recommended_action

# Near-duplicate content clustering (astroturf detection)
clusters = await svc.content_similarity_cluster(post_ids=["p1","p2","p3"], threshold=0.8)
# Returns: cluster_count, duplicate_rate, coordination_suspected, clusters
```

---

## Persona and Bot Analysis

```python
# Bot/authenticity analysis
persona = await svc.persona_analysis(handle="@suspicious_account")
# Returns: bot_indicators, bot_probability, persona_type (AUTHENTIC/BOT/SUSPECTED_INAUTHENTIC)

# Posting cadence anomaly detection
cadence = await svc.cadence_anomaly_detection(handle="@suspicious_account", days=30)
# Returns: burst_windows, fixed_interval_suspected, anomaly_score, anomaly_detected
```

---

## Network and Community Analysis

```python
# Social graph topology
graph = await svc.social_graph_analysis(handle="@actor")
# Returns: degree, clustering_coefficient, is_bridge_node, network_centrality_estimate

# Community partitioning across a handle set
communities = await svc.community_detection(
    handles=["@a", "@b", "@c", "@d", "@e"],
    algorithm="label_propagation",
)
# Returns: community_count, modularity_estimate, bridge_accounts, communities

# Cross-platform narrative drift
drift = await svc.cross_platform_narrative_analysis(
    topic="election_fraud",
    platforms=["TWITTER", "TELEGRAM", "FACEBOOK"],
)
# Returns: per-platform sentiment, narrative_drift_score, cross_platform_coordinated
```

---

## Influence Operation Detection

```python
# State-sponsored operation indicators
io = await svc.influence_operation_detection(campaign_id="campaign-001")
# Returns: indicators, operation_probability, attribution_confidence, state_sponsored_suspected
```

---

## Trending Topics and Radicalization Scan

```python
# Trending topics by post velocity
trending = await svc.topic_trending_analysis(window_hours=24)

# Radicalization content scan
scan = await svc.radicalization_indicator_scan(post_ids=["p1", "p2", "p3"])
# Returns: flagged_count, flagged_posts, radicalisation_rate
```

---

## Keyword Expansion

```python
expanded = await svc.keyword_expansion(
    seed_terms=["protest", "uprising", "rally"],
    top_k=20,
)
# Returns: expanded_terms with similarity scores, coverage_estimate
# Note: wire Ollama 'nomic-embed-text' for true semantic expansion
```

---

## Intelligence Export

```python
# JSON / CSV export
export = await svc.export_intelligence(fmt="json")

# STIX 2.1 bundle (compatible with stix2.parse())
bundle = await svc.stix_export(bundle_id="bundle-001", include_networks=True)
# Returns: STIX bundle with Indicator, ThreatActor, and Relationship objects

# Full SOCINT report
report = await svc.socint_report(topic="election_misinformation", period="7d")
```

---

## Explainability and Transparency

```python
# EU AI Act Art. 13 score explanation
explanation = await svc.score_explanation(
    result_id="<disinfo_check_id or persona_analysis_id>",
    score_field="disinfo_score",
)
# Returns: score_value, contributing_indicators, feature_importance, narrative
```

---

## Compliance and PII Minimisation

```python
# Check platform ToS compliance
compliance = await svc.platform_policy_compliance(platform="TWITTER")

# Audit and purge records beyond retention window
audit = await svc.pii_retention_audit(retention_days=90)
# Returns: records_audited, records_purged, purge_breakdown
```

---

## AI Agent Registration and Guardrails

```python
# Register an AI agent
agent = svc.register_socint_agent(
    agent_id="agent-1", tenant_id="tenant-a",
    name="MonitorBot", runtime="bytewax", role="collector", scope="public_posts",
)

# Validate an agent action (blocks harassment, doxxing, evasion scopes)
svc.validate_agent_action(
    tenant_id="tenant-a",
    privileged_scope=False,
    human_approval_recorded=True,
)
```

---

## Dashboard and Health

```python
summary = svc.dashboard_summary("tenant-a")
health  = await svc.health_check()
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-socint/dashboard` | `intel_socint:view` | Overview |
| `/intel-socint/authorities` | `intel_socint:authorities` | Governance |
| `/intel-socint/topics` | `intel_socint:topics` | Planning |
| `/intel-socint/sources` | `intel_socint:sources` | Collection |
| `/intel-socint/posts` | `intel_socint:posts` | Collection |
| `/intel-socint/signals` | `intel_socint:signals` | Analysis |
| `/intel-socint/influence` | `intel_socint:influence` | Analysis |
| `/intel-socint/networks` | `intel_socint:networks` | Analysis |

---

## Composability

Reference this capability in `.apg` source files:

```apg
use intel_socint;
```

It composes with: `auth`, `audl`, `ntfy`, `nlpc`, `grph`.

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or
environment variables prefixed with `INTEL_SOCINT_`.

---

## Improvement Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 prioritised improvements:

- P0: Streaming ingestion (Bytewax/Kafka), persistent storage (SQLAlchemy async)
- P1: LLM-backed sentiment (Ollama), real-time alerting, graph DB, PII enforcement
- P2: STIX 2.1 export, explainable AI, temporal anomaly, LSH clustering, Prometheus
- P3: Multilingual NLP, influence decay, federated query, adaptive keyword expansion
