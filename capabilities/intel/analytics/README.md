# APG Intelligence Analytics

`intel_analytics` is an executable APG capability for governed,
evidence-backed intelligence analytics.  It can be composed into generated APG
applications that need threat analytics, fraud analytics, public-safety
analytics, incident analytics, strategic analytics, operational analytics, or
risk analytics.

## What It Provides

- Authority, workspace, dataset, feature-set, model, run, insight, dashboard,
  narrative, recommendation, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, dataset
  lineage, model validation, evidence, approvals, Bytewax lifecycle routing,
  and AI-agent guardrails.
- Full suite of async analytics methods: statistical analysis, pattern
  recognition, anomaly detection (global and adaptive rolling), cluster
  analysis, geospatial analysis, temporal analysis and forecasting, link
  analysis with community detection, network centrality (degree, betweenness,
  PageRank-proxy), Granger causality, Bayesian risk scoring, ensemble scoring,
  dataset lineage traversal, provenance validation, visual analytics export,
  and Ollama-powered narrative draft generation.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/analytics/app.py
./.venv/bin/pytest -q capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_analytics --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.analytics import IntelligenceAnalyticsService

svc = IntelligenceAnalyticsService(tenant_id="tenant-a", actor_id="analyst-1")

# Governance chain
authority = svc.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "analytics-scope", "confidential",
    "approver-1", "2027-12-31", "evidence-auth",
)
workspace = svc.record_workspace(
    "ws-1", "tenant-a", "analytical", "Primary Workspace",
    "confidential", "auth-1", "evidence-ws",
)
dataset = svc.register_dataset(
    "ds-1", "tenant-a", "ws-1", "structured",
    "source://raw/2026", "owner-1", "ds-0", "standard", "evidence-ds",
)
```

## Async Analytics Methods

### Descriptive / Inferential Statistics
```python
result = await svc.statistical_analysis("ds-1", "descriptive")
normalised = await svc.data_normalise("ds-1", method="z_score")
```

### Pattern Recognition
```python
patterns = await svc.pattern_recognition([1.2, 1.5, 1.1, 9.9, 1.3], "statistical")
pat_rec = await svc.pattern_recognise("ds-1", algorithm="kmeans")
```

### Anomaly Detection
```python
# Global Z-score
anomalies = await svc.anomaly_statistical(values, sigma_threshold=2.5)

# Adaptive rolling baseline (new — reduces false positives on drifting signals)
rolling = await svc.anomaly_rolling(time_series, window=50, sigma=2.5)

# Batch time-series
batch = await svc.anomaly_detection_batch(time_series)
```

### Temporal Analysis and Forecasting
```python
bins = await svc.temporal_analysis(events, period="day")
pattern = await svc.temporal_pattern(events, granularity="hour")

# Holt-Winters double exponential smoothing (new)
forecast = await svc.forecast_temporal(values, periods_ahead=5)
```

### Cluster and Spatial Analysis
```python
clusters = await svc.cluster_analysis(entity_ids, features)
spatial = await svc.spatial_cluster(geo_points, eps_km=5.0)
geo = await svc.geospatial_analysis(geo_data, "density")
```

### Link and Network Analysis
```python
links = await svc.link_analysis(entities, relationships)
extended = await svc.link_analysis_extended(entities, relationships, include_communities=True)

# Register a graph then compute metrics
await svc.register_network("net-1", nodes, edges)
degree_cent = await svc.network_centrality("net-1")
betweenness = await svc.network_betweenness("net-1")   # new — Brandes algorithm

# Community detection (new — Louvain Phase 1 modularity optimisation)
communities = await svc.detect_communities(entities, relationships, resolution=1.0)

# Inline centrality without pre-registering
centrality = await svc.network_centrality_compute(entities, relationships)
```

### Causal Inference
```python
# Granger causality test (new — OLS residual variance comparison)
causality = await svc.granger_causality(series_x, series_y, max_lag=5)
# Returns: causal_direction (x_causes_y | y_causes_x | bidirectional | none)
```

### Risk Scoring
```python
# Bayesian Beta-Binomial risk update (new — incremental posterior updating)
risk = await svc.bayesian_risk_update(
    "entity-42",
    evidence_observations=[
        {"positive": True, "weight": 1.0},
        {"positive": False, "weight": 0.5},
    ],
)
# Returns posterior_mean, credible_interval_95, bayes_factor, risk_label

# Ensemble scoring across runs (new — softmax-weighted aggregation)
ensemble = await svc.ensemble_score(["run-1", "run-2", "run-3"], weights=[1.0, 0.8, 0.6])
```

### Governance and Provenance
```python
# Full provenance chain validation (new)
prov = await svc.validate_provenance("insight-1")
# Walks: insight → run → model → feature_set → dataset → workspace → authority

# Dataset lineage traversal (new)
upstream = await svc.lineage_traverse("ds-3", direction="upstream")
downstream = await svc.lineage_traverse("ds-0", direction="downstream")
```

### Narrative Generation
```python
# Ollama-powered first-draft narrative (new — requires OLLAMA_BASE_URL)
narrative = await svc.generate_narrative_draft("analysis-id-xyz", model="llama3.2:3b")
# narrative["requires_approval"] is always True — human review mandatory
```

### Export and Reporting
```python
exported = await svc.data_visualisation_export(analysis_id, fmt="csv")
report = await svc.analytics_report(analysis_id)
visual = await svc.visual_analytics(analysis_id, chart_type="line")
```

### Workflow Orchestration
```python
# Sequential multi-step workflow
steps = await svc.analytical_workflow("ds-1", ["statistical", "pattern", "anomaly"])

# Confidence summary
summary = await svc.insight_confidence_summary()
```

## Guardrails

The capability is evidence-led and compliance-first.  It does not implement
hallucinated insights, training-data leakage, privacy bypass, unsupported
automated decisions, unapproved model deployment, autonomous dissemination, or
cross-tenant analytics.  AI-agent actions that request those scopes are denied
by the rule engine.

Narrative drafts generated by `generate_narrative_draft` always carry
`requires_approval: True` and must pass through `record_review()` before
dissemination.  Provenance validation (`validate_provenance`) checks for
broken chain links and expired authorities at dissemination time.

## Streaming

Events are routed to the Bytewax+NATS streaming pipeline under the subject
`apg.intel.analytics.lifecycle`.  Batch ingestion via `validate_batch`.
Cross-capability eventing targets NATS JetStream subjects:
- `apg.intel.analytics.insight.created`
- `apg.intel.analytics.anomaly.detected`
- `apg.intel.analytics.model.run.completed`

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/intel/analytics/*.py \
    capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/analytics --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/analytics --json
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Stream Analytics via NATS JetStream** [Streaming Architecture]
- **I2. Graph-Native Betweenness and PageRank Centrality** [Graph Analytics]
- **I3. Explainable Anomaly Attribution** [Explainability / XAI]
- **I4. Temporal Forecasting with Holt-Winters Smoothing** [Predictive Analytics]
- **I5. Multi-Tenant Privacy-Preserving Differential Statistics** [Privacy Engineering]
- **I6. Semantic Entity Disambiguation and Merging** [Data Quality / Entity Resolution]
- **I7. Causal Graph Construction from Temporal Co-occurrence** [Causal Inference]
- **I8. Automated Insight Narrative Generation via Local LLM (Ollama)** [Natural Language Generation]
- **I9. Confidence-Weighted Ensemble Scoring** [Model Governance]
- **I10. Network Community Detection (Louvain-Approximation)** [Graph Analytics]
- **I11. Provenance Chain Validation and Evidence Integrity Checking** [Governance / Chain of Custody]
- **I12. Adaptive Sigma Threshold via Rolling Baseline** [Anomaly Detection]
- **I13. Cross-Capability Event Bus Integration (NATS Pub/Sub)** [System Integration / Composability]
- **I14. Risk Score Aggregation with Bayesian Updating** [Risk Analytics]
- **I15. Automated Dataset Lineage Graph Traversal** [Data Lineage / Governance]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
