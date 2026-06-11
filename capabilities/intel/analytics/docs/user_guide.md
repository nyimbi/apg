# Intelligence Analytics — User Guide

**Capability ID**: `intel_analytics` | **Domain**: `intel` | **Version**: `1.2.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

`intel_analytics` is the governed intelligence analytics runtime for APG
applications.  It provides a full analytical stack — from raw dataset
ingestion through feature engineering, model management, anomaly detection,
network analysis, causal inference, risk scoring, and narrative generation —
within a strict evidence-backed, tenant-isolated governance framework.

All analytical operations emit structured audit events routed through
Bytewax+NATS, ensuring a complete, tamper-evident audit trail suitable for
judicial and regulatory review.

---

## Concepts

### Governance Chain

Every analysis product traces back through an unbroken chain:

```
Authority
  └── Workspace
        └── Dataset
              └── FeatureSet
                    └── Model
                          └── Run
                                └── Insight
                                      └── Dashboard / Narrative / Recommendation
```

The `validate_provenance(insight_id)` method verifies this chain at any point,
checking for broken links, expired authorities, and missing evidence
references.

### Tenant Isolation

All data structures are keyed by `(tenant_id, item_id)`.  No method returns
data belonging to another tenant.  Cross-tenant aggregation requires
differential privacy (`differential_statistics`, planned in v1.3).

### Evidence-Led Operations

Every mutating operation requires an `evidence_reference` parameter.  The
rule engine will deny operations with missing evidence or unsupported
enumerated values.  See `capability_contract.py` for the full allowed-values
list.

---

## Installation

```bash
pip install apg-intel-analytics
# or for development:
cd capabilities/intel/analytics
pip install -e .
```

---

## Quick Start

```python
import asyncio
from capabilities.intel.analytics import IntelligenceAnalyticsService

async def main():
    svc = IntelligenceAnalyticsService(tenant_id="acme", actor_id="analyst-1")

    # 1. Establish governance chain
    svc.record_authority(
        "auth-1", "acme", "mission_order", "threat-scope",
        "confidential", "approver-1", "2027-12-31", "ev-001",
    )
    svc.record_workspace(
        "ws-1", "acme", "analytical", "Threat Workspace",
        "confidential", "auth-1", "ev-002",
    )
    svc.register_dataset(
        "ds-1", "acme", "ws-1", "structured", "source://signals",
        "owner-1", "ds-root", "standard", "ev-003",
    )
    svc.record_feature_set(
        "fs-1", "acme", "ds-1", "numerical",
        "features://signals/v1", 0.87, "analyst-1", "ev-004",
    )
    svc.record_model(
        "mdl-1", "acme", "fs-1", "statistical",
        "anomaly_detection", "validation://v1", "medium", "ev-005",
    )
    svc.record_run(
        "run-1", "acme", "mdl-1", "batch",
        "results://run1", 0.82, "analyst-1", "ev-006",
    )

    # 2. Run analytics
    stats = await svc.statistical_analysis("ds-1", "descriptive")
    print(f"Mean confidence: {stats['mean']}")

    anomalies = await svc.anomaly_rolling(
        [{"t": f"2026-01-{i:02d}T00:00:00Z", "v": float(i % 7)} for i in range(1, 31)],
        window=10, sigma=2.0,
    )
    print(f"Rolling anomalies detected: {anomalies['anomaly_count']}")

    # 3. Validate provenance before dissemination
    svc.record_insight(
        "ins-1", "acme", "run-1", "anomaly",
        stats["analysis_id"], 0.82, "analyst-1", "ev-007",
    )
    prov = await svc.validate_provenance("ins-1")
    if not prov["valid"]:
        raise ValueError(f"Broken provenance: {prov['broken_links']}")

asyncio.run(main())
```

---

## Analytical Methods Reference

### Statistical Analysis

| Method | Description |
|--------|-------------|
| `statistical_analysis(dataset_id, analysis_type)` | Descriptive statistics over feature confidence scores |
| `data_normalise(dataset_id, method)` | Min-max or Z-score normalisation of feature scores |
| `insight_confidence_summary()` | Per-type confidence aggregation for the tenant |

### Pattern Recognition

| Method | Description |
|--------|-------------|
| `pattern_recognition(data_points, algorithm)` | Outlier detection + lag-1 autocorrelation |
| `pattern_recognise(dataset_id, algorithm)` | Dataset-linked pattern detection |

### Anomaly Detection

| Method | Description |
|--------|-------------|
| `anomaly_detection_batch(time_series)` | Global Z-score anomaly detection on `{t,v}` series |
| `anomaly_statistical(values, sigma_threshold)` | Z-score flagging on a plain float list |
| `anomaly_rolling(time_series, window, sigma)` | **Adaptive rolling-baseline** detection — reduces false positives on drifting signals |

**When to use `anomaly_rolling` vs `anomaly_statistical`**:
- Use `anomaly_statistical` for stationary signals with a stable mean.
- Use `anomaly_rolling` for operational intelligence streams where baseline
  activity levels shift over time (e.g., network traffic, event counts with
  day-of-week seasonality).

### Temporal Analysis and Forecasting

| Method | Description |
|--------|-------------|
| `temporal_analysis(events, period)` | Event frequency binning by period label |
| `temporal_pattern(events, granularity)` | Hourly / daily / weekly activity distribution |
| `forecast_temporal(values, periods_ahead, alpha, beta)` | **Holt-Winters double exponential smoothing** point forecasts with 95% confidence intervals |

**Forecast example**:
```python
# Predict next 3 periods based on 12 months of historical event counts
forecast = await svc.forecast_temporal(
    values=[120, 135, 128, 145, 162, 155, 170, 180, 175, 190, 200, 195],
    periods_ahead=3,
    alpha=0.3,
    beta=0.1,
)
for f in forecast["forecasts"]:
    print(f"Period +{f['horizon']}: {f['point']}  [{f['lower_95']}, {f['upper_95']}]")
```

### Cluster and Spatial Analysis

| Method | Description |
|--------|-------------|
| `cluster_analysis(entity_ids, features)` | K-means-style partitioning by feature confidence |
| `spatial_cluster(geo_points, eps_km)` | DBSCAN-style spatial clustering with Haversine distance |
| `geospatial_analysis(geo_data, analysis_type)` | Centroid and bounding-box computation |

### Link and Network Analysis

| Method | Description |
|--------|-------------|
| `link_analysis(entities, relationships)` | Degree centrality and hub identification |
| `link_analysis_extended(entities, relationships, include_communities)` | Link analysis with connected-component community labels |
| `register_network(network_id, nodes, edges)` | Persist a graph for repeated analysis |
| `network_centrality(network_id)` | Degree centrality on a stored network |
| `network_betweenness(network_id)` | **Brandes betweenness centrality** — exposes bridge nodes invisible to degree centrality |
| `network_centrality_compute(entities, relationships)` | Inline degree + betweenness proxy without pre-registering |
| `detect_communities(entities, relationships, resolution)` | **Louvain Phase 1** modularity-based community detection |

**Betweenness vs Degree centrality**:
Degree centrality counts direct connections.  Betweenness counts how often a
node lies on shortest paths between other pairs.  In criminal network analysis,
a broker with few direct contacts but many brokered connections has low degree
but high betweenness — betweenness surfaces these structurally critical nodes.

**Community detection parameters**:
- `resolution=1.0` — default, balanced community sizes.
- `resolution > 1.0` — smaller, tighter communities.
- `resolution < 1.0` — larger, more aggregated communities.

### Causal Inference

| Method | Description |
|--------|-------------|
| `granger_causality(series_x, series_y, max_lag)` | **Granger F-test** for temporal predictive causation |

```python
causality = await svc.granger_causality(
    series_x=[...],   # possible cause
    series_y=[...],   # possible effect
    max_lag=5,
)
# causality["causal_direction"]: x_causes_y | y_causes_x | bidirectional | none
# causality["optimal_lag"]: lag at which causal signal is strongest
```

**Interpretation**: Granger causality does not establish mechanistic causation.
It establishes that series_x contains information that statistically precedes
changes in series_y.  Treat as evidence for further investigation, not as a
definitive causal claim.

### Risk Scoring

| Method | Description |
|--------|-------------|
| `bayesian_risk_update(entity_id, evidence_observations, prior_alpha, prior_beta)` | **Beta-Binomial Bayesian** posterior risk score with incremental updating |
| `ensemble_score(run_ids, weights)` | **Softmax-weighted ensemble** confidence scoring across multiple model runs |

**Bayesian risk example**:
```python
# First encounter — weak prior (1 positive in 10 prior observations)
risk = await svc.bayesian_risk_update(
    "suspect-42",
    evidence_observations=[
        {"positive": True, "weight": 1.0},   # confirmed association
        {"positive": False, "weight": 2.0},  # two alibi confirmations
    ],
    prior_alpha=1.0,
    prior_beta=9.0,
)
print(risk["posterior_mean"])      # updated probability of risk
print(risk["credible_interval_95"])  # (lower, upper) 95% credible interval
print(risk["bayes_factor"])        # evidence strength vs prior
print(risk["risk_label"])          # "low" | "medium" | "high"

# Subsequent session — the stored posterior is retrieved automatically
risk2 = await svc.bayesian_risk_update(
    "suspect-42",
    evidence_observations=[{"positive": True, "weight": 1.0}],
)
# risk2["posterior_alpha"] > risk["posterior_alpha"] — incremental update
```

### Governance and Provenance

| Method | Description |
|--------|-------------|
| `validate_provenance(insight_id)` | Full chain validation: insight → authority, with expiry check |
| `lineage_traverse(dataset_id, direction, max_depth)` | DAG walk of dataset lineage references |

**Provenance validation**:
```python
prov = await svc.validate_provenance("insight-42")
if not prov["valid"]:
    if prov["authority_expired"]:
        print("Authority has expired — re-authorisation required")
    for link in prov["broken_links"]:
        print(f"Missing: {link}")
```

**Lineage traversal**:
```python
# What contributed to this dataset?
upstream = await svc.lineage_traverse("ds-5", direction="upstream")

# What does this dataset feed?
downstream = await svc.lineage_traverse("ds-1", direction="downstream")
# Use downstream for impact analysis: if ds-1 is retracted, these are affected
```

### Narrative Generation

| Method | Description |
|--------|-------------|
| `generate_narrative_draft(analysis_id, model)` | Ollama-powered first-draft narrative; always requires human approval |

```python
# Requires OLLAMA_BASE_URL environment variable
import os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

narrative = await svc.generate_narrative_draft("stat_ds-1_descriptive", model="llama3.2:3b")
assert narrative["requires_approval"] is True  # always True — cannot be overridden

# Review the draft before recording it
svc.record_narrative(
    "narr-1", "acme", "ins-1", "analytical_brief",
    narrative["draft"][:200], "approver-1", "ev-008",
)
svc.record_review("rev-1", "acme", "narr-1", "reviewer-1", "approved", "ev-009")
```

When `OLLAMA_BASE_URL` is not set, the method returns a structured template
for manual completion rather than raising an error.

### Export and Reporting

| Method | Description |
|--------|-------------|
| `data_visualisation_export(analysis_id, fmt)` | Export analysis result as json / csv / geojson / pdf_summary |
| `analytics_report(analysis_id)` | Structured report linking analysis result to tenant insights |
| `visual_analytics(analysis_id, chart_type)` | Vega-Lite spec descriptor for bar / line / scatter / heatmap / network |

### Workflow Orchestration

```python
# Sequential multi-step pipeline
results = await svc.analytical_workflow(
    "ds-1",
    steps=["statistical", "pattern", "anomaly", "normalise"],
)
for step_result in results:
    print(step_result["step"], step_result["result"]["analysis_id"])
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-analytics/dashboard` | `intel_analytics:view` | Overview |
| `/intel-analytics/authorities` | `intel_analytics:authorities` | Governance |
| `/intel-analytics/workspaces` | `intel_analytics:workspaces` | Planning |
| `/intel-analytics/datasets` | `intel_analytics:datasets` | Data |
| `/intel-analytics/features` | `intel_analytics:features` | Data |
| `/intel-analytics/models` | `intel_analytics:models` | Analysis |
| `/intel-analytics/runs` | `intel_analytics:runs` | Analysis |
| `/intel-analytics/insights` | `intel_analytics:insights` | Analysis |
| `/intel-analytics/networks` | `intel_analytics:networks` | Network |
| `/intel-analytics/risk` | `intel_analytics:risk` | Risk |
| `/intel-analytics/narratives` | `intel_analytics:narratives` | Dissemination |

---

## Capability Composition

Reference `intel_analytics` in `.apg` source files:

```apg
use intel_analytics;
```

The capability exposes the following workflow interfaces for composition:

- `analytics_authority_workflow`
- `analytics_workspace_workflow`
- `analytics_dataset_workflow`
- `analytics_feature_workflow`
- `analytics_model_workflow`
- `analytics_run_workflow`
- `analytics_insight_workflow`

Cross-capability event subjects (NATS JetStream):

- `apg.intel.analytics.insight.created`
- `apg.intel.analytics.anomaly.detected`
- `apg.intel.analytics.model.run.completed`
- `apg.intel.analytics.risk.updated`

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INTEL_ANALYTICS_DB_URL` | None | PostgreSQL connection URL |
| `INTEL_ANALYTICS_TENANT` | `default` | Default tenant ID |
| `INTEL_ANALYTICS_AUDIT_STREAM` | `apg.intel.analytics.lifecycle` | NATS subject for audit events |
| `OLLAMA_BASE_URL` | None | Ollama base URL for narrative generation |

---

## Dependencies

**Required**:
- `auth` — tenant authentication
- `audt` — audit event routing
- `ntfy` — notification dispatch

**Optional** (enable richer features):
- `nlpc` — NLP composition for narrative enhancement
- `grph` — graph database backend for large networks (>50k nodes)
- Ollama (local) — narrative draft generation

---

## Testing

```bash
# Unit tests
./.venv/bin/pytest -vxs capabilities/intel/analytics/tests/

# Type checking
./.venv/bin/pyright capabilities/intel/analytics/

# Compile check
./.venv/bin/python -m py_compile capabilities/intel/analytics/*.py

# Full lifecycle audit
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/analytics --json
```

---

## Further Reading

- `service.py` — Full business logic with all async methods
- `models.py` — Dataclass models for all domain objects
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic request/response schemas
- `capability_contract.py` — Supported enumerated values and rule engine
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned enhancements
- `SPECIFICATION.md` — Formal capability specification
- `PLAN.md` — Implementation plan and decision log
