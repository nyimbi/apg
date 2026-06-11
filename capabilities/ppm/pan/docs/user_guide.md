# Portfolio Analytics — User Guide

**Capability ID**: `ppm_pan` | **Domain**: `ppm` | **Version**: `1.1.0`

---

## Introduction

Portfolio Analytics (`ppm_pan`) provides executive-grade portfolio visibility for project management offices (PMOs) and portfolio directors. It aggregates data across all projects in one or more portfolios to produce strategic dashboards, investment efficiency metrics, delivery health indicators, and AI-assisted narratives — all scoped to a tenant and governed by a deterministic rule engine.

---

## Getting Started

### Installation

```bash
pip install apg-ppm-pan
```

### Minimal Setup

```python
import asyncio
from capabilities.ppm.pan.service import PortfolioAnalyticsService

svc = PortfolioAnalyticsService(tenant_id="acme", actor_id="alice")

# Create a portfolio
svc.create_portfolio(
    portfolio_id="port-001",
    tenant_id="acme",
    name="Digital Transformation 2026",
    status="active",
    classification="internal",
    owner_id="alice",
    approval_reference="APPROVAL-2026-001",
    evidence_reference="GOV-BOARD-MIN-2026-03",
)

# Run a portfolio overview
overview = asyncio.run(svc.portfolio_overview("port-001", "2026-06-11"))
print(overview)
```

---

## Core Concepts

### Tenant Scoping
Every operation requires a `tenant_id`. The service constructor sets a default tenant; individual CRUD methods accept an explicit `tenant_id` for multi-tenant deployments.

### Governance Rules
Write operations are validated by the deterministic rule engine (`evaluate_capability_rules`). Common denial reasons:
- `tenant_context_required` — `tenant_id` is empty
- `portfolio_write_requires_approval` — no `approval_reference` supplied
- `classification_downgrade_denied` — attempting to lower a portfolio's classification level
- `scenario_override_requires_analyst` — scenario created without a named analyst

### Project Registry
Advanced analytics (EVM, velocity, bottleneck detection) rely on projects being registered in the internal `_project_registry` dict. Register a project like this:

```python
svc._project_registry["acme:proj-001"] = {
    "project_id": "proj-001",
    "portfolio_id": "port-001",
    "tenant_id": "acme",
    "budget": 500_000,
    "actual_cost": 210_000,
    "planned_value": 230_000,
    "earned_value": 195_000,
    "budget_at_completion": 500_000,
    "projected_return": 1_200_000,
    "progress_pct": 42,
    "health": "amber",
    "demand_fte": 8.0,
    "supply_fte": 6.5,
    "role_demand": {"senior_developer": 4.0, "data_engineer": 2.0, "ba": 2.0},
    "role_supply": {"senior_developer": 3.0, "data_engineer": 2.5, "ba": 1.0},
    "completed_date": None,
}
```

---

## Analytics Methods

### Portfolio Overview
```python
overview = await svc.portfolio_overview("port-001", "2026-06-11")
# Returns: project_count, total_budget, budget_utilisation_pct,
#          health_distribution, avg_progress_pct, avg_alignment_score
```

### Strategic Alignment Scoring
```python
result = await svc.strategic_alignment_score(
    project_id="proj-001",
    strategic_objectives=[
        {"objective_id": "obj-1", "weight": 3, "score": 8.5},
        {"objective_id": "obj-2", "weight": 2, "score": 6.0},
    ]
)
# Returns: composite_alignment_score (0–10), alignment_pct
```

### Risk-Return Analysis
```python
# First record raw data points:
svc.analyse_risk_return(
    analysis_id="rr-001", tenant_id="acme", portfolio_id="port-001",
    risk_category="technology", return_metric="npv",
    risk_score=6.5, return_value=800_000,
    analysis_period="2026-Q2", evidence_reference="BIZ-CASE-001",
)

# Then compute the matrix:
matrix = await svc.risk_return_analysis("port-001")
# Returns: quadrant_distribution, avg_risk_score, avg_return_value, frontier
```

### Earned Value Metrics (EVM)
Requires `planned_value`, `earned_value`, `actual_cost`, `budget_at_completion` on each project.

```python
evm = await svc.earned_value_metrics("port-001", as_of_date="2026-06-11")
# Returns per-project SPI, CPI, SV, CV, EAC, TCPI and portfolio-level aggregates
```

Key indicators:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| SPI | EV / PV | < 1.0 = behind schedule |
| CPI | EV / AC | < 1.0 = over budget |
| EAC | BAC / CPI | Revised cost estimate |
| TCPI | (BAC-EV) / (BAC-AC) | Required CPI to meet budget |
| SV | EV - PV | Negative = schedule slippage |
| CV | EV - AC | Negative = cost overrun |

### Portfolio Bubble Chart
```python
chart = await svc.portfolio_bubble_chart(
    portfolio_id="port-001",
    x_metric="risk_score",
    y_metric="return_value",
    size_metric="budget",
)
# Returns normalised (0–1) bubble positions; swap axes without re-querying
```

### Delivery Velocity Trend
```python
velocity = await svc.delivery_velocity_trend("port-001", window_weeks=4)
# Returns trend ('improving'|'flat'|'declining'), slope, per-window counts
# velocity_alert=True when 2+ consecutive declining windows detected
```

### RAG Escalation Check
```python
escalations = await svc.rag_escalation_check(
    tenant_id="acme",
    red_threshold_snapshots=2,
)
# Portfolios with avg_alignment < 4 for >= 2 snapshots get escalation records
# Each escalation includes a pre-built ntfy notification payload
```

### Benchmark Gap Analysis
```python
gaps = await svc.benchmark_gap_analysis(
    portfolio_id="port-001",
    benchmark_types=["industry_average", "target", "best_in_class"],
)
# Returns tornado-sorted list of gaps; largest_gap surfaces the worst deviation
```

### Portfolio Balance Score (Three Horizons)
```python
balance = await svc.portfolio_balance_score(
    portfolio_id="port-001",
    h1_target_pct=70.0,
    h2_target_pct=20.0,
    h3_target_pct=10.0,
)
# Returns horizon_split, balance_deviation_pct, balance_rating
# balance_rating: 'excellent' | 'good' | 'needs_rebalancing'
```

Classification heuristic (alignment scores drive horizon assignment):
- **H1** (run-the-business): strategic_fit < 0.4 AND innovation_index < 0.4
- **H2** (growth): strategic_fit ≥ 0.4 OR innovation_index in [0.4, 0.7)
- **H3** (transformation): innovation_index ≥ 0.7

### Resource Bottleneck Detector
```python
bottlenecks = await svc.resource_bottleneck_detector(
    portfolio_id="port-001",
    period="2026-Q3",
    top_n=5,
)
# Returns top_bottlenecks ranked by utilisation ratio
# severity: 'critical' (>1.3) | 'high' (>1.1) | 'moderate' (>1.0) | 'ok'
```

Requires `role_demand` and `role_supply` dicts on each project in the registry. Falls back to aggregate FTE when role-level data is absent.

### Portfolio Lifecycle Advance
```python
transition = await svc.portfolio_lifecycle_advance(
    portfolio_id="port-001",
    target_stage="under_review",
    actor_id="alice",
    evidence_reference="GOV-REVIEW-2026-Q2",
)
# Validates legal transitions; raises PermissionError on illegal moves
# Returns full lifecycle_history list for audit trail
```

Legal state machine:
```
proposed -> approved -> active -> under_review -> closed
                      -> archived
                under_review -> active  (re-activation)
```

### Executive Portfolio Report
```python
report = await svc.executive_portfolio_report("port-001", "2026-Q2")
# Composes: portfolio_overview + risk_return + investment_efficiency + benefits tracking
```

### AI Narrative Generation
Requires a running Ollama instance (`OLLAMA_BASE_URL` environment variable).

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_NARRATIVE_MODEL=llama3.1:8b  # optional; default is llama3.1:8b
```

```python
narrative = await svc.generate_portfolio_narrative(
    portfolio_id="port-001",
    period="2026-Q2",
    style="formal",          # 'formal' | 'concise' | 'risk_focused'
)
print(narrative["narrative"])
```

Falls back to a structured template narrative when Ollama is not available. `model_used` in the response indicates whether the LLM or `"template_fallback"` was used.

### Intel Domain Sync
```python
sync_result = await svc.sync_to_intel_domain(
    portfolio_id="port-001",
    intel_service=None,      # pass an intel service instance for direct ingestion
)
# Without intel_service: signal is queued via audit stream for async pickup
# With intel_service: calls intel_service.ingest_portfolio_signal(payload)
```

The signal payload contains `rag_status`, `avg_alignment_score`, `avg_risk_score`, `critical_projects`, and `severity` — everything the intel domain needs to model portfolio health as an enterprise threat signal.

---

## Portfolio Optimisation

```python
optimised = await svc.portfolio_optimisation(
    budget_constraint=5_000_000,
    resource_constraint=50.0,   # FTE
)
# Greedy knapsack by ROI ratio (projected_return / budget)
# Returns: selected_projects, excluded_projects, portfolio_roi_pct, remaining_budget
```

---

## Exporting Data

```python
# JSON export
export_json = await svc.export_portfolios(format="json")

# CSV export
export_csv = await svc.export_portfolios(format="csv")
print(export_csv["content"])
```

---

## Governance and Compliance

```python
# Check all portfolios for governance compliance
compliance = await svc.portfolio_compliance_check()
# Returns: no_owner_count, no_budget_count, compliance_rate_pct

# Retrieve audit trail
events = await svc.get_audit_events()
# Returns list of {tenant_id, event_type, reference_id, processor}
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary("acme")
# Returns counts of all artefacts: portfolios, alignment_scores, heat_maps,
# performance_snapshots, scenarios, reports, agents, audit_events
```

---

## Flask-AppBuilder Integration

The capability registers Blueprint routes automatically when loaded by the APG shell. To mount it standalone:

```python
from capabilities.ppm.pan.app import create_app
app = create_app()
app.run(debug=True, port=5010)
```

UI routes are prefixed `/ppm-pan/` and require the `ppm_pan` permission set.

---

## Configuration Reference

All settings are tenant-scoped. Override via the `conf` capability or env vars prefixed `PPM_PAN_`.

| Key | Default | Description |
|-----|---------|-------------|
| `PPM_PAN_DB_URL` | SQLite in-memory | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | (unset) | Enables AI narrative generation |
| `OLLAMA_NARRATIVE_MODEL` | `llama3.1:8b` | Ollama model for narrative generation |

---

## Testing

```bash
uv run pytest -vxs capabilities/ppm/pan/tests/
```

Key test files:
- `tests/test_service.py` — service unit and integration tests
- `tests/test_contract.py` — governance rule engine tests

---

## Composability

```apg
use ppm_pan;
```

| Integrates with | Direction | Purpose |
|-----------------|-----------|---------|
| ppm_res | inbound | Capacity and skill data for heat maps |
| ppm_pac | inbound | Earned value and cost actuals |
| ppm_pbl | inbound | Planned baseline for EVM |
| intel | outbound | Portfolio health signals via sync_to_intel_domain |
| ntfy | outbound | RAG escalation notification payloads |

---

## Further Reading

- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed improvement specifications
- `service.py` — Full implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `cap_spec.md` — Formal capability specification
