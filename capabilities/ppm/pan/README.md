# Portfolio Analytics

## Overview
Portfolio Analytics (pan) delivers executive-grade visibility across the project portfolio: strategic alignment scoring, risk-return matrices, capacity heat maps, performance scorecards, benchmark comparisons, and scenario analysis. All analytics are tenant-scoped, approval-gated for writes, and emitted as events for downstream consumption.

## Capability ID
`ppm_pan`

## Provides
| Service | Description |
|---------|-------------|
| portfolio_performance_dashboard | Aggregated KPI views across the portfolio |
| strategic_alignment_scoring | Weighted multi-criteria alignment scores per dimension |
| risk_return_analysis | Risk category vs return metric bubble and matrix views |
| capacity_heat_map | Utilisation heatmaps by resource type, skill, department, or geography |
| portfolio_investment_analysis | NPV, IRR, ROI, and payback period across initiatives |
| project_pipeline_reporting | Funnel view from idea to in-flight to completed |
| benchmark_comparison | Industry, peer, historical, target, and best-in-class benchmarks |
| portfolio_optimisation_recommendations | AI-assisted portfolio rebalancing suggestions |
| executive_portfolio_briefings | Scheduled PDF/email summaries |
| scenario_analysis | Modelled what-if scenarios with analyst attribution |

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication and role-based access |
| audl | Audit logging of analytical decisions |
| mten | Tenant scoping |
| conf | Feature flags and thresholds |
| ntfy | Executive briefing delivery |
| nlpc | NLP-powered search and summarisation |
| moni | Dashboard health monitoring |
| mqeb | Analytics event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| alignment.supported_dimensions | 6 | Strategic, risk, resource, financial, innovation, sustainability |
| alignment.supported_scoring_methods | 5 | AHP, TOPSIS, weighted criteria, etc. |
| risk_return.supported_return_metrics | 6 | NPV, IRR, ROI, payback, B/C ratio, EV/EBITDA |
| governance.classification_downgrade_denied | true | Data classification control |
| governance.scenario_override_requires_analyst | true | Integrity control |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-pan/dashboard | GET | Portfolio analytics dashboard | ppm_pan:view |
| /ppm-pan/portfolios | GET/POST | Portfolio list and creation | ppm_pan:portfolios |
| /ppm-pan/alignment | GET/POST | Strategic alignment scorecard | ppm_pan:alignment |
| /ppm-pan/risk-return | GET/POST | Risk-return matrix | ppm_pan:risk |
| /ppm-pan/capacity | GET/POST | Capacity heat map | ppm_pan:capacity |
| /ppm-pan/performance | GET/POST | Performance scoreboard | ppm_pan:performance |
| /ppm-pan/scenarios | GET/POST | Scenario analysis | ppm_pan:scenarios |
| /ppm-pan/reports | GET/POST | Report builder | ppm_pan:reports |
| /ppm-pan/agents | GET/POST | Agent workbench | ppm_pan:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| portfolio_write_requires_approval | approval_present=False | deny |
| classification_downgrade_denied | classification_downgrade=True | deny |
| cross_tenant_access_denied | cross_tenant_access=True | deny |
| scenario_analyst_required | analyst_present=False | deny |
| analytics_batch_requires_bytewax | event_stream != "bytewax" | deny |
| privileged_agent_action_requires_human_approval | privileged + no approval | deny |

## Data Models
- **Portfolio** — id, name, status, classification, owner_id, approval_reference
- **AlignmentScore** — id, portfolio_id, dimension, scoring_method, score, rationale
- **RiskReturnAnalysis** — id, portfolio_id, risk_category, return_metric, risk_score, return_value
- **CapacityHeatMap** — id, portfolio_id, dimension, snapshot_period, heat_map_data
- **PerformanceSnapshot** — id, portfolio_id, period, metrics, benchmark_type, actual_value
- **ScenarioAnalysis** — id, portfolio_id, scenario_name, assumptions, projected_outcome, analyst_id
- **PortfolioReport** — id, portfolio_id, dashboard_type, format, report_data
- **PortfolioAgent** — id, name, runtime, role, scope

## Streaming Events
- `portfolio_created`, `portfolio_updated`, `alignment_score_calculated`
- `risk_return_analysed`, `capacity_heat_map_generated`, `performance_snapshot_taken`
- `benchmark_comparison_run`, `scenario_analysed`, `report_generated`, `agent_registered`

## Edge Cases Handled
- Classification downgrade of a portfolio is denied to prevent data exposure
- Scenario analysis requires a named analyst to prevent unattributed projections
- Portfolio creation requires approval even for draft status to prevent shadow portfolios
- Heat maps are stored as serialised JSON blobs to support arbitrary dimension grids
- Alignment scores are per-dimension to support partial scoring workflows

## Composability Notes
- Consumes capacity data from **ppm_res** for heat map generation
- Consumes EV and cost data from **ppm_pac** and **ppm_pbl** for performance snapshots
- Feeds the executive dashboard layer alongside **intel** domain dashboards
