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
| earned_value_management | SPI, CPI, EAC, TCPI per project with portfolio-level aggregation |
| delivery_velocity_trending | Rolling completion rate trend with decline alerting |
| three_horizons_balance | McKinsey H1/H2/H3 investment balance scoring |
| resource_bottleneck_detection | Role-level over-allocation detection and severity ranking |
| intel_domain_sync | Portfolio health signal injection into enterprise intelligence pipeline |

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
| /ppm-pan/evm | GET | Earned value metrics | ppm_pan:performance |
| /ppm-pan/velocity | GET | Delivery velocity trend | ppm_pan:performance |
| /ppm-pan/balance | GET | Three-horizons balance score | ppm_pan:alignment |
| /ppm-pan/bottlenecks | GET | Resource bottleneck detector | ppm_pan:capacity |
| /ppm-pan/escalations | GET | RAG escalation check | ppm_pan:admin |

## Key Service Methods

### Core (synchronous)
- `describe()` — return capability contract
- `evaluate(context)` — evaluate governance rules
- `create_portfolio()` — create a new portfolio record
- `get_portfolio()` / `list_portfolios()` — retrieve portfolios
- `score_alignment()` / `list_alignment_scores()` — record alignment scores
- `analyse_risk_return()` — record risk-return entries
- `generate_heat_map()` / `generate_report()` — artefact creation
- `snapshot_performance()` — performance snapshot
- `run_scenario()` — scenario analysis record
- `register_agent()` / `validate_agent_action()` — agent lifecycle

### Async analytics
- `portfolio_overview(portfolio_id, as_of_date)` — budget, health, progress aggregation
- `strategic_alignment_score(project_id, objectives)` — weighted composite scoring
- `risk_return_analysis(portfolio_id)` — efficient frontier with quadrant distribution
- `capacity_demand_chart(portfolio_id, period)` — FTE gap analysis
- `resource_heat_map(portfolio_id, period)` — utilisation heat map
- `portfolio_health_dashboard()` — cross-portfolio RAG status
- `investment_efficiency_index(portfolio_id)` — IEI / NPV / ROI
- `benefits_realisation_tracking(project_id, benefit_id, actual_value)` — variance tracking
- `portfolio_optimisation(budget_constraint, resource_constraint)` — greedy knapsack selection
- `executive_portfolio_report(portfolio_id, period)` — composite executive report
- `portfolio_performance_report(period)` — SPI/CPI aggregate
- `risk_return_summary()` — tenant-wide risk/return summary
- `export_portfolios(format)` — JSON/CSV export
- `capacity_heat_map_summary()` — utilisation distribution
- `scenario_comparison(scenario_ids)` — side-by-side scenario comparison
- `alignment_heatmap()` — alignment by strategic dimension
- `health_check()` — service liveness check
- `portfolio_compliance_check()` — governance compliance audit
- `bulk_create_portfolios(portfolio_specs)` — bulk portfolio creation

### World-class additions (v1.1+)
- `earned_value_metrics(portfolio_id, as_of_date)` — SPI, CPI, EAC, TCPI, SV, CV per project and portfolio
- `portfolio_bubble_chart(portfolio_id, x_metric, y_metric, size_metric)` — normalised multi-axis bubble chart data
- `delivery_velocity_trend(portfolio_id, window_weeks)` — rolling completion rate with linear regression slope
- `rag_escalation_check(tenant_id, red_threshold_snapshots)` — automated RED escalation with ntfy payload
- `benchmark_gap_analysis(portfolio_id, benchmark_types)` — tornado-ranked gap analysis across benchmark types
- `portfolio_balance_score(portfolio_id, h1/h2/h3_target_pct)` — McKinsey Three Horizons balance scoring
- `resource_bottleneck_detector(portfolio_id, period, top_n)` — role-level over-allocation severity ranking
- `portfolio_lifecycle_advance(portfolio_id, target_stage, actor_id, evidence_reference)` — governed stage transitions with history
- `generate_portfolio_narrative(portfolio_id, period, style)` — Ollama-powered executive prose narrative
- `sync_to_intel_domain(portfolio_id, intel_service)` — portfolio health signal injection into intel pipeline

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

## Project Registry Fields (for EVM and velocity methods)
Projects registered in `_project_registry` support the following optional fields for advanced analytics:

| Field | Type | Used by |
|-------|------|---------|
| planned_value | float | earned_value_metrics |
| earned_value | float | earned_value_metrics |
| actual_cost | float | earned_value_metrics, portfolio_overview |
| budget_at_completion | float | earned_value_metrics |
| completed_date | ISO date str | delivery_velocity_trend |
| role_demand | dict[str, float] | resource_bottleneck_detector |
| role_supply | dict[str, float] | resource_bottleneck_detector |
| demand_fte | float | capacity_demand_chart, portfolio_optimisation |
| supply_fte | float | capacity_demand_chart |
| projected_return | float | investment_efficiency_index, portfolio_optimisation |
| planned_benefits | dict[str, float] | benefits_realisation_tracking |
| health | green/amber/red | portfolio_overview, sync_to_intel_domain |

## Streaming Events
- `portfolio_created`, `portfolio_updated`, `alignment_score_calculated`
- `risk_return_analysed`, `capacity_heat_map_generated`, `performance_snapshot_taken`
- `benchmark_comparison_run`, `scenario_analysed`, `report_generated`, `agent_registered`
- `earned_value_metrics_computed`, `bubble_chart_generated`, `delivery_velocity_computed`
- `rag_escalation_check_run`, `benchmark_gap_analysed`, `portfolio_balance_scored`
- `resource_bottleneck_detected`, `portfolio_lifecycle_advanced`
- `portfolio_narrative_generated`, `intel_signal_synced`

## Edge Cases Handled
- Classification downgrade of a portfolio is denied to prevent data exposure
- Scenario analysis requires a named analyst to prevent unattributed projections
- Portfolio creation requires approval even for draft status to prevent shadow portfolios
- Heat maps are stored as serialised JSON blobs to support arbitrary dimension grids
- Alignment scores are per-dimension to support partial scoring workflows
- EVM methods return `None` for SPI/CPI when PV or AC is zero to avoid division artefacts
- Lifecycle transitions validate legal state machine paths; illegal transitions raise PermissionError
- Narrative generation falls back to structured template when Ollama is unavailable
- Intel sync queues the signal in the audit stream when no intel_service is injected

## Composability Notes
- Consumes capacity data from **ppm_res** for heat map generation
- Consumes EV and cost data from **ppm_pac** and **ppm_pbl** for performance snapshots
- Feeds the executive dashboard layer alongside **intel** domain dashboards
- `sync_to_intel_domain` bridges portfolio RAG status into **intel** threat signal pipeline
- `rag_escalation_check` outputs notification payloads consumable by **ntfy** capability

## See Also
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed improvement specifications
- `service.py` — Full implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `docs/user_guide.md` — End-user guide
