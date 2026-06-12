# Prescriptive Analytics

## Overview
The Prescriptive Analytics capability (`bia_psa`) provides optimisation engines (LP, IP, GA, RL, stochastic, robust), decision support with explainability, recommendation action management with approval workflows, what-if analysis, sensitivity analysis, Monte Carlo simulation, and cost-benefit analysis — all tenant-scoped with mandatory governance and full audit.

**v2.0** adds 15 world-class enhancements covering stochastic programming, multi-objective Pareto optimisation, RL policy training, Benders decomposition, SHAP-style explainability, rolling-horizon re-optimisation, and full financial quantification via DCF-integrated CBA.

## Capability ID
`bia_psa`

## Provides
- `optimisation_engine` — 7+ optimisation types: LP, IP, GA, RL, stochastic, robust, decomposition-based
- `decision_support_system` — Typed decision recording with mandatory explainability
- `recommendation_actions` — Generated recommendations with approval-gated actioning
- `whatif_analysis` — Baseline-anchored scenario parameter sweeps
- `constraint_management` — Hard/soft/preference constraint enforcement + conflict detection
- `multi_objective_analysis` — Pareto-optimal multi-objective optimisation (NSGA-II)
- `allocation_optimisation` — Resource allocation and scheduling optimisation
- `process_improvement_recommendations` — Process change and investment recommendations
- `stochastic_programming` — SAA-based stochastic LP and multi-stage scenario trees
- `robust_optimisation` — Ellipsoidal uncertainty set LP with price-of-robustness reporting
- `rl_policy_engine` — Q-learning and PPO policy training and application
- `cost_benefit_analysis` — DCF engine: NPV, IRR, MIRR, profitability index, payback

## Requires
| Capability | Reason |
|------------|--------|
| auth | User identity and permission checks |
| audl | Audit all decisions and recommendations |
| mten | Tenant context enforcement |
| conf | Runtime configuration |
| mqeb | Streaming analytics lifecycle events |
| moni | Operational monitoring of optimisation runs |
| wflo | Approval workflow for recommendations |
| bia_pda | Predictive model outputs as optimisation inputs |

## Configuration
| Option | Default | Description |
|--------|---------|-------------|
| max_variables | 10,000 | Optimisation variable limit |
| max_constraints | 50,000 | Constraint limit per optimisation |
| require_approval | true | Recommendations require approval before action |
| require_explainability | true | Decisions must have rationale |
| max_recommendations_per_run | 50 | Output cap per optimisation run |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/bia/psa/optimisations | GET/POST | List/create optimisations | bia_psa:optimise |
| /api/bia/psa/optimisations/\<id\>/run | POST | Run optimisation | bia_psa:optimise |
| /api/bia/psa/optimisations/\<id\>/profile | POST | Profile run performance | bia_psa:optimise |
| /api/bia/psa/optimisations/\<id\>/conflicts | GET | Detect constraint conflicts (IIS) | bia_psa:optimise |
| /api/bia/psa/optimisations/stochastic | POST | SAA stochastic LP | bia_psa:optimise |
| /api/bia/psa/optimisations/robust | POST | Robust LP with uncertainty sets | bia_psa:optimise |
| /api/bia/psa/optimisations/pareto | POST | Multi-objective Pareto front | bia_psa:optimise |
| /api/bia/psa/optimisations/rolling | POST | Rolling-horizon re-optimisation | bia_psa:optimise |
| /api/bia/psa/optimisations/benders | POST | Benders decomposition | bia_psa:optimise |
| /api/bia/psa/optimisations/genetic | POST | Genetic algorithm with adaptive mutation | bia_psa:optimise |
| /api/bia/psa/resource-allocation | POST | Resource allocation LP | bia_psa:optimise |
| /api/bia/psa/rl/train | POST | Train RL policy | bia_psa:optimise |
| /api/bia/psa/rl/\<policy_id\>/apply | POST | Apply RL policy to state | bia_psa:optimise |
| /api/bia/psa/scenario-trees | POST | Build multi-stage scenario tree | bia_psa:optimise |
| /api/bia/psa/scenario-trees/\<id\>/solve | POST | Solve scenario tree (EVPI, EVSS) | bia_psa:optimise |
| /api/bia/psa/recommendations | GET/POST | List/generate | bia_psa:recommendations |
| /api/bia/psa/recommendations/\<id\>/approve | POST | Approve | bia_psa:approve |
| /api/bia/psa/recommendations/\<id\>/act | POST | Act on recommendation | bia_psa:approve |
| /api/bia/psa/recommendations/\<id\>/explain | GET | SHAP/LIME/counterfactual explanation | bia_psa:recommendations |
| /api/bia/psa/whatif | GET/POST | What-if analysis | bia_psa:whatif |
| /api/bia/psa/decisions | GET/POST | Decision log | bia_psa:decisions |
| /api/bia/psa/cba | POST | Cost-benefit analysis (NPV, IRR, MIRR) | bia_psa:optimise |
| /api/bia/psa/shadow-prices/\<lp_run_id\> | GET | Shadow price / dual variable report | bia_psa:optimise |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| unapproved_recommendation_action_denied | approval_state=pending | deny |
| rejected_recommendation_cannot_be_acted | approval_state=rejected | deny |
| whatif_requires_baseline | No baseline model | deny |
| hard_constraint_violation_denied | Hard constraint violated | deny |
| decision_explainability_required | No rationale | deny |
| archived_analysis_read_only | state=archived | deny |

## Data Models
- `OptimisationResponse` — id, name, optimisation_type, state, objective_type, constraints, variables, result
- `RecommendationResponse` — id, optimisation_id, recommendation_type, approval_state, actions, impact_estimate
- `WhatIfResponse` — id, name, baseline_model_id, parameters, state, results
- `DecisionRecord` — id, decision_type, recommendation_id, rationale, decided_by, outcome
- `ParetoFront` (`bia_psa_pareto_front`) — point_id, objective_values, decision_variables, dominated_hypervolume
- `RLPolicy` (`bia_psa_rl_policy`) — policy_id, algorithm, episodes, weights (JSON), env_spec
- `PerfProfile` (`bia_psa_perf_profile`) — phase_timings_ms, bottleneck_phase, memory_peak_mb, recommendation
- `ScenarioTree` — stages, branching_factor, probability-weighted adjacency list, EVPI, EVSS

## Streaming Events
- `optimisation_started`, `optimisation_completed`, `decision_recorded`
- `recommendation_generated`, `recommendation_approved`, `recommendation_rejected`
- `whatif_simulated`, `constraint_added`, `constraint_violated`, `allocation_optimised`
- `rl_policy_trained`, `pareto_front_computed`, `scenario_tree_built`, `cba_computed`

---

## World-Class Enhancements (v2.0)

| # | Enhancement | Category | Key Output |
|---|-------------|----------|------------|
| 1 | **Stochastic LP via SAA** | Solver Depth | Mean optimal ± CI, VaR, CVaR over N scenario samples |
| 2 | **Multi-Objective Pareto Front (NSGA-II)** | Solver Depth | Full Pareto front with dominated-hypervolume metric |
| 3 | **Reinforcement Learning Policy Optimisation** | AI/ML Integration | Tabular Q-learning + PPO; persist trained policy weights |
| 4 | **Automatic Bound-Tightening (AC-3 + Interval Arithmetic)** | Solver Efficiency | Tightened bounds, reduction %, preprocessing time; hooks into `run_optimisation` |
| 5 | **Dual Price / Shadow Price Reporting** | Decision Explainability | Per-constraint marginal value, RHS ranging, 100% rule check |
| 6 | **Genetic Algorithm with Adaptive Mutation** | Solver Depth | Tournament selection, self-adaptive σ-mutation, convergence curve; strategies: `fixed\|adaptive\|cma_es` |
| 7 | **Robust LP with Ellipsoidal Uncertainty Sets** | Solver Depth | SOCP reformulation, robustness_level ∈ [0,1], price-of-robustness |
| 8 | **Benders Decomposition** | Solver Scalability | Master IP + parallel async LP subproblems; per-iteration gap history |
| 9 | **SHAP-style Recommendation Explainability** | Decision Explainability | Shapley values, waterfall data, counterfactual, NL summary; methods: `shap\|lime\|counterfactual` |
| 10 | **Resource Allocation Optimiser** | Vertical Feature | Auto-formulated transportation/assignment LP; Gantt-compatible schedule output |
| 11 | **Rolling-Horizon Re-optimisation** | Temporal Dynamics | MPC-style receding window; period decisions, cumulative objective, stability metric |
| 12 | **Multi-Stage Scenario Tree** | Solver Depth | Branching uncertainty paths; EVPI and EVSS; reduction methods: `moment_matching\|forward_selection\|kmeans` |
| 13 | **Cost-Benefit Analysis with DCF** | Decision Support | NPV, IRR (Newton-Raphson), MIRR, profitability index, discounted payback, sensitivity ±10%/20% |
| 14 | **Constraint Conflict Detection (IIS)** | Model Quality | Deletion-filter IIS algorithm; conflict strength metric; suggested relaxations |
| 15 | **Optimisation Performance Profiler** | Operational Excellence | Per-phase timings (load, presolve, LP relaxation, branching, cuts, primal); auto-surfaced when `duration_ms > threshold` |

---

## New Methods

### 1. Stochastic Linear Programme (SAA)

```python
svc = PrescriptiveAnalyticsService(tenant_id="acme", actor_id="analyst1")

result = await svc.stochastic_linear_programme(
    tenant_id="acme",
    scenario_samples=[
        {"objective": [1.2, 0.8], "rhs": [100, 80]},
        {"objective": [1.0, 1.1], "rhs": [95, 85]},
    ],
    coefficients={"constraint_matrix": [[1, 2], [3, 1]], "rhs": [100, 80]},
    bounds={"lower": [0.0, 0.0], "upper": [50.0, 40.0]},
    n_samples=200,
    method="simplex",
)
# result["mean_optimal_value"]  -> float
# result["confidence_interval"] -> [lower, upper]
# result["var_95"]              -> Value-at-Risk at 95th percentile
# result["cvar_95"]             -> Conditional-VaR
```

### 2. Multi-Objective Pareto Front

```python
pareto = await svc.pareto_optimisation(
    tenant_id="acme",
    objectives=[
        {"name": "cost",         "direction": "minimize", "coefficients": [1.0, 2.0]},
        {"name": "service_level","direction": "maximize", "coefficients": [0.5, 1.5]},
    ],
    constraints=[{"lhs": [1, 1], "relation": "<=", "rhs": 100}],
    variables=[{"name": "x1", "lower": 0, "upper": 60}, {"name": "x2", "lower": 0, "upper": 60}],
    population_size=100,
    generations=200,
)
# pareto["front"]               -> list of {point_id, objective_values, decision_variables}
# pareto["dominated_hypervolume"]-> float (quality metric)
```

### 3. Resource Allocation Optimiser

```python
allocation = await svc.resource_allocation(
    tenant_id="acme",
    resources=[
        {"id": "r1", "type": "analyst", "capacity": 40, "cost_per_unit": 150.00},
        {"id": "r2", "type": "analyst", "capacity": 40, "cost_per_unit": 130.00},
    ],
    tasks=[
        {"id": "t1", "demand": 25, "priority": 1, "deadline": "2026-06-30"},
        {"id": "t2", "demand": 30, "priority": 2, "deadline": "2026-07-15"},
    ],
    capacities={"r1": 40, "r2": 40},
    costs={"r1": 150.00, "r2": 130.00},
    time_horizon=30,
    owner_id="pm1",
)
# allocation["assignment_matrix"]      -> {task_id: {resource_id: units}}
# allocation["utilisation_by_resource"]-> {resource_id: pct}
# allocation["unmet_demand"]           -> Decimal
# allocation["total_cost"]             -> Decimal
# allocation["gantt_schedule"]         -> list of {task, resource, start, end}
```

### 4. Constraint Conflict Detection (IIS)

```python
conflict = await svc.detect_constraint_conflicts(
    tenant_id="acme",
    model_id="opt-uuid-here",
)
# conflict["is_feasible"]          -> bool
# conflict["iis_constraints"]      -> list of constraint descriptors in minimal infeasible set
# conflict["conflict_pairs"]       -> [(c_i, c_j, conflict_strength), ...]
# conflict["suggested_relaxations"]-> [{"constraint_id": ..., "relax_by": float}, ...]
```

### 5. Cost-Benefit Analysis with DCF

```python
cba = await svc.cost_benefit_analysis(
    tenant_id="acme",
    initial_investment=500_000.00,
    cash_flows=[120_000, 150_000, 180_000, 200_000, 220_000],  # years 1-5
    discount_rate=0.10,
    inflation_rate=0.03,
    risk_adjustments=[
        {"scenario": "base",      "weight": 0.6},
        {"scenario": "downside",  "weight": 0.25, "cash_flow_multiplier": 0.75},
        {"scenario": "upside",    "weight": 0.15, "cash_flow_multiplier": 1.20},
    ],
    owner_id="cfo1",
)
# cba["npv"]                    -> Decimal
# cba["irr"]                    -> float (Newton-Raphson)
# cba["mirr"]                   -> float
# cba["profitability_index"]    -> float
# cba["discounted_payback_yrs"] -> float
# cba["break_even_point"]       -> float
# cba["npv_sensitivity"]        -> {"-20%": ..., "-10%": ..., "+10%": ..., "+20%": ...}
```

---

## Quick Start

```python
from capabilities.bia.psa.service import PrescriptiveAnalyticsService

svc = PrescriptiveAnalyticsService(tenant_id="acme", actor_id="user1")

# Solve a linear programme
lp = await svc.linear_programme(
    tenant_id="acme",
    coefficients={
        "objective": [2.0, 3.0],
        "constraint_matrix": [[1, 2], [3, 1]],
        "rhs": [14, 14],
    },
    bounds={"lower": [0.0, 0.0], "upper": [None, None]},
    method="simplex",
)
print(lp["optimal_x"], lp["objective_value"])

# Generate and approve a recommendation
rec = await svc.generate_recommendation(
    tenant_id="acme",
    optimisation_id=lp["id"],
    name="Increase x2 allocation",
    recommendation_type="resource_reallocation",
    description="x2 has higher margin; increase allocation by 15%.",
    owner_id="user1",
    impact_estimate={"revenue_increase_pct": 8.5},
)
await svc.approve_recommendation("acme", rec["id"], approver_id="manager1")
await svc.act_on_recommendation("acme", rec["id"], actor_id="user1")
```

## Edge Cases Handled
- Recommendations must be approved before any action — no bypass path exists
- Rejected recommendations cannot be acted on — new recommendation required
- What-if analysis requires a baseline to ensure deltas are meaningful
- Hard constraint violations abort optimisation with explicit revision instruction
- Archived analyses are read-only — new analysis required for further work
- Decision explainability is mandatory — empty rationale is rejected at enforcement layer
- IIS detection returns empty `iis_constraints` for feasible models (no false positives)
- Rolling-horizon will not advance past `horizon_periods` — caller controls re-invocation

## Composability Notes
- Consumes `bia_pda` model forecasts as optimisation objectives or stochastic parameter distributions
- `wflo` manages multi-step recommendation approval chains
- `bia_rpt` generates decision and recommendation audit reports
- `moni` tracks recommendation acceptance rates, optimisation run times, and perf profile trends
- `rl_policy` trained here can be consumed by `bia_dsa` for real-time decision scoring
