# Prescriptive Analytics — World-Class Improvements

**Capability**: `bia_psa` | **Domain**: `bia` | **Date**: 2026-06-11

---

## 1. Stochastic LP via Sample-Average Approximation (SAA)

**Category**: Solver Depth

**Justification**: Deterministic LP ignores parameter uncertainty. Production optimisers (AIMMS, Gurobi Cloud) solve SAA formulations that average over N sampled realisations of uncertain parameters, producing solutions robust to demand/cost variability. Current `linear_programme()` has no stochastic variant.

**Implementation**: Add `stochastic_linear_programme(tenant_id, scenario_samples, coefficients, bounds, n_samples, method)`. Draw `n_samples` from parameter distributions, solve each LP, return mean optimal value ± confidence interval, value-at-risk (VaR), and conditional-VaR. Back probability calculations with exact quantile aggregation, not just percentile approximation.

**Competitor Reference**: IBM CPLEX Stochastic Optimizer, AIMMS SP module.

---

## 2. Multi-Objective Pareto Front Generation

**Category**: Solver Depth

**Justification**: Real decisions balance competing objectives (cost vs. service level). NSGA-II and ε-constraint methods enumerate the Pareto front so decision-makers choose a preferred trade-off point rather than accepting a single collapsed scalar. No Pareto capability exists today.

**Implementation**: Add `pareto_optimisation(tenant_id, objectives, constraints, variables, population_size, generations)`. Implement NSGA-II with non-dominated sorting + crowding distance. Return full front as list of `{point_id, objective_values, decision_variables}` plus dominated-hypervolume metric. Store front as `bia_psa_pareto_front` entity.

**Competitor Reference**: Pyomo + NSGA-II (DEAP), Gurobi multi-objective, MATLAB gamultiobj.

---

## 3. Reinforcement Learning Policy Optimisation

**Category**: AI/ML Integration

**Justification**: Static LP/IP cannot adapt to sequential decisions under non-stationary conditions (inventory replenishment, dynamic pricing). RL-based prescriptive systems (used by Amazon for inventory, Uber for surge pricing) generate policies that improve over time. The capability claims RL support but has no implementation.

**Implementation**: Add `train_rl_policy(tenant_id, env_spec, algorithm, episodes, reward_fn)` and `apply_rl_policy(tenant_id, policy_id, state)`. Env spec encodes state/action/reward structure. Implement tabular Q-learning and a PPO shell that dispatches to a local Ollama-served RL agent. Persist trained policy weights (JSON-serialised) as `bia_psa_rl_policy`.

**Competitor Reference**: AWS SageMaker RL, Google Vertex AI RL, RLlib.

---

## 4. Automatic Bound-Tightening via Constraint Propagation

**Category**: Solver Efficiency

**Justification**: Loose variable bounds increase branch-and-bound search trees by orders of magnitude. Commercial solvers (Gurobi, CPLEX) run preprocessing passes that tighten bounds before solving. Current IP/LP ignores this, making large problems slow.

**Implementation**: Add `tighten_bounds(tenant_id, model_id)` preprocessing step called automatically before `run_optimisation()` for `integer_programming` type. Implement AC-3 arc-consistency + interval arithmetic propagation. Return `{original_bounds, tightened_bounds, reduction_pct, preprocessing_time_ms}`. Integrates as a hook in `run_optimisation`.

**Competitor Reference**: CPLEX presolve, Gurobi bound tightening.

---

## 5. Dual Price / Shadow Price Reporting with Economic Interpretation

**Category**: Decision Explainability

**Justification**: Dual values (shadow prices) quantify the marginal cost of each constraint — critical for resource pricing decisions. The current `linear_programme()` returns zeroed dual values, defeating the primary economic value of LP for operations managers.

**Implementation**: Enhance `linear_programme()` to compute exact dual variables using the KKT conditions of the simplex final tableau. Add `shadow_price_report(tenant_id, lp_run_id)` returning per-constraint `{marginal_value, allowable_increase, allowable_decrease, economic_interpretation}`. Include right-hand-side ranging (100% rule check for simultaneous changes).

**Competitor Reference**: LINDO shadow price report, SAS OR dual analysis output.

---

## 6. Genetic Algorithm with Adaptive Mutation Rate

**Category**: Solver Depth

**Justification**: Fixed mutation rates in GA cause premature convergence on early generations or slow convergence on later ones. Adaptive GA (as in NSGA-III and CMA-ES) tracks population diversity and adjusts mutation probability dynamically, converging 2-4× faster on combinatorial problems.

**Implementation**: Add `genetic_optimisation(tenant_id, chromosome_spec, fitness_fn_desc, population_size, max_generations, crossover_rate, mutation_strategy)`. Implement tournament selection, single-point + uniform crossover, and self-adaptive σ-mutation. Track diversity metric (average Hamming distance). Expose convergence curve in result. `mutation_strategy`: `fixed | adaptive | cma_es`.

**Competitor Reference**: DEAP adaptive GA, pymoo CMA-ES.

---

## 7. Robust Optimisation with Ellipsoidal Uncertainty Sets

**Category**: Solver Depth

**Justification**: Worst-case robust LP (Ben-Tal & Nemirovski) converts an uncertain LP into a deterministic cone programme solvable without sampling, giving hard guarantees rather than probabilistic ones. Used in finance (portfolio CVaR), energy scheduling, and supply chain design.

**Implementation**: Add `robust_linear_programme(tenant_id, coefficients, uncertainty_sets, bounds, robustness_level)`. Transform uncertain constraints using ellipsoidal uncertainty sets to second-order cone constraints. `robustness_level` ∈ [0,1] interpolates between nominal and fully robust solutions. Return nominal vs. robust optimal values and the "price of robustness".

**Competitor Reference**: CVXPY robust LP, Mosek SOCP, Gurobi cone programming.

---

## 8. Decomposition-Based Solving (Benders / Dantzig-Wolfe)

**Category**: Solver Scalability

**Justification**: Large-scale MIP instances (>10k variables) become intractable for direct branch-and-bound. Benders decomposition splits the problem into a master IP and LP subproblems, enabling parallel subproblem solving. Used in production by logistics companies (UPS, DHL route optimisation).

**Implementation**: Add `benders_decompose(tenant_id, master_problem, subproblem_list, max_iterations, convergence_tol)`. Implement classical Benders: solve master, generate optimality/feasibility cuts from subproblems, iterate. Return per-iteration gap history and final solution. Expose `parallelism` parameter to run subproblems concurrently via `asyncio.gather`.

**Competitor Reference**: CPLEX Benders, Google OR-Tools CP-SAT decomposition.

---

## 9. Explainable Recommendation Engine with SHAP-style Attribution

**Category**: Decision Explainability

**Justification**: Black-box recommendations create adoption resistance. Regulators (EU AI Act Article 86) require explanations for automated decisions. SHAP values attribute each recommendation score to input features, enabling "why this action?" audits. Current MCDA returns scores without feature attribution.

**Implementation**: Add `explain_recommendation(tenant_id, rec_id, explanation_method)` where `method ∈ {shap, lime, counterfactual}`. For SHAP: compute exact Shapley values over criterion weights via cooperative game theory formula. Return `{feature_contributions, waterfall_chart_data, counterfactual_nearest_alternative, natural_language_summary}`.

**Competitor Reference**: IBM OpenScale, Salesforce Einstein Explainability, Microsoft InterpretML.

---

## 10. Resource Allocation Optimiser with Capacity Constraints

**Category**: Vertical Feature

**Justification**: Resource allocation (staff scheduling, budget allocation, machine assignment) is the #1 use case for prescriptive analytics in enterprise. A specialised interface surfaces the LP formulation transparently, making it accessible without solver expertise.

**Implementation**: Add `resource_allocation(tenant_id, resources, tasks, capacities, costs, time_horizon, owner_id)`. Formulate automatically as transportation/assignment LP. Resources: `{id, type, capacity, cost_per_unit}`. Tasks: `{id, demand, priority, deadline}`. Return `{assignment_matrix, utilisation_by_resource, unmet_demand, total_cost}` as `Decimal` values. Include Gantt-compatible schedule output.

**Competitor Reference**: Oracle Fusion Resource Optimizer, SAP IBP Supply Optimisation.

---

## 11. Rolling-Horizon Re-optimisation

**Category**: Temporal Dynamics

**Justification**: One-shot optimisation becomes stale as data changes. Rolling-horizon (receding-horizon) control re-solves with a sliding window, incorporating new observations — standard practice in MPC (Model Predictive Control) for manufacturing and energy management.

**Implementation**: Add `rolling_horizon_optimisation(tenant_id, model_id, horizon_periods, re_solve_interval, data_update_fn_desc)`. Simulate N periods: at each step, update parameter estimates, re-solve LP/IP for the remaining horizon, execute only the first-period decision, advance time. Return `{period_decisions, cumulative_objective, stability_metric}` (stability = variance in period-over-period decisions).

**Competitor Reference**: Siemens SIMATIC MPC, Honeywell Profit Controller.

---

## 12. Scenario Tree Construction for Multi-Stage Stochastic Programming

**Category**: Solver Depth

**Justification**: Two-stage SP collapses all uncertainty to one point in time. Real decisions are sequential under evolving uncertainty (multi-stage). Scenario trees represent branching uncertainty paths; solvers exploit the tree structure for polynomial-time dynamic programming.

**Implementation**: Add `build_scenario_tree(tenant_id, stages, branching_factor, parameter_distributions, reduction_method)` and `solve_scenario_tree(tenant_id, tree_id, objective)`. `reduction_method ∈ {moment_matching, forward_selection, kmeans}`. Store tree as adjacency list with probability-weighted nodes. Return `{here_and-now_decisions, recourse_actions_by_node, expected_value_of_perfect_information (EVPI), expected_value_of_stochastic_solution (EVSS)}`.

**Competitor Reference**: GAMS stochastic programming, SPInE toolkit.

---

## 13. Cost-Benefit Analysis Engine with Discounted Cash Flow

**Category**: Decision Support

**Justification**: Decision support without financial quantification is advisory at best. A DCF-integrated CBA engine translates optimisation outcomes into NPV, IRR, and payback period — the language of capital allocation committees. Current `decision_tree_analysis` only computes simple EMV.

**Implementation**: Add `cost_benefit_analysis(tenant_id, initial_investment, cash_flows, discount_rate, inflation_rate, risk_adjustments, owner_id)`. All monetary values as `Decimal`. Compute: NPV, IRR (Newton-Raphson), modified IRR (MIRR), profitability index, discounted payback period, break-even point. Apply `risk_adjustments` as scenario weights. Return sensitivity of NPV to ±10%/20% discount rate and cash-flow changes.

**Competitor Reference**: Oracle Primavera CBA, Microsoft Azure FinOps ROI Calculator.

---

## 14. Constraint Conflict Detection and Diagnosis

**Category**: Model Quality

**Justification**: Infeasible models (conflicting constraints) are the most common modelling error and the hardest to diagnose. Irreducible Infeasible Subsystem (IIS) isolation — as in CPLEX `refineConflict` — identifies the minimal set of conflicting constraints, cutting debug time from hours to seconds.

**Implementation**: Add `detect_constraint_conflicts(tenant_id, model_id)`. Implement a deletion filter IIS algorithm: iteratively remove constraints and re-check feasibility; report the minimal infeasible set. For each conflicting pair, compute a "conflict strength" metric (degree of violation if both are relaxed to equality). Return `{is_feasible, iis_constraints, conflict_pairs, suggested_relaxations}`.

**Competitor Reference**: CPLEX `refineConflict`, Gurobi `computeIIS`, SCIP conflict analysis.

---

## 15. Optimisation Performance Profiler with Bottleneck Attribution

**Category**: Operational Excellence

**Justification**: Slow optimisation runs block interactive decision support. Without profiling, engineers cannot distinguish constraint density, variable count, solver choice, or data retrieval as the bottleneck. Commercial platforms (Gurobi Compute Server, AIMMS PRO) expose per-phase timing.

**Implementation**: Add `profile_optimisation(tenant_id, model_id)`. Instrument `run_optimisation` with per-phase timers: data loading, preprocessing (bound tightening, scaling), LP relaxation, branching, cut generation, heuristic primal, cleanup. Return `{phase_timings_ms, bottleneck_phase, memory_peak_mb, recommendation: str}`. Persist as `bia_psa_perf_profile` for trend analysis across runs. Automatically surface when `duration_ms > threshold`.

**Competitor Reference**: Gurobi Compute Server profiling, CPLEX performance tuning advisor.
