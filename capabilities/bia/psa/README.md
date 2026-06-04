# Prescriptive Analytics

## Overview
The Prescriptive Analytics capability (bia_psa) provides optimisation engines (LP, IP, GA, RL), decision support with explainability, recommendation action management with approval workflows, and what-if analysis — all tenant-scoped with mandatory governance and full audit.

## Capability ID
`bia_psa`

## Provides
- optimisation_engine: 7 optimisation types with constraint and objective management
- decision_support_system: Typed decision recording with mandatory explainability
- recommendation_actions: Generated recommendations with approval-gated actioning
- whatif_analysis: Baseline-anchored scenario parameter sweeps
- constraint_management: Hard/soft/preference constraint enforcement
- multi_objective_analysis: Pareto-optimal multi-objective optimisation
- allocation_optimisation: Resource allocation and scheduling optimisation
- process_improvement_recommendations: Process change and investment recommendations

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
| /api/bia/psa/optimisations/<id>/run | POST | Run optimisation | bia_psa:optimise |
| /api/bia/psa/recommendations | GET/POST | List/generate | bia_psa:recommendations |
| /api/bia/psa/recommendations/<id>/approve | POST | Approve | bia_psa:approve |
| /api/bia/psa/recommendations/<id>/act | POST | Act on recommendation | bia_psa:approve |
| /api/bia/psa/whatif | GET/POST | What-if analysis | bia_psa:whatif |
| /api/bia/psa/decisions | GET/POST | Decision log | bia_psa:decisions |

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
- OptimisationResponse: id, name, optimisation_type, state, objective_type, constraints, variables, result
- RecommendationResponse: id, optimisation_id, recommendation_type, approval_state, actions, impact_estimate
- WhatIfResponse: id, name, baseline_model_id, parameters, state, results
- DecisionRecord: id, decision_type, recommendation_id, rationale, decided_by, outcome

## Streaming Events
- optimisation_started, optimisation_completed, decision_recorded
- recommendation_generated, recommendation_approved, recommendation_rejected
- whatif_simulated, constraint_added, constraint_violated, allocation_optimised

## Edge Cases Handled
- Recommendations must be approved before any action can be taken — no bypass
- Rejected recommendations cannot be acted on — new recommendation required
- What-if analysis requires a baseline to ensure delta is meaningful
- Hard constraint violations abort the optimisation with explicit revision instruction
- Archived analyses are read-only — new analysis required for further work
- Decision explainability is mandatory — rationale cannot be empty

## Composability Notes
- Consumes bia_pda model forecasts as optimisation objectives
- wflo manages multi-step recommendation approval chains
- bia_rpt generates decision and recommendation audit reports
- moni tracks recommendation acceptance rates and optimisation run times
