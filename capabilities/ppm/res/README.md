# Resource Management

## Overview
Resource Management (res) manages the full resource lifecycle: pool registration, skill cataloguing with evidence-backed proficiency, allocation to projects with over-allocation controls, capacity planning, utilisation band tracking, demand forecasting, leave management, and cost rate governance with finance-approval gates.

## Capability ID
`ppm_res`

## Provides
| Service | Description |
|---------|-------------|
| resource_pool_management | Register and manage human, equipment, facility, and other resource types |
| skill_matching_engine | Match resources to requirements using configurable algorithms |
| capacity_planning | Staffing, hiring, contractor, training, and re-deployment plans |
| utilisation_tracking | Real-time utilisation snapshots with under/optimal/near/over/critical bands |
| demand_forecasting | 30d to multi-year demand vs supply FTE gap analysis |
| resource_allocation_workflow | Allocation with over-allocation approval gates |
| leave_and_availability_management | Leave recording and availability calendar |
| cost_rate_management | Standard, billing, overtime, contractor, and blended rates |
| resource_demand_vs_supply_analysis | Gap analysis across horizons and skill filters |
| hiring_and_contractor_planning | Capacity plan types for staffing and contractor scenarios |

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Audit logging |
| mten | Tenant scoping |
| conf | Configuration and thresholds |
| ntfy | Over-allocation and demand alert notifications |
| wflo | Leave and over-allocation approval workflows |
| schd | Integration with scheduling engine |
| nlpc | Natural language skill search |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| resources.supported_types | 7 | human, equipment, material, facility, software_license, subcontractor, budget_pool |
| skills.supported_proficiency_levels | 5 | beginner through master |
| allocations.supported_matching_algorithms | 5 | exact, weighted, availability, cost-optimised, balanced |
| governance.over_allocation_requires_manager_approval | true | Capacity control |
| governance.cost_rate_change_requires_finance_approval | true | Financial control |
| governance.skill_proficiency_fabrication_denied | true | Integrity control |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-res/resources | GET/POST | Resource pool | ppm_res:resources |
| /ppm-res/skills | GET/POST | Skill catalog | ppm_res:skills |
| /ppm-res/skill-match | POST | Skill matching | ppm_res:skill_match |
| /ppm-res/allocations | GET/POST | Allocation console | ppm_res:allocations |
| /ppm-res/capacity | GET/POST | Capacity planning | ppm_res:capacity |
| /ppm-res/utilisation | GET/POST | Utilisation tracker | ppm_res:utilisation |
| /ppm-res/demand | GET/POST | Demand forecast | ppm_res:demand |
| /ppm-res/availability | GET | Availability calendar | ppm_res:availability |
| /ppm-res/rates | GET/POST | Cost rate table | ppm_res:rates |
| /ppm-res/agents | GET/POST | Agent workbench | ppm_res:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| resource_type_supported | resource_type not in list | deny |
| over_allocation_requires_approval | over_allocated + no approval | deny |
| skill_proficiency_fabrication_denied | skill_proficiency_fabrication=True | deny |
| cost_rate_finance_approval_required | finance_approval_present=False | deny |
| leave_approval_required | approval_present=False | deny |
| cross_tenant_resource_access_denied | cross_tenant_access=True | deny |
| resource_batch_requires_bytewax | event_stream != "bytewax" | deny |

## Data Models
- **Resource** — id, name, resource_type, status, department, owner_id, cost_rate, cost_rate_type
- **ResourceSkill** — id, resource_id, skill_name, proficiency_level, years_experience
- **ResourceAllocation** — id, resource_id, project_id, task_id, allocation_pct, manager_approval_reference
- **CapacityPlan** — id, plan_type, name, horizon, demand_data, supply_data, gap_analysis
- **UtilisationSnapshot** — id, resource_id, snapshot_period, allocated_hours, utilisation_pct, utilisation_band
- **DemandForecast** — id, horizon, resource_type, forecast_demand_fte, current_supply_fte, gap_fte
- **LeaveRecord** — id, resource_id, leave_type, start_date, end_date, approval_reference
- **CostRate** — id, resource_id, rate_type, rate_amount, currency, effective_date, finance_approval_reference
- **ResourceAgent** — id, name, runtime, role, scope

## Streaming Events
- `resource_created`, `resource_updated`, `skill_added`, `allocation_confirmed`
- `allocation_cancelled`, `capacity_plan_published`, `utilisation_snapshot_taken`
- `demand_forecast_generated`, `over_allocation_detected`, `leave_recorded`
- `cost_rate_updated`, `agent_registered`

## Edge Cases Handled
- Skill proficiency claims without evidence are rejected to prevent padding CVs
- Over-allocation (>100%) is permitted only with explicit manager approval
- Finance approval is required for any cost rate change, not just initial setup
- Utilisation bands are auto-calculated from allocated/available hours ratio
- Skill matching uses exact set intersection for "exact_skill_match" algorithm
- Demand gap is auto-calculated as forecast_demand_fte minus current_supply_fte

## Composability Notes
- Skill matching feeds **ppm_pps** resource assignment recommendations
- Cost rates feed **ppm_pac** labour cost transactions
- Utilisation data feeds **ppm_pan** capacity heat maps
- Leave records block **ppm_pps** allocations during absence periods

## World-Class Enhancements (v2.0)

1. **Skill Ontology Graph** — RDF-style skill inheritance (react isa frontend isa software_engineering); replaces fragile prefix fuzzy match
2. **Continuous Utilisation Time-Series** — daily-granularity time-series store enabling rolling trends, anomaly detection, and burn-rate projection
3. **CP-SAT Optimal Assignment** — OR-Tools constraint solver replaces greedy team_builder; optimises skill coverage, balance, cost, and leave avoidance simultaneously
4. **Real-Time Over-Allocation Webhooks** — background monitor fires ntfy events at 80/95/100% thresholds without requiring an explicit check call
5. **Evidence-Verified Skill Proficiency Pipeline** — async verification queue with claimed → verified state machine; prevents non-empty-string bypass
6. **Multi-Currency Cost Normalisation** — CurrencyNormalisationService converts all rates to tenant base currency at time-of-record exchange rates
7. **Leave Impact Propagation** — leave creation auto-computes impacted tasks, delay risk, and backfill cost; publishes leave_impact_computed for ppm_pps
8. **Carbon / Sustainability Accounting** — carbon_kg_per_day on equipment/facility resources; feeds ESG dashboards and CSRD compliance reporting
9. **Probabilistic Demand Forecasting** — Monte Carlo over historical variance and pipeline win rates; outputs P50/P80/P90 FTE bands instead of a point estimate
10. **Skill Endorsement Social Graph** — peer endorsement model gated on expert proficiency; endorsement count and network centrality as additional signals
11. **Allocation Marketplace (Internal Gig Board)** — bench resources (>30% idle) post availability; PMs browse and request; estimated 10-15% bench cost reduction
12. **Bi-Temporal Cost Rate Versioning** — valid_time + transaction_time model; supports retroactive corrections without losing decision-time audit trail
13. **Role-Based Capacity Pools** — allocations target role/grade pools (e.g. senior_engineer); engine resolves best available member at scheduling time
14. **Automated Capacity Plan Generation** — generate_capacity_plan computes demand/supply from live data and applies hire/contract/redeploy/train closure strategies
15. **RBAC Delegation Chains** — time-scoped delegation from resource manager to team lead; full chain validated at _enforce call time for SOX compliance

## New Methods

### `bulk_allocate_resources` — batch project staffing

```python
result = await svc.bulk_allocate_resources(
    allocation_specs=[
        {"resource_id": "r-001", "project_id": "p-42", "task_id": "t-7",
         "start_date": "2026-07-01", "end_date": "2026-09-30",
         "allocation_pct": 80, "evidence_reference": "sow-2026-42"},
        {"resource_id": "r-002", "project_id": "p-42", "task_id": "t-8",
         "start_date": "2026-07-01", "end_date": "2026-08-31",
         "allocation_pct": 50},
    ],
    tenant_id="tenant-acme",
)
# {"created_count": 2, "error_count": 0, "allocations": [...], "errors": []}
```

### `team_builder` — skill-matched team assembly

```python
team = await svc.team_builder(
    project_id="p-42",
    required_skills=["python", "postgresql", "fastapi"],
    team_size=3,
    tenant_id="tenant-acme",
)
# {"team_id": "team-p-42-2026-06-12", "actual_size": 3,
#  "skill_coverage": 3, "members": [...]}
```

### `demand_gap_analysis` — live capacity deficit check

```python
gap = await svc.demand_gap_analysis(tenant_id="tenant-acme")
# {"total_demand_units": 42.5, "total_capacity_units": 38.0,
#  "gap": 4.5, "status": "deficit", "computed_at": "2026-06-12"}
```
