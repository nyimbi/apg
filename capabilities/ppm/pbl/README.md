# Project Baseline Management

## Overview
Project Baseline Management (pbl) establishes and protects the scope, schedule, and cost baselines for projects. It enforces formal change control, calculates earned value metrics, detects variance threshold breaches, and prevents retroactive baseline manipulation — providing the performance measurement baseline required for EVM compliance.

## Capability ID
`ppm_pbl`

## Provides
| Service | Description |
|---------|-------------|
| scope_baseline_management | Approved scope baseline with WBS linkage |
| schedule_baseline_management | Approved schedule baseline with milestone lock-in |
| cost_baseline_management | Approved cost baseline with budget-at-completion |
| change_control_workflow | Formal change request, impact assessment, and approval pipeline |
| earned_value_analysis | PV, EV, AC, SV, CV, SPI, CPI, EAC, ETC, VAC |
| earned_schedule_analysis | ES, SPI(t), SV(t), IEAC(t) — time-domain EVM metrics |
| baseline_variance_tracking | Schedule and cost variance with threshold alerting |
| change_impact_assessment | Schedule, cost, scope, and risk impact quantification |
| baseline_approval_workflow | Designated approver enforcement for baseline promotion |
| integrated_baseline_review | Cross-baseline consistency checks with IBR health index |
| performance_measurement_baseline | PMB for EVM reporting |
| baseline_lock_management | Write-protect approved baselines; enforce freeze periods |
| cr_dependency_graph | DAG linking change requests via blocks/depends_on/supersedes |
| completion_forecasting | EAC (3 methods), TCPI, VAC with method recommendation |
| portfolio_baseline_summary | Cross-project CPI/SPI rollup with risk-tiered project list |
| baseline_deviation_scoring | Composite BDS for dashboard KPI triage |
| baseline_version_history | Append-only version snapshots with diff capability |

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication and role enforcement |
| audl | Immutable audit log for baseline changes |
| mten | Tenant context |
| conf | Variance thresholds and workflow config |
| ntfy | Variance breach notifications |
| wflo | Change request approval workflow |
| comp | EVM compliance and regulatory reporting |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| baselines.supported_types | 7 | scope, schedule, cost, quality, resource, risk, integrated |
| change_control.supported_priorities | 5 | low to emergency |
| earned_value.supported_forecasting_methods | 4 | typical, atypical, scheduled, custom |
| governance.retroactive_baseline_edit_denied | true | Integrity control |
| governance.ev_manipulation_denied | true | EVM integrity |
| governance.change_control_bypass_denied | true | Mandatory change process |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-pbl/baselines | GET/POST | Baseline list and creation | ppm_pbl:baselines |
| /ppm-pbl/baselines/lock | POST | Lock/unlock baseline | ppm_pbl:admin |
| /ppm-pbl/baselines/ibr | GET | Integrated Baseline Review | ppm_pbl:baselines |
| /ppm-pbl/baselines/versions | GET | Baseline version history | ppm_pbl:baselines |
| /ppm-pbl/baselines/freeze | POST | Set freeze period | ppm_pbl:admin |
| /ppm-pbl/changes | GET/POST | Change request queue | ppm_pbl:changes |
| /ppm-pbl/changes/graph | GET | CR dependency graph | ppm_pbl:changes |
| /ppm-pbl/changes/link | POST | Link change requests | ppm_pbl:changes |
| /ppm-pbl/impact | GET/POST | Impact assessment | ppm_pbl:impact |
| /ppm-pbl/ev | GET/POST | Earned value dashboard | ppm_pbl:ev |
| /ppm-pbl/ev/forecast | GET | EAC/TCPI/VAC forecast | ppm_pbl:ev |
| /ppm-pbl/ev/earned-schedule | GET | Earned Schedule metrics | ppm_pbl:ev |
| /ppm-pbl/variance | GET | Variance report | ppm_pbl:reports |
| /ppm-pbl/portfolio | GET | Portfolio baseline summary | ppm_pbl:portfolio |
| /ppm-pbl/bds | GET | Baseline Deviation Scores | ppm_pbl:reports |
| /ppm-pbl/approvals | GET/POST | Approval console | ppm_pbl:approve |
| /ppm-pbl/agents | GET/POST | Agent workbench | ppm_pbl:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| retroactive_baseline_edit_denied | retroactive=True on edit | deny |
| change_control_bypass_denied | change_control_bypass=True | deny |
| ev_manipulation_denied | ev_manipulation=True | deny |
| baseline_approval_requires_designated_approver | designated_approver=False | deny |
| change_baseline_required | baseline_present=False | deny |
| change_impact_required | impact_present=False | deny |
| change_approval_required | approval_present=False on implement | deny |
| cross_tenant_baseline_access_denied | cross_tenant_access=True | deny |

## Data Models
- **ProjectBaseline** — id, project_id, baseline_type, status, name, owner_id, approval_reference
- **ChangeRequest** — id, baseline_id, change_type, priority, status, submitter_id, impact_reference
- **ChangeImpactAssessment** — id, change_request_id, impact_areas, schedule_impact_days, cost_impact_amount
- **EarnedValueSnapshot** — id, baseline_id, snapshot_date, pv, ev, ac, bac, eac, etc
- **VarianceReport** — id, baseline_id, schedule_variance, cost_variance, spi, cpi, threshold_breached
- **BaselineApproval** — id, reference_id, reviewer_id, designated_approver, status
- **BaselineAgent** — id, name, runtime, role, scope

## Streaming Events
- `baseline_created`, `baseline_approved`, `baseline_superseded`
- `change_request_submitted`, `change_impact_assessed`, `change_request_approved`
- `change_request_rejected`, `change_implemented`, `ev_snapshot_taken`
- `variance_threshold_breached`, `agent_registered`

## Edge Cases Handled
- Retroactive baseline edits are blocked; a change request must be used instead
- EV manipulation (manually overriding PV/EV without actual data) is rejected by rule engine
- Designated approver requirement prevents self-approval of baselines
- SPI/CPI < 0.9 automatically flags threshold_breached in variance reports
- Change requests to non-existent baselines are rejected before business logic runs

## Composability Notes
- Receives cost data from **ppm_pac** for cost baseline creation
- Receives schedule from **ppm_pps** for schedule baseline creation
- EV snapshots feed **ppm_pan** portfolio performance dashboards
- Change requests can trigger notifications via **ntfy** and workflow steps via **wflo**

---

## World-Class Enhancements (v2.0)

1. **Integrated Baseline Review (IBR) Automation** — cross-validates scope/schedule/cost baselines with a scored IBR health index (0–100).
2. **Rolling Wave Baseline Segments** — `set_rolling_wave_baseline()` tracks committed vs planning horizons; enables EVM on the committed segment.
3. **Probabilistic Cost and Schedule Reserves** — `set_reserve_analysis()` stores management/contingency reserves and P80/P90 confidence bounds.
4. **Time-Phased Budget Distribution (S-Curve)** — `set_time_phased_budget()` drives period-by-period PV; `take_ev_snapshot()` auto-resolves correct PV from the S-curve.
5. **Retroactive Integrity Audit Trail** — structured `_audit_diff()` captures `{field, before, after}` diffs in a tamper-evident append-only log.
6. **Baseline Lock Mechanism** — `lock_baseline()` / `unlock_baseline()` with explicit owner tracking; locked baselines reject all mutations.
7. **Change Request Dependency Graph** — `link_change_requests()` + `get_cr_dependency_graph()` model blocks/depends_on/supersedes relationships as a DAG.
8. **Earned Schedule (ES) Metrics** — `earned_schedule_metrics()` computes ES, SV(t), SPI(t), and IEAC(t); SPI(t) correctly converges to 1.0 at completion.
9. **Multi-Baseline Portfolio View** — `portfolio_baseline_summary()` rolls up CPI, SPI, EAC, and risk tiers (red/amber/green) across all tenant projects.
10. **Automated Variance Threshold Escalation** — `configure_variance_escalation()` triggers notifications and optional CR freeze on consecutive threshold breaches.
11. **Baseline Freeze Periods** — `set_freeze_period()` blocks CR submissions during reporting lock windows; `emergency` priority CRs are exempt.
12. **WBS-Linked Scope Baseline** — `set_scope_baseline()` now accepts `wbs_elements` with control accounts, enabling proper PMB construction.
13. **Variance At Completion (VAC) Forecasting** — `forecast_completion()` returns EAC under three methods (typical/atypical/scheduled), VAC, and TCPI with recommended method.
14. **Baseline Deviation Score (BDS)** — `baseline_deviation_scores()` computes a weighted 0–100 health score per project, ranked worst-first for dashboard KPIs.
15. **Baseline Version History and Diff** — `get_baseline_version_history()` preserves full baseline snapshots; `diff_baseline_versions()` returns structured diffs between any two versions.

---

## New Methods

### `integrated_baseline_review(project_id, tenant_id=None)`
Cross-validates all three baseline types and returns a scored report.

```python
svc = ProjectBaselineService(tenant_id="acme")
ibr = await svc.integrated_baseline_review(project_id="proj-001")
# ibr["ibr_health_index"]  -> 85.0
# ibr["dimensions"]["completeness"]["pass"]  -> True
# ibr["dimensions"]["cost_integrity"]["pass"] -> True
```

### `forecast_completion(project_id, tenant_id=None)`
Computes EAC under three methods, TCPI, and VAC from the latest EV snapshot.

```python
fc = await svc.forecast_completion(project_id="proj-001")
# fc["eac_typical"]       -> 125000.00   (BAC / CPI)
# fc["eac_atypical"]      -> 118000.00   (AC + remaining at planned rate)
# fc["tcpi"]              -> 0.9412      (CPI needed to finish on budget)
# fc["vac"]               -> -5000.00
# fc["recommended_method"] -> "typical"
```

### `baseline_deviation_scores(tenant_id=None)`
Ranks all active projects by composite BDS (0–100, lower is better) for portfolio triage.

```python
bds = await svc.baseline_deviation_scores()
# bds["scores"][0]  -> {"project_id": "proj-007", "bds": 72.4, "tier": "red", ...}
# bds["scores"][-1] -> {"project_id": "proj-002", "bds": 8.1,  "tier": "green", ...}
# Wire directly to dashboard KPI tiles; sort by -bds for worst-first triage.
```
