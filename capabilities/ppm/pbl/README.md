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
| baseline_variance_tracking | Schedule and cost variance with threshold alerting |
| change_impact_assessment | Schedule, cost, scope, and risk impact quantification |
| baseline_approval_workflow | Designated approver enforcement for baseline promotion |
| integrated_baseline_review | Cross-baseline consistency checks |
| performance_measurement_baseline | PMB for EVM reporting |

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
| /ppm-pbl/changes | GET/POST | Change request queue | ppm_pbl:changes |
| /ppm-pbl/impact | GET/POST | Impact assessment | ppm_pbl:impact |
| /ppm-pbl/ev | GET/POST | Earned value dashboard | ppm_pbl:ev |
| /ppm-pbl/variance | GET | Variance report | ppm_pbl:reports |
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
