# Project Accounting

## Overview
Project Accounting (pac) provides complete financial tracking for projects: cost capture, revenue recognition under multiple WIP methods, milestone billing, budget control, and profitability reporting. Every transaction is tenant-scoped, approval-gated, and streamed via Bytewax for real-time financial visibility.

## Capability ID
`ppm_pac`

## Provides
| Service | Description |
|---------|-------------|
| project_cost_tracking | Capture actual, committed, and forecast costs by type |
| revenue_recognition_workflow | Recognise revenue via percentage completion, completed contract, EV, and other GAAP/IFRS methods |
| wip_accounting_workflow | Post WIP adjustments with auditor sign-off |
| milestone_billing_workflow | Raise milestone and progress invoices with approval gates |
| project_profitability_reporting | Gross/contribution/net margin per project account |
| budget_vs_actual_analysis | Real-time budget variance with controller-approval override path |
| cost_variance_alerts | Event-driven alerts on cost overruns |
| cash_flow_forecasting | Forward-looking cash flow from committed and forecast costs |
| multi_currency_project_accounting | USD, EUR, GBP, KES, and 6 more currencies |
| audit_trail_maintenance | Immutable audit events via Bytewax stream |

## Requires
| Capability | Reason |
|------------|--------|
| auth | User authentication and permission enforcement |
| audl | Immutable audit logging of all transactions |
| mten | Tenant isolation and context propagation |
| conf | Runtime configuration and feature flags |
| ntfy | Budget-overrun and approval notifications |
| wflo | Multi-step approval workflows |
| moni | Operational health monitoring |
| comp | Regulatory compliance (IFRS 15, ASC 606) |
| mqeb | Event publishing to Bytewax stream |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| tenant_id | "default" | Tenant context identifier |
| costs.supported_cost_types | 8 types | Allowable cost categories |
| revenue.supported_wip_methods | 5 methods | Revenue recognition methods |
| billing.supported_billing_types | 6 types | Invoice styles |
| governance.backdated_transaction_requires_justification | true | Audit control |
| governance.revenue_recognition_requires_approval | true | Segregation of duties |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /ppm-pac/dashboard | GET | Accounting dashboard | ppm_pac:view |
| /ppm-pac/accounts | GET/POST | Project account list and creation | ppm_pac:accounts |
| /ppm-pac/costs | GET/POST | Cost transaction ledger | ppm_pac:costs |
| /ppm-pac/revenue | GET/POST | Revenue recognition console | ppm_pac:revenue |
| /ppm-pac/wip | GET/POST | WIP accounting workbench | ppm_pac:wip |
| /ppm-pac/billing | GET/POST | Milestone billing console | ppm_pac:billing |
| /ppm-pac/budgets | GET/POST | Budget control console | ppm_pac:budgets |
| /ppm-pac/profitability | GET | Profitability report | ppm_pac:reports |
| /ppm-pac/variance | GET | Variance analysis | ppm_pac:reports |
| /ppm-pac/approvals | GET/POST | Approval queue | ppm_pac:approve |
| /ppm-pac/agents | GET/POST | Agent workbench | ppm_pac:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| cost_amount_positive | amount_positive=False | deny |
| revenue_approval_required | approval_present=False | deny |
| negative_revenue_denied | amount_positive=False on revenue | deny |
| wip_auditor_required | auditor_present=False | deny |
| backdated_cost_requires_justification | backdated=True, no justification | deny |
| budget_override_requires_controller | controller_approval_present=False | deny |
| cross_tenant_cost_access_denied | cross_tenant_access=True | deny |
| cost_batch_requires_bytewax | event_stream != "bytewax" | deny |
| privileged_agent_action_requires_human_approval | privileged_scope + no approval | deny |

## Data Models
- **ProjectAccount** — id, tenant_id, project_id, name, status, currency, budget_amount, owner_id
- **CostTransaction** — id, account_id, cost_type, transaction_type, amount, period_reference
- **RevenueRecognition** — id, account_id, revenue_type, wip_method, amount, approval_reference
- **WipAdjustment** — id, account_id, adjustment_amount, auditor_id
- **MilestoneInvoice** — id, account_id, billing_type, amount, milestone_reference, approval_reference
- **BudgetOverride** — id, account_id, original_budget, revised_budget, controller_approval_reference
- **AccountingApproval** — id, reference_id, approval_type, reviewer_id, status
- **AccountingAgent** — id, name, runtime, role, scope

## Streaming Events
- `project_account_created`, `cost_transaction_recorded`, `revenue_recognised`
- `wip_adjustment_posted`, `milestone_invoice_raised`, `budget_variance_detected`
- `approval_submitted`, `approval_completed`, `profitability_report_generated`, `agent_registered`

## Edge Cases Handled
- Backdated transactions require explicit justification to prevent retroactive manipulation
- Negative revenue recognition is rejected at the rule-engine level
- WIP adjustments require a named auditor for segregation of duties
- Budget overrides need controller-level approval separate from the PM
- Cross-tenant cost access is structurally denied via tenant-keyed storage
- Cost batch submissions must route through Bytewax; direct queue writes are rejected

## Composability Notes
- Pairs with **ppm_tex** (time entries feed actual costs automatically)
- Pairs with **ppm_pbl** (EV snapshots use cost baseline data)
- Pairs with **ppm_res** (resource cost rates underpin labour cost transactions)
- Downstream consumers can subscribe to `apg.ppm.pac.lifecycle` via Bytewax
