# Project Accounting

## Overview
Project Accounting (pac) provides complete financial tracking for projects: cost capture, revenue recognition under multiple WIP methods, milestone billing, earned value management, budget control, cash flow forecasting, and profitability reporting. Every transaction is tenant-scoped, approval-gated, and streamed via Bytewax for real-time financial visibility.

## Capability ID
`ppm_pac`

## Provides
| Service | Description |
|---------|-------------|
| project_cost_tracking | Capture actual, committed, and forecast costs by type and cost code |
| revenue_recognition_workflow | Recognise revenue via percentage completion, completed contract, EV, and other GAAP/IFRS methods |
| wip_accounting_workflow | Post WIP adjustments with auditor sign-off |
| milestone_billing_workflow | Raise milestone and progress invoices with approval gates |
| project_profitability_reporting | Gross/contribution/net margin per project account |
| budget_vs_actual_analysis | Real-time budget variance with controller-approval override path |
| cost_variance_alerts | Event-driven alerts on cost overruns with configurable thresholds |
| cash_flow_forecasting | Period-by-period cash flow projection from committed POs and milestone schedules |
| earned_value_management | Full EVM suite: PV, EV, AC, SPI, CPI, EAC, ETC with trend analysis |
| three_point_eac | PERT-weighted EAC with P50/P80/P90 confidence bands |
| period_cost_summary | Per-period burn rate and time-to-completion forecasting |
| accruals | Period-end accrual posting with auto-reversal tracking |
| intercompany_recharge | Cross-project cost recharging with transfer pricing markup |
| variance_classification | Heuristic root-cause classification of budget variances |
| ledger_reconciliation | External ERP ledger reconciliation with tolerance matching |
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
| budget.warn_threshold_pct | 80 | Budget utilisation warn threshold |
| budget.critical_threshold_pct | 95 | Budget utilisation critical threshold |
| eac.tolerance_pct | 0.5 | Ledger reconciliation tolerance |

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
| /ppm-pac/cashflow | GET | Cash flow forecast | ppm_pac:reports |
| /ppm-pac/ev-trend | GET | Earned value trend analysis | ppm_pac:reports |
| /ppm-pac/accruals | GET/POST | Period-end accruals | ppm_pac:costs |
| /ppm-pac/recharges | POST | Intercompany cost recharges | ppm_pac:admin |
| /ppm-pac/reconcile | POST | External ledger reconciliation | ppm_pac:admin |

## Key Service Methods

### Core Accounting
- `create_account()` — Open a new project accounting record
- `get_account()` / `list_accounts()` — Retrieve accounts
- `record_cost()` — Post a direct cost transaction
- `project_budget_setup()` — Define itemised budget lines
- `cost_code_create()` — Create a cost code for granular tracking
- `record_timesheet_cost()` — Post labour cost from timesheet hours
- `record_expense()` — Record an approved project expense
- `purchase_order_project()` — Raise a supplier purchase order

### Revenue and WIP
- `recognise_revenue()` — Post a revenue recognition entry
- `revenue_recognition_project()` — Auto-recognise using a WIP method
- `post_wip_adjustment()` — Post an auditor-signed WIP adjustment
- `raise_invoice()` — Issue a milestone billing invoice

### Analytics and EVM
- `earned_value_analysis()` — Full EVM metrics for a period
- `ev_trend_analysis()` — CPI/SPI trend across stored snapshots
- `three_point_eac()` — PERT-weighted EAC with confidence percentiles
- `project_profitability()` — Full P&L: revenue, costs, margins
- `project_cost_report()` — Budget vs actual by cost code
- `budget_vs_actual()` — Quick budget variance summary
- `period_cost_summary()` — Per-period burn rate and forecast

### Planning and Control
- `cash_flow_forecast()` — Period-by-period inflow/outflow projection
- `check_budget_thresholds()` — Alert evaluation by cost code
- `post_accrual()` — Period-end accrual with auto-reversal tracking
- `classify_variance()` — Root-cause classification of budget variances
- `override_budget()` — Controller-approved budget revision

### Interoperability
- `create_intercompany_recharge()` — Cross-project cost recharge with markup
- `reconcile_with_external_ledger()` — ERP ledger reconciliation
- `invoice_project_cost()` — Allocate supplier invoice across cost codes
- `export_accounting_data()` — JSON/CSV export

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
| accrual_reversal_period_different | period == reversal_period | deny |
| intercompany_self_recharge_denied | from_project == to_project | deny |

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
- `cash_flow_forecast_generated`, `ev_trend_analysed`, `three_point_eac_computed`
- `accrual_posted`, `intercompany_recharge_posted`, `reconciliation_complete`
- `budget_threshold_warn`, `budget_threshold_critical`, `variance_classified`
- `period_cost_summary_generated`

## Edge Cases Handled
- Backdated transactions require explicit justification to prevent retroactive manipulation
- Negative revenue recognition is rejected at the rule-engine level
- WIP adjustments require a named auditor for segregation of duties
- Budget overrides need controller-level approval separate from the PM
- Cross-tenant cost access is structurally denied via tenant-keyed storage
- Cost batch submissions must route through Bytewax; direct queue writes are rejected
- Accrual and reversal periods must differ; same-period accruals are rejected
- Intercompany recharges cannot be self-directed (from == to is rejected)
- Ledger reconciliation uses configurable tolerance to handle minor rounding differences

## Composability Notes
- Pairs with **ppm_tex** (time entries feed actual costs automatically)
- Pairs with **ppm_pbl** (EV snapshots use cost baseline data; percent_complete source)
- Pairs with **ppm_res** (resource cost rates underpin labour cost transactions)
- Downstream consumers can subscribe to `apg.ppm.pac.lifecycle` via Bytewax
- `reconcile_with_external_ledger` integrates with ERP adapters (SAP, Xero, Sage)
- `create_intercompany_recharge` pairs with `ppm_pac` instances across entity tenants
