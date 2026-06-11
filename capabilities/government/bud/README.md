# Budget & Financial Planning

## Overview
Programme budgeting, vote accounting, commitment control, MTEF rolling envelopes, PBB scorecards, fiscal risk management, IPSAS-aligned reporting, inter-government fiscal transfers, expenditure anomaly detection, and parliamentary estimates generation for government entities. Enforces appropriation limits, prevents over-commitment, and ensures every budget revision carries a treasury notification reference.

## Capability ID
`government_bud`

## Provides
- `budget_programme_workflow` — Record and manage programme budget entries
- `vote_accounting_workflow` — Maintain vote account balances and transactions
- `budget_revision_workflow` — Process reallocations, virements, and supplementary estimates
- `commitment_control_workflow` — Gate expenditures behind available vote balances
- `expenditure_recording_workflow` — Record actual expenditures against commitments
- `fiscal_reporting_workflow` — Generate budget outturn, variance, and Treasury reports
- `budget_approval_workflow` — Approval chain for budget items
- `budget_review_workflow` — Governance review of budget decisions
- `budget_agent_workflow` — Automated budget analytics agents
- `treasury_submission_workflow` — Treasury submission packaging and tracking
- `mtef_planning_workflow` — Medium-Term Expenditure Framework rolling envelope automation
- `pbb_scorecard_workflow` — Programme-Based Budgeting KPI scoring and reallocation signals
- `fiscal_risk_workflow` — Fiscal risk register and contingent liability modelling
- `igft_allocation_workflow` — Inter-Government Fiscal Transfer formula-based allocation
- `arrears_management_workflow` — Payment arrears registry and prioritised settlement plans
- `anomaly_detection_workflow` — ML-assisted expenditure anomaly detection (local Ollama)
- `ipsas_reporting_workflow` — IPSAS accrual-basis financial statement generation
- `parliamentary_estimates_workflow` — Parliamentary estimates package compilation

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication and RBAC |
| audl | Immutable audit log of all budget transactions |
| mten | Multi-tenant data isolation |
| conf | Runtime configuration management |
| ntfy | Notify approvers and Treasury of budget events |
| wflo | Approval workflow orchestration |
| comp | PFMA/budget circular compliance checks |
| moni | Operational monitoring and alerting |
| mqeb | Event streaming via bytewax + NATS |

## Configuration
| Key | Default | Description |
|---|---|---|
| tenant_id | default | Tenant identifier |
| governance.commitment_without_balance_denied | true | Block over-commitment |
| governance.revision_without_treasury_approval_denied | true | Require treasury notification |
| governance.negative_vote_balance_denied | true | Prevent negative balances |
| igft.constitutional_floor_pct | 15.0 | Minimum IGFT allocation as % of shareable revenue |
| anomaly.sensitivity | 0.8 | Expenditure anomaly detection sensitivity (0-1) |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-bud/dashboard | GET | Budget dashboard summary | government_bud:view |
| /government-bud/budgets | GET/POST | List/create budgets | government_bud:budgets |
| /government-bud/votes | GET/POST | Vote account ledger | government_bud:votes |
| /government-bud/revisions | GET/POST | Budget revisions | government_bud:revisions |
| /government-bud/commitments | GET/POST | Commitment control queue | government_bud:commitments |
| /government-bud/expenditures | GET/POST | Expenditure ledger | government_bud:expenditures |
| /government-bud/reports | GET/POST | Fiscal reports | government_bud:reports |
| /government-bud/treasury | GET/POST | Treasury submissions | government_bud:treasury |
| /government-bud/agents | GET/POST | Budget automation agents | government_bud:admin |
| /government-bud/mtef | GET/POST | MTEF envelope planning | government_bud:mtef |
| /government-bud/pbb | GET/POST | PBB scorecards | government_bud:pbb |
| /government-bud/risks | GET/POST | Fiscal risk register | government_bud:risks |
| /government-bud/igft | GET/POST | IGFT allocation engine | government_bud:igft |
| /government-bud/arrears | GET/POST | Payment arrears registry | government_bud:arrears |
| /government-bud/anomalies | GET | Expenditure anomaly detection | government_bud:anomaly |
| /government-bud/ipsas | GET | IPSAS accrual report | government_bud:reports |
| /government-bud/parliament | GET | Parliamentary estimates package | government_bud:parliament |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| tenant_context_required | tenant_context_present=False | deny |
| commitment_balance_required | sufficient_balance=False | deny |
| negative_vote_balance_denied | negative_balance=True | deny |
| revision_treasury_required | treasury_notification_present=False | deny |
| cross_vote_reallocation_requires_approval | cross_vote=True, approval=False | deny |
| igft_floor_enforcement | total_allocated < constitutional_floor | reject allocation |
| mtef_sector_shares_must_sum_100 | sum(sector_shares) != 100 | assert error |

## Key Service Methods

### Synchronous (Core Budget Execution)
- `describe()` — Capability contract
- `evaluate()` — Policy rule evaluation
- `record_budget()` — Record programme budget entry
- `create_budget_ceiling()` — Set vote ceiling
- `requisition()` — Raise department requisition
- `commitment_check()` — Check available balance pre-commitment
- `payment_approval()` — Approve payment against commitment
- `budget_revision()` — Submit budget revision
- `expenditure_report()` — Department expenditure report
- `budget_vs_actual()` — BVA analysis for a vote
- `supplementary_budget()` — Appropriate supplementary budget
- `treasury_single_account()` — Record TSA movement
- `public_finance_report()` — PFM summary report
- `record_vote()` — Record vote account
- `record_revision()` — Low-level revision record
- `record_commitment()` — Record commitment against vote
- `record_expenditure()` — Record expenditure against commitment
- `generate_report()` — Generate fiscal report
- `record_approval()` — Record budget approval
- `record_review()` — Record governance review
- `register_agent()` — Register budget automation agent
- `validate_agent_action()` — Policy-gate agent operations
- `validate_batch()` — Validate batch processing via bytewax
- `vote_balance_summary()` — Vote balance snapshot
- `dashboard_summary()` — Dashboard KPI snapshot
- `multi_year_budget_plan()` — Multi-year plan
- `inter_agency_transfer()` — Cross-agency transfer
- `cash_flow_projection()` — Monthly cash-flow forecast
- `commitment_liquidation()` — Liquidate commitment
- `audit_trail_report()` — Paginated audit events
- `budget_utilisation_analysis()` — Vote-type utilisation breakdown
- `outstanding_commitments_report()` — Open commitments list
- `performance_budget_link()` — Link vote to KPIs
- `variance_alert()` — Threshold-based variance alerts
- `donor_funded_budget()` — Register donor project
- `fiscal_year_close()` — Year-end closing procedures
- `internal_audit_schedule()` — Quarterly audit plan
- `procurement_linkage()` — Link commitment to contract
- `grants_management()` — Conditional grants with disbursement schedules
- `debt_management()` — Public debt instrument recording
- `revenue_projection()` — Revenue stream forecasting

### Async (Advanced Analytics & Automation)
- `ml_budget_variance_predict()` — Ollama-powered variance risk prediction
- `mtef_rolling_envelope()` — MTEF three-year sector ceilings
- `pbb_scorecard()` — PBB KPI composite scoring
- `reconcile_tsa_with_expenditures()` — TSA vs expenditure reconciliation
- `register_fiscal_risk()` — Fiscal risk register entry
- `compute_contingent_liability_exposure()` — Aggregate expected contingent exposure
- `stress_test_budget()` — Macro-fiscal scenario stress testing
- `compute_igft_allocation()` — Formula-based IGFT allocation per county/unit
- `detect_expenditure_anomalies()` — Heuristic + optional ML anomaly detection
- `generate_parliamentary_estimates()` — Parliamentary estimates package
- `register_payment_arrear()` — Register overdue payment as arrear
- `generate_arrears_payment_plan()` — Priority-ranked arrears settlement plan
- `generate_ipsas_accrual_report()` — IPSAS 1/24 accrual financial statements

## Streaming Events (NATS)
| Event | NATS Subject |
|---|---|
| budget_recorded | apg.government.bud.lifecycle |
| commitment_recorded | apg.government.bud.commitment.{tenant_id} |
| mtef_envelope_set | apg.government.bud.mtef |
| pbb_scorecard_computed | apg.government.bud.pbb |
| tsa_reconciliation_completed | apg.government.bud.tsa.reconciliation |
| fiscal_risk_registered | apg.government.bud.risk |
| igft_allocation_computed | apg.government.bud.igft |
| expenditure_anomalies_detected | apg.government.bud.anomaly |
| parliamentary_estimates_generated | apg.government.bud.parliament.submission |
| payment_arrear_registered | apg.government.bud.arrears |
| ipsas_accrual_report_generated | (audit log) |

## Data Models
- `BudgetProgramme` — id, tenant_id, budget_type, fund_source, vote_id, total_amount, fiscal_year, status
- `VoteAccount` — id, tenant_id, vote_code, allocated_amount, committed_amount, available_balance
- `BudgetRevision` — id, tenant_id, budget_id, revision_type, amount_change, treasury_notification_reference
- `CommitmentRecord` — id, tenant_id, vote_id, commitment_type, amount, approval_reference
- `ExpenditureRecord` — id, tenant_id, commitment_id, expenditure_type, amount
- `FiscalReport` — id, tenant_id, budget_id, report_type, fiscal_period, status
- `BudgetApproval`, `BudgetReview`, `BudgetAgent`

## Edge Cases Handled
- Commitment attempted when vote balance is zero — denied with `insufficient_vote_balance`
- Budget revision without treasury notification reference — denied
- Cross-vote reallocation without separate approval — denied
- Agent attempting to suppress audit evidence — denied
- Batch processing routed to non-bytewax stream — denied
- IGFT allocation below constitutional floor — rejected with floor breach flag
- TSA debit with no matching expenditure record — flagged as unmatched reconciliation item
- Expenditure anomaly with z-score > 3 and round-number amount — dual-reason suspicion flag

## Composability
Composes with:
- `government_con` — commitments triggered by contract awards
- `government_csr` — fee collections feed into vote accounts
- `government_tax` — revenue receipts update AIA vote balances
- `intel` — fiscal report output feeds executive dashboard
- `intel_alerts` — variance and anomaly events trigger alert workflows

## Local AI Integration
When `OLLAMA_BASE_URL` is set, the following methods use locally hosted models (no data leaves the Ministry):
- `ml_budget_variance_predict()` — budget overspend risk
- `detect_expenditure_anomalies()` — ML-assisted anomaly scoring
- `stress_test_budget()` — scenario narrative generation (optional)
