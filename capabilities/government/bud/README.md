# Budget Management

## Overview
Programme budgeting, vote accounting, commitment control, budget revisions, fiscal reporting, and Treasury submission for government entities. Enforces appropriation limits, prevents over-commitment, and ensures every budget revision carries a treasury notification reference.

## Capability ID
`government_bud`

## Provides
- budget_programme_workflow: Record and manage programme budget entries
- vote_accounting_workflow: Maintain vote account balances and transactions
- budget_revision_workflow: Process reallocations, virements, and supplementary estimates
- commitment_control_workflow: Gate expenditures behind available vote balances
- expenditure_recording_workflow: Record actual expenditures against commitments
- fiscal_reporting_workflow: Generate budget outturn, variance, and Treasury reports
- budget_approval_workflow: Approval chain for budget items
- budget_review_workflow: Governance review of budget decisions
- budget_agent_workflow: Automated budget analytics agents
- treasury_submission_workflow: Treasury submission packaging and tracking

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
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Default | Description |
|---|---|---|
| tenant_id | default | Tenant identifier |
| governance.commitment_without_balance_denied | true | Block over-commitment |
| governance.revision_without_treasury_approval_denied | true | Require treasury notification |
| governance.negative_vote_balance_denied | true | Prevent negative balances |

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

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| tenant_context_required | tenant_context_present=False | deny |
| commitment_balance_required | sufficient_balance=False | deny |
| negative_vote_balance_denied | negative_balance=True | deny |
| revision_treasury_required | treasury_notification_present=False | deny |
| cross_vote_reallocation_requires_approval | cross_vote=True, approval=False | deny |

## Data Models
- BudgetProgramme: id, tenant_id, budget_type, fund_source, vote_id, total_amount, fiscal_year, status
- VoteAccount: id, tenant_id, vote_code, allocated_amount, committed_amount, available_balance
- BudgetRevision: id, tenant_id, budget_id, revision_type, amount_change, treasury_notification_reference
- CommitmentRecord: id, tenant_id, vote_id, commitment_type, amount, approval_reference
- ExpenditureRecord: id, tenant_id, commitment_id, expenditure_type, amount
- FiscalReport: id, tenant_id, budget_id, report_type, fiscal_period, status
- BudgetApproval, BudgetReview, BudgetAgent

## Streaming Events
- budget_recorded, vote_recorded, budget_revision_recorded, commitment_recorded
- expenditure_recorded, fiscal_report_generated, budget_approved, treasury_submission_recorded

## Edge Cases Handled
- Commitment attempted when vote balance is zero — denied with `insufficient_vote_balance`
- Budget revision without treasury notification reference — denied
- Cross-vote reallocation without separate approval — denied
- Agent attempting to suppress audit evidence — denied
- Batch processing routed to non-bytewax stream — denied

## Composability Notes
Composes with `government_con` (commitments triggered by contract awards), `government_csr` (fee collections feed into vote accounts), and `government_tax` (revenue receipts update AIA vote balances). The fiscal report output feeds into `intel` dashboard for executive visibility.
