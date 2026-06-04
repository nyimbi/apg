# Tax Administration

## Overview
Taxpayer registration, return filing, assessment, objections, debt collection, and audit case management. Implements the full tax administration lifecycle from TIN issuance through audit closure, with strict controls on duplicate PINs, objection deadlines, and debt collection procedures.

## Capability ID
`government_tax`

## Provides
- taxpayer_registration_workflow: TIN/PIN issuance and taxpayer onboarding
- return_filing_workflow: Monthly, quarterly, and annual return submission
- tax_assessment_workflow: Self-assessment and authority-raised assessments
- objection_management_workflow: Taxpayer objection within statutory deadline
- debt_collection_workflow: Demand notice, payment plan, and legal proceedings
- audit_case_management_workflow: Desk, field, IT, and forensic audit management
- tax_review_workflow: Governance review of tax decisions
- tax_agent_workflow: Automated return processing and compliance agents
- tax_refund_workflow: VAT and income tax refund processing
- compliance_risk_scoring_workflow: Risk-based compliance monitoring

## Requires
| Capability | Reason |
|---|---|
| auth | Taxpayer and assessor RBAC |
| audl | Immutable tax transaction audit trail |
| mten | Tenant-scoped taxpayer data isolation |
| conf | Tax rate and threshold configuration |
| ntfy | Filing deadline reminders and assessment notices |
| wflo | Assessment and objection approval workflow |
| comp | Tax Act compliance checks |
| schd | Return filing deadline scheduling |
| moni | Compliance monitoring and late filing detection |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.duplicate_pin_denied | One PIN per taxpayer identity |
| governance.objection_outside_deadline_denied | Statutory objection period enforced |
| governance.debt_collection_without_demand_notice_denied | Demand notice mandatory before collection |
| governance.taxpayer_data_confidentiality_enforced | Tax secrecy provisions enforced |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-tax/registrations | GET/POST | Taxpayer registration | government_tax:register |
| /government-tax/returns | GET/POST | Return filing | government_tax:returns |
| /government-tax/assessments | GET/POST | Tax assessment | government_tax:assess |
| /government-tax/objections | GET/POST | Objection management | government_tax:object |
| /government-tax/debt-collection | GET/POST | Debt collection | government_tax:collect |
| /government-tax/audits | GET/POST | Audit case management | government_tax:audit |
| /government-tax/refunds | GET/POST | Tax refunds | government_tax:refunds |
| /government-tax/compliance | GET | Compliance dashboard | government_tax:compliance |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| duplicate_pin_denied | duplicate_pin=True | deny |
| objection_deadline_enforced | within_deadline=False | deny |
| debt_collection_demand_required | demand_notice_issued=False | deny |
| audit_auditor_required | auditor_present=False | deny |
| return_period_required | period_present=False | deny |

## Data Models
- TaxpayerRegistration: id, tenant_id, tax_type, tax_pin, national_id, taxpayer_name, status
- TaxReturn: id, tenant_id, return_type, taxpayer_pin, period, gross_income, tax_liability, status
- TaxAssessment: id, tenant_id, return_id, assessment_type, assessed_amount, assessor_id, status
- TaxObjection: id, assessment_id, taxpayer_pin, grounds, amount_disputed, status, filed_date
- DebtCollectionCase: id, taxpayer_pin, assessment_id, collection_method, amount_owed, demand_notice_reference
- AuditCase: id, tenant_id, taxpayer_pin, audit_type, auditor_id, period_under_review, findings
- TaxReview, TaxAgent

## Streaming Events
- taxpayer_registered, tax_return_filed, tax_assessed, objection_filed
- objection_determined, debt_collection_initiated, payment_received, audit_case_opened, audit_completed

## Edge Cases Handled
- Objection filed after statutory deadline — denied with `objection_deadline_passed`; taxpayer must apply for extension
- Debt collection initiated without demand notice — denied regardless of debt age
- Duplicate PIN registration (same national ID with different names) — blocked and flagged for investigation
- Tax secrecy: taxpayer data never exposed to other tenants even via aggregate reports
- VAT refund audit triggers separate audit case type with different evidence requirements

## Composability Notes
Composes with `government_bud` (tax revenue feeds exchequer and AIA vote accounts), `government_csr` (taxpayer returns submitted through citizen portal), `government_law` (tax fraud cases create law enforcement dockets), `government_con` (withholding tax on government contracts), and `intel` (tax gap analysis and evasion pattern intelligence).
