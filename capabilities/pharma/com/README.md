# Commercial Operations & Pharmacovigilance

## Overview
Manages pharmaceutical field force activities including territory management, sales rep assignments, physician call recording, PDMA-compliant sample dispensing, HCP interaction tracking, aggregate spend management, and commercial planning. Enforces Sunshine Act / Open Payments reporting and PDMA compliance rules at every transactional boundary.

Includes a pharmacovigilance (PV) layer covering Individual Case Safety Report (ICSR) lifecycle, MedDRA term coding, adverse event signal detection (ROR disproportionality), regulatory submission packaging (ICH E2B R3 / EMA / FDA / PMDA), duplicate ICSR detection, signal triage scoring, Open Payments report generation, and CAPA management.

## Capability ID
`pharma_com`

## Provides
- territory_management_workflow: Create and manage sales territories with product alignment
- sales_rep_management_workflow: Assign and track field representatives with certification validation
- call_activity_workflow: Record physician calls with outcome and product discussion capture
- sample_management_workflow: PDMA-compliant sample dispensing with electronic signature and lot tracking
- hcp_interaction_workflow: Log all HCP interactions with value-threshold spend detection
- commercial_plan_workflow: Draft, approve, and track commercial plans with territory quotas
- target_segmentation_workflow: Tier physicians and set call frequency targets
- aggregate_spend_workflow: Track and cap cumulative HCP spend per fiscal year
- pdma_compliance_workflow: Enforce PDMA rules at sample dispensing and interaction recording
- commercial_dashboard_workflow: Consolidated field force KPI dashboard
- icsr_lifecycle_workflow: Create, update, and submit Individual Case Safety Reports (E2B R3)
- signal_detection_workflow: ROR disproportionality analysis over the ICSR corpus
- meddra_coding_workflow: Verbatim term mapping to MedDRA PT/LLT hierarchy
- regulatory_submission_workflow: Package and submit ICSRs to EMA, FDA, PMDA, Health Canada, TGA
- open_payments_workflow: CMS Sunshine Act annual report generation and validation
- signal_triage_workflow: Composite priority scoring (0–100) with tier assignment
- duplicate_detection_workflow: Probabilistic deduplication of ICSR records
- capa_workflow: Corrective and preventive action creation from compliance violations

## Requires
| Capability | Reason |
|------------|--------|
| auth | Identity and access control for all operations |
| audl | Immutable audit trail for compliance |
| mten | Multi-tenant context enforcement |
| conf | Runtime configuration management |
| ntfy | Alert notifications for compliance violations and signal escalation |
| wflo | Plan approval workflow and ICSR submission workflow |
| comp | PDMA and Sunshine Act compliance enforcement |
| schd | Call frequency scheduling |
| mqeb | Event streaming via Bytewax |
| qms | CAPA routing and quality management integration |
| pvi | Pharmacovigilance signal handoff |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| compliance.aggregate_spend_cap | Maximum annual HCP spend (USD) | 500.0 |
| spend.receipt_required_above | Receipt threshold amount | 25.0 |
| spend.pre_approval_required_above | Pre-approval threshold | 100.0 |
| targets.review_cycle_months | Target tier review frequency | 6 |
| pv.ror_threshold | ROR signal detection threshold | 2.0 |
| pv.min_case_count | Minimum cases for signal flag | 3 |
| pv.meddra_release | MedDRA release version | 26.1 |
| pv.signal_triage_escalate_threshold | Triage score for escalate tier | 60 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-com/api/v1/territories | GET | List territories | pharma_com:territories |
| /pharma-com/api/v1/territories | POST | Create territory | pharma_com:territories |
| /pharma-com/api/v1/reps | GET | List sales reps | pharma_com:reps |
| /pharma-com/api/v1/calls | POST | Record call | pharma_com:calls |
| /pharma-com/api/v1/samples | POST | Dispense sample | pharma_com:samples |
| /pharma-com/api/v1/interactions | POST | Record HCP interaction | pharma_com:interactions |
| /pharma-com/api/v1/plans | POST | Create commercial plan | pharma_com:plans |
| /pharma-com/api/v1/spend/summary | GET | Aggregate spend summary | pharma_com:spend |
| /pharma-com/api/v1/icsrs | POST | Create ICSR | pharma_com:pv |
| /pharma-com/api/v1/icsrs/signals | GET | Detect adverse event signals | pharma_com:pv |
| /pharma-com/api/v1/icsrs/meddra | POST | Encode MedDRA term | pharma_com:pv |
| /pharma-com/api/v1/submissions | POST | Initiate regulatory submission | pharma_com:submissions |
| /pharma-com/api/v1/open-payments | GET | Generate Open Payments report | pharma_com:reporting |
| /pharma-com/api/v1/signals/triage | POST | Compute signal triage score | pharma_com:pv |
| /pharma-com/api/v1/icsrs/duplicates | GET | Detect duplicate ICSRs | pharma_com:pv |
| /pharma-com/api/v1/capa | POST | Create CAPA from violation | pharma_com:capa |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| sample_pdma_compliance_required | Sample dispensed without PDMA compliance | Deny — complete PDMA workflow |
| sample_signature_required | HCP signature missing | Deny — capture signature |
| aggregate_spend_cap_enforced | Annual HCP spend exceeds cap | Deny — escalate to compliance |
| spend_pre_approval_required | Spend above 100 USD without pre-approval | Deny — obtain pre-approval |
| rep_certification_required | Rep lacks valid certification | Deny — complete certification |
| icsr_suspect_product_required | ICSR submitted with no suspect products | Deny — add suspect product |
| signal_ror_threshold | ROR >= configured threshold AND cases >= min_count | Emit signal_detected event |
| submission_authority_validated | Submission target not in supported list | Deny — correct authority code |
| open_payments_receipt_required | Spend record > $10 without receipt | Flag validation error on report |
| capa_due_date_required | CAPA created without due date | Deny — set due date |

## Data Models
- Territory: territory_type, name, owner_id, product_ids, approval_reference
- SalesRep: rep_type, territory_id, quota, certification_reference
- CallRecord: physician_id, call_type, products_discussed, outcome
- SampleDispensing: sample_type, lot_number, expiry_date, pdma_compliant, hcp_signature_reference
- HcpInteraction: hcp_id, interaction_type, spend_amount, spend_category
- CommercialPlan: plan_period, territory_ids, product_ids, total_quota
- TargetPhysician: physician_id, tier, call_frequency_per_quarter
- AggregateSpendRecord: hcp_id, category, amount, fiscal_year
- ICSR: reporter_type, suspect_products, adverse_reactions, meddra_codes, seriousness_criteria, causality_assessment
- RegulatorySubmission: authority, submission_type, included_icsrs, tracking_number, status
- SignalTriageScore: product_id, reaction_term, composite_score, tier, components
- CAPA: violation_type, violation_reference, root_cause, corrective_action, preventive_action, due_date

## Streaming Events
- territory_created, territory_updated, rep_assigned, call_recorded
- sample_dispensed, sample_reconciled, interaction_recorded
- spend_recorded, plan_approved, compliance_flag_raised
- pdma_violation_detected, aggregate_spend_cap_exceeded
- icsr_created, meddra_term_encoded, signal_detected, signal_escalated
- regulatory_submission_initiated, open_payments_report_generated
- duplicate_icsrs_detected, capa_created

## Edge Cases Handled
- Aggregate spend cap tracked per HCP per fiscal year, not per transaction
- Sample dispensing blocked if PDMA workflow not completed regardless of signature presence
- Territory updates do not require re-approval, only initial creation does
- Cross-tenant isolation enforced at every read/write operation
- Plan approval workflow decoupled from creation to support multi-level approval chains
- ROR uses rule-of-three guard: signals suppressed below min_case_count even if ROR is high
- MedDRA coding falls back to partial substring match when exact term not found
- ICSR deduplication runs before regulatory submission packaging
- Regulatory submissions validate ICSR existence before assembling envelope
- Open Payments report flags missing receipts as validation errors, does not block report generation
- Signal triage score components are individually capped to prevent single-factor dominance

## Composability Notes
Composes with `pharma_rec` for Sunshine Act reporting obligations. Feeds spend data to `grc` for compliance reporting. Territory data feeds `pharma_com` forecasting into `pharma_sup` demand planning. ICSR signals route to `pvi` capability. CAPAs route to `qms` for quality lifecycle management. Triage scores feed `intel` dashboard for real-time PV monitoring.
