# Government Contracts and Procurement

## Overview
End-to-end public procurement process covering tender management, bid evaluation, contract award, contract lifecycle management, variation control, performance monitoring, and PPDA compliance. Enforces the Public Procurement and Disposal Act requirements including debarment register and mandatory notifications.

## Capability ID
`government_con`

## Provides
- tender_management_workflow: Publish and manage procurement tenders
- evaluation_workflow: Structured bid evaluation against defined criteria
- contract_award_workflow: Award contracts with PPDA notification
- contract_lifecycle_workflow: Manage signed contract from inception to close
- contract_variation_workflow: Process approved variations with PPDA notification
- contract_performance_workflow: Monitor and record contractor performance
- ppda_compliance_workflow: PPDA submission, threshold tracking, annual reporting
- procurement_review_workflow: Governance review of procurement decisions
- procurement_agent_workflow: Automated compliance and evaluation agents
- debarment_register_workflow: Maintain and enforce bidder debarment register

## Requires
| Capability | Reason |
|---|---|
| auth | Procurement officer and approver RBAC |
| audl | Immutable procurement audit trail |
| mten | Tenant-scoped procurement isolation |
| conf | Procurement thresholds and approval limits |
| ntfy | PPDA notifications and award notices |
| wflo | Evaluation and approval workflow |
| comp | PPDA Act compliance checks |
| moni | Contract performance monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.award_without_evaluation_denied | Require completed evaluation before award |
| governance.single_source_requires_justification | Direct procurement needs written justification |
| governance.debarred_bidder_denied | Block debarred bidders from evaluation |
| ppda_compliance.debarment_register_enabled | Maintain active debarment register |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-con/tenders | GET/POST | Tender management | government_con:tenders |
| /government-con/evaluations | GET/POST | Evaluation workbench | government_con:evaluate |
| /government-con/awards | GET/POST | Contract awards | government_con:award |
| /government-con/contracts | GET/POST | Contract ledger | government_con:contracts |
| /government-con/variations | GET/POST | Contract variations | government_con:vary |
| /government-con/ppda | GET/POST | PPDA compliance | government_con:ppda |
| /government-con/debarment | GET/POST | Debarment register | government_con:debarment |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| award_evaluation_required | approved_evaluation_present=False | deny |
| single_source_requires_justification | method=direct, justification=False | deny |
| debarred_bidder_denied | bidder_debarred=True | deny |
| variation_ppda_notification_required | ppda_notification=False | deny |
| contract_signed_by_required | signed_by=False | deny |

## Data Models
- Tender: id, tenant_id, procurement_method, ppda_threshold, title, status, justification
- TenderEvaluation: id, tender_id, bidder_id, criteria, score, evaluator_id
- ContractAward: id, tender_id, awarded_to, awarded_amount, ppda_notification_reference
- GovernmentContract: id, award_id, contract_type, contract_value, start_date, end_date, status
- ContractVariation: id, contract_id, variation_type, value_change, ppda_notification_reference
- ContractPerformance: id, contract_id, performance_status, reviewer_id, period
- DebarredBidder: id, bidder_id, reason, debarred_until
- PpdaCompliance, ProcurementReview, ProcurementAgent

## Streaming Events
- tender_published, tender_awarded, contract_signed, contract_varied
- contract_performance_recorded, ppda_submission_recorded, bidder_debarred, tender_cancelled

## Edge Cases Handled
- Direct procurement without written justification — denied
- Award attempted before evaluation is complete — denied
- Debarred bidder included in evaluation — denied at evaluation stage
- Contract variation without PPDA notification — denied
- Batch procurement events routed to non-bytewax processor — denied

## Composability Notes
Composes with `government_bud` (contract awards create budget commitments), `government_per` (construction contracts require building permits), `government_cas` (procurement complaints create cases), and `intel` (tender pattern analysis for anti-corruption intelligence).

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements: Government Contracts & Procurement (government_con)
- **I2.** Overview
- **I3.** Full Async Service Layer
- **I4.** Persistent PostgreSQL-Backed Repository
- **I5.** Structured Audit Trail with Immutable Event Store
- **I6.** AI-Powered Bid Collusion Detection
- **I7.** Real-Time PPDA Notification Gateway
- **I8.** Conflict-of-Interest Declaration Workflow
- **I9.** Multi-Criteria Weighted Scoring Engine
- **I10.** Contract Expiry & Renewal Alert Engine
- **I11.** Vendor Due-Diligence & Sanctions Screening
- **I12.** E-Procurement Portal Integration (IFMIS/G2B)
- **I13.** Procurement Plan vs. Actuals Variance Reporting
- **I14.** Digital Contract Signing with e-Signature
- **I15.** Anti-Corruption Pattern Intelligence Feed

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
