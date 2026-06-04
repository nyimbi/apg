# Permits Management

## Overview
Building permits, environmental permits, conditional approvals, inspection scheduling, and compliance monitoring. Prevents construction before permit issuance, enforces occupation certificate requirements, and triggers enforcement actions on condition breaches.

## Capability ID
`government_per`

## Provides
- permit_application_workflow: Permit application intake and assessment
- permit_issuance_workflow: Issue permit with conditions attached
- conditional_approval_workflow: Track and enforce permit conditions
- inspection_scheduling_workflow: Construction-phase inspection management
- permit_compliance_monitoring_workflow: Ongoing compliance against conditions
- permit_revocation_workflow: Revoke permits for serious breaches
- permits_review_workflow: Governance review of permit decisions
- permits_agent_workflow: Automated condition monitoring agents
- permit_transfer_workflow: Permit transfer on property sale
- enforcement_action_workflow: Stop-work orders and enforcement notices

## Requires
| Capability | Reason |
|---|---|
| auth | Applicant and inspector RBAC |
| audl | Permit decision audit trail |
| mten | Tenant-scoped permit registry |
| conf | Permit type configuration and fee schedules |
| ntfy | Condition due date alerts and inspection notices |
| wflo | Application and approval workflow |
| geos | Site mapping and boundary checking |
| schd | Inspection scheduling |
| comp | Planning Act and environmental compliance |
| moni | Condition expiry and compliance monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.construction_before_permit_denied | Commencement without permit always blocked |
| governance.occupation_before_final_inspection_denied | Occupation certificate requires final inspection pass |
| governance.condition_breach_triggers_enforcement | Automatic enforcement on major breach |
| governance.duplicate_permit_denied | One active permit per holder, type, and site |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-per/applications | GET/POST | Permit applications | government_per:apply |
| /government-per/permits | GET | Permit register | government_per:permits |
| /government-per/conditions | GET/POST | Permit conditions | government_per:conditions |
| /government-per/inspections | GET/POST | Inspection schedule | government_per:inspect |
| /government-per/compliance | GET/POST | Compliance monitoring | government_per:compliance |
| /government-per/enforcement | GET/POST | Enforcement actions | government_per:enforce |
| /government-per/map | GET | Permit site map | government_per:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| application_fee_required | fee_paid=False | deny |
| construction_before_permit_denied | permit_active=False | deny |
| occupation_final_inspection_required | final_inspection_passed=False | deny |
| duplicate_permit_denied | duplicate_detected=True | deny |
| condition_permit_required | permit_present=False | deny |

## Data Models
- PermitApplication: id, tenant_id, permit_type, applicant_id, site_reference, fee_paid, status
- Permit: id, tenant_id, permit_number, holder_id, site_reference, expiry_date, status
- PermitCondition: id, permit_id, condition_type, due_date, responsible_party, fulfilled
- PermitInspection: id, permit_id, inspection_type, inspector_id, scheduled_date, outcome
- ComplianceRecord: id, permit_id, compliance_status, officer_id, narrative
- EnforcementAction: id, permit_id, compliance_id, action_type, officer_id
- PermitReview, PermitsAgent

## Streaming Events
- permit_application_submitted, permit_issued, permit_condition_recorded, inspection_scheduled
- inspection_outcome_recorded, permit_compliance_updated, permit_revoked, enforcement_action_initiated

## Edge Cases Handled
- Construction commencement before permit is active — always denied
- Occupation without final inspection pass — denied regardless of other inspections
- Duplicate permit for same site and holder — blocked even if previous expired
- Pre-commencement conditions not fulfilled — permit cannot proceed to construction phase
- Environmental permit conditions have different enforcement pathways than building permits

## Composability Notes
Composes with `government_lic` (contractors need valid licence before permit issued), `government_con` (construction contracts reference building permits), `government_cas` (planning complaints become cases), `government_bud` (permit fees credit planning fund vote accounts), and `intel` (permit pattern analysis for urban planning intelligence).
