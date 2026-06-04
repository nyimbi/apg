# Licensing and Permits

## Overview
Business and professional licence applications, renewals, inspections, revocations, and fee collection with full compliance monitoring. Enforces that licences cannot be renewed if the last inspection failed, prevents duplicate licences, and requires formal notice before revocation.

## Capability ID
`government_lic`

## Provides
- licence_application_workflow: Application intake and document verification
- licence_issuance_workflow: Issue licence after approved application
- inspection_scheduling_workflow: Schedule and record licence inspections
- licence_renewal_workflow: Renewal with inspection pre-requisite check
- fee_collection_workflow: Application, renewal, and penalty fee collection
- licence_revocation_workflow: Revocation with notice period enforcement
- licensing_review_workflow: Governance review of licensing decisions
- licensing_agent_workflow: Automated renewal notification and compliance agents
- licence_status_tracking_workflow: Real-time licence status monitoring
- compliance_monitoring_workflow: Ongoing compliance against licence conditions

## Requires
| Capability | Reason |
|---|---|
| auth | Applicant and officer RBAC |
| audl | Licensing decision audit trail |
| mten | Tenant-scoped licence registry |
| conf | Licence type configuration and fee schedules |
| ntfy | Renewal reminders and inspection notifications |
| wflo | Application and approval workflow |
| schd | Inspection scheduling |
| comp | Regulatory compliance checks |
| moni | Expiry and compliance monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.licence_without_payment_denied | Application fee must be paid before processing |
| governance.expired_licence_operation_denied | Operations under expired licence blocked |
| governance.inspection_fail_blocks_renewal | Failed inspection prevents renewal |
| governance.duplicate_licence_denied | One active licence per holder per type |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-lic/applications | GET/POST | Licence applications | government_lic:apply |
| /government-lic/licences | GET | Licence register | government_lic:licences |
| /government-lic/inspections | GET/POST | Inspection schedule | government_lic:inspect |
| /government-lic/renewals | GET/POST | Licence renewals | government_lic:renew |
| /government-lic/fees | GET/POST | Fee collection | government_lic:fees |
| /government-lic/revocations | GET/POST | Revocations | government_lic:revoke |
| /government-lic/compliance | GET | Compliance dashboard | government_lic:compliance |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| application_fee_required | fee_paid=False | deny |
| duplicate_licence_denied | duplicate_detected=True | deny |
| renewal_inspection_fail_blocks | last_inspection_failed=True | deny |
| revocation_notice_required | notice_served=False | deny |
| revocation_approval_required | approval_present=False | deny |

## Data Models
- LicenceApplication: id, tenant_id, licence_type, applicant_id, status, fee_paid
- Licence: id, tenant_id, licence_type, licence_number, holder_id, expiry_date, status
- LicenceInspection: id, licence_id, inspection_type, inspector_id, scheduled_date, outcome
- LicenceRenewal: id, licence_id, renewal_type, new_expiry_date, renewal_fee_paid
- FeeRecord: id, application_id, fee_type, amount, receipt_number, paid
- LicenceRevocation: id, licence_id, reason, notice_served, approval_reference
- LicensingReview, LicensingAgent

## Streaming Events
- licence_application_submitted, licence_issued, inspection_scheduled, inspection_outcome_recorded
- licence_renewed, fee_collected, licence_suspended, licence_revoked, licence_expired

## Edge Cases Handled
- Renewal attempted after failed inspection — blocked until inspection passed
- Duplicate licence for same holder and type — blocked even if previous expired
- Revocation without notice period — denied regardless of severity of breach
- Late renewal creates `late_fee` obligation automatically
- Multiple inspection types can be active simultaneously for complex facilities

## Composability Notes
Composes with `government_csr` (licence applications submitted through citizen portal), `government_bud` (licence fees credited to AIA vote accounts), `government_cas` (licence complaints create cases), `government_con` (contractor registration uses professional licence), and `government_per` (building permits require valid contractor licence).
