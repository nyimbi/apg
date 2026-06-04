# Case Management

## Overview
Citizen case intake, assignment, workflow routing, SLA tracking, escalation, and outcome recording for government service delivery. Handles complaints, enquiries, applications, and regulatory referrals across all intake channels with full audit trail.

## Capability ID
`government_cas`

## Provides
- case_intake_workflow: Multi-channel citizen case intake
- case_assignment_workflow: Route cases to officers, teams, or agencies
- case_routing_workflow: Intelligent routing based on case type and priority
- sla_tracking_workflow: Monitor SLA deadlines and trigger alerts
- case_escalation_workflow: Escalate breached or complex cases
- case_outcome_workflow: Record and approve case resolutions
- case_notification_workflow: Notify citizens of case status changes
- case_review_workflow: Governance review of case handling quality
- case_agent_workflow: Automated case routing and triage agents
- citizen_case_portal_workflow: Citizen-facing case status tracking

## Requires
| Capability | Reason |
|---|---|
| auth | Officer authentication and RBAC |
| audl | Immutable audit of all case actions |
| mten | Tenant-scoped case isolation |
| conf | SLA thresholds and routing rules |
| ntfy | Citizen SMS/email notifications |
| wflo | Case state machine and approval workflows |
| nlpc | Case text search and auto-classification |
| srch | Full-text case search |
| moni | SLA breach monitoring and alerting |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.citizen_data_privacy_enforced | GDPR/data protection compliance |
| governance.sla_breach_triggers_escalation | Auto-escalate on SLA breach |
| sla.supported_sla_categories | statutory, urgent, standard, ministerial, court_ordered |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-cas/intake | POST | Open new case | government_cas:create |
| /government-cas/cases | GET | List cases queue | government_cas:cases |
| /government-cas/assignments | POST | Assign case | government_cas:assign |
| /government-cas/escalations | POST | Escalate case | government_cas:escalate |
| /government-cas/sla | GET | SLA tracking view | government_cas:sla |
| /government-cas/outcomes | POST | Record outcome | government_cas:outcomes |
| /government-cas/search | GET | Search cases | government_cas:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| tenant_context_required | tenant_context_present=False | deny |
| cross_tenant_case_denied | cross_tenant=True | deny |
| outcome_approval_required | approval_present=False | deny |
| escalation_supervisor_required | supervisor_present=False | deny |
| unauthenticated_submission_denied | authenticated=False | deny (implicit) |

## Data Models
- CitizenCase: id, tenant_id, case_type, intake_channel, citizen_id, priority, status, subject
- CaseAssignment: id, tenant_id, case_id, assignment_type, assignee_id
- CaseEscalation: id, tenant_id, case_id, escalation_reason, supervisor_id
- SlaRecord: id, tenant_id, case_id, sla_category, due_date, breached
- CaseOutcome: id, tenant_id, case_id, outcome_type, approval_reference
- CaseNotification, CaseReview, CaseAgent

## Streaming Events
- case_opened, case_assigned, case_escalated, case_sla_breached
- case_outcome_recorded, case_closed, case_notification_sent, case_reopened

## Edge Cases Handled
- SLA breach auto-escalation when `sla_breach_triggers_escalation` is enabled
- Duplicate case detection via citizen_id + case_type matching
- Cross-tenant case access always denied regardless of user role
- Outcome recording without approval reference — denied
- Citizen privacy data never exposed across tenant boundaries

## Composability Notes
Composes with `government_csr` (portal applications escalate to case management), `government_lic` (licence complaints create cases), `government_law` (case outcomes may trigger prosecution referrals), and `intel` (case pattern analytics for policy intelligence).
