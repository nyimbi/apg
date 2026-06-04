# Mine Safety & Compliance

## Overview
Manages mine safety operations including incident reporting and investigation, hazard identification and risk assessment, risk register maintenance, permit-to-work issuance, corrective action tracking, compliance obligation registers, safety audits, and safety statistics reporting. Enforces statutory requirements including mandatory investigation before closing LTI and above incidents, stop-work authority for extreme risks, and issuer qualification checks for high-risk permits.

## Capability ID
`mining_saf`

## Provides
| Service | Description |
|---|---|
| incident_reporting_workflow | Full incident lifecycle from report to close with investigation gating |
| hazard_identification_workflow | Hazard record with risk assessment and control measures |
| risk_register_management | Enterprise risk register with inherent and residual risk ratings |
| permit_to_work_workflow | PTW issuance, validity checking, and closure |
| corrective_action_tracking | CA creation, assignment, due-date tracking, and overdue flagging |
| compliance_register_management | Regulatory obligation tracking |
| safety_audit_workflow | Internal and external audit management |
| emergency_drill_management | Emergency drill scheduling and results recording |
| safety_statistics_reporting | LTIFR, incident counts, open CA summary |
| stop_work_authority_workflow | Stop-work invocation gating for extreme risks |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication |
| audl | Immutable audit trail for statutory compliance |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration |
| ntfy | Immediate notifications for serious incidents and overdue CAs |
| wflo | Incident investigation and CA approval workflows |
| comp | Regulatory compliance obligation tracking |
| moni | Real-time safety metric monitoring |
| mqeb | Event streaming for safety dashboards |

## Configuration
| Key | Default | Description |
|---|---|---|
| incidents.immediate_notification_required | true | Fatality/LTI triggers immediate notification |
| incidents.investigation_required_for_lti_and_above | true | LTI, DO, fatality require investigation before close |
| hazards.risk_assessment_required | true | Risk assessment mandatory for all hazards |
| permits_to_work.issuer_qualification_required | true | PTW issuer must hold statutory qualification |
| permits_to_work.isolation_verification_required | true | Isolation must be verified before PTW issue |
| governance.open_extreme_risk_stop_work_trigger | true | Extreme hazards require stop-work before submission |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-saf/incidents | GET/POST | List/report incidents | mining_saf:view/write |
| /api/mining-saf/incidents/:id | GET | Get incident | mining_saf:view |
| /api/mining-saf/incidents/:id/investigate | POST | Open investigation | mining_saf:write |
| /api/mining-saf/incidents/:id/close | POST | Close incident | mining_saf:write |
| /api/mining-saf/incidents/:id/notify-regulatory | POST | Send regulatory notification | mining_saf:write |
| /api/mining-saf/hazards | GET/POST | List/identify hazards | mining_saf:view/write |
| /api/mining-saf/hazards/:id | GET | Get hazard | mining_saf:view |
| /api/mining-saf/hazards/:id/close | POST | Close hazard | mining_saf:write |
| /api/mining-saf/risk-register | GET/POST | List/add entries | mining_saf:view/write |
| /api/mining-saf/permits | GET/POST | List/issue PTWs | mining_saf:view/ptw_issue |
| /api/mining-saf/permits/:id | GET | Get PTW | mining_saf:view |
| /api/mining-saf/permits/:id/close | POST | Close PTW | mining_saf:ptw_issue |
| /api/mining-saf/permits/:id/valid | GET | Validity check | mining_saf:view |
| /api/mining-saf/corrective-actions | GET/POST | List/create CAs | mining_saf:view/write |
| /api/mining-saf/corrective-actions/:id/close | POST | Close CA | mining_saf:write |
| /api/mining-saf/corrective-actions/flag-overdue | POST | Flag overdue CAs | mining_saf:write |
| /api/mining-saf/statistics | GET | Safety statistics | mining_saf:reports |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| incident_immediate_notification | Fatality without notification | DENY |
| lti_investigation_required | Close LTI without investigation | DENY |
| extreme_risk_stop_work_trigger | Extreme hazard without stop-work | DENY |
| expired_ptw_access_denied | Access with expired PTW | DENY |
| ptw_issuer_qualification_required | Unqualified issuer | DENY |
| ptw_isolation_verification_required | Isolation not verified | DENY |
| stop_work_investigation_required | Resume after stop-work without investigation | DENY |
| delete_closed_incident_denied | Delete closed incident | DENY — archive instead |
| corrective_action_assignee_required | CA without assignee | DENY |
| corrective_action_due_date_required | CA without due date | DENY |

## Data Models
| Model | Key Fields |
|---|---|
| IncidentCreate/Response | incident_type, occurred_at, location, mine_area, description, investigation_id, status |
| HazardCreate/Response | hazard_category, risk_rating (inherent/residual), control_measures[], stop_work_invoked |
| RiskRegisterEntryCreate/Response | consequence, likelihood, inherent/residual_risk_rating, controls[], risk_owner_id |
| PermitToWorkCreate/Response | ptw_type, valid_from/to, issuer_id, isolation_points[], workers[] |
| CorrectiveActionCreate/Response | source_type, source_id, assignee_id, due_date, status, verified_by |

## Streaming Events
- `incident_reported` / `incident_escalated` / `incident_closed`
- `incident_investigation_opened`
- `hazard_identified` / `hazard_risk_assessed`
- `permit_issued` / `permit_closed`
- `corrective_action_assigned` / `corrective_action_overdue`
- `stop_work_authority_invoked`
- `emergency_drill_completed`

## Edge Cases Handled
- Fatalities trigger critical-level escalation log at report time regardless of notification flag
- Extreme risk hazards blocked at service layer if stop-work not invoked — not just a warning
- Expired PTW validity check is real-time (compares valid_to against utcnow())
- Cross-tenant access rejected with AssertionError
- Overdue CA flagging is idempotent; scan can be run on schedule without double-flagging

## Composability Notes
- Incident data feeds `mining_env` exceedance records for environmental incidents
- Stop-work events feed `mining_pro` blast management hold status
- Corrective actions link to `mining_eqp` work orders for equipment-related defects
- Compliance obligations integrate with `comp` regulatory calendar
- Training matrix feeds auth role-based access checks for PTW issuance
