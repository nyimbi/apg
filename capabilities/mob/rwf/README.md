# Remote Workforce

## Overview
The Remote Workforce (RWF) capability provides a complete remote and hybrid work governance runtime. It manages remote work policy authoring, activation, and employee acknowledgment; VPN access provisioning with MFA enforcement and split-tunneling prevention; consent-based productivity tracking; equipment requisition with per-employee limits; digital onboarding orchestration with step tracking; remote compliance checks; and remote incident management — all governed by tenant-scoped deterministic rules with full audit trails.

## Capability ID
`mob_rwf`

## Provides
| Service | Description |
|---------|-------------|
| remote_work_policy_management | Author, version, activate, and retire remote work policies |
| vpn_access_governance | Provision, revoke, and session-track VPN access with MFA enforcement |
| productivity_tracking_workflow | Consent-gated employee productivity metric recording and aggregation |
| equipment_requisition_workflow | Request, approve, ship, deliver, and return remote work equipment |
| digital_onboarding_workflow | Manager-approved onboarding with step-by-step completion tracking |
| remote_compliance_monitoring | Record and track remote compliance check results with next-due scheduling |
| remote_incident_management | Raise, track, and resolve remote workforce security and policy incidents |
| onboarding_step_orchestration | Orchestrate 8 standard onboarding steps with progress tracking |
| policy_acknowledgment_workflow | Employee acknowledgment recording with IP/device audit trail |
| remote_workforce_analytics | Dashboard aggregation of policies, VPN, equipment, onboarding, and incidents |

## Requires
| Capability | Reason |
|------------|--------|
| auth | User authentication and token validation |
| audl | Audit trail for all state-changing operations |
| mten | Multi-tenancy enforcement |
| conf | Runtime configuration management |
| ntfy | Incident and compliance alert notifications |
| nlpc | NLP for policy content search and summarisation |
| moni | Operational monitoring of VPN sessions and incidents |
| wflo | Multi-stage approval for policy activation and equipment requisition |
| schd | Scheduling next compliance check due dates |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| vpn.mfa_required | true | Enforce MFA before VPN provisioning |
| vpn.split_tunneling_allowed | false | Deny split-tunneling requests |
| vpn.max_session_hours | 12 | Maximum VPN session duration |
| productivity.tracking_consent_required | true | Require explicit employee consent |
| productivity.aggregation_only | true | Only store aggregated metrics, not raw events |
| equipment.max_items_per_employee | 5 | Equipment per-employee limit |
| compliance.check_interval_days | 30 | Interval between required compliance checks |
| governance.onboarding_requires_manager_approval | true | Manager approval gates onboarding start |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/mob/rwf/contract | GET | Return capability contract | mob_rwf:view |
| /api/mob/rwf/policies | GET | List work policies | mob_rwf:policies:list |
| /api/mob/rwf/policies | POST | Create work policy | mob_rwf:policies:create |
| /api/mob/rwf/policies/<id> | GET | Get policy detail | mob_rwf:policies:view |
| /api/mob/rwf/policies/<id> | PUT | Update policy | mob_rwf:policies:edit |
| /api/mob/rwf/policies/<id>/activate | POST | Activate policy | mob_rwf:policies:activate |
| /api/mob/rwf/policies/<id>/acknowledge | POST | Record acknowledgment | mob_rwf:policies:acknowledge |
| /api/mob/rwf/policies/<id>/acknowledgments | GET | List acknowledgments | mob_rwf:policies:view |
| /api/mob/rwf/vpn | GET | List VPN access | mob_rwf:vpn:list |
| /api/mob/rwf/vpn | POST | Provision VPN access | mob_rwf:vpn:provision |
| /api/mob/rwf/vpn/<id> | DELETE | Revoke VPN access | mob_rwf:vpn:revoke |
| /api/mob/rwf/vpn/<id>/sessions | POST | Start VPN session | mob_rwf:vpn:connect |
| /api/mob/rwf/vpn/sessions/<id>/end | POST | End VPN session | mob_rwf:vpn:connect |
| /api/mob/rwf/productivity | GET | List productivity metrics | mob_rwf:productivity:view |
| /api/mob/rwf/productivity | POST | Record metric | mob_rwf:productivity:write |
| /api/mob/rwf/productivity/<emp_id>/summary | GET | Get employee summary | mob_rwf:productivity:view |
| /api/mob/rwf/equipment | GET | List equipment | mob_rwf:equipment:list |
| /api/mob/rwf/equipment | POST | Request equipment | mob_rwf:equipment:request |
| /api/mob/rwf/equipment/<id>/approve | POST | Approve requisition | mob_rwf:equipment:approve |
| /api/mob/rwf/equipment/<id>/ship | POST | Mark shipped | mob_rwf:equipment:manage |
| /api/mob/rwf/equipment/<id>/deliver | POST | Mark delivered | mob_rwf:equipment:manage |
| /api/mob/rwf/equipment/<id>/return | POST | Mark returned | mob_rwf:equipment:manage |
| /api/mob/rwf/onboarding | GET | List onboarding records | mob_rwf:onboarding:list |
| /api/mob/rwf/onboarding | POST | Start onboarding | mob_rwf:onboarding:start |
| /api/mob/rwf/onboarding/<id> | GET | Get onboarding record | mob_rwf:onboarding:view |
| /api/mob/rwf/onboarding/<id>/steps | POST | Complete step | mob_rwf:onboarding:manage |
| /api/mob/rwf/compliance | GET | List compliance checks | mob_rwf:compliance:view |
| /api/mob/rwf/compliance | POST | Record check result | mob_rwf:compliance:record |
| /api/mob/rwf/incidents | GET | List incidents | mob_rwf:incidents:view |
| /api/mob/rwf/incidents | POST | Raise incident | mob_rwf:incidents:report |
| /api/mob/rwf/incidents/<id>/resolve | POST | Resolve incident | mob_rwf:incidents:resolve |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| write_requires_policy | operation_type=write, policy_attached=False | deny |
| work_policy_activation_requires_approval | operation=activate_work_policy, approval_present=False | deny |
| policy_acknowledgment_requires_active_policy | operation=acknowledge_policy, policy_state=draft | deny |
| vpn_requires_mfa | operation=provision_vpn, mfa_verified=False | deny |
| vpn_split_tunneling_denied | operation=provision_vpn, split_tunneling_requested=True | deny |
| productivity_tracking_requires_consent | operation=record_productivity, consent_given=False | deny |
| equipment_limit_per_employee | operation=request_equipment, equipment_limit_exceeded=True | deny |
| onboarding_requires_manager_approval | operation=start_onboarding, manager_approval_present=False | deny |
| revoked_vpn_blocks_session | vpn_state=revoked | deny |
| cross_tenant_access_denied | cross_tenant_access=True | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| WorkPolicyResponse | id, name, policy_type, state, version, acknowledgment_count |
| PolicyAcknowledgmentResponse | id, policy_id, employee_id, acknowledged_at, ip_address, device_id |
| VpnAccessResponse | id, employee_id, vpn_protocol, state, split_tunneling_enabled, expires_at |
| VpnSessionResponse | id, vpn_access_id, started_at, ended_at, bytes_in, bytes_out, duration_seconds |
| ProductivityMetricResponse | id, employee_id, metric_type, value, period_start, period_end, consent_given |
| EquipmentRequisitionResponse | id, employee_id, equipment_type, quantity, state, asset_tag |
| OnboardingRecordResponse | id, employee_id, state, completed_steps, pending_steps, completed_at |
| OnboardingStepResponse | id, onboarding_id, step_type, completed_by, completed_at |
| ComplianceCheckResponse | id, employee_id, check_type, result, next_due_at |
| RemoteIncidentResponse | id, employee_id, incident_type, severity, state, resolved_by |

## Streaming Events
- `work_policy_created` / `work_policy_activated` / `work_policy_acknowledged`
- `vpn_access_provisioned` / `vpn_access_revoked`
- `vpn_session_started` / `vpn_session_ended`
- `productivity_metric_recorded`
- `equipment_requested` / `equipment_approved` / `equipment_delivered` / `equipment_returned`
- `onboarding_started` / `onboarding_step_completed` / `onboarding_completed`
- `compliance_check_completed`
- `remote_incident_raised` / `remote_incident_resolved`

## Edge Cases Handled
- Policy acknowledgment blocked for draft and retired policies — only active policies can be acknowledged
- VPN split tunneling denied at rule-engine level even if client requests it
- Productivity tracking completely blocked without prior employee consent
- Equipment approval increments the per-employee count; return decrements it, restoring capacity
- Onboarding auto-transitions to `completed` when all pending steps are cleared
- Revoked and suspended VPN access blocks new session creation
- Policy version counter increments on every update, enabling change history
- Compliance next-due date automatically scheduled 30 days from check date
- VPN sessions track bytes transferred and compute duration on close

## Composability Notes
- `mob_mdm` device enrolment state can gate VPN provisioning for corporate devices
- `mob_map` biometric enrollment can be triggered as an onboarding step
- `wflo` handles multi-stage approval for equipment and policy activation
- `schd` schedules next compliance check due dates
- `ntfy` dispatches alerts for open incidents and overdue compliance checks
- `nlpc` enables semantic search over work policy content
- All events feed `mqeb` for downstream analytics in `moni` dashboards
