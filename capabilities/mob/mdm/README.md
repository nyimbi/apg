# Mobile Device Management

## Overview
The Mobile Device Management (MDM) capability provides an enterprise-grade device lifecycle management runtime. It covers device enrolment across multiple platforms and methods; deterministic policy creation, activation, and assignment; continuous compliance evaluation with automatic alert generation; silent app distribution; remote wipe with mandatory dual approval; MDM configuration profile deployment; and a device inventory registry — all tenant-scoped with full audit trails.

## Capability ID
`mob_mdm`

## Provides
| Service | Description |
|---------|-------------|
| device_enrolment_workflow | Enrol devices via DEP, zero-touch, QR, NFC, manual, and bulk CSV |
| mdm_policy_enforcement | Create, version, activate, and assign security/network/app/kiosk policies |
| compliance_monitoring | Continuous per-device compliance evaluation with severity-based findings |
| remote_wipe_workflow | Dual-approval remote wipe (full, selective, corporate, factory reset) |
| app_distribution_workflow | Required/available/blocked silent app install/removal |
| mdm_profile_deployment | Deploy config, certificate, VPN, WiFi, email, and custom profiles |
| device_lock_workflow | Lock, unlock, lost mode, and activation lock actions |
| enrolment_state_machine | Enrolled → suspended → unenrolled → wiped state transitions |
| corporate_wipe_workflow | Selective corporate data removal preserving personal data (BYOD) |
| device_inventory_registry | Full device inventory with asset tags, assignments, and last-seen tracking |

## Requires
| Capability | Reason |
|------------|--------|
| auth | User authentication and token validation |
| audl | Audit trail for all device and policy operations |
| mten | Multi-tenancy enforcement |
| conf | Runtime configuration |
| ntfy | Alert and notification dispatch |
| comp | Regulatory compliance framework integration |
| moni | Operational monitoring for fleet health |
| wflo | Approval workflow for policies and wipe requests |
| mqeb | Event streaming via Bytewax |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| devices.approval_required_for_enrolment | true | Require approval before device enrolment |
| policies.approval_required | true | Require approval before policy activation |
| remote_actions.wipe_requires_dual_approval | true | Two approvers required for any wipe |
| compliance.evaluation_interval_minutes | 60 | How often compliance is re-evaluated |
| compliance.grace_period_hours | 24 | Grace period before non-compliant blocking |
| governance.cross_tenant_access_denied | true | Prevent cross-tenant access |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /api/mob/mdm/contract | GET | Return capability contract | mob_mdm:view |
| /api/mob/mdm/devices | GET | List enrolled devices | mob_mdm:devices:list |
| /api/mob/mdm/devices | POST | Enrol a device | mob_mdm:enrolment:manage |
| /api/mob/mdm/devices/bulk-enrol | POST | Bulk enrol devices from list | mob_mdm:enrolment:manage |
| /api/mob/mdm/devices/<id> | GET | Get device detail | mob_mdm:devices:view |
| /api/mob/mdm/devices/<id> | PUT | Update device | mob_mdm:devices:edit |
| /api/mob/mdm/devices/<id>/unenrol | POST | Unenrol device | mob_mdm:enrolment:manage |
| /api/mob/mdm/devices/<id>/suspend | POST | Suspend device | mob_mdm:devices:manage |
| /api/mob/mdm/devices/<id>/health-score | GET | Get device health score | mob_mdm:devices:view |
| /api/mob/mdm/policies | GET | List policies | mob_mdm:policies:list |
| /api/mob/mdm/policies | POST | Create policy | mob_mdm:policies:create |
| /api/mob/mdm/policies/<id> | GET | Get policy | mob_mdm:policies:view |
| /api/mob/mdm/policies/<id> | PUT | Update policy | mob_mdm:policies:edit |
| /api/mob/mdm/policies/<id>/activate | POST | Activate policy | mob_mdm:policies:activate |
| /api/mob/mdm/policies/assign | POST | Assign policy to device | mob_mdm:policies:assign |
| /api/mob/mdm/device-groups | POST | Create device group | mob_mdm:groups:manage |
| /api/mob/mdm/device-groups/<id>/devices | POST | Add device to group | mob_mdm:groups:manage |
| /api/mob/mdm/device-groups/<id>/assign-policy | POST | Assign policy to group | mob_mdm:policies:assign |
| /api/mob/mdm/compliance | GET | List compliance records | mob_mdm:compliance:view |
| /api/mob/mdm/compliance | POST | Run compliance evaluation | mob_mdm:compliance:evaluate |
| /api/mob/mdm/apps | GET | List app distributions | mob_mdm:apps:list |
| /api/mob/mdm/apps | POST | Distribute app | mob_mdm:apps:distribute |
| /api/mob/mdm/remote-actions/wipes | GET | List wipe requests | mob_mdm:remote:wipe |
| /api/mob/mdm/remote-actions/wipes | POST | Request wipe | mob_mdm:remote:wipe |
| /api/mob/mdm/remote-actions/wipes/<id>/execute | POST | Execute wipe | mob_mdm:remote:wipe |
| /api/mob/mdm/remote-actions/wipes/<id>/cancel | POST | Cancel pending wipe | mob_mdm:remote:wipe |
| /api/mob/mdm/profiles | GET | List MDM profiles | mob_mdm:profiles:list |
| /api/mob/mdm/profiles | POST | Create MDM profile | mob_mdm:profiles:create |
| /api/mob/mdm/profiles/<id>/deploy/<device_id> | POST | Deploy profile to device | mob_mdm:profiles:deploy |
| /api/mob/mdm/alerts | GET | List MDM alerts | mob_mdm:alerts:view |
| /api/mob/mdm/alerts/<id>/resolve | POST | Resolve alert | mob_mdm:alerts:manage |
| /api/mob/mdm/certificates | GET | List tracked certificates | mob_mdm:certs:view |
| /api/mob/mdm/certificates | POST | Track a certificate | mob_mdm:certs:manage |
| /api/mob/mdm/audit-log | GET | Query audit log | mob_mdm:audit:view |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| write_requires_policy | operation_type=write, policy_attached=False | deny |
| device_type_must_be_supported | operation=enrol_device, device_type_supported=False | deny |
| enrolment_requires_approval | operation=enrol_device, approval_present=False | deny |
| policy_activation_requires_approval | operation=activate_policy, approval_present=False | deny |
| wipe_requires_dual_approval | operation=request_wipe, dual_approval_present=False | deny |
| unenrolled_device_blocks_app_install | operation=distribute_app, device_enrolled=False | deny |
| non_compliant_device_blocks_access | operation=grant_access, device_compliance_state=non_compliant | deny |
| suspended_device_blocks_all_actions | device_state=suspended | deny |
| wiped_device_blocks_all_actions | device_state=wiped | deny |
| cross_tenant_access_denied | cross_tenant_access=True | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| DeviceResponse | id, tenant_id, serial_number, device_type, os_platform, enrolment_state, compliance_state |
| PolicyResponse | id, name, policy_type, state, version, platform_targets |
| PolicyAssignmentResponse | id, policy_id, device_id, assigned_by, state |
| ComplianceRecordResponse | id, device_id, compliance_state, findings, evaluated_by, next_evaluation_at |
| AppDistributionResponse | id, app_bundle_id, device_id, distribution_type, state |
| WipeRequestResponse | id, device_id, wipe_type, approval_reference, second_approval_reference, state |
| MdmProfileResponse | id, name, profile_type, platform, state, deployed_to_count |
| MdmAlertResponse | id, device_id, alert_type, severity, message, resolved |

## Streaming Events
- `device_enrolled` / `device_unenrolled` / `device_suspended` / `device_wiped`
- `policy_created` / `policy_activated` / `policy_assigned` / `policy_assigned_to_group`
- `compliance_evaluated` / `compliance_state_changed`
- `app_distributed` / `app_removed`
- `profile_deployed` / `profile_removed`
- `wipe_requested` / `wipe_completed` / `wipe_cancelled`
- `device_locked` / `device_unlocked`
- `mdm_alert_raised`
- `device_group_created` / `device_added_to_group`
- `device_health_score_computed`
- `certificate_tracked` / `certificate_expiring_soon` / `certificate_expired`

## Edge Cases Handled
- Wipe execution transitions device to `wiped` state, blocking all further actions
- Non-compliant devices trigger automatic alerts at `high` severity
- Policy version counter increments on every update — version history is preserved
- Compliance evaluation only runs on enrolled devices (blocked otherwise)
- App distribution is blocked for unenrolled, suspended, and wiped devices
- Dual approval enforced at rule-engine level for all wipe types including selective
- Corporate wipe on BYOD preserves personal partition — wipe_type=corporate_wipe
- Profile deploy increments deployed_to_count rather than creating duplicate records
- Bulk enrolment skips duplicate serials rather than raising hard errors — returns counts
- Wipe cancellation only permitted in `pending` state; completed wipes cannot be rolled back
- Certificate tracking auto-raises `high` alert at ≤30 days and `critical` at ≤0 days remaining
- Device health score degrades immediately on any open `critical`/`high` alert
- Group policy assignment propagates to all current group members; failures are per-device isolated

## Composability Notes
- Exposes `device_enrolled` status consumed by `mob_map` for biometric enrollment gating
- Integrates with `comp` for regulatory compliance framework mapping
- Feeds `wflo` for multi-stage approval workflows (policy activation, wipe requests)
- Emits events to `mqeb` consumed by `moni` for fleet health dashboards
- Alert generation feeds `ntfy` for push/email/sms escalation
- `audl` receives every state-changing operation for compliance audit trails
