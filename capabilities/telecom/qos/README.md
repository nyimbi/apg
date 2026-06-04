# Quality of Service

## Overview
QoS policy management and enforcement covering bearer QoS, traffic shaping and policing, SLA parameter measurement, real-time degradation detection with root cause analysis, automated and manual remediation workflows, and PCRF/PCEF integration for policy enforcement on network elements.

## Capability ID
`telecom_qos`

## Provides
- qos_policy_management_workflow: Policy creation, modification, and conflict resolution
- traffic_prioritisation_workflow: DPI-based traffic classification and marking
- sla_enforcement_workflow: Per-customer SLA parameter measurement
- degradation_detection_workflow: Real-time QoS degradation detection
- root_cause_analysis_workflow: Evidence-backed root cause attribution
- auto_remediation_workflow: Configurable auto-remediation with disruptive action gating
- qos_reporting_workflow: QoS performance reporting
- qos_agent_workflow: QoS automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Policy change audit trail |
| mten | Tenant isolation |
| conf | QoS configuration |
| ntfy | SLA breach and degradation notifications |
| moni | Real-time monitoring |
| mqeb | Event streaming |
| wflo | Remediation approval workflows |

## Configuration
| Key | Description |
|-----|-------------|
| policies.conflict_detection | Enabled by default |
| degradation.confidence_threshold | 0.85 minimum confidence |
| remediation.human_approval_for_disruptive | Required for bearer re-establishment etc. |
| sla.measurement_interval_seconds | 60-second measurement cycle |
| governance.qos_downgrade_requires_approval | Always required |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-qos/policies | GET/POST | QoS policy console | telecom_qos:policies |
| /telecom-qos/traffic | GET/POST | Traffic classification | telecom_qos:traffic |
| /telecom-qos/enforcement | GET/POST | Enforcement status | telecom_qos:enforcement |
| /telecom-qos/sla | GET/POST | SLA measurement | telecom_qos:sla |
| /telecom-qos/degradation | GET/POST | Degradation console | telecom_qos:degradation |
| /telecom-qos/root-cause | GET/POST | Root cause analysis | telecom_qos:degradation |
| /telecom-qos/remediation | GET/POST | Remediation management | telecom_qos:remediation |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| qos_policy_approval_required | no approval on create | deny |
| qos_conflict_check_required | conflict not checked | deny |
| qos_downgrade_approval_required | downgrade without approval | deny |
| degradation_confidence_required | no confidence score | deny |
| disruptive_remediation_approval_required | disruptive + no approval | deny |
| cross_tenant_qos_denied | cross-tenant agent scope | deny |
| unapproved_policy_change_denied | agent changes policy | deny |

## Data Models
- QosPolicy: id, tenant_id, policy_type, qos_class, name, parameters, approval_reference, status
- QosTrafficClassification: id, tenant_id, traffic_type, classification, policy_id, flow_reference
- QosEnforcementRecord: id, tenant_id, policy_id, ne_reference, status, enforced_at
- QosSlasMeasurement: id, tenant_id, sla_parameter, measured_value, target_value, customer_id, is_breach
- QosDegradation: id, tenant_id, cause, confidence_score, description, affected_resource, status
- QosRootCause: id, tenant_id, degradation_id, root_cause_description, confidence_score
- QosRemediation: id, tenant_id, degradation_id, remediation_type, is_disruptive, approval_reference, status
- QosAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- qos_policy_activated, qos_policy_changed, sla_breach_detected, degradation_detected
- root_cause_identified, remediation_triggered, remediation_completed, traffic_anomaly_detected, qos_agent_registered

## Edge Cases Handled
- SLA breach direction is parameter-dependent: latency/loss/jitter → breach if measured > target; throughput/availability → breach if measured < target
- QoS downgrade (reducing GBR, increasing latency target) requires explicit approval regardless of policy creator permissions
- Disruptive remediations (bearer re-establishment) require approval even when degradation confidence is 0.99
- Policy conflict detection is performed client-side before submission — server blocks if conflict_checked=False
- Non-disruptive remediations (load balancing, traffic steering) can be auto-triggered without approval

## Composability Notes
Consumes performance data from telecom_per (KPI breaches trigger degradation detection). Pushes policy changes through telecom_pro (config push to PCRF). SLA breach data feeds telecom_bil (SLA credit) and telecom_per (compliance tracking). Degradation root causes feed telecom_net (alarm correlation).
