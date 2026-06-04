# Network Management

## Overview
Network operations centre capability providing fault management with alarm correlation, performance monitoring with threshold alerting, configuration change management with freeze period enforcement, SLA monitoring, and NOC shift handover management. Designed for 24×7 NOC operations with a dark-themed UI.

## Capability ID
`telecom_net`

## Provides
- fault_management_workflow: Alarm raise, acknowledge, clear, suppress lifecycle
- performance_management_workflow: KPI collection, threshold alerting, trending
- configuration_management_workflow: Change request, approval, execution, rollback
- sla_monitoring_workflow: SLA measurement, breach detection, penalty reporting
- noc_operations_workflow: Shift management, handover, escalation
- alarm_correlation_workflow: Cross-domain alarm grouping
- change_management_workflow: Full ITIL-aligned change lifecycle
- net_agent_workflow: Network operations automation agents

## Requires
| Capability | Reason |
|------------|--------|
| auth | NOC operator authentication |
| audl | Change and alarm audit trail |
| mten | Tenant isolation |
| conf | Threshold and SLA configuration |
| ntfy | SLA breach and critical alarm notifications |
| wflo | Change approval workflows |
| moni | Infrastructure monitoring |
| mqeb | Event streaming |
| schd | Scheduled maintenance windows |

## Configuration
| Key | Description |
|-----|-------------|
| faults.supported_severities | critical/major/minor/warning/informational |
| configuration.change_freeze_enabled | Freeze period enforcement |
| configuration.rollback_enabled | Auto-rollback on failure |
| sla.breach_alerting | Breach notifications enabled |
| governance.alarm_suppression_requires_approval | Cannot suppress without approval |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-net/alarms | GET/POST | Alarm console | telecom_net:faults |
| /telecom-net/faults | GET/POST | Fault ticket queue | telecom_net:faults |
| /telecom-net/performance | GET/POST | Performance console | telecom_net:performance |
| /telecom-net/config-changes | GET/POST | Change management | telecom_net:config |
| /telecom-net/sla | GET/POST | SLA monitoring | telecom_net:sla |
| /telecom-net/noc | GET | NOC operations view | telecom_net:noc |
| /telecom-net/escalations | GET/POST | Escalation management | telecom_net:escalations |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| fault_severity_not_supported | unknown severity | deny |
| alarm_suppression_approval_required | suppress without approval | deny |
| config_change_approval_required | change without approval | deny |
| change_freeze_period_active | in freeze, no emergency override | deny |
| handover_notes_required | empty handover notes | deny |
| unapproved_config_change_denied | agent changes without approval | deny |

## Data Models
- NetAlarm: id, tenant_id, ne_reference, severity, category, status, raised_at, cleared_at
- NetFaultTicket: id, tenant_id, alarm_id, title, severity, assigned_to, escalation_level, status
- NetPerformanceRecord: id, tenant_id, ne_reference, metric_type, value, threshold, domain
- NetConfigChange: id, tenant_id, ne_reference, change_type, description, status, approval_reference
- NetSlaRecord: id, tenant_id, sla_type, customer_id, target_value, actual_value, period, status
- NetNocHandover: id, tenant_id, shift, handing_over_operator, taking_over_operator, notes, open_alarms_count
- NetAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- alarm_raised, alarm_cleared, fault_ticket_opened, fault_ticket_resolved
- performance_threshold_breached, config_change_approved, config_change_completed
- sla_breach_detected, noc_escalation_triggered, net_agent_registered

## Edge Cases Handled
- Emergency changes in freeze periods require explicit emergency_change type, not just override flag
- Alarm suppression always requires approval — no agent can suppress autonomously
- SLA status=breached when actual < target (availability); reversed for latency metrics
- NOC handover notes are mandatory — empty string is rejected, not silently accepted
- Performance records above threshold fire audit events immediately, not on batch flush

## Composability Notes
Feeds alarm and performance data to telecom_ana (analytics) and telecom_per (KPI management). Config changes consume resource data from telecom_inv. SLA breach events trigger telecom_bil (SLA credits) and comp (regulatory SLA reporting).
