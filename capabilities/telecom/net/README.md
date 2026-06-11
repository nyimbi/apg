# Network Management

## Overview
Network operations centre capability providing fault management with alarm correlation, performance monitoring with threshold alerting, configuration change management with freeze period enforcement, SLA monitoring, NOC shift handover management, post-incident review generation, capacity trend forecasting, and SLA penalty calculation. Designed for 24×7 NOC operations with a dark-themed UI.

## Capability ID
`telecom_net`

## Provides
- fault_management_workflow: Alarm raise, acknowledge, clear, suppress lifecycle
- performance_management_workflow: KPI collection, threshold alerting, trending, capacity forecasting
- configuration_management_workflow: Change request, approval, execution, rollback, drift detection
- sla_monitoring_workflow: SLA measurement, breach detection, penalty calculation, benchmarking
- noc_operations_workflow: Shift management, handover, escalation, workload analysis
- alarm_correlation_workflow: Single-domain and cross-domain alarm grouping
- change_management_workflow: Full ITIL-aligned change lifecycle
- net_agent_workflow: Network operations automation agents
- pir_workflow: Post-incident review generation
- ne_health_workflow: Composite NE health scoring for topology view

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
| /telecom-net/correlations | GET/POST | Alarm correlations | telecom_net:faults |
| /telecom-net/topology | GET | NE health topology view | telecom_net:view |
| /telecom-net/agents | GET/POST | Automation agents | telecom_net:admin |

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

## Key Service Methods

### Fault Management
| Method | Description |
|--------|-------------|
| `raise_alarm` | Raise a network alarm from a network element |
| `update_alarm_status` | Acknowledge, clear, or update alarm lifecycle status |
| `suppress_alarm` | Suppress alarm — requires explicit approval reference |
| `fault_alert` | Raise alarm and auto-open ticket for critical/major faults |
| `fault_correlation` | Correlate a batch of alerts to find root events |
| `correlate_alarms` | Correlate specific alarm IDs into a single fault event |
| `cross_domain_correlation` | Multi-domain alarm correlation for shared upstream failures |

### Fault Tickets
| Method | Description |
|--------|-------------|
| `open_fault_ticket` | Open a fault ticket from a raised alarm |
| `resolve_fault_ticket` | Resolve and close a fault ticket |
| `escalate_fault` | Escalate ticket to a higher support tier or vendor |
| `trouble_ticket_create` | Create trouble ticket with priority-to-severity mapping |
| `trouble_ticket_update` | Update ticket with work note and optional status change |

### Root Cause & Post-Incident
| Method | Description |
|--------|-------------|
| `root_cause_analysis` | Record RCA findings for a fault ticket |
| `generate_pir` | Generate a Post-Incident Review report for a resolved ticket |

### Performance & Capacity
| Method | Description |
|--------|-------------|
| `record_performance` | Record a KPI metric from a network element |
| `performance_threshold_crossing` | Process a threshold-crossing event and auto-raise alarm |
| `performance_analytics` | Compute network performance KPIs for a period |
| `capacity_trend_forecast` | Forecast days-to-threshold-breach via linear trend |
| `ne_health_score` | Composite NE health score (0–100) for topology view |

### Configuration & Maintenance
| Method | Description |
|--------|-------------|
| `submit_config_change` | Submit a change request with approval reference |
| `complete_config_change` | Mark a config change as completed |
| `backup_config` | Back up running configuration of a network element |
| `network_configuration_backup` | Versioned config backup with retention (last 10) |
| `detect_configuration_drift` | Compare running config to approved baseline; raises alarm on drift |
| `planned_maintenance` | Schedule a maintenance window with conflict detection |
| `create_maintenance_window` | Create a maintenance window for alarm suppression |
| `close_maintenance_window` | Close a maintenance window and re-enable alarm processing |
| `firmware_upgrade` | Schedule firmware upgrade; auto-creates maintenance window |

### SLA
| Method | Description |
|--------|-------------|
| `record_sla` | Record an SLA measurement; auto-flags breaches |
| `sla_penalty_calculation` | Compute contractual penalty for a breached SLA record |
| `multi_tenant_sla_benchmark` | Cross-tenant SLA compliance ranking (admin-scoped) |

### NOC Operations
| Method | Description |
|--------|-------------|
| `record_noc_handover` | Record a NOC shift handover |
| `noc_shift_report` | Generate a NOC shift summary report |
| `noc_workload_analysis` | Analyse alarm volumes per shift for staffing recommendations |

### Dashboard & Reporting
| Method | Description |
|--------|-------------|
| `network_health_dashboard` | Comprehensive NOC health snapshot |
| `dashboard_summary` | Lightweight count-based dashboard summary |
| `network_compliance_report` | Compliance report against a named standard |
| `performance_analytics` | KPI aggregation for a reporting period |
| `export_network_data` | Export alarms and tickets as JSON or CSV |

### Automation Agents
| Method | Description |
|--------|-------------|
| `register_agent` | Register a network operations automation agent |
| `validate_agent_action` | Enforce agent scope and approval policies |
| `ml_network_fault_predict` | Ollama-backed ML fault prediction (optional) |

## Streaming Events
- alarm_raised, alarm_cleared, fault_ticket_opened, fault_ticket_resolved
- performance_threshold_breached, config_change_approved, config_change_completed
- sla_breach_detected, noc_escalation_triggered, net_agent_registered

## Edge Cases Handled
- Emergency changes in freeze periods require explicit emergency_change type, not just an override flag
- Alarm suppression always requires approval — no agent can suppress autonomously
- SLA status=breached when actual < target (availability); reversed for latency metrics
- NOC handover notes are mandatory — empty string is rejected, not silently accepted
- Performance records above threshold fire audit events immediately, not on batch flush
- Conflicting maintenance windows on the same NE are rejected at scheduling time
- Config drift detection auto-raises a `configuration_error` alarm without requiring manual intervention
- `generate_pir` computes MTTR from ticket timestamps; gracefully handles missing timestamps

## Composability Notes
Feeds alarm and performance data to `telecom_ana` (analytics) and `telecom_per` (KPI management). Config changes consume resource data from `telecom_inv`. SLA breach events and penalty credit notes target `telecom_bil` (billing). PIR reports feed `comp` (regulatory SLA reporting). NE health scores drive the `telecom_net` topology view component.
