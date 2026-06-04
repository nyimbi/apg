# Smart Metering & AMI

## Overview
Smart Metering & AMI manages the full lifecycle of advanced metering infrastructure from meter registration through interval data collection, tamper detection with evidence workflows, remote connect/disconnect with approval controls, demand response event coordination with customer opt-out, and data quality flagging. It also monitors AMI head-end connectivity ratios across communication technologies.

## Capability ID
`energy_met`

## Provides
| Service | Description |
|---|---|
| `meter_registry` | Register and maintain smart meter inventory by type, technology and customer |
| `ami_head_end_management` | Monitor AMI head-end connectivity ratio and protocol health |
| `interval_data_collection` | Collect and store interval readings with quality flags |
| `tamper_detection` | Detect, classify and investigate tamper events with evidence |
| `remote_connect_disconnect` | Issue and track remote commands with approval controls |
| `demand_response_coordination` | Manage DR events, track opt-outs, record actual reductions |
| `data_quality_management` | Flag and resolve reading quality issues |
| `meter_data_export` | Export interval data for billing and market settlement |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and permission checks |
| `audl` | Audit trail for tamper events and remote commands |
| `mten` | Multi-tenant meter data isolation |
| `conf` | Head-end protocol and interval configuration |
| `ntfy` | Tamper alerts and DR event notifications |
| `wflo` | Disconnect approval and tamper investigation workflows |
| `moni` | AMI head-end health monitoring |
| `mqeb` | Event streaming for tamper and command lifecycle |
| `schd` | Scheduled DR events and batch read jobs |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `readings.retention_days` | int | 730 | Interval data retention period |
| `commands.retry_limit` | int | 3 | Max command retry attempts |
| `commands.approval_required_for_disconnect` | bool | true | Disconnect requires approval |
| `demand_response.opt_out_allowed` | bool | true | Customers can opt out of DR |
| `demand_response.notification_required` | bool | true | Notify customers before DR event |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-met/api/v1/dashboard` | GET | Dashboard summary | `energy_met:view` |
| `/energy-met/api/v1/meters` | GET | List meters | `energy_met:meters` |
| `/energy-met/api/v1/meters` | POST | Register meter | `energy_met:meters` |
| `/energy-met/api/v1/meters/<id>` | GET | Meter detail | `energy_met:meters` |
| `/energy-met/api/v1/readings` | POST | Submit interval reading | `energy_met:readings` |
| `/energy-met/api/v1/tamper` | GET | List tamper events | `energy_met:tamper` |
| `/energy-met/api/v1/tamper` | POST | Report tamper | `energy_met:tamper` |
| `/energy-met/api/v1/tamper/<id>/resolve` | PUT | Resolve tamper | `energy_met:tamper` |
| `/energy-met/api/v1/commands` | POST | Issue remote command | `energy_met:commands` |
| `/energy-met/api/v1/commands/<id>/acknowledge` | PUT | Acknowledge command | `energy_met:commands` |
| `/energy-met/api/v1/demand-response` | POST | Create DR event | `energy_met:demand_response` |
| `/energy-met/api/v1/demand-response/<id>/opt-out` | POST | Meter opt-out | `energy_met:demand_response` |
| `/energy-met/api/v1/data-quality` | POST | Set quality flag | `energy_met:data_quality` |
| `/energy-met/api/v1/head-end` | POST | Update head-end status | `energy_met:admin` |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `meter_type_supported` | meter_type not in supported list | deny |
| `meter_serial_required` | serial_present=False | deny |
| `reading_meter_active` | meter.status != active | deny |
| `disconnect_approval_required` | disconnect command without approval | deny |
| `firmware_update_approval_required` | firmware update without approval | deny |
| `tamper_evidence_required` | evidence_present=False on tamper report | deny |
| `dr_opt_out_respected` | customer_opted_out=True on DR activation | deny |
| `dr_notification_required` | notification_sent=False on DR creation | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `privileged_met_agent_requires_human_approval` | agent disconnect without human approval | deny |

## Data Models
| Model | Key Fields |
|---|---|
| `SmartMeter` | id, serial_number, meter_type, communication_technology, status, customer_id, load_limit_kw |
| `IntervalReading` | id, meter_id, reading_type, interval_length, value, unit, quality_flag |
| `TamperEvent` | id, meter_id, tamper_type, detected_at, evidence_reference, status |
| `RemoteCommand` | id, meter_id, command_type, status, issued_by, approved_by, retry_count |
| `DemandResponseEvent` | id, event_type, target_reduction_kw, actual_reduction_kw, meter_ids, opt_out_meter_ids |
| `DataQualityFlag` | id, reading_id, quality_flag, reason, substitute_value |
| `AmiHeadEndStatus` | id, head_end_name, protocol, connected_meters, total_meters, communication_ratio |

## Streaming Events
- `meter_registered` / `meter_status_changed`
- `interval_reading_received`
- `tamper_event_detected`
- `remote_command_sent` / `remote_command_executed`
- `demand_response_event_created` / `demand_response_event_completed`
- `data_quality_flag_set`
- `ami_head_end_heartbeat`

## Edge Cases Handled
- Readings rejected for inactive, tampered or disconnected meters
- Disconnect command requires explicit approval — on_demand_read does not
- Firmware update treated as privileged command with separate approval rule
- DR opt-out list checked per meter before activating DR event
- Head-end marked "degraded" when communication ratio drops below 90%
- Quality flag substitution value stored separately from original reading

## Composability Notes
- Interval data feeds `energy_bil` for consumption billing
- Tamper events escalate to `intel` threat detection workflows
- DR events coordinate with `energy_grd` for system-level demand management
- Disconnected meters update `energy_dis` customer outage counts
- AMI head-end health feeds `moni` operational dashboards
