# Service Provisioning

## Overview
Service activation and provisioning engine covering workflow orchestration, network resource reservation, configuration push to network elements via multiple protocols (NETCONF, RESTCONF, CLI, REST API), end-to-end activation verification, automated rollback on failure, and bulk provisioning with pre-approval gating.

## Capability ID
`telecom_pro`

## Provides
- service_activation_workflow: End-to-end service activation orchestration
- network_resource_allocation: Conflict-checked resource reservation and release
- configuration_push_workflow: Multi-protocol config push with dry-run
- activation_confirmation_workflow: E2E test and confirmation recording
- rollback_workflow: Automated and manual rollback on failure
- bulk_provisioning_workflow: Pre-approved bulk service activation
- pro_agent_workflow: Provisioning automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Provisioning event audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | Activation and failure notifications |
| wflo | Workflow state management |
| mqeb | Event streaming |
| moni | NE health monitoring |
| schd | Scheduled bulk job execution |

## Configuration
| Key | Description |
|-----|-------------|
| workflows.timeout_minutes | 60-minute workflow timeout |
| workflows.max_retries | Maximum 3 retries |
| resources.reservation_ttl_minutes | 30-minute reservation TTL |
| network_elements.health_check_before_push | Mandatory NE health check |
| config_push.dry_run_enabled | Dry run before live push |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-pro/workflows | GET/POST | Workflow console | telecom_pro:workflows |
| /telecom-pro/resources | GET/POST | Resource management | telecom_pro:resources |
| /telecom-pro/config-push | GET/POST | Config push console | telecom_pro:config_push |
| /telecom-pro/activation | GET/POST | Activation management | telecom_pro:activation |
| /telecom-pro/rollback | GET/POST | Rollback console | telecom_pro:rollback |
| /telecom-pro/bulk | GET/POST | Bulk provisioning | telecom_pro:bulk |
| /telecom-pro/network-elements | GET/POST | NE health console | telecom_pro:network_elements |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| workflow_type_not_supported | unknown type | deny |
| order_reference_required | no order reference | deny |
| resource_conflict_check_required | no conflict check | deny |
| dry_run_bypass_denied | dry_run_bypassed=True | deny |
| activation_verification_required | verification not completed | deny |
| bulk_provisioning_approval_required | no approval reference | deny |
| cross_tenant_provisioning_denied | cross-tenant scope | deny |

## Data Models
- ProWorkflow: id, tenant_id, workflow_type, order_reference, status, retry_count, started_at
- ProResourceReservation: id, tenant_id, workflow_id, resource_type, resource_value, reserved_at, expires_at, released
- ProConfigPush: id, tenant_id, workflow_id, ne_reference, push_method, template_reference, dry_run_completed, status
- ProActivation: id, tenant_id, workflow_id, service_reference, status, verification_completed, e2e_test_passed
- ProRollback: id, tenant_id, workflow_id, trigger, description, status, triggered_at
- ProBulkJob: id, tenant_id, workflow_type, item_count, approval_reference, status
- ProAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- workflow_queued, resource_reserved, config_push_dispatched, config_push_completed
- service_activated, activation_confirmed, workflow_failed, rollback_triggered, rollback_completed, pro_agent_registered

## Edge Cases Handled
- Dry run cannot be bypassed even by privileged agents — hard rule, not configurable
- Resource reservations expire after TTL; expired reservations are auto-released
- Rollback preserves the original workflow record with status=rolled_back for audit
- Bulk jobs require a separate approval from the order approval to prevent reuse of stale approvals
- NE health check failure blocks config push regardless of workflow priority

## Composability Notes
Receives provisioning tasks from telecom_ord (order decomposition). Reserves resources from telecom_inv (IPAM, circuit). Pushes configuration to NEs tracked in telecom_inv. Activation confirmation triggers telecom_cus (lifecycle event) and telecom_bil (charge activation).
