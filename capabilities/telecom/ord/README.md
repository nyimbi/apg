# Order Management

## Overview
End-to-end service order management covering order capture, validation, decomposition into provisioning tasks, orchestration, fallout management, number portability, bulk order processing, and real-time order tracking. Enforces duplicate detection and requires explicit approval for bulk operations.

## Capability ID
`telecom_ord`

## Provides
- order_capture_workflow: Multi-channel service order intake
- order_validation_workflow: Pre-provisioning checks and constraint validation
- order_decomposition_workflow: Order → parallel task decomposition
- provisioning_orchestration_workflow: Task dependency-aware execution
- fallout_management_workflow: Automated retry with escalation threshold
- order_tracking_workflow: Real-time status with customer notifications
- number_portability_workflow: Donor/recipient portability request management
- ord_agent_workflow: Order automation agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Order event audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | Order status notifications |
| wflo | Approval and state workflows |
| mqeb | Event streaming |
| schd | Bulk order scheduling |
| comp | Portability regulatory compliance |

## Configuration
| Key | Description |
|-----|-------------|
| orders.sla_hours | Priority-based SLA: emergency=1h, urgent=2h, high=4h |
| fallout.max_retries | Maximum 3 auto-retries before escalation |
| fallout.escalation_threshold_minutes | Escalate after 30 minutes in fallout |
| decomposition.parallel_execution | Tasks run in parallel where no dependency |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-ord/orders | GET/POST | Order console | telecom_ord:orders |
| /telecom-ord/orders/<id> | GET | Order detail | telecom_ord:orders |
| /telecom-ord/decomposition | GET/POST | Task decomposition | telecom_ord:decomposition |
| /telecom-ord/tasks | GET/POST | Task queue | telecom_ord:tasks |
| /telecom-ord/fallout | GET/POST | Fallout management | telecom_ord:fallout |
| /telecom-ord/portability | GET/POST | Number portability | telecom_ord:portability |
| /telecom-ord/bulk | GET/POST | Bulk orders | telecom_ord:bulk |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| order_type_not_supported | unknown order type | deny |
| duplicate_order_detected | is_duplicate=True | deny |
| customer_reference_required | no customer_id | deny |
| order_must_be_valid_for_decomposition | not yet validated | deny |
| msisdn_required_for_portability | no MSISDN | deny |
| bulk_order_approval_required | no approval reference | deny |

## Data Models
- OrdOrder: id, tenant_id, order_type, customer_id, channel, priority, status, submitted_at
- OrdTask: id, tenant_id, order_id, task_type, status, depends_on, assigned_to, completed_at
- OrdFallout: id, tenant_id, order_id, fallout_category, description, retry_count, resolution, status
- OrdPortabilityRequest: id, tenant_id, order_id, msisdn, donor_operator, recipient_operator, status
- OrdBulkOrder: id, tenant_id, order_type, item_count, approval_reference, status
- OrdAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- order_submitted, order_validated, order_decomposed, task_completed
- order_fallout, order_retry, provisioning_completed, order_completed, order_cancelled, ord_agent_registered

## Edge Cases Handled
- Decomposition requires validated status — submitted-but-not-validated orders cannot be decomposed
- Fallout retry counter increments each retry; exceeding max_retries triggers escalation flag
- Portability requires both MSISDN and donor_operator — partial portability requests denied
- Bulk order approval is separate from individual order approval to prevent privilege escalation
- Task depends_on is stored as a string reference, not a foreign key — allows cross-service dependencies

## Composability Notes
Triggers telecom_pro (provisioning workflows) on decomposition. Validates customer data against telecom_cus. Checks network resource availability against telecom_inv. Order completion triggers telecom_bil (charge setup) and telecom_cus (lifecycle event).
