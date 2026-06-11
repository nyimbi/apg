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
- cost_estimation_workflow: Pre-order itemised cost breakdown
- contract_lifecycle_workflow: e-Signature, renewal, and expiry management
- jeopardy_prediction_workflow: ML-assisted at-risk order scoring
- metrics_export_workflow: Prometheus / JSON operational metrics

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
| orders.sla_hours | Priority-based SLA: emergency=1h, urgent=2h, high=4h, normal=24h, low=72h |
| fallout.max_retries | Maximum 3 auto-retries before escalation |
| fallout.escalation_threshold_minutes | Escalate after 30 minutes in fallout |
| decomposition.parallel_execution | Tasks run in parallel where no dependency |
| idempotency.ttl_hours | Idempotency key TTL for safe order submission replay (default 24h) |

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
| /telecom-ord/contracts | GET/POST | Contract management | telecom_ord:contracts |
| /telecom-ord/metrics | GET | Prometheus metrics | telecom_ord:ops |
| /telecom-ord/webhooks | POST | Webhook registration | telecom_ord:orders |

## Service Methods

### Core Lifecycle
| Method | Description |
|--------|-------------|
| `submit_order()` | Submit a new service order |
| `validate_order()` | Mark order validated after pre-checks |
| `decompose_order()` | Decompose validated order into tasks |
| `create_task()` | Create a provisioning task |
| `complete_task()` | Mark task completed |
| `complete_order()` | Mark order fully completed |
| `record_fallout()` | Record order fallout event |
| `retry_fallout()` | Increment retry and re-queue |
| `resolve_fallout()` | Document fallout resolution |

### Async Workflows
| Method | Description |
|--------|-------------|
| `capture_order()` | Multi-product order bundle capture |
| `order_validation()` | Run structured validation checks |
| `credit_check_order()` | Customer credit bureau check |
| `contract_creation()` | Create service contract |
| `order_fallout()` | Error-code-classified fallout recording |
| `order_amendment()` | Amend in-flight order parameters |
| `order_cancellation()` | Cancel with reason and audit |
| `order_analytics()` | Completion/fallout/cancel rates by period |
| `order_sla_monitoring()` | Breaching and at-risk order detection |
| `bulk_order_import()` | CSV bulk import with per-row results |

### World-Class Enhancements (v1.1)
| Method | Description |
|--------|-------------|
| `validate_portability_eligibility()` | E.164 + operator + concurrent-port checks |
| `register_webhook()` | HMAC-signed lifecycle event webhooks |
| `predict_order_jeopardy()` | Risk scoring: score, band, recommended action |
| `estimate_order_cost()` | Itemised cost estimate before submission |
| `confirm_contract_signature()` | e-Signature confirmation → contract active |
| `renew_contract()` | Extend active contract, stacking end dates |
| `get_task_execution_plan()` | DAG adjacency + topological execution groups |
| `export_metrics()` | Prometheus text or JSON operational metrics |
| `replay_order()` | Reconstruct order history from audit log |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| order_type_not_supported | unknown order type | deny |
| duplicate_order_detected | is_duplicate=True | deny |
| customer_reference_required | no customer_id | deny |
| order_must_be_valid_for_decomposition | not yet validated | deny |
| msisdn_required_for_portability | no MSISDN | deny |
| bulk_order_approval_required | no approval reference | deny |
| contract_signature_state | not pending_signature | deny renew/confirm |
| order_must_be_amendable | completed or cancelled | deny amendment |

## Data Models
- OrdOrder: id, tenant_id, order_type, customer_id, channel, priority, status, submitted_at, completed_at
- OrdTask: id, tenant_id, order_id, task_type, status, depends_on, assigned_to, started_at, completed_at
- OrdFallout: id, tenant_id, order_id, fallout_category, description, retry_count, resolution, resolved_at, status
- OrdPortabilityRequest: id, tenant_id, order_id, msisdn, donor_operator, recipient_operator, status, submitted_at, porting_date
- OrdBulkOrder: id, tenant_id, order_type, item_count, approval_reference, status, submitted_by, submitted_at
- OrdAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- order_submitted, order_validated, order_decomposed, task_completed
- order_fallout, order_retry, provisioning_completed, order_completed
- order_cancelled, ord_agent_registered, contract_created, contract_signed
- contract_renewed, webhook_registered, jeopardy_predicted, metrics_exported

## Edge Cases Handled
- Decomposition requires validated status — submitted-but-not-validated orders cannot be decomposed
- Fallout retry counter increments each retry; exceeding max_retries triggers escalation flag
- Portability requires both MSISDN and donor_operator — partial portability requests denied
- Portability eligibility check rejects concurrent active port for the same MSISDN
- Bulk order approval is separate from individual order approval to prevent privilege escalation
- Task depends_on is stored as a string reference; `get_task_execution_plan` builds the DAG and detects cycles
- Amendments are tenant-partitioned (no cross-tenant data leakage in list stores)
- Contract renewal stacks from current end_date, not today, to avoid gaps in consecutive renewals

## Composability Notes
Triggers telecom_pro (provisioning workflows) on decomposition. Validates customer data against telecom_cus. Checks network resource availability against telecom_inv. Order completion triggers telecom_bil (charge setup) and telecom_cus (lifecycle event). Cost estimation integrates with telecom_bil tariff catalogue.
