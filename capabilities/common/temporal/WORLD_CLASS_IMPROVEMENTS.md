# World-Class Improvements — Temporal Capability

**Capability**: `temporal` | **Domain**: common
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Batch Workflow Operations

The current API requires one HTTP round-trip per workflow action. A `batch_start_workflows` method accepting a list of `StartWorkflowRequest` objects reduces orchestration overhead by ~70% for fan-out patterns (e.g., triggering per-tenant reconciliation). Temporal's native concurrency handles the async fan-out; the service layer only needs to gather results and surface per-item errors without aborting the batch.

## 2. Workflow Update (Temporal Update API)

Temporal 1.24+ exposes `workflow.update()` — a synchronous RPC that validates, executes a handler *inside* the workflow, and returns a result, all in one call. This is strictly more powerful than fire-and-forget signals for cases like "reserve inventory in an order workflow and confirm atomically." The service is missing this primitive entirely. Add `update_workflow(workflow_id, update_name, *, args)` with proper retry semantics.

## 3. Structured Workflow Metadata / Search Attributes

Workflows are launched with opaque `input_data` dicts. Registering typed search attributes (string, keyword, datetime, int) on the Temporal namespace enables `list_workflows` to accept server-side indexed filter expressions instead of pulling all workflows and filtering in Python. This alone can cut UI latency from O(n) to O(log n) for large installations.

## 4. Saga / Compensation Pattern Support

Long-running business processes fail mid-flight. The service has no native compensation primitives. Add `register_compensation(workflow_id, activity_name, rollback_args)` and `trigger_compensation(workflow_id)` backed by a Temporal child workflow that executes registered rollback activities in reverse order. This makes distributed transactions first-class, not an afterthought in each workflow definition.

## 5. Workflow Version / Patch Management

Temporal's `workflow.patched()` API enables safe in-flight code upgrades without draining workers. The service should expose `list_workflow_patches(workflow_type)` and `describe_patch_status(workflow_type, patch_id)` so operators can track live/deprecated code paths and schedule drains before deploying breaking changes.

## 6. Durable Timers with Named Cancellation

`schedule_workflow` covers cron, but one-shot "call me back in 30 days" timers are a core Temporal primitive. Add `create_timer(workflow_id, timer_name, fire_after_seconds)` and `cancel_timer(workflow_id, timer_name)` backed by `workflow.sleep` signals. Named timers are inspectable via queries, making them vastly more operable than anonymous sleeps buried in workflow code.

## 7. Continue-As-New Lifecycle Hook

Workflows that accumulate unbounded history (e.g., a perpetual event-processing loop) hit Temporal's 50 000-event history limit and terminate unexpectedly. Expose `configure_continue_as_new(workflow_type, max_history_events)` so the worker automatically continues workflows before the limit, resetting history while preserving business state. Without this, multi-year workflows are a reliability timebomb.

## 8. Worker Versioning / Build-ID Groups

Temporal 1.26 introduces worker versioning via build-ID sets, enabling blue/green worker deploys with no task misrouting. Add `add_compatible_build_id(task_queue, new_build_id, existing_build_id)` and `get_worker_task_reachability(task_queue, build_ids)` to the service so APG deployments can safely roll workers without manual coordination.

## 9. Nexus Service Calls

Temporal Nexus (GA in 2025) allows workflows to call cross-namespace or external HTTP services with exactly-once delivery. Add `call_nexus_endpoint(endpoint_name, operation, *, input_data)` backed by Temporal's `nexus.Client`. This lets APG workflows compose with external SaaS APIs (payment processors, identity providers) with the same durability guarantees as internal activities.

## 10. Structured Concurrency via Child Workflows

Complex orchestrations currently require stuffing all logic into a single workflow. Add `start_child_workflow(parent_workflow_id, child_workflow_type, *, input_data, wait_for_completion)` and `list_child_workflows(parent_workflow_id)`. Parent/child links are tracked via Temporal's parent-close policy, giving the orchestration a proper tree structure that is visible in the Temporal UI.

## 11. Dead-Letter Queue / Poison Pill Handling

Activities that fail beyond `max_attempts` currently land in Temporal's "failed" state with no operator intervention path. Add `get_stuck_workflows(*, stalled_for_seconds)` (uses visibility API with `ExecutionTime > threshold`) and `retry_workflow_from_last_activity(workflow_id)` to give ops teams a structured recovery path instead of requiring manual Temporal UI triage.

## 12. Workflow Tagging / Label System

Searchability beyond status and workflow type is critical for multi-tenant installations. Add `set_workflow_labels(workflow_id, labels: dict[str, str])` backed by Temporal search attributes with a `apg_label_*` prefix convention, and `search_workflows_by_label(labels)` for indexed lookups. This enables "show me all Purchase Order workflows for tenant X that are awaiting finance approval" in a single server-side query.

## 13. Execution Replay / Test Harness Hooks

Temporal's replay tester verifies that new workflow code produces identical history decisions as recorded histories. Expose `replay_workflow_history(workflow_type, history_json)` which runs `temporalio.testing.WorkflowEnvironment.start_time_skipping()` against the provided history file. This makes regression testing of workflow code changes a first-class service operation, not a manual developer step.

## 14. Observability: Structured OpenTelemetry Spans

Every public method in `TemporalService` currently logs via Python `logging` only. Wrap each method with an OpenTelemetry span carrying `workflow.id`, `workflow.type`, `tenant.id`, and `temporal.namespace` as span attributes. Temporal's SDK has native OTel support; the service should activate it and propagate context through all `connect()` / `start_workflow()` / `complete_task()` calls so distributed traces appear in Grafana/Jaeger without code changes at the call site.

## 15. Temporal Cloud / Multi-Region Namespace Support

Current `connect()` assumes single-region self-hosted. Temporal Cloud uses mTLS with API key auth and a `<namespace>.<account>.tmprl.cloud:7233` endpoint. Add `connect_cloud(api_key, account_id, namespace)` which configures `TLSConfig` and sets the correct SNI, and expose `get_cloud_usage_metrics(namespace)` calling the Temporal Cloud gRPC API. This unblocks SaaS deployment without any service-level code changes.
