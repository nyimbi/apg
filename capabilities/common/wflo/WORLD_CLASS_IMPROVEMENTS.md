# Workflow Low-Code (wflo) — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Async-First Service Layer

All service methods are synchronous, blocking any async call sites in FastAPI / ASGI hosts. Convert the entire WfloService to async, using `asyncio.Lock` per tenant for mutation isolation. This eliminates thread contention and enables true concurrent execution monitoring without thread pools.

## 2. Persistent Storage Adapter Pattern

The in-memory `dict` stores are a prototype-grade affordance. Introduce a `WfloRepository` protocol with `async get / put / query / delete` contract, then ship a `PostgresWfloRepository` via SQLAlchemy async core. This makes the service stateless and horizontally scalable with zero API changes.

## 3. Workflow Version Diffing and Migration

When `create_workflow_definition` is called for an existing name at a higher version, automatically diff steps between versions, detect removed/reordered steps, and produce a structured migration plan. Guard against in-flight executions being invalidated by version changes without explicit migration approval.

## 4. Real BPMN 2.0 Parser

The `bpmn_import` regex parser discards sequence flows, gateways, pools, lanes, boundary events, and data objects. Replace it with a proper BPMN XML parser (e.g. `xml.etree.ElementTree` with namespace handling) that round-trips the full element graph into steps, gateways, and event subscriptions.

## 5. Visual Designer State Serialization (JSON-to-Graph)

The UI has a designer route but no graph serialization contract. Add `serialize_designer_state` / `deserialize_designer_state` methods that emit a canonical `{nodes, edges, metadata}` JSON suitable for React Flow or similar canvas renderers. This makes the capability genuinely low-code rather than API-only.

## 6. Conditional Expression Evaluator

`inclusive_gateway` stores conditions as raw strings but never evaluates them. Integrate a safe expression evaluator (e.g. `simpleeval` or a hand-rolled subset) so runtime branching decisions are deterministic and unit-testable without executing arbitrary code.

## 7. SLA Deadline Tracking with Real Timestamps

`sla_enforce` compares only whether a task was claimed, ignoring wall-clock age. Store task `created_at` and compute elapsed minutes against `sla_minutes`. Return precise breach severity levels (`warning`, `critical`) based on configurable threshold ratios.

## 8. Process Mining Integration

Add a `process_mine` method that replays the `WorkflowAuditEventRecord` stream for an execution and produces a Petri-net compatible event log in XES format. This enables PM4Py-style conformance checking and variant discovery without a separate data pipeline.

## 9. Multi-Tenant Isolation at Storage Layer

Tenant isolation is currently enforced by post-fetch comparisons. Move it upstream: namespace all storage keys with tenant ID, so a cross-tenant lookup is structurally impossible. Applies to both in-memory dict and the Postgres adapter.

## 10. Webhook / Notification Dispatch

`emit_event` writes to an in-memory dict but never fans out. Add an `EventDispatcher` protocol with `async dispatch(event)` and ship HTTP webhook, internal message bus (Redis Streams or Bytewax), and no-op implementations. Wire it into `emit_event` so downstream systems receive real-time execution signals.

## 11. Parallel Gateway Join Synchronization

`parallel_gateway` adds branch steps but `complete_execution` has no join semantics — it can complete before all branches finish. Add a `join_policy` field (`all`, `any`, `n_of_m`) to the gateway record and enforce it in `complete_execution` by counting completed tasks per `parallel_group`.

## 12. Bulk Execution Scheduling

Add a `schedule_bulk_executions` method accepting a list of `{definition_id, correlation_id, payload, scheduled_at}` records. Validate each against policy in a single pass, persist as `ScheduledExecutionRecord`, and return a batch receipt. This enables scheduled/cron-driven process launches without N separate API calls.

## 13. Execution Replay and Idempotency

Re-submitting a `start_execution` with the same `correlation_id` creates a duplicate. Add idempotency enforcement: detect duplicate correlation IDs, return the existing execution record, and emit a `duplicate_start_attempted` audit event. This is critical for at-least-once delivery environments.

## 14. Role-Based Access Control (RBAC) on Service Methods

All methods accept an `actor` string but never check whether that actor has permission to perform the operation. Introduce an `WfloAccessPolicy` protocol with `can(actor, operation, resource) -> bool` checked before each mutation. Ship a role-map default implementation backed by the capability contract.

## 15. Structured Error Catalog

`_raise_policy` emits an untyped `PermissionError` with a freeform string. Replace it with a typed `WfloError` hierarchy (`WfloPermissionError`, `WfloNotFoundError`, `WfloValidationError`, `WfloPolicyError`) carrying machine-readable `code`, `detail`, and `context` fields. This lets API layers map errors to HTTP status codes and client SDKs handle them programmatically.
