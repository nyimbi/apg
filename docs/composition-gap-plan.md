# APG Composition Gap Closure Plan

## Research Basis
See: docs/research/composition-systems-gap-analysis.md

## Gap Summary (Priority Order)

Based on analysis of Temporal, Camunda 8, MuleSoft Anypoint, AWS Step Functions, Zapier/Make, Apache Camel, Netflix Conductor, Prefect/Dagster, n8n, and Boomi:

### CRITICAL GAPS (data loss / integration blocker)

#### G1: Durable Workflow Execution
**What best systems do**: Temporal, Camunda/Zeebe, Conductor use append-only event sourcing — every state transition is an immutable journal entry. A crash mid-step replays from the last journal entry with exactly-once guarantees.
**APG current state**: JSON file write (non-atomic). kill -9 mid-write loses state.
**Plan**: 
- Integrate with `capabilities/common/audl/` — its chain-hashed append-only event log IS the durable journal
- Each workflow step transition writes an `AuditEvent` via `AuditLoggingService`
- Crash recovery replays from the audl event log
- Timeline: Phase 1 (immediate)

#### G2: Non-HTTP Connector Types
**What best systems do**: MuleSoft covers SAP RFC, EDI X12/EDIFACT, HL7 MLLP, AMQP, MQTT, gRPC, SFTP, JMS. Apache Camel has 300+ components. APG's connector generator only reads OpenAPI (HTTP).
**APG current state**: HTTP-only via OpenAPI spec → Bearer token stub
**Plan**:
- Add `protocol` field to connector DSL: `protocol: kafka | mqtt | grpc | sftp | amqp | http`  
- Add `compiler/connector_generator.py` templates per protocol
- Kafka connector: `confluent-kafka-python` producer/consumer stubs
- MQTT connector: `paho-mqtt` publish/subscribe stubs
- gRPC connector: protobuf stub generator
- Timeline: Phase 2

#### G3: OAuth Token Lifecycle
**What best systems do**: Zapier, Make, n8n, Boomi all manage per-connected-account OAuth2 credentials with automatic token refresh and encrypted at-rest storage.
**APG current state**: `Bearer {api_key}` from env var only
**Plan**:
- Add `auth_type: oauth2 | api_key | basic | jwt` to connector DSL
- Add `APG_OAUTH_TOKENS` PostgreSQL table: `(connector_name, tenant_id, access_token_encrypted, refresh_token_encrypted, expires_at)`
- Add `_refresh_oauth_token(connector_name)` to generated app
- Timeline: Phase 2

### HIGH GAPS

#### G4: Workflow Versioning
**What best systems do**: Temporal pins running instances to the workflow code version that started them. New deployments don't break in-flight runs.
**APG current state**: Deploying a new spec overwrites everything; in-flight runs use whatever code is live
**Plan**:
- Add `version` field to workflow DSL
- Store `spec_version` in each workflow run record
- Timeline: Phase 3

#### G5: External Event Signals (waits: implemented)
**What best systems do**: Temporal `workflow.wait_for_signal()`, Camunda message correlation, Step Functions `.waitForTaskToken`
**APG current state**: `waits:` field declared in DSL and AST but generated app has no signal endpoint
**Plan**:
- Add `POST /workflows/runs/{id}/signal/{event_name}` endpoint
- Implement run-state polling in `_execute_workflow_steps` for wait steps
- Timeline: Phase 1

#### G6: Sub-Workflow Composition
**What best systems do**: All platforms allow invoking another workflow as a step, with result passing
**APG current state**: Cannot fan out into reusable sub-processes
**Plan**:
- Add `call_workflow: other_workflow_name` as a step action type
- Compiler generates code to invoke `run_workflow(name, input)` inline
- Timeline: Phase 2

#### G7: Visual Debug Timeline
**What best systems do**: Temporal Web UI, Camunda Operate, Step Functions console show per-step event history with timestamps, input/output, and retry counts
**APG current state**: `/ui/debug/{run_id}` shows a static snapshot; no event history
**Plan**:
- Write step events to audl event log with timestamps
- `/ui/debug/{run_id}` renders a visual timeline from the event log
- Timeline: Phase 1 (requires G1)

### MEDIUM GAPS

#### G8: Connector Pagination/Cursor Iteration
**What best systems do**: MuleSoft, n8n auto-paginate cursor/page-based APIs transparently
**APG current state**: Generated stubs make one HTTP call; silent truncation if API paginates
**Plan**: Add `pagination: cursor | page | offset` to connector operation DSL

#### G9: Per-Tenant Credential Isolation  
**Plan**: OAuth token store keyed by `(connector_name, tenant_id)`

#### G10: Rate Limiting in Connector Stubs
**Plan**: Token bucket per connector instance, configurable via `rate_limit: 100/minute` in connector DSL

## Marketplace Plan

### Connector Marketplace UI
- Route: `GET /ui/marketplace`
- Shows: all registered connectors (from `APG_CONNECTOR_REGISTRY`), installed status, operations count
- Actions: "Generate stub" links to `/ui/marketplace/{name}/generate`
- Data: populated from scanning `connectors/` directory on startup

### Marketplace Registry
- `connectors/MANIFEST.json` — versioned registry of available connectors
- `apg connector list --marketplace` — shows available vs installed connectors

## Implementation Phases

### Phase 1 (Now — close stop hook gaps)
1. Grammar: add saga/emit_events/subscribe_events keywords ✅ (this task)
2. Durable saga via audl event log
3. External event signal endpoint
4. Marketplace UI page
5. Visual workflow debug timeline (event-based)

### Phase 2 (Next sprint)
6. Non-HTTP connector protocols (Kafka, MQTT, gRPC stubs)
7. OAuth token lifecycle management
8. Sub-workflow composition

### Phase 3 (Future)
9. Workflow versioning
10. Connector pagination
11. Per-tenant credential isolation
12. Rate limiting

## Files to Modify

| File | Change |
|------|--------|
| `spec/apg.g4` | Add saga/emit_events/subscribe_events to member_name |
| `compiler/code_generator.py` | Durable saga audl integration, signal endpoint, marketplace route |
| `compiler/connector_generator.py` | Protocol-aware stub generation |
| `compiler/templates/marketplace.html.j2` | Marketplace UI page |
| `compiler/templates/workflow_timeline.html.j2` | Visual debug timeline |
| `connectors/MANIFEST.json` | Connector registry |
