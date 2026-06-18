# Workflow Composition Systems: Gap Analysis for APG
**Date:** 2026-06-15  
**Author:** Research analysis for Datacraft / APG project  
**Scope:** 10 world-class workflow/composition platforms vs. APG's current capabilities

---

## 1. Executive Summary — What APG Cannot Do (Priority Order)

APG is a DSL compiler that generates a single-file Python HTTP server. Its workflow model covers state-machine declarations with steps, timers, guards, human tasks, compensation, circuit breakers, and JSON/PostgreSQL persistence. That is a solid foundation but leaves the following gaps, ranked by implementation impact:

| Priority | Gap | Severity |
|---|---|---|
| 1 | **No durable execution / event-sourced state** — JSON file write is not crash-safe; workflow progress lost on kill-9 | Critical |
| 2 | **No non-HTTP protocol connectors** — cannot speak SAP RFC/IDoc, MQTT, AMQP, ISO8583, SWIFT, FIX, FTP, SFTP, JMS natively | Critical |
| 3 | **No streaming / CDC integration** — cannot consume Bytewax topics, Kinesis streams, Debezium CDC events as workflow triggers | High |
| 4 | **No connector OAuth/token lifecycle** — connectors are static `Bearer {api_key}`; no PKCE, refresh, token rotation, per-tenant creds | High |
| 5 | **No visual workflow debugger / event timeline** — no UI equivalent to Temporal Web UI, Camunda Operate, or Step Functions console | High |
| 6 | **No versioning of in-flight workflows** — cannot patch a live workflow definition without losing running instances | High |
| 7 | **No cross-workflow saga coordination** — compensation runs within a single workflow; cannot coordinate rollback across multiple child workflows | Medium |
| 8 | **No connector pagination / rate-limiting** — generated stubs call one URL once; no cursor/page iteration, no token-bucket throttle | Medium |
| 9 | **No EDI / B2B / mainframe connectivity** — X12, EDIFACT, AS2, CICS, RPC-encoded SOAP | Medium |
| 10 | **No multi-tenant connector credential isolation** — one `APG_CONNECTOR_AUTH` env var; no per-tenant OAuth credential store | Medium |
| 11 | **No marketplace / versioned connector registry** — `scan_connectors()` reads local dir; no version pinning, no publish/install lifecycle | Low–Medium |
| 12 | **No BPMN / visual design tool** — all composition is code/DSL; no drag-and-drop process modeller | Low |

---

## 2. Platform-by-Platform Analysis

---

### 2.1 Temporal.io

**Website:** https://temporal.io  
**Model:** Durable execution — workflows are plain code (Go, Java, TypeScript, Python, .NET, PHP) whose full execution state is persisted as an append-only event history on the Temporal Service.

#### Core Composition Model
Workflows are ordinary functions. The SDK intercepts every `await` and records it as an event. On crash/restart the worker replays the event history to restore state without re-executing side effects. No DSL is required.

```python
# Temporal Python SDK example
@workflow.defn
class OrderFulfillmentWorkflow:
    @workflow.run
    async def run(self, order_id: str) -> str:
        # Each activity call is a durable checkpoint
        payment = await workflow.execute_activity(
            authorize_payment, order_id,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        if not payment["ok"]:
            await workflow.execute_activity(void_order, order_id)
            return "cancelled"
        await workflow.execute_activity(reserve_stock, order_id)
        return "fulfilled"
```

#### Connector / Integration Ecosystem
- No native connector library (Temporal is an execution engine, not an iPaaS).
- Activities wrap any I/O — HTTP, gRPC, database, messaging, SAP, etc.
- SDK workers are deployed by the team; the team writes activity code in any supported language.
- Temporal Cloud (SaaS) adds metrics, multi-region namespaces, audit logs.

#### Durability & Fault Tolerance
- Event-sourced append-only history per workflow execution.
- Workflow logic: **effectively exactly-once** (replay deduplicate re-executions of workflow code).
- Activities: **at-least-once** — must be idempotent; SDK provides `activity.heartbeat()` for long-running activities.
- Timers durable across server restarts — `await asyncio.sleep(30 * 24 * 3600)` works.
- History size limit (default 50 000 events / 50 MB) — use `continue_as_new` for very long workflows.

#### Event-Driven Capabilities
- **Signals**: external systems inject events into a running workflow. A payment portal sends `workflow.signal("approved")` after human approval.
- **Queries**: read workflow state without mutating it.
- **Update**: transactional signal+query (added 2023).
- **Schedules**: cron-style, jitter, catch-up, pause/resume.

#### Human-in-the-Loop
`workflow.wait_condition(lambda: self._approved)` blocks indefinitely; a Signal from any external source (web portal, email link) unblocks it. Timeout via `workflow.sleep()`. Pattern is natural code — no special BPMN element needed.

#### Multi-Tenant Architecture
Namespaces provide isolation. Each namespace has independent task queues, workflow histories, quotas. Temporal Cloud supports namespace-per-tenant billing. OSS Temporal: operators manage namespace provisioning.

#### Observability & Debugging
- **Temporal Web UI**: timeline of every event in every workflow execution. Drill down to individual activity retries.
- **tctl / temporal CLI**: query, signal, cancel, list workflows.
- **OpenTelemetry** export to Jaeger/Zipkin/Tempo.
- **Prometheus** metrics (workflow started/completed/failed, activity schedule-to-start latency, etc.).
- **Replay testing**: `TestWorkflowEnvironment` lets you unit-test workflow logic including time-skipping and injected failures.

#### Workflow Versioning
Two strategies:
1. **`workflow.get_version()`** / `workflow.patched()` — adds a version branch that applies to new executions while existing ones continue on old path.
2. **Worker Deployments** — tag workers with a Build ID; route new workflow starts to new workers; drain old workers as old executions complete.

#### Saga / Compensation
Compensation is idiomatic Python `try/finally`:
```python
try:
    await workflow.execute_activity(reserve_inventory, ...)
    await workflow.execute_activity(charge_payment, ...)
except Exception:
    await workflow.execute_activity(release_inventory, ...)
    raise
```
No special saga DSL. The event history guarantees compensation activities run exactly as written.

#### What APG Cannot Match (vs. Temporal)
- Crash-safe durable execution (APG JSON file is NOT atomic; a kill during write loses state)
- True saga across child workflows
- Durable timers surviving process restart without polling
- Workflow versioning for in-flight executions
- Production-grade observability UI with event timeline replay

---

### 2.2 Camunda 8

**Website:** https://camunda.com  
**Engine:** Zeebe (cloud-native, partitioned, log-based BPMN/DMN execution engine)

#### Core Composition Model
Processes are modelled as BPMN 2.0 XML diagrams. The Web Modeler and Desktop Modeler provide drag-and-drop editing. Zeebe executes the BPMN; process state is stored in Zeebe's replicated append-only log.

```xml
<!-- BPMN fragment: gateway with compensation -->
<bpmn:serviceTask id="ReserveInventory" name="Reserve Inventory">
  <bpmn:extensionElements>
    <zeebe:taskDefinition type="reserve-inventory" />
  </bpmn:extensionElements>
</bpmn:serviceTask>
<bpmn:boundaryEvent id="CompensateReserveInventory" attachedToRef="ReserveInventory">
  <bpmn:compensateEventDefinition />
</bpmn:boundaryEvent>
<bpmn:serviceTask id="ReleaseInventory" name="Release Inventory" isForCompensation="true">
  <bpmn:extensionElements>
    <zeebe:taskDefinition type="release-inventory" />
  </bpmn:extensionElements>
</bpmn:serviceTask>
```

#### Connector Ecosystem
- 50+ out-of-the-box connectors (Slack, SendGrid, AWS Lambda, GitHub, HTTP, REST, Bytewax, RabbitMQ, AWS SQS, Google PubSub, Salesforce, etc.)
- Connector SDK (Java) for custom connectors
- Marketplace: https://marketplace.camunda.com
- Connectors are configured in the BPMN properties panel — no code for standard integrations
- Inbound connectors (webhooks, message queues) trigger process instances

#### Durability & Fault Tolerance
- Zeebe uses a Raft-based replicated log (event sourcing). All process instance state is durable.
- Processes survive broker restarts; partitioned for horizontal scale.
- Retry policies configurable per service task.
- Incident management: failed jobs become "incidents" visible in Operate; operators can resolve and retry.

#### Event-Driven Capabilities
- BPMN Message events and Signal events
- Bytewax connector (inbound/outbound) — process triggered by Bytewax message, or publishes to Bytewax
- Timer events (ISO 8601 durations and cycles)
- Boundary events (timer, error, message, signal, compensation) attached to any task

#### Human-in-the-Loop
- **Tasklist** application: users claim and complete user tasks
- **Forms** (Camunda Form JSON schema) attached to BPMN User Tasks
- Candidate groups / candidate users / due dates / follow-up dates
- Task Tester for isolated developer testing of a single task

#### Multi-Tenant Architecture
- Camunda 8 SaaS: cluster-per-tenant or multi-tenant cluster with tenant-id isolation (added 8.3+)
- Self-Managed: namespace-level isolation with separate Zeebe partitions

#### Observability & Debugging
- **Operate**: real-time view of running process instances, incident drill-down, variable inspection, historical search
- **Optimize**: process analytics — cycle times, bottleneck heatmaps, KPI dashboards
- Metrics exported via Prometheus; traces via OpenTelemetry
- Task Tester: execute a single task in isolation against a real Zeebe cluster

#### Saga / Compensation
Native BPMN Compensation Events — the standard mechanism. A compensation intermediate throw event triggers all compensation handlers of completed activities in scope, in reverse order. This is the standard BPMN 2.0 pattern; Camunda executes it natively.

#### What APG Cannot Match (vs. Camunda 8)
- Visual BPMN modeller with real-time collaboration
- First-class BPMN compensation events (APG has compensation field but execution is custom Python)
- Incident-based workflow repair (APG has no "incident" concept; a failed step is just an error)
- Bytewax/SQS inbound triggers (APG event subscriptions are in-process only)
- Process analytics (Optimize)

---

### 2.3 MuleSoft Anypoint Platform

**Website:** https://www.mulesoft.com  
**Model:** API-led connectivity with three architectural layers (System APIs → Process APIs → Experience APIs)

#### Core Composition Model
Mule applications are built in Anypoint Studio (Eclipse-based IDE) or Anypoint Code Builder (VS Code extension). Flows are defined as XML sequences of components (sources, processors, routers, transformers, error handlers). DataWeave 2.0 is the transformation language.

```xml
<!-- Mule flow fragment: SAP to Salesforce sync -->
<flow name="sap-to-sfdc-sync">
  <sap:inbound-endpoint type="function" functionName="BAPI_CUSTOMER_GETLIST"/>
  <ee:transform>
    <ee:message>
      <ee:set-payload>
        <![CDATA[%dw 2.0
        output application/json
        ---
        payload.CUSTOMERLIST map (c) -> {
          Name: c.NAME1,
          AccountNumber: c.KUNNR
        }]]>
      </ee:set-payload>
    </ee:message>
  </ee:transform>
  <salesforce:create type="Account"/>
</flow>
```

#### Connector Ecosystem
- **1500+ connectors** in Anypoint Exchange
- Categories: SAP (RFC, IDoc, BAPI), EDI (X12, EDIFACT, HL7, AS2), Mainframe (IBM CICS), databases, SaaS (Salesforce, ServiceNow, Workday, SAP S/4HANA), messaging (JMS, IBM MQ, Bytewax, AMQP, MQTT), cloud (AWS, Azure, GCP), protocols (FTP, SFTP, FTPS, SMTP, POP3)
- SAP Connector: certified, uses SAP JCo; supports RFC calls, IDoc send/receive, BAPI
- EDI: X12 and EDIFACT parsing/generation with full schema validation
- Connector SDK (Java/XML) for custom connectors; Connector Generator from OAS/WSDL
- Each connector handles: OAuth 2.0 (PKCE, client credentials, auth code), API key, basic, NTLM, Kerberos

#### Durability & Fault Tolerance
- Mule runtime: in-memory transaction management with XA transactions for database/JMS
- CloudHub 2.0: Kubernetes-based, auto-restart on crash
- Persistent VM queues for async message reliability
- Object Store v2: distributed key-value store for workflow state between flow invocations

#### Event-Driven Capabilities
- Anypoint MQ: managed cloud messaging (pub/sub, queues, FIFO)
- Bytewax connector (inbound trigger or outbound publish)
- Webhooks, scheduled polling (cron), MQTT inbound
- Event-driven integration at Process API layer

#### Human-in-the-Loop
Not native — MuleSoft is an integration platform, not a BPM engine. Human approval workflows typically delegate to Salesforce, ServiceNow, or an embedded Camunda instance.

#### Multi-Tenant Architecture
- Anypoint Platform: organizations → business groups → environments
- Runtime Manager: deploy separate Mule apps per tenant or use shared app with tenant routing
- CloudHub 2.0 Private Spaces: network-isolated deployment zones

#### Observability & Debugging
- Anypoint Monitoring: custom dashboards, custom metrics, log correlation, distributed tracing
- Runtime Manager: deployment health, CPU/heap metrics, restart controls
- API Analytics: per-API request/response analytics, SLA alerting
- DataWeave Playground for interactive transformation testing

#### Connector OAuth / Token Lifecycle
Full OAuth 2.0 support built into every OAuth connector:
- Authorization Code flow with PKCE
- Client Credentials flow
- Token refresh (automatic before expiry)
- Per-environment credential stores
- Secrets Manager integration (AWS SM, Azure KV, HashiCorp Vault)

#### Connector Pagination
Built into connectors: automatic cursor-based pagination, page-number iteration, result accumulation. DataWeave `do ... while` or recursive functions for custom cases.

#### Saga / Compensation
Implemented via Mule's Try scope with On Error Continue/On Error Propagate, combined with compensating sub-flows. Not native saga; requires explicit error handler wiring. No built-in saga coordinator.

#### What APG Cannot Match (vs. MuleSoft)
- SAP RFC/IDoc/BAPI connectivity (requires JCo native library)
- EDI X12/EDIFACT parsing/generation
- IBM MQ, AMQP, JMS connectivity
- OAuth 2.0 full lifecycle (refresh, PKCE, per-environment)
- Automatic connector pagination
- DataWeave transformation language (APG has no transformation DSL)
- 1500+ production-grade connectors

---

### 2.4 AWS Step Functions

**Website:** https://aws.amazon.com/step-functions/  
**Model:** Serverless state machine orchestration using Amazon States Language (ASL) — JSON/YAML DSL

#### Core Composition Model
State machines are defined in ASL (JSON). Each state is a node; transitions are explicit. Workflow Studio provides visual drag-and-drop editing in the AWS Console and VS Code.

```json
{
  "Comment": "Order fulfillment with saga compensation",
  "StartAt": "ReserveInventory",
  "States": {
    "ReserveInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:::function:reserve-inventory",
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "RevertOrder"}],
      "Next": "ProcessPayment"
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:::function:process-payment",
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "ReleaseInventory"}],
      "End": true
    },
    "ReleaseInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:::function:release-inventory",
      "Next": "RevertOrder"
    },
    "RevertOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:::function:revert-order",
      "End": true
    }
  }
}
```

#### Connector / Integration Ecosystem
- **270+ AWS service integrations** (SDK integrations) — invoke any AWS service directly from a state without Lambda
- Optimistic integrations: call Lambda, DynamoDB, SQS, SNS, ECS, Glue, EMR, Batch, SageMaker, Bedrock, etc.
- For non-AWS systems: wrap in Lambda
- EventBridge integration for event-driven triggers
- No native non-AWS connector library for SaaS/ERP/messaging

#### Durability & Fault Tolerance
- State is persisted by the Step Functions service (not by the caller)
- Standard Workflows: at-least-once execution; events stored for 90 days; durable across AZ failure
- Express Workflows: at-least-once, high throughput, 5-minute max
- Built-in retry with exponential backoff: `Retry: [{ErrorEquals: [...], IntervalSeconds: 1, MaxAttempts: 3, BackoffRate: 2.0}]`
- Multi-AZ by default

#### Event-Driven Capabilities
- EventBridge rules trigger Step Functions executions
- SQS, SNS, Bytewax (MSK), API Gateway as triggers via EventBridge Pipes
- Callback patterns: `waitForTaskToken` — task pauses until external system calls `SendTaskSuccess`/`SendTaskFailure`
- Heartbeat timeouts for long-running tasks

#### Human-in-the-Loop
`waitForTaskToken` pattern: the state machine pauses; a task token is sent to an SQS queue or email; when a human approves via a portal that calls `SendTaskSuccess(taskToken, output)`, the workflow resumes. No built-in UI; teams build approval portals.

#### Multi-Tenant Architecture
IAM-based isolation per AWS account; multiple state machines per account; cross-account execution via IAM roles. No native tenant-ID routing; teams implement per-tenant state machines or use context variables.

#### Observability & Debugging
- **Execution console**: visual graph with each state highlighted (green/red/in-progress), input/output at every state, full event history
- **CloudWatch Logs**: execution history, step-level duration, errors
- **X-Ray tracing**: distributed traces across Lambda, DynamoDB, etc.
- **CloudWatch Metrics**: executions started/succeeded/failed/throttled, step duration

#### Workflow Versioning
- State machine **versions**: published versions are immutable snapshots; aliases (like `LIVE`) point to a version; routing config can split traffic between versions (10% v2, 90% v1)
- Running executions continue on the version they started on

#### Saga / Compensation
ASL `Catch` on each state routes to compensating tasks on failure. Parallel saga branches use the `Parallel` state type. Compensation is explicit in ASL — no automatic reverse-order execution (unlike BPMN compensation events). Teams implement the saga graph manually.

#### What APG Cannot Match (vs. Step Functions)
- Visual state graph with per-execution live highlighting
- `waitForTaskToken` durable pause (APG's `waits` field is declarative but execution is in-process)
- 270+ native AWS service integrations without any code
- Workflow versions with traffic splitting for canary deployments
- CloudWatch/X-Ray integration

---

### 2.5 Zapier / Make (Integromat)

**Websites:** https://zapier.com / https://make.com  
**Model:** No-code/low-code trigger-action automation; visual node-based flow editor

#### Core Composition Model

**Zapier**: Linear trigger → multi-step action chains ("Zaps"). Branching via Paths app. No loops natively; Filter steps skip actions. Copilot (AI) generates Zaps from natural language descriptions.

**Make**: Scenario-based visual canvas. Supports routers (branches), iterators (loops over arrays), aggregators (collect multiple items), error handlers, HTTP modules, data stores. More expressive than Zapier.

```
# Zapier Zap (pseudo-notation)
Trigger: New row in Google Sheets
  ├─ Action: Find or create Salesforce Contact
  ├─ Filter: Only if email contains "@enterprise.com"
  └─ Action: Send Slack message
```

```
# Make Scenario (pseudo-notation)
Webhook trigger
  → HTTP: GET /api/orders (paginated with iterator)
  → Router:
      Branch 1 (filter: status=pending) → Salesforce: Create Lead
      Branch 2 (filter: status=shipped) → SendGrid: Send email
  → Error handler → Slack: Alert on failure
```

#### Connector / Integration Ecosystem
- **Zapier**: 7000+ app integrations; OAuth managed by Zapier; each app exposes triggers and actions
- **Make**: 2000+ pre-integrated apps; more actions/triggers per app than Zapier; HTTP module for custom APIs
- Both: native AI nodes (OpenAI, Claude, Gemini, Mistral) since 2024
- Neither has SAP RFC, mainframe, MQTT, or ISO8583 connectors
- No EDI/B2B support

#### Durability & Fault Tolerance
- Zapier: task history retained; failed tasks can be replayed manually; no durable execution guarantee
- Make: execution log with error details; can replay from failure point manually
- Neither provides event-sourced exactly-once guarantees
- Both: SaaS multi-tenant with uptime SLAs (~99.9%)

#### Event-Driven Capabilities
- Webhooks as triggers (both)
- Scheduled triggers (both)
- Polling triggers (both — Zapier polls every 1–15 min on free tier; Make on schedule)
- Make: watch modules for real-time webhooks on supported apps

#### Human-in-the-Loop
No native human task management. Workarounds: email a link that triggers a webhook; use a Google Form; integrate with an approval SaaS. Not a first-class feature.

#### Observability & Debugging
- Zapier: Task History (per-Zap execution logs with input/output per step)
- Make: Execution History (visual scenario replay with data at each node)
- Neither: metrics dashboards, Prometheus/OpenTelemetry

#### Saga / Compensation
Not supported. No concept of saga or distributed transaction management.

#### Connector OAuth / Token Lifecycle
Zapier and Make manage OAuth tokens on behalf of users — users authenticate once via OAuth consent screen; platforms store and refresh tokens automatically. This is a key advantage over APG's static env-var approach.

#### What APG Cannot Match (vs. Zapier/Make)
- 7000-connector marketplace with OAuth token management
- Visual no-code editor accessible to non-developers
- Automatic OAuth token refresh per connected account
- Iterator/aggregator patterns for paginated API results
- AI-generated workflow from natural language

---

### 2.6 Apache Camel

**Website:** https://camel.apache.org  
**Model:** EIP (Enterprise Integration Patterns) framework — routes defined in Java DSL, XML DSL, YAML DSL, or Groovy. 350+ components.

#### Core Composition Model
A Route is a pipeline: from a source component, through processors (transform, filter, enrich, route, split, aggregate), to a destination component. EIPs are the building blocks.

```java
// Java DSL: Bytewax → Content-Based Router → JMS or REST
from("bytewax:orders?groupId=order-processor")
    .unmarshal().json(OrderEvent.class)
    .choice()
        .when(simple("${body.amount} > 10000"))
            .to("jms:queue:high-value-orders")
        .when(simple("${body.status} == 'cancelled'"))
            .to("direct:compensate")
        .otherwise()
            .to("rest:POST:/orders/process")
    .end();

// YAML DSL equivalent
- from:
    uri: "bytewax:orders"
    steps:
      - unmarshal:
          json: {}
      - choice:
          when:
            - simple: "${body.amount} > 10000"
              steps:
                - to: "jms:queue:high-value-orders"
```

#### Component Ecosystem
**350+ components** including:
- Messaging: Bytewax, MQTT (Paho), AMQP, JMS (ActiveMQ, IBM MQ, WebSphere MQ), NATS, RabbitMQ, STOMP, ZeroMQ
- Protocols: FTP/SFTP/FTPS, SMTP/POP3/IMAP, HTTP/HTTPS, gRPC, WebSocket, TCP, UDP
- Legacy/Enterprise: SAP (via camel-sap), IBM CICS (via StickerMap), SWIFT (third-party), EDI (camel-edi), HL7 MLLP, DICOM
- Cloud: AWS (S3, SQS, SNS, DynamoDB, Kinesis, Lambda), Azure (Event Hub, Service Bus, Blob), GCP (PubSub, BigQuery)
- Databases: JDBC, JPA, MongoDB, Cassandra, Couchbase, Elasticsearch
- Files: CSV, XML, JSON, Avro, Protobuf, CBOR, flat-file (fixed-width)
- AI: LangChain4J integration (Camel 4.x)

#### Durability & Fault Tolerance
- Dead Letter Channel: undeliverable messages routed to DLQ
- Error handlers: `defaultErrorHandler`, `deadLetterChannel`, `loggingErrorHandler`
- Retry: configurable attempts, delay, backoff
- Idempotent Consumer EIP: deduplication using in-memory or JPA/Redis backing store
- Transaction support: JMS XA transactions, JDBC transactions
- Camel Quarkus / Spring Boot: production-grade deployment with health checks

#### Event-Driven Capabilities
- Direct, SEDA (in-VM async), VM queues
- Polling consumers (file, DB, FTP) with configurable intervals
- Push consumers (Bytewax, MQTT, JMS, webhooks)
- Event-driven via CDI events on Quarkus
- Splitter, Aggregator, Resequencer, Correlation Slip EIPs

#### Human-in-the-Loop
Not a feature of Camel. Camel is an integration framework; human tasks require integration with a BPM system (e.g., a Camel route triggers a Camunda user task).

#### Observability & Debugging
- **Kaoto**: visual designer for Apache Camel routes (drag-and-drop, YAML output)
- JMX MBeans for runtime metrics
- **Micrometer** / Prometheus metrics
- OpenTelemetry / Jaeger distributed tracing
- Camel Management Console (in Camel Quarkus Dev UI)
- Route dump for runtime inspection

#### Saga / Compensation
**Saga EIP** is a first-class pattern in Apache Camel:
```java
from("direct:order")
    .saga()
        .compensation("direct:cancelOrder")
        .option("orderId", header("orderId"))
        .to("direct:reserveInventory")
        .to("direct:processPayment");
```
The saga EIP supports both in-memory and LRA (Long Running Action / MicroProfile LRA) coordination protocols. Compensation routes are called automatically on failure.

#### What APG Cannot Match (vs. Apache Camel)
- 350 components covering MQTT, AMQP, JMS, Bytewax inbound/outbound, SFTP, FTP, gRPC, HL7 MLLP
- SAP connector (via camel-sap)
- EDI processing (camel-edi)
- Saga EIP with automatic compensation routing
- Idempotent Consumer deduplication
- Content-Based Router, Splitter, Aggregator, Resequencer EIPs
- XML/YAML/Groovy DSL options alongside Java

---

### 2.7 Netflix Conductor / Orkes

**Website:** https://orkes.io / https://github.com/conductor-oss/conductor  
**Model:** JSON DSL workflow definitions; stateless polyglot workers poll task queues; Conductor server manages state

#### Core Composition Model
Workflows are JSON documents defining a DAG of tasks. Workers (any language) poll for tasks via REST/gRPC. The server maintains all state.

```json
{
  "name": "order_fulfillment",
  "version": 1,
  "tasks": [
    {
      "name": "reserve_inventory",
      "taskReferenceName": "reserveInventoryRef",
      "type": "SIMPLE",
      "inputParameters": {"orderId": "${workflow.input.orderId}"}
    },
    {
      "name": "process_payment",
      "taskReferenceName": "processPaymentRef",
      "type": "SIMPLE",
      "inputParameters": {
        "orderId": "${workflow.input.orderId}",
        "amount": "${reserveInventoryRef.output.totalAmount}"
      }
    },
    {
      "name": "compensate_inventory",
      "taskReferenceName": "compensateInventoryRef",
      "type": "SIMPLE",
      "startDelay": 0
    }
  ]
}
```

#### Task Types
- `SIMPLE` — polled by external workers
- `HTTP` — makes an HTTP call directly
- `INLINE` — executes JavaScript in the server (SpEL-like)
- `WAIT` — durable pause until signal or timeout
- `HUMAN` — native human task (Orkes)
- `DO_WHILE` — loop with condition
- `FORK_JOIN` — parallel branches with join
- `DYNAMIC` — task name resolved at runtime from workflow input
- `EVENT` — publish/subscribe to events (SQS, NATS, Conductor, AMQP)
- `SUB_WORKFLOW` — nest another workflow definition
- `SWITCH` — multi-way branch
- `SET_VARIABLE` — pure state mutation
- `TERMINATE` — end workflow from inside a branch
- `LLM tasks` (Orkes) — call LLM, generate text, store in vector DB

#### Durability & Fault Tolerance
- All workflow state stored in the Conductor server's backing database (Redis + Elasticsearch, or Postgres on Orkes)
- Worker crashes are safe: task re-queued after acknowledgment timeout
- Configurable retries per task with exponential backoff
- Workers are stateless — horizontal scaling by adding workers

#### Event-Driven Capabilities
- `EVENT` task publishes/subscribes to SQS, NATS, Conductor internal, AMQP, Azure Service Bus (Orkes)
- External triggers via Conductor's Start Workflow API (REST/gRPC)
- Webhooks trigger workflow starts
- Schedules (cron) for recurring workflows

#### Human-in-the-Loop (Orkes)
Native `HUMAN` task type added in Orkes 2023:
- Assignees specified at definition time or dynamically
- Responsibility chain: escalate to next person after timeout
- Human tasks visible in Orkes Playground UI
- Full history of assignment and completion events

#### Multi-Tenant Architecture
- OSS: application-level namespacing
- Orkes Cloud: full multi-tenant with dedicated namespaces, RBAC, audit logs, quota per namespace

#### Observability & Debugging
- Conductor/Orkes UI: workflow execution graph with per-task status, input/output, retry count
- Task queue depths
- Search across all executions (Elasticsearch-backed)
- Metrics via Prometheus (Orkes adds more metrics than OSS)
- Execution audit log

#### Saga / Compensation
Compensation requires explicit wiring in the workflow DAG — a failed branch triggers a FORK of compensating tasks. Orkes added a `COMPENSATE` task pattern in 2024 that is closer to BPMN compensation semantics. No automatic reverse-order execution like Temporal's try/finally.

#### What APG Cannot Match (vs. Conductor/Orkes)
- Native `HUMAN` task type with responsibility chains
- `SUB_WORKFLOW` for workflow composition / fan-out
- `DO_WHILE` and `DYNAMIC` task types
- EVENT task for async pub/sub integration
- Stateless polyglot worker model (workers in Java, Python, Go, JS, C# independently)
- Elasticsearch-backed workflow search

---

### 2.8 Prefect / Dagster

**Websites:** https://prefect.io / https://dagster.io  
**Model:** Python-native data pipeline orchestration; task-centric (Prefect) or asset-centric (Dagster)

#### Core Composition Model

**Prefect**:
```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=10)
def extract_data(source: str) -> list:
    return requests.get(source).json()

@task
def transform(records: list) -> list:
    return [{"id": r["id"], "name": r["name"].upper()} for r in records]

@flow(name="ETL Pipeline")
def etl_pipeline(source: str):
    raw = extract_data(source)
    clean = transform(raw)
    load(clean)
```

**Dagster** (asset-centric):
```python
from dagster import asset, AssetIn

@asset
def raw_customers(context) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM customers", conn)

@asset(ins={"raw_customers": AssetIn()})
def enriched_customers(raw_customers: pd.DataFrame) -> pd.DataFrame:
    return raw_customers.merge(demographics, on="customer_id")
```

#### Durability & Fault Tolerance
- Prefect: run state stored in Prefect Cloud or Server; task results cached in configured result storage (S3, GCS, local); task retries with configurable policies
- Dagster: asset materialization history in Dagster's event log (SQLite or Postgres); re-materialization of specific failed assets without rerunning the full pipeline
- Neither: event-sourced durable execution like Temporal; both rely on the orchestration server being available

#### Event-Driven Capabilities
- **Prefect Automations**: trigger flows from state changes (e.g., another flow completes), webhooks, schedule
- **Dagster Sensors**: polls for external conditions (new S3 file, DB row count, API response) and triggers asset jobs
- **Dagster Asset Sensors**: trigger downstream assets when upstream is materialized

#### Observability & Debugging
- **Prefect UI**: flow run timeline, task state graph, log viewer, retry history
- **Dagster Launchpad / Asset Catalog**: asset lineage graph, materialization history, per-asset metadata, freshness checks
- Both: Prometheus metrics, structured logging

#### Saga / Compensation
Neither Prefect nor Dagster has a saga pattern. Compensation is via Python try/except in the flow function. No distributed transaction coordination.

#### What APG Cannot Match (vs. Prefect/Dagster)
- Asset-based lineage graph and data freshness tracking (Dagster)
- Sensor-based event triggers from external data sources
- First-class caching of task/asset results (avoid recomputation)
- Data quality checks built into asset definitions
- Python-native dynamic mapping (fan-out over arbitrary collections at runtime)

---

### 2.9 n8n

**Website:** https://n8n.io  
**Model:** Open-source (fair-code) visual workflow automation; self-hostable; 400+ nodes; code nodes (JavaScript/Python)

#### Core Composition Model
Workflows are JSON node graphs edited in a visual canvas. Each node has a type (trigger, action, transform, code). Nodes pass arrays of items between them (n8n's item model). Code nodes run arbitrary JS or Python.

```json
{
  "nodes": [
    {"id": "1", "type": "n8n-nodes-base.webhook", "parameters": {"httpMethod": "POST", "path": "order"}},
    {"id": "2", "type": "n8n-nodes-base.if", "parameters": {"conditions": {"number": [{"value1": "={{$json.amount}}", "operation": "larger", "value2": 1000}]}}},
    {"id": "3", "type": "n8n-nodes-base.httpRequest", "parameters": {"url": "https://api.payment.com/charge"}},
    {"id": "4", "type": "n8n-nodes-base.code", "parameters": {"jsCode": "return items.map(i => ({json: {...i.json, processed: true}}));"}}
  ],
  "connections": {"1": {"main": [[{"node": "2"}]]}, "2": {"main": [[{"node": "3"}], [{"node": "4"}]]}}
}
```

#### Connector / Integration Ecosystem
- **400+ built-in nodes**
- Categories: CRM (Salesforce, HubSpot, Pipedrive), Project Mgmt (Jira, Notion, Asana, Linear), Comms (Slack, Discord, Telegram, WhatsApp), Databases (Postgres, MySQL, MongoDB, Redis, Supabase), Cloud (AWS, GCP, Azure), Finance (Stripe, QuickBooks, Xero), Marketing (Mailchimp, SendGrid), AI (OpenAI, Anthropic, Gemini, Groq, Vertex, Hugging Face, LangChain)
- HTTP Request node for any REST API
- No SAP RFC, mainframe, EDI, SWIFT, MQTT, ISO8583, FIX connectors
- Custom nodes via npm packages

#### Durability & Fault Tolerance
- Execution history stored in DB (SQLite or Postgres)
- Can replay/retry failed executions manually
- No event-sourced durable execution; if the n8n server is killed mid-execution, the run is marked as "crashed"
- n8n Cloud: managed, SLA 99.9%

#### Event-Driven Capabilities
- Webhook triggers (HTTP inbound)
- Scheduled triggers (cron)
- Polling triggers (interval-based)
- Bytewax trigger (via HTTP or custom node)
- Event Bus via NATS (community)

#### Human-in-the-Loop
No native human task management. Workarounds: send email with approval link → webhook resumes workflow. n8n 2024 added basic waiting functionality (`Wait` node pauses until webhook callback).

#### Multi-Tenant Architecture
- OSS: single-tenant
- n8n Cloud: multi-tenant with team/workspace isolation
- Self-hosted Enterprise: RBAC, SSO (LDAP, SAML), credential sharing, audit logs

#### Observability & Debugging
- Execution list with status, duration, item counts
- Node-by-node output inspection (data visible next to each node after run)
- Error workflow: a separate workflow triggered on failure
- No Prometheus metrics out-of-the-box (Enterprise adds log streaming)

#### Saga / Compensation
No saga support. Error workflow provides a compensation hook but is not a structured rollback mechanism.

#### Connector OAuth / Token Lifecycle
n8n manages OAuth 2.0 tokens for all OAuth-based nodes. Users authenticate once via the Credentials UI; n8n stores and refreshes tokens. Per-workspace credential isolation.

#### What APG Cannot Match (vs. n8n)
- Visual canvas editor with instant node-by-node output inspection
- OAuth token management per connected account
- 400+ pre-built nodes without custom code
- `Wait` node for async pause/resume via webhook
- Code nodes (arbitrary JS/Python inline)
- AI Agent node (LLM + tools on one canvas)

---

### 2.10 Boomi AtomSphere

**Website:** https://boomi.com  
**Model:** Cloud-native iPaaS; Atom-based runtime; drag-and-drop process designer; 80+ application connectors + 22 technology connectors

#### Core Composition Model
Processes are defined in Boomi's web UI as flowcharts connecting shapes (Start, Connector, Map, Decision, Branch, etc.). Atoms (lightweight JVM runtimes) execute processes on-premises, in cloud, or hybrid.

```
[Start] → [Connector: Salesforce GET Contacts]
        → [Map: Salesforce Contact → SAP Customer XML]
        → [Connector: SAP IDoc Send: DEBMAS06]
        → [Notify: Success email]
```

#### Connector / Integration Ecosystem
- **80+ application connectors**: Salesforce, SAP, Oracle ERP, Workday, ServiceNow, NetSuite, Microsoft Dynamics, SAP S/4HANA, QuickBooks, Marketo, HubSpot
- **22 technology connectors**: HTTP/SOAP, FTP/SFTP, JDBC, LDAP, Disk (local file), AS2, MLLP (HL7), JMS, AMQP, MongoDB, Elasticsearch, Amazon S3/SQS/DynamoDB
- B2B/EDI: X12, EDIFACT, Tradacoms, HL7 — with Trading Partner Management UI
- Connector SDK for custom connectors
- SAP: IDoc, BAPI, RFC (via certified SAP connector)

#### Durability & Fault Tolerance
- Atom runtime: local process queue with retry on transient failure
- Molecule (clustered Atom): high-availability with automatic failover
- Boomi Cloud: multi-tenant, managed, auto-scaled
- Process reporting: all executions logged with shape-level detail
- Dead letter handling: configurable retry + alerting

#### Event-Driven Capabilities
- Listen operation: HTTP listener, JMS listener, AS2 listener, SFTP polling, scheduled
- Real-time integration via API service component
- Event Streams (Boomi): managed pub/sub within the Boomi platform (2023+)
- No native Bytewax consumer/producer (use HTTP or JDBC workaround)

#### Human-in-the-Loop
Boomi Flow (now standalone): drag-and-drop UI builder for approval workflows and portals. Can be embedded in Boomi integration processes. Not as powerful as Camunda Tasklist but covers basic approval routing.

#### Multi-Tenant Architecture
- Boomi Enterprise: account → environments (Development, Test, Production) → deployment
- Molecule atoms: share execution across multiple tenants with runtime isolation
- API Management: gateway with per-consumer rate limiting and OAuth

#### Observability & Debugging
- **Process Reporting**: shape-level execution trace with input/output documents at each step
- **Atom Management**: CPU, heap, queue depth metrics
- **Alerting**: email/webhook on process failure
- **Test mode**: run a process in test mode with sample data; inspect at every shape
- No Prometheus/OpenTelemetry natively; Boomi Observe (add-on) adds deeper APM

#### B2B / EDI Management
Best-in-class in this category among the 10 platforms:
- Trading Partner Management: onboard EDI partners with document standards, acknowledgment profiles, communication methods
- Full AS2 managed file transfer
- X12/EDIFACT schema library built-in
- Interchange envelope management (ISA/GS for X12, UNB/UNG for EDIFACT)
- 997/999 functional acknowledgments generated automatically

#### Saga / Compensation
No saga support. Error handling via Try/Catch shape and Notify shape. Compensation is manual: separate process triggered on failure.

#### What APG Cannot Match (vs. Boomi)
- B2B/EDI Trading Partner Management (X12, EDIFACT, AS2)
- SAP IDoc/BAPI/RFC connector (certified)
- HL7 MLLP for healthcare
- AS2 managed file transfer
- Trading partner onboarding with document standard profiles
- Boomi Flow for no-code approval UI

---

## 3. Gap Matrix — Feature × Platform

Legend: **Y** = full support, **P** = partial/limited, **N** = not supported, **APG** = APG's current state

| Feature | Temporal | Camunda 8 | MuleSoft | Step Functions | Zapier/Make | Apache Camel | Conductor/Orkes | Prefect/Dagster | n8n | Boomi | **APG** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Durable execution (event-sourced) | **Y** | **Y** | P | **Y** | N | N | **Y** | P | N | P | **N** |
| JSON/YAML DSL workflow | P (code) | P (BPMN) | P (XML) | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | N (visual) | **Y** |
| Visual workflow designer | P (UI only) | **Y** | **Y** | **Y** | **Y** | P (Kaoto) | **Y** | **Y** | **Y** | **Y** | **N** |
| Per-execution event timeline | **Y** | **Y** | P | **Y** | P | N | **Y** | P | P | **Y** | **N** |
| Workflow versioning (in-flight) | **Y** | **Y** | P | **Y** | N | N | P | P | N | N | **N** |
| Saga / compensation (native) | **Y** | **Y** | P | P | N | **Y** | P | N | N | N | P (declared, not engine-enforced) |
| Child/sub-workflows | **Y** | **Y** | **Y** | **Y** | P | **Y** | **Y** | **Y** | P | **Y** | **N** |
| Durable timers (survive restart) | **Y** | **Y** | P | **Y** | N | N | **Y** | N | N | N | **N** |
| Signals / external events into workflow | **Y** | **Y** | P | P (callback token) | N | P | P | P | P | N | P (in-process only) |
| Human task (native) | P (code) | **Y** | N | P | N | N | **Y** | N | N | P (Boomi Flow) | P (declared) |
| MQTT connector | N | P | **Y** | N | N | **Y** | N | N | N | N | **N** |
| Bytewax inbound trigger | N | **Y** | **Y** | P (MSK) | N | **Y** | P | P | N | N | **N** |
| SAP RFC/IDoc | N | N | **Y** | N | N | **Y** | N | N | N | **Y** | **N** |
| EDI X12/EDIFACT | N | N | **Y** | N | N | P | N | N | N | **Y** | **N** |
| JMS/AMQP/IBM MQ | N | P | **Y** | P | N | **Y** | P | N | N | **Y** | **N** |
| SFTP/FTP connector | N | P | **Y** | N | N | **Y** | N | N | N | **Y** | **N** |
| gRPC connector | N | N | P | P | N | **Y** | P | N | N | N | **N** |
| OAuth 2.0 token lifecycle | N | P | **Y** | **Y** | **Y** | P | P | N | **Y** | **Y** | **N** |
| Connector pagination | N | N | **Y** | N | P | P | N | N | P | P | **N** |
| Per-tenant connector credentials | P (namespace) | P | **Y** | P | **Y** | N | P | N | P | P | **N** |
| Connector marketplace/registry | N | **Y** | **Y** | N | **Y** | P | N | N | P | P | P (scan only) |
| Multi-tenant isolation | **Y** | **Y** | **Y** | P | **Y** | N | **Y** | P | P | **Y** | P (tenant_id field) |
| Prometheus/OpenTelemetry | **Y** | **Y** | **Y** | P | N | **Y** | **Y** | **Y** | N | P | **N** |
| 400+ pre-built connectors | N | P (50+) | **Y** (1500+) | P (AWS) | **Y** | **Y** (350+) | N | N | **Y** (400+) | P (100+) | **N** |
| Streaming/CDC integration | N | P | **Y** | P | N | **Y** | N | P | N | N | **N** |
| Data lineage / asset tracking | N | N | P | N | N | N | N | **Y** | N | N | **N** |
| B2B/EDI trading partner mgmt | N | N | **Y** | N | N | N | N | N | N | **Y** | **N** |
| ISO8583 / payment switching | N | N | N | N | N | N | N | N | N | N | **Y** (capability) |

---

## 4. Types of Systems APG CANNOT Compose (With Examples)

### 4.1 Message-Oriented Middleware
**Systems:** Bytewax, AWS SQS/SNS, Azure Service Bus, RabbitMQ, IBM MQ, NATS (durable), ActiveMQ, AMQP brokers  
**Why APG cannot:** APG event subscriptions are in-process Python dict lookups. They cannot consume from an external broker topic, acknowledge messages, handle backpressure, or implement consumer groups.  
**Example use case:** "When a payment event arrives on the `payments.captured` Bytewax topic, trigger the `FulfillOrder` workflow."  
**What's needed:** A Bytewax consumer adapter that bridges broker messages into APG event emissions.

### 4.2 IoT and Industrial Protocols
**Systems:** MQTT brokers (Mosquitto, HiveMQ, AWS IoT Core), OPC-UA servers, Modbus devices, Siemens S7 PLCs, BACNET (building automation)  
**Why APG cannot:** These use binary/topic-based protocols over TCP that are not HTTP.  
**Example use case:** "When sensor temperature exceeds threshold on MQTT topic `factory/sensor/01/temp`, trigger `MaintenanceAlert` workflow."  
**What's needed:** MQTT subscriber component that emits APG events.

### 4.3 ERP and Enterprise Systems via Native Protocols
**Systems:** SAP ECC/S4HANA (RFC, IDoc, BAPI), Oracle E-Business Suite (Oracle AQ), JD Edwards, Microsoft Dynamics (COM/DCOM)  
**Why APG cannot:** SAP RFC requires the SAP JCo (Java Connector) or PyRFC native library and its own socket protocol (not HTTP). IDocs are XML documents delivered over CPIC/RFC or AS2. APG's connector generator only reads OpenAPI specs and emits HTTP stubs.  
**Example use case:** "When a PurchaseOrder is approved in APG, create an SAP IDoc of type ORDERS05 in the SAP backend."  
**What's needed:** PyRFC-based SAP connector or delegation to an MuleSoft/Camel adapter.

### 4.4 EDI and B2B Document Exchange
**Systems:** X12 (North America), EDIFACT (international), HL7 (healthcare), RosettaNet (supply chain), TRADACOMS (UK retail)  
**Why APG cannot:** EDI messages are fixed-width or segment-based documents, not JSON/HTTP. AS2 transport uses S/MIME encryption and MDN acknowledgments. APG has no EDI parser, schema library, or AS2 transport.  
**Example use case:** "Receive an X12 850 Purchase Order from a trading partner via AS2, validate against the 850 schema, transform to an APG procurement record, and send an X12 997 Functional Acknowledgment."  
**What's needed:** X12/EDIFACT parser library + AS2 transport layer.

### 4.5 Financial Industry Protocols
**Systems:** SWIFT MT/MX (ISO 20022), FIX protocol (equities/FX trading), ISO8583 (card payment switching), ACH (NACHA), SEPA XML  
**Why APG cannot:** These are domain-specific binary or fixed-format protocols with strict validation, checksums, and network-layer requirements.  
**Note:** APG has an ISO8583 capability (`capabilities/fintech/switch/`) but it is a standalone module, not integrated into the workflow/connector system.  
**Example use case:** "Route an ISO20022 pacs.008 SWIFT payment message through the workflow engine with compliance screening."

### 4.6 Streaming Data Platforms
**Systems:** Bytewax Streams, AWS Kinesis, Azure Event Hub, Apache Flink, Apache Spark Streaming, ksqlDB  
**Why APG cannot:** APG workflows are request-response triggered. Stream processing involves stateful windowed aggregations over infinite event streams — fundamentally different execution model from APG's step-by-step workflow.  
**Example use case:** "Aggregate all `payment_attempted` events in a 5-minute tumbling window and trigger a `FraudAlert` workflow if more than 10 failures from the same card."

### 4.7 Database Change-Data-Capture (CDC)
**Systems:** Debezium + Bytewax, AWS DMS, Azure Data Factory CDC, Oracle GoldenGate  
**Why APG cannot:** CDC requires reading the database replication log (Postgres WAL, MySQL binlog). APG has no replication log reader.  
**Example use case:** "Whenever a row is updated in the `orders` table of a legacy system, emit an APG event and trigger the `SyncOrderToERP` workflow."

### 4.8 Healthcare-Specific Protocols
**Systems:** HL7 v2 (MLLP transport), FHIR R4 (REST but with complex profile validation), DICOM (imaging), X12 837/835 (medical claims)  
**Why APG cannot:** HL7 v2 uses MLLP (Minimal Lower Layer Protocol) over bare TCP. APG has no MLLP transport. FHIR validation requires the FHIR spec schema library.  
**Example use case:** "Receive an HL7 ADT^A01 admission message, parse the PID segment, create a patient record in APG, trigger the `PatientOnboarding` workflow."

### 4.9 Email and Messaging Protocol Inbounds
**Systems:** IMAP/POP3 (email polling), SMTP inbound relay, WhatsApp Business API webhooks (already partially supported), Telegram Bot API  
**Why APG cannot:** IMAP is a stateful protocol; polling an inbox and parsing MIME attachments is not a workflow connector in APG's current model.  
**Example use case:** "Poll an IMAP inbox for invoices; when a PDF invoice arrives, OCR it and create a PurchaseRequest workflow."

### 4.10 Legacy File Transfer and Batch Systems
**Systems:** SFTP/FTP servers, IBM MQ file transfer, mainframe JES spool files, flat-file EDI drops  
**Why APG cannot:** APG has no file-system polling component or FTP connector.  
**Example use case:** "Every hour, poll an SFTP directory for new payroll files, parse fixed-width records, and trigger `ProcessPayslip` workflows for each employee."

---

## 5. Prioritized Gap List — What to Implement First for Maximum Impact

Prioritization criteria: (a) number of real customer use cases unblocked, (b) implementation effort, (c) whether it is a foundational enabler for other gaps.

### Tier 1 — Foundational (blocks everything else)

**5.1 Durable Workflow State (Event-Sourced Persistence)**  
_Effort: High | Impact: Critical_  
Replace JSON file writes with an append-only event log per workflow run (minimum: Postgres-backed). Workflow replay must restore state from events, not from a snapshot. This eliminates data loss on process crash and enables exactly-once workflow logic semantics. Consider: adopt Temporal as the backend execution engine (APG DSL compiles to Temporal workflow code) rather than building a custom event store.

**5.2 External Message Bus Integration**  
_Effort: Medium | Impact: Critical_  
Bytewax consumer + NATS consumer as APG event sources. A workflow declared with `subscribe_events: [bytewax://payments.captured]` should register a Bytewax consumer group that feeds `emit_apg_event()`. This unblocks all event-driven enterprise integration patterns.

**5.3 OAuth 2.0 Connector Credential Store**  
_Effort: Medium | Impact: High_  
Replace static `APG_CONNECTOR_AUTH` env var with a credential store supporting: client_credentials, authorization_code+PKCE, refresh tokens, per-tenant isolation. Minimum viable: encrypt and store tokens in Postgres, auto-refresh before expiry.

### Tier 2 — Connector Ecosystem Expansion

**5.4 MQTT Connector**  
_Effort: Low–Medium | Impact: High_  
`paho-mqtt` Python client wrapped as an APG connector. Enables IoT, smart meter, device telemetry use cases.

**5.5 SFTP/FTP Connector**  
_Effort: Low | Impact: High_  
`paramiko`-based SFTP polling connector. Enables EDI file drops, payroll file processing, any legacy batch file integration.

**5.6 SAP Connector (PyRFC)**  
_Effort: High | Impact: Medium–High (depends on customer base)_  
`pyrfc` Python wrapper around SAP JCo. Enables RFC calls, BAPI invocations, IDoc send/receive. Required for any ERP customer running SAP.

**5.7 Connector Pagination Support**  
_Effort: Low | Impact: Medium_  
Add `paginated: true`, `cursor_field: "next_page_token"`, `page_size: 100` to the connector generator. Generated stub iterates until no next page. Eliminates the current silent truncation on paginated APIs.

**5.8 Connector Rate Limiting**  
_Effort: Low | Impact: Medium_  
Token-bucket rate limiter in the base `APGConnector` class. Configure per connector: `rate_limit: {requests_per_minute: 60}`.

### Tier 3 — Workflow Engine Enhancements

**5.9 Workflow Versioning**  
_Effort: Medium | Impact: High_  
Pin running workflow instances to the DSL version that started them. New instances use the latest version. Requires versioned workflow definitions stored in DB, not re-parsed from `.apg` file on every run.

**5.10 Sub-Workflow / Child Workflow Composition**  
_Effort: Medium | Impact: High_  
A workflow step type `sub_workflow: OrderFulfillment` that spawns a child workflow run, optionally awaiting its completion. Enables fan-out patterns, reusable sub-processes, saga coordination across workflows.

**5.11 Workflow Event Timeline API**  
_Effort: Medium | Impact: High_  
Per-run event log endpoint: `GET /workflows/{run_id}/events` returns every step transition, guard evaluation, timer fire, human task assignment, compensation trigger — with timestamps. Feeds a minimal UI timeline (could be built with htmx in the generated app).

**5.12 `waitForExternalEvent` with Durable Token**  
_Effort: Medium | Impact: Medium_  
Generate a one-time callback token for a `waits` step. Calling `POST /workflows/{run_id}/signal?token=XXX&event=payment_confirmed` resumes the workflow. Currently, APG's `waits` field is declared but the external signalling mechanism is not implemented in the generated app.

### Tier 4 — Quality & Operations

**5.13 Connector Marketplace with Version Pinning**  
_Effort: Low–Medium | Impact: Medium_  
Extend `scan_connectors()` to a proper registry with connector name + semver pinning in the `.apg` spec: `connector PaymentsAPI version "^2.1.0"`. Package connectors as Python packages installable via `pip install apg-connector-stripe`.

**5.14 EDI Parser Library Integration**  
_Effort: High | Impact: Medium (healthcare/supply chain verticals)_  
Integrate `pydifact` (EDIFACT) and a Python X12 library. Add an `edi` connector type that parses EDI documents into structured APG records.

**5.15 Prometheus Metrics Endpoint**  
_Effort: Low | Impact: Medium_  
`GET /metrics` in the generated app: workflow run counts by state, step latency histograms, circuit breaker open/closed counts, connector error rates. Standard Prometheus text format.

---

## 6. All Sources

### Temporal.io
- [Durable Execution Platform — Temporal](https://temporal.io/product)
- [What is Durable Execution — Temporal Blog](https://temporal.io/blog/what-is-durable-execution)
- [Temporal Workflow Execution — Docs](https://docs.temporal.io/workflow-execution)
- [Mastering Durable Execution in Distributed Systems](https://temporal.io/blog/durable-execution-in-distributed-systems-increasing-observability)
- [Temporal Use Cases and Design Patterns](https://docs.temporal.io/evaluate/use-cases-design-patterns)
- [Human-in-the-Loop AI Agent — Temporal Docs](https://docs.temporal.io/ai-cookbook/human-in-the-loop-python)
- [Implementing Saga Pattern with Temporal](https://devtechtools.org/en/blog/implementing-saga-pattern-temporal-distributed-transactions)
- [Child Workflows — Temporal Docs](https://docs.temporal.io/child-workflows)
- [Worker Versioning — Temporal Docs](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
- [Observability — Temporal Docs](https://docs.temporal.io/evaluate/development-production-features/observability)
- [Temporal Fundamentals Part V: Workflow Patterns — Keith Tenzer](https://keithtenzer.com/temporal/Temporal_Fundamentals_Workflow_Patterns/)
- [GitHub — temporalio/temporal](https://github.com/temporalio/temporal)

### Camunda 8
- [What is Camunda 8 — Docs](https://docs.camunda.io/docs/8.6/components/concepts/what-is-camunda-8/)
- [Camunda 8 Components — Docs](https://docs.camunda.io/docs/8.7/components/)
- [Workflow Patterns — Camunda 8 Docs](https://docs.camunda.io/docs/components/concepts/workflow-patterns/)
- [Human Task Orchestration Guide — Camunda 8 Docs](https://docs.camunda.io/docs/guides/orchestrate-human-tasks/)
- [Saga Pattern Realization With Camunda](https://karolinduerr.github.io/BA-SagaPattern/Camunda/Camunda_General/)
- [Compensation Events in Camunda 8 — Bank Example](https://camunda.com/blog/2025/06/how-a-bank-uses-compensation-events-camunda-8/)
- [Task Tester — Camunda Blog](https://camunda.com/blog/2026/03/getting-fast-feedback-with-camunda-task-tester/)
- [Camunda Connectors Overview — Academy](https://academy.camunda.com/c8-connectors-overview)
- [How to Build a Camunda 8 Connector](https://camunda.com/blog/2022/11/how-to-build-camunda-platform-8-connector/)
- [How Camunda 8 Supports Process Automation at Enterprise Scale](https://camunda.com/blog/2022/09/how-camunda-8-supports-process-automation-at-enterprise-scale/)

### MuleSoft Anypoint
- [MuleSoft Anypoint Platform — Salesforce](https://www.salesforce.com/mulesoft/anypoint-platform/)
- [API-led Connectivity — MuleSoft Blog](https://blogs.mulesoft.com/learn-apis/api-led-connectivity/)
- [Anypoint Connectors Overview — MuleSoft Docs](https://docs.mulesoft.com/connectors/introduction/introduction-to-anypoint-connectors)
- [SAP Connector 5.9 — MuleSoft Docs](https://docs.mulesoft.com/sap-connector/latest/)
- [SAP Integration With MuleSoft — Medium](https://medium.com/another-integration-blog/sap-integration-with-mulesoft-9d3c3b5a3d39)
- [EDI Transactions with Anypoint X12 Connector — MuleSoft Blog](https://blogs.mulesoft.com/dev-guides/api-connectors-templates/edi-transactions-with-anypoint-x12-connector/)
- [MuleSoft Connectors — Complete Guide](https://www.nexgenarchitects.com/blog-posts/mulesoft-connectors)
- [API-led Connectivity Whitepaper](https://www.integsoft.cz/resources/files/anypoint-platform/API-led-connectivity-whitepaper.pdf)

### AWS Step Functions
- [Amazon States Language — AWS Docs](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-amazon-states-language.html)
- [Workflow Studio — AWS Docs](https://docs.aws.amazon.com/step-functions/latest/dg/workflow-studio.html)
- [Saga Orchestration Pattern — AWS Prescriptive Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/saga-orchestration.html)
- [Implementing Saga Pattern with AWS Lambda and Step Functions — Medium](https://hexshift.medium.com/implementing-the-saga-pattern-for-distributed-transactions-with-aws-lambda-and-step-functions-117ce2530149)
- [AWS Step Functions Saga Pattern Implementation](https://awsforengineers.com/blog/aws-step-functions-saga-pattern-implementation/)
- [Enhanced Local IDE Experience for Step Functions — AWS Blog](https://aws.amazon.com/blogs/compute/introducing-an-enhanced-local-ide-experience-for-aws-step-functions/)

### Zapier / Make
- [Zapier vs Make (Integromat) 2026](https://agence-scroll.com/en/blog/zapier-vs-integromat)
- [n8n vs Zapier vs Make: 2024 Decision Guide](https://cloudminister.com/blog/n8n-vs-zapier-vs-make-integromat-which-workflow-automation-tool-is-right-for-you/)
- [Zapier vs Integromat: Automation Platforms Compared](https://saasvssaas.com/zapier-vs-integromat/)

### Apache Camel
- [Apache Camel EIPs — Official Docs](https://camel.apache.org/components/4.18.x/eips/enterprise-integration-patterns.html)
- [Apache Camel — Wikipedia](https://en.wikipedia.org/wiki/Apache_Camel)
- [Integration Patterns with Apache Camel — Baeldung](https://www.baeldung.com/camel-integration-patterns)
- [Understanding EIPs with Apache Camel — Medium](https://medium.com/@lakshmeeshachar/understanding-enterprise-integration-patterns-eips-with-apache-camel-the-ultimate-guide-ebb0f93d854d)
- [Java DSL — Apache Camel Docs](https://camel.apache.org/manual/java-dsl.html)
- [Endpoint DSL — Apache Camel Docs](https://camel.apache.org/manual/Endpoint-dsl.html)
- [Defining Camel Routes in Quarkus — Docs](https://camel.apache.org/camel-quarkus/3.27.x/user-guide/defining-camel-routes.html)
- [Red Hat Build of Apache Camel for Enterprise Integration](https://www.redhat.com/en/resources/accelerate-enterprise-integration-datasheet)

### Netflix Conductor / Orkes
- [Netflix Conductor: A Microservices Orchestrator — Netflix TechBlog](https://netflixtechblog.com/netflix-conductor-a-microservices-orchestrator-2e8d4771bf40)
- [Netflix Conductor — GeeksforGeeks](https://www.geeksforgeeks.org/system-design/netflix-conductor-microservices-orchestration/)
- [Orkes Whitepaper: From Netflix Conductor to Enterprise Scale](https://orkes.io/whitepapers/orkes-whitepaper-from-netflix-conductor-to-enterprise-scale/)
- [Exploring Netflix/Orkes Conductor — Arpit Rathore](https://arpitrathore.com/exploring-netflixorkes-conductor-a-workflow-orchestration-powerhouse)
- [Orkes Brings GenAI and Human-in-the-Loop — BusinessWire](https://www.businesswire.com/news/home/20231108649301/en/Orkes-Brings-Generative-AI-and-Human-in-the-Loop-Capabilities-to-Microservices-and-Workflow-Orchestration)
- [Orkes Raises $60M — BusinessWire](https://www.businesswire.com/news/home/20260423550324/en/Orkes-Raises-$60M-as-Developers-Increasingly-Use-Its-Platform-to-Deploy-AI-Confidently-in-Production)
- [Orkes Conductor Documentation](https://orkes.io/content/)
- [Orkes Platform](https://orkes.io/platform)

### Prefect / Dagster
- [Dagster vs Prefect: Key Differences 2024 — Orchestra](https://www.getorchestra.io/guides/dagster-vs-prefect-key-differences-2024)
- [Dagster vs Prefect: Compare Modern Orchestration Tools — Dagster](https://dagster.io/vs/dagster-vs-prefect)
- [Data Pipeline Orchestration: Airflow vs Dagster vs Prefect 2026](https://reintech.io/blog/data-pipeline-orchestration-airflow-dagster-prefect-2026)
- [Orchestration Showdown: Dagster vs Prefect vs Airflow — ZenML](https://www.zenml.io/blog/orchestration-showdown-dagster-vs-prefect-vs-airflow)
- [Decoding Data Orchestration Tools — FreeAgent Engineering](https://engineering.freeagent.com/2025/05/29/decoding-data-orchestration-tools-comparing-prefect-dagster-airflow-and-mage/)

### n8n
- [n8n Features — Official](https://n8n.io/features/)
- [n8n Complete Overview 2024 — No Code Alliance](https://nocodealliance.org/tool-overview/n8n)
- [n8n: Open-Source Workflow Automation — IJCAONLINE](https://www.ijcaonline.org/archives/volume187/number63/n8n-an-open-source-workflow-automation-for-enterprise-integration-and-ai-orchestration/)
- [Inside n8n: Fair-Code Platform Leads AI-Powered Automation — Medium](https://medium.com/@takafumi.endo/inside-n8n-how-a-fair-code-open-source-platform-leads-ai-powered-workflow-automation-e8128890d496)

### Boomi
- [What is Boomi AtomSphere Platform — TechTarget](https://www.techtarget.com/searchcloudcomputing/definition/Dell-Boomi)
- [Boomi AtomSphere Platform: Important Aspects — Preludesys](https://preludesys.com/aspects-of-boomi-atomsphere-platform/)
- [Dell Boomi iPaaS Integration Services — ApiX-Drive](https://apix-drive.com/en/blog/other/dell-boomi-ipaas-integration-services)
- [Boomi Integration Platform Explained — NeosAlpha](https://neosalpha.com/boomi-integration-platform-explained/)
- [Dell Boomi Integration Platform Guide — Bluent](https://www.bluent.com/blog/boomi-integration-explained)

### Cross-Platform and Integration Theory
- [IoT and Event Streaming at Scale with Bytewax and MQTT — APG](https://www.apg.io/blog/iot-with-bytewax-connect-mqtt-and-rest-proxy/)
- [OPC UA, MQTT, and Bytewax — Kai Waehner](https://www.kai-waehner.de/blog/2022/02/11/opc-ua-mqtt-apache-bytewax-the-trinity-of-data-streaming-in-industrial-iot/)
- [Designing Durable Event-Driven Workflows — Medium](https://medium.com/@nileshsharma_4675/designing-durable-event-driven-workflows-making-systems-resilient-and-reliable-484d88b8a12f)
- [DSL-Based Workflow Orchestration: Introduction and Architecture — Medium](https://medium.com/@nareshvenkat14/dsl-based-workflow-orchestration-part-1-introduction-architecture-9d0112f77e00)
- [State of Open Source Workflow Orchestration Systems 2025 — PracData](https://www.pracdata.io/p/state-of-workflow-orchestration-ecosystem-2025)
- [Workflow Orchestration: Enterprise Automation at Scale — BMC](https://www.bmc.com/blogs/workflow-orchestration/)
- [Limits of the Event-Driven Orchestrator — Data People Etc.](https://stkbailey.substack.com/p/limits-of-the-event-driven-orchestrator)
- [Temporal vs Apache Airflow: Workflow Orchestration Compared — Xgrid](https://www.xgrid.co/resources/temporal-vs-apache-airflow-workflow-orchestration/)
- [Dapr Workflow Features and Concepts](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-features-concepts/)

---

## Appendix: APG Current Capability Inventory (Reference Baseline)

Documented from source inspection of `/Users/nyimbiodero/src/pjs/apg`:

| Capability | Implementation |
|---|---|
| Workflow DSL | `workflow { steps, human_tasks, assignments, guards, timers, waits, retry_policy, compensation }` |
| Persistence | JSON file (atomic `os.replace` with `.tmp`); optional PostgreSQL `_pg_save_workflow_run()` |
| Event pub/sub | In-process `APG_EVENT_SUBSCRIPTIONS` dict; `emit_apg_event()` triggers subscribers synchronously |
| Circuit breaker | Per-step counter in `CIRCUIT_BREAKERS` dict; threshold + reset_timeout configurable |
| Connector generator | `apg connector generate --spec openapi.yaml` → Python class with HTTP stub + circuit breaker |
| Connector auth | Static `Bearer {api_key}` or `Authorization` header; from env var |
| Multi-tenancy | `tenant_id` field on entities; `TENANT_SCOPED_ENTITIES` set; `_tenant_id()` thread-local |
| Human tasks | Declared in `human_tasks:` list; stored as `status=waiting` in run; no external UI integration |
| Compensation | Declared in `compensation:` dict; executed as workflow step names; no reverse-order guarantee |
| Timers | ISO 8601 duration strings stored in `timers:` dict; enforced via `_step_failure_budget()` |
| Workflow resume | `POST /workflows/{name}/runs/{id}/resume` — replays from last completed step |
| Durable timers | NOT durable — timers are computed relative to `started_at` on resume; do not survive kill |
| Sub-workflows | NOT supported — a single workflow cannot spawn another |
| Workflow versioning | NOT supported — definition re-parsed from ENTITIES on every run |
| Bytewax/MQTT/AMQP | NOT supported |
| SAP/EDI/SWIFT | NOT supported |
| OAuth lifecycle | NOT supported |
| Visual designer | NOT supported |
| Metrics endpoint | NOT supported |
| Temporal integration | Partial — `apg_activities.py` decorates functions with `@activity.defn` when temporalio installed, but no generated workflow code targets Temporal |
