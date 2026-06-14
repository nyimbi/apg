# Composition Systems Analysis 2026
## Benchmarking APG Against World-Class Composition Platforms

**Author:** Nyimbi Odero  
**Date:** 2026-06-15  
**Scope:** Comparative analysis of composition paradigms, gap analysis for APG, and a prioritized closure roadmap.

---

## 1. Introduction

APG is a DSL compiler that generates deployable Python/Flask applications from `.apg` source files. It ships 266+ pre-built capabilities spanning fintech, healthcare, SCM, HCM, government, intelligence, and 20+ other domains. The composition engine has a defined dependency graph (1,900 require-edges, 2,050 provide-entries), a deployment-tier ordering, and active capability contracts keyed on provides/requires service name pairs.

This document benchmarks APG against the world's most mature composition platforms, identifies concrete gaps, categorizes system types APG cannot currently compose, and proposes a prioritized closure roadmap.

---

## 2. World's Best Composition Systems — Deep Analysis

### 2.1 Salesforce Platform

**Paradigm:** Declarative, event-driven, low-code orchestration over a shared multi-tenant data model.

**Core composition primitives:**
- **Flow Builder** — point-and-click visual DAG editor for business processes. Flows branch, loop, call Apex, invoke external HTTP, query/mutate Salesforce objects, and send notifications. Executes synchronously (screen flows), asynchronously (autolaunched flows), or on schedule.
- **Platform Events** — publish/subscribe event bus built into the data model. Any object mutation can emit a platform event; subscribers react in real time with no polling.
- **MuleSoft Anypoint** (owned by Salesforce) — provides the integration layer for external system composition.
- **Apex Triggers** — low-level hooks that fire on DML operations, enabling reactive composition within the data tier.
- **Einstein Automate** — AI-assisted flow generation that suggests steps from plain-language descriptions.

**Composition strengths:** Tight coupling between data model and process model. A record change automatically propagates through subscribed flows. No separate event broker to manage. Change-Data-Capture (CDC) events are first-class citizens.

**Scale:** Millions of orgs, each with isolated flow graphs. Multi-tenant isolation is enforced at the platform layer — one org's flows cannot interfere with another's.

---

### 2.2 ServiceNow Now Platform

**Paradigm:** Workflow-first platform where all records, approvals, and integrations are modeled as workflow nodes.

**Core composition primitives:**
- **Flow Designer** — NLU-assisted drag-and-drop workflow editor with typed triggers (record change, schedule, inbound webhook, REST). Native support for parallel branches, wait-for-approval steps, and subflows.
- **IntegrationHub** — pre-built "spokes" (connectors) for Slack, Jira, AWS, etc. Each spoke exposes typed actions that can be dropped into a flow. No custom code required for common integrations.
- **Scripted REST APIs** — compose external APIs as first-class Now Platform objects.
- **Event Management** — real-time event correlation from monitoring tools mapped to ITSM records.
- **Predictive Intelligence** — ML classifiers embedded as composition nodes (route ticket to correct team based on text).

**Composition strengths:** Human-in-the-loop is a first-class concept. Approval chains with SLA timers, escalation paths, and audit trails are built into the workflow primitives. Workflows can suspend indefinitely waiting for a human decision without polling.

---

### 2.3 Microsoft Power Platform

**Paradigm:** Citizen-developer composition via connectors, a shared data layer (Dataverse), and visual flow builders.

**Core composition primitives:**
- **Power Automate** — 1,000+ pre-built connectors. Flows compose connectors declaratively. Supports parallel branches, do-until loops, condition splits, and error handling scopes.
- **Dataverse** — shared normalized data model that any connector can read/write. Provides cross-application consistency without custom integration code.
- **Power Apps** — low-code front-end composition tied to Dataverse tables.
- **Azure Integration Services** — Logic Apps, Service Bus, Event Grid, API Management form the enterprise-grade composition backbone below the citizen-developer layer.
- **Connector SDK** — schema-driven connector auto-generation: define an OpenAPI spec, and Power Automate generates a connector with typed inputs/outputs that appear in the visual editor.

**Composition strengths:** The connector SDK reduces integration time for new external systems to hours. The OpenAPI → connector pipeline is the industry benchmark for schema-driven connector generation.

---

### 2.4 OutSystems / Mendix

**Paradigm:** Full-stack low-code with model-driven composition of UI, data, and logic.

**Core composition primitives (OutSystems):**
- **Service Actions** — typed, versioned cross-application service calls. Strong contracts enforced at compile time.
- **Events** — lightweight publish/subscribe within the OutSystems platform.
- **Forge** — marketplace of pre-built components (UI widgets, integrations, business modules) that can be dropped into any app.
- **Impact Analysis** — when a service contract changes, the platform identifies every consumer and flags breaking changes before deploy.

**Core composition primitives (Mendix):**
- **Marketplace** — 1,200+ reusable modules with version pinning and compatibility metadata.
- **Integration with Kafka, MuleSoft, AWS** — native connectors for event streaming and API management.
- **MendixAI** — natural language to module scaffold.

**Composition strengths (shared):** Strong module versioning and impact analysis. The Forge/Marketplace model creates a composable ecosystem with reuse incentives. Contract-first development with compiler-enforced breaking-change detection.

---

### 2.5 AWS Step Functions / EventBridge

**Paradigm:** Infrastructure-level durable state machine composition with event-driven fan-out.

**Core composition primitives:**
- **Step Functions** — JSON/YAML state machine definitions (Amazon States Language). States: Task, Choice, Parallel, Map, Wait, Pass, Fail. Execution history persisted for 90 days. Native retry/timeout/catch per state.
- **EventBridge** — schema registry + event bus. Events match rules and fan-out to 20+ targets (Lambda, SQS, Step Functions, ECS, HTTP endpoints). Schema-driven: auto-generates typed bindings for TypeScript, Python, Java.
- **EventBridge Pipes** — point-to-point enrichment pipelines: source → filter → enrich → target.
- **Express Workflows** — high-throughput (100k events/sec), short-lived (< 5 min) state machines.
- **Standard Workflows** — exactly-once, durable, long-running (up to 1 year).

**Composition strengths:** Declarative retry/timeout/circuit-breaker at the state level — not in application code. Map state enables parallel fan-out over arrays with concurrency limits. Time-travel debugging via execution history: step through every state transition with inputs/outputs. Cross-account event bus sharing enables multi-organization composition.

---

### 2.6 Temporal.io

**Paradigm:** Durable execution — workflows are code (Go, Python, Java, TypeScript) that executes reliably despite failures.

**Core composition primitives:**
- **Workflows** — long-running coroutines. State is automatically persisted as an event log; on failure, re-execution replays from the log rather than from scratch.
- **Activities** — individual units of work (API calls, DB writes). Each activity has configurable retry policies, heartbeat timeouts, and schedule-to-start deadlines.
- **Child Workflows** — sub-workflow composition with parent-child lifecycle coupling or detached execution.
- **Signals** — external events sent into a running workflow (e.g., "payment received"). The workflow suspends waiting for a signal with no polling.
- **Queries** — read current workflow state without altering it.
- **Schedules** — cron-style recurring workflow triggers.
- **Nexus** — cross-namespace and cross-cluster composition: call activities in a remote namespace as if local.

**Composition strengths:** Durable execution eliminates the need to manually checkpoint long-running processes. A workflow that spans 6 months of human approvals is just an `await` in Python. Time-travel debugging is native — replay any historical execution exactly. Compensating transactions (sagas) are first-class patterns in the SDK.

---

### 2.7 Apache Camel / MuleSoft

**Paradigm:** Integration-first composition via message routing patterns (Enterprise Integration Patterns — Hohpe & Woolf).

**Core composition primitives (Apache Camel):**
- **Routes** — from(source).to(target) pipelines with transformers, filters, splitters, aggregators, and choice routers.
- **Components** — 300+ pre-built connectors (Kafka, S3, HTTP, JDBC, AMQP, gRPC, etc.).
- **Enterprise Integration Patterns** — Content-Based Router, Message Filter, Splitter, Aggregator, Dead Letter Channel, Idempotent Consumer all implemented as first-class route DSL nodes.
- **Camel K** — serverless Camel routes as Kubernetes CRDs.

**Core composition primitives (MuleSoft Anypoint):**
- **RAML / AsyncAPI** — design-first API specifications that auto-generate mule flows and typed connectors.
- **Exchange** — asset marketplace with 1,500+ connectors, API fragments, and reusable templates.
- **API Policies** — rate limiting, OAuth, caching applied as composition layers on any API without code changes.

**Composition strengths:** The EIP vocabulary gives developers a shared language for common integration patterns. Dead letter channels, idempotent consumers, and transactional outbox patterns are configuration, not code. MuleSoft's design-first approach (RAML → connector) is the gold standard for API-driven connector generation.

---

### 2.8 Unix Pipes — The Original Composition Model

**Paradigm:** Single-purpose programs composed via stdio streams. Each program reads from stdin, writes to stdout; the shell glues them into pipelines.

**Core composition primitives:**
- **stdin/stdout/stderr** — universal interface. Any program that speaks text can compose with any other.
- **Pipe operator (`|`)** — connects stdout of one process to stdin of the next. Buffered, streaming, zero-copy on Linux.
- **Named pipes / FIFOs** — persistent pipe handles for non-linear composition.
- **Exit codes** — composable error propagation (`&&`, `||`, `set -e`).
- **Environment variables** — ambient context injection without parameter threading.

**Composition strengths:** The smallest possible composition surface area. No framework, no schema, no registry. Programs have no awareness of the pipeline they participate in — pure composition by convention. Composability is a by-product of good single-responsibility design.

**Lessons for APG:** The Unix model succeeds because every program has one clear responsibility and one universal interface. APG capabilities should aspire to this: a single entry-point service contract, universal context injection via tenant/auth headers, and zero framework lock-in at the composition boundary.

---

## 3. What Each System Can Do That APG Cannot (Specific Gaps)

### 3.1 Visual Flow Designers

**Who has it:** Salesforce Flow Builder, ServiceNow Flow Designer, Power Automate, OutSystems, Mendix.

**What APG has:** DSL text files, `composition/studio` directory (exists but underdeveloped), VSCode extension skeleton.

**The gap:** No drag-and-drop composition surface. Business analysts cannot build workflows without writing `.apg` syntax. The `composition/orchestration` capability has `WorkflowOrchestrationService.define_workflow()` with typed parameters, but no visual editor to drive it.

---

### 3.2 Real-Time Event Streaming Composition

**Who has it:** Salesforce Platform Events, EventBridge, Temporal signals, Apache Camel.

**What APG has:** `composition_events` capability (Bytewax-based event streaming bus, tier-7 in deployment order). It exists in code (`capabilities/composition/events/service.py`) with Redis-backed pub/sub, but its integration into the `.apg` DSL is absent — you cannot write `on event X -> capability Y` in a `.apg` file today.

**The gap:** No DSL-level event subscription syntax. Capabilities emit and consume events at the Python service layer, not at the composition declaration layer.

---

### 3.3 Multi-Tenant Composition Isolation

**Who has it:** Salesforce (per-org isolation), ServiceNow (per-instance), OutSystems (per-environment).

**What APG has:** `mten` (multi-tenancy) capability is a Tier-1 foundation (117 dependents). `guard_tenant_id` is used in services. But multi-tenant composition — where tenant A's workflow cannot trigger tenant B's capability — is not enforced at the composition routing layer.

**The gap:** The `composition/gateway` has circuit-breaker and TLS support, but no tenant-scoped routing that prevents cross-tenant capability invocation at runtime.

---

### 3.4 Cross-Capability Rollback and Compensation (Sagas)

**Who has it:** Temporal (first-class Saga pattern), AWS Step Functions (Catch/Compensate states), MuleSoft (transactional scopes).

**What APG has:** `WorkflowOrchestrationService` stores `_compensations` dict and `define_workflow()` accepts `compensation_steps`. The data structure exists but no saga execution engine drives it.

**The gap:** Compensation logic is stored but not automatically triggered on failure. A failed `fintech_payments → fintech_wallets → fintech_kyc` chain has no automatic rollback.

---

### 3.5 Time-Travel Debugging of Composed Flows

**Who has it:** AWS Step Functions (90-day execution history, step-by-step replay), Temporal (event log replay), ServiceNow (flow execution audit trail).

**What APG has:** `audl` (audit log) capability is Tier-1 with 220 dependents — every capability writes to it. But there is no tool to reconstruct a workflow execution from audit events and replay it.

**The gap:** The audit data exists; the debugger that navigates it does not.

---

### 3.6 Declarative Retry / Timeout / Circuit-Breaker

**Who has it:** AWS Step Functions (per-state Retry/Catch), Temporal (per-activity RetryPolicy), Apache Camel (Dead Letter Channel), Power Automate (Configure run after).

**What APG has:** `composition/gateway` has `advanced_circuit_breaker.py`. The code exists but is not surfaced in the `.apg` DSL. You cannot write `retry: 3, timeout: 30s, backoff: exponential` on a capability call in a `.apg` file.

**The gap:** Resilience primitives are implemented in Python but not DSL-accessible. Every capability developer must wire them manually.

---

### 3.7 Schema-Driven Connector Auto-Generation

**Who has it:** Power Platform Connector SDK (OpenAPI → connector in hours), MuleSoft (RAML → Mule flow), EventBridge (AsyncAPI → typed event bindings).

**What APG has:** `capability_contract.py` defines PROVIDES/REQUIRES lists as Python strings. The `int_api` capability manages external API integration. But there is no pipeline where an OpenAPI spec auto-generates a capability stub.

**The gap:** Adding a new external system (e.g., a bank's SOAP API) requires hand-writing a capability. The `capability_contract_factory.py` exists but does not ingest OpenAPI/AsyncAPI specs.

---

### 3.8 Composition Marketplace / Store

**Who has it:** OutSystems Forge (1,200+ components), Mendix Marketplace (1,200+ modules), Power Platform AppSource, MuleSoft Exchange (1,500+ assets).

**What APG has:** `marketplace/` directory exists in the repo root, and `capabilities/composition/registry` is a Tier-8 deployment. But there is no hosted index, version-pinned distribution format, or discovery UI for APG capabilities.

**The gap:** APG's 266 capabilities are a local filesystem artifact. There is no way for a third party to publish a capability, or for an APG user to install one by name.

---

## 4. Types of Systems APG Cannot Currently Compose

### 4.1 Streaming / Real-Time Data Pipelines

Systems like Kafka, Flink, Spark Streaming, Bytewax operate on continuous, unbounded event streams. APG's `composition_events` capability is Bytewax-backed and exists in code, but `.apg` files model discrete CRUD entities (`table Patient { ... }`), not streams. There is no `stream`, `window`, `aggregation`, or `watermark` concept in the DSL.

**Example gap:** Cannot write a `.apg` file that computes a rolling 5-minute sum of M-Pesa transactions and triggers a fraud alert when the threshold is crossed.

### 4.2 Long-Running Human-in-the-Loop Workflows (>24 hours)

Processes like loan approvals (days), regulatory submissions (weeks), grant disbursement (months) require durable suspension — the system waits for a human decision without holding a process or database connection. APG's `wflo` capability has `_suspended` dict in memory, but there is no durable execution layer (like Temporal's event log) that survives process restarts.

**Example gap:** A 30-day loan underwriting workflow that waits for credit bureau response, KYC document upload, and committee approval cannot be expressed as a single durable APG composition.

### 4.3 Cross-Organization Boundary Compositions

Compositions that span organizations — bank ↔ telecoms ↔ regulator — require federated identity, cross-org event routing, and contractual data sharing agreements enforced at the protocol level. APG has `ztna` (zero-trust network access) and `mten` (multi-tenancy), but no concept of a cross-org composition contract where org A grants org B access to a specific capability endpoint.

**Example gap:** A SACCOS-to-bank-to-CBK regulatory reporting chain where each party is a separate APG deployment cannot share typed events without custom integration code.

### 4.4 ML Inference Pipelines

Pipelines that combine feature engineering, model inference, post-processing, and feedback loops are structurally different from CRUD workflows. APG has `aicr` (AI/ML core), `mlcm` (ML model lifecycle), and `ragn` (RAG). But there is no DSL for declaring a feature store → model inference → confidence threshold → fallback → logging pipeline as a composable APG flow.

**Example gap:** A credit scoring pipeline that calls three models, ensembles results, checks regulatory explainability requirements, and logs the decision cannot be expressed as a `.apg` composition.

### 4.5 IoT Sensor Composition

IoT composition involves high-frequency, schema-sparse, time-series data from heterogeneous devices. APG has `iotd` (IoT device management, Tier-6) and `edge` (edge computing, Tier-8), but the DSL has no primitives for time-series aggregation, device shadow state, or sensor fusion patterns.

**Example gap:** A precision agriculture `.apg` file cannot compose soil sensor readings with weather API data to trigger irrigation commands at sub-minute resolution.

### 4.6 Multi-Cloud Compositions

Compositions that span AWS, Azure, and GCP — e.g., inference on Azure AI, storage on S3, identity on Google IAM — require cloud-neutral abstraction at the connector level. APG's `conn` (connectors) capability exists but has no multi-cloud abstraction layer or portable IAM binding.

### 4.7 Regulatory Approval Chains

Regulated processes (drug trials, securities filings, central bank submissions) require tamper-evident multi-party approval with legal-grade audit trails, time-stamped digital signatures, and regulatory authority as a participant in the workflow. APG has `grc_pol`, `grc_aud`, `esgn` (e-signature), and `comp` (compliance engine), but no capability that models a regulatory authority as a composition participant with a signed handoff protocol.

---

## 5. Gap Closure Plan — Prioritized by Value / Effort

### Priority 1 (High Value, Moderate Effort): DSL-Level Resilience Primitives

**What to build:** Add `retry`, `timeout`, `circuit_breaker`, and `on_failure` keywords to the `.apg` grammar at the capability-call level.

```apg
// Proposed syntax
workflow LoanApproval {
    step kyc_check {
        capability: fintech_kyc.verify;
        retry: 3, backoff: exponential, timeout: 10s;
        on_failure: compensate;
    }
}
```

**Architecture:** The compiler's `code_generator.py` already templates Python output. The `composition/gateway`'s `advanced_circuit_breaker.py` is the runtime target. The grammar extension in `compiler/parser.py` and AST node in `compiler/ast_builder.py` are the only new surfaces needed.

**Effort:** ~2 weeks. No new infrastructure. Immediately improves every capability composition.

---

### Priority 2 (High Value, Moderate Effort): DSL-Level Event Subscriptions

**What to build:** Add `on event` and `emit event` syntax to the `.apg` DSL, backed by the existing `composition_events` (Bytewax/Redis) capability.

```apg
on event fintech_payments.payment_received {
    trigger: fintech_wallets.credit_wallet;
    filter: event.amount > 0;
}
```

**Architecture:** The `composition_events` service exists. The `mqeb` (message queue / event bus) is Tier-5. The compiler generates a subscription registration call at app startup. No new runtime components needed — wire DSL to existing infrastructure.

**Effort:** ~3 weeks. Very high leverage — unlocks reactive composition across all 266 capabilities.

---

### Priority 3 (High Value, High Effort): Durable Saga Execution Engine

**What to build:** Drive the existing `compensation_steps` structure in `WorkflowOrchestrationService` with a persistent saga log backed by the `audl` capability.

**Architecture:** On workflow step failure, the orchestration engine reads `compensation_steps` in reverse order and executes each. The `audl` event log provides the execution history needed for replay. A saga coordinator coroutine polls `audl` for incomplete compensation chains on startup (crash recovery). This mirrors Temporal's approach without requiring a separate Temporal cluster.

**Effort:** ~4 weeks. Enables reliable multi-capability transactions (e.g., payment + wallet + ledger as an atomic saga).

---

### Priority 4 (High Value, High Effort): Capability Marketplace with OpenAPI Ingestion

**What to build:** Two components:
1. A CLI command `apg marketplace publish <capability>` that packages a capability as a versioned artifact and uploads to a registry (the existing `composition_registry` capability).
2. A CLI command `apg capability from-openapi <spec.yaml>` that generates a capability stub (models, views, service skeleton, capability_contract) from an OpenAPI 3.x or AsyncAPI 2.x spec.

**Architecture:** `capability_contract_factory.py` is the right insertion point for OpenAPI ingestion. The `apig` (API gateway) capability provides the runtime hosting. The `regy` (registry) capability provides the metadata store. The `marketplace/` directory is the right location for the CLI tooling.

**Effort:** ~6 weeks. Reduces new integration time from days to hours. Directly competitive with Power Platform Connector SDK.

---

### Priority 5 (Medium Value, Low Effort): Audit-Based Flow Debugger

**What to build:** A CLI command `apg debug workflow <execution_id>` that queries the `audl` capability's event store and reconstructs the step-by-step execution trace of a workflow, showing each capability invoked, inputs, outputs, duration, and errors.

**Architecture:** `audl` already captures events from all 220+ dependent capabilities. The debugger is a read-only query tool — no new write paths. Output as a rich terminal table (similar to `git log --graph`) or exportable JSON.

**Effort:** ~1 week. High developer experience value for zero infrastructure cost.

---

### Priority 6 (Medium Value, High Effort): Durable Long-Running Workflows

**What to build:** A persistence layer for `wflo` suspended states backed by PostgreSQL (APG's exclusive data store) rather than in-memory dicts. A signal delivery mechanism (via `mqeb`) that allows external events to resume a suspended workflow.

**Architecture:** Replace `_suspended: dict` in `WorkflowOrchestrationService` with a `wflo_suspended_executions` PostgreSQL table. Use `schd` (scheduler) capability for heartbeat/timeout monitoring. Use `mqeb` for signal delivery. This gives APG Temporal-like durable execution semantics without a Temporal cluster.

**Effort:** ~5 weeks. Enables the 30-day loan approval class of workflows.

---

### Priority 7 (Lower Value, Low Effort): Tenant-Scoped Composition Routing

**What to build:** Middleware in `composition/gateway` that validates `tenant_id` on every inter-capability call and rejects cross-tenant invocations.

**Architecture:** `guard_tenant_id` is already imported in capability services. The gateway needs a routing table (`tenant_id → allowed_capability_set`) enforced as a request filter. The `composition_access` capability (Tier-8) is the right home for this policy.

**Effort:** ~1 week. Closes a compliance gap for SaaS deployments.

---

## 6. APG's Unique Composition Advantages

### 6.1 Semantic Capability Contracts

APG's `PROVIDES` / `REQUIRES` model (1,900 edges, 2,050 service declarations) is a machine-readable service graph. No other platform in this survey has a static, compiler-verified dependency graph across all platform capabilities. Salesforce flows are composed at runtime; Power Automate connectors have no global dependency graph. APG can statically detect a missing dependency before deployment — the composability audit that generated `COMPOSABILITY.md` proves this.

### 6.2 Single-File Deployment Model

A `.apg` file is a complete application specification — schema, UI, security, theme, and composition in one place. Salesforce requires Flows + Objects + Apex + Profiles as separate artifacts. ServiceNow requires tables + flows + business rules + UI actions. APG's single-source model makes composition explicit and version-controllable as one file per application.

### 6.3 Africa-First Mobile Money Connectors

`fintech_mobile`, `fintech_payments`, `fintech_wallets`, `fintech_agency`, `fintech_remittance` in the fintech domain are built with M-Pesa, Airtel Money, MTN MoMo, and similar mobile money primitives as first-class citizens. No other composition platform in this survey ships production-ready East/West African mobile money composition out of the box. This is a moat — replicating it requires domain knowledge that Salesforce, ServiceNow, and AWS do not have.

### 6.4 DSL-Driven Composition (No Visual Clutter)

Visual flow editors introduce accidental complexity: nodes drift, labels truncate, parallel branches become spaghetti. APG's text DSL is diff-able, grep-able, and composable with standard developer tooling (git, CI, editors). The VSCode extension provides IntelliSense without sacrificing the text model. This is the Unix pipe philosophy applied to enterprise composition: simplicity as a feature, not a limitation.

### 6.5 26-Tier Dependency-Ordered Deployment

The `COMPOSABILITY.md` deployment tier system gives APG something no other platform in this survey provides out of the box: a correct partial-order deployment plan for 266 capabilities derived from their dependency graph. Operators can deploy in tier order with confidence that no capability starts before its dependencies. Salesforce and ServiceNow manage this implicitly in their hosted platforms; APG makes it explicit for self-hosted deployments.

### 6.6 Domain Depth Across Africa-Relevant Verticals

With capabilities spanning government (`government_ele` for elections, `government_tax`), agriculture, mining, healthcare (including `healthcare_tel` for telemedicine), and intelligence (`intel_sigint`, `intel_humint`, `intel_osint`), APG covers verticals that no general-purpose composition platform addresses. The composition graph across these domains is unique intellectual property.

---

## 7. Sources and References

The following sources informed this analysis:

- Salesforce Platform Events documentation: https://developer.salesforce.com/docs/atlas.en-us.platform_events.meta/platform_events/
- Salesforce Flow Builder reference: https://help.salesforce.com/s/articleView?id=sf.flow_ref.htm
- ServiceNow Flow Designer documentation: https://docs.servicenow.com/bundle/washingtondc-build-workflows/page/administer/flow-designer/reference/flow-designer-overview.html
- Microsoft Power Platform Connector SDK: https://learn.microsoft.com/en-us/connectors/custom-connectors/define-openapi-definition
- Power Automate documentation: https://learn.microsoft.com/en-us/power-automate/
- OutSystems Forge: https://www.outsystems.com/forge/
- Mendix Marketplace: https://marketplace.mendix.com/
- AWS Step Functions developer guide: https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html
- Amazon EventBridge documentation: https://docs.aws.amazon.com/eventbridge/latest/userguide/
- Temporal.io documentation: https://docs.temporal.io/
- Temporal Nexus (cross-namespace composition): https://docs.temporal.io/nexus
- Apache Camel component reference: https://camel.apache.org/components/latest/
- Enterprise Integration Patterns (Hohpe & Woolf, 2003): https://www.enterpriseintegrationpatterns.com/
- MuleSoft Anypoint Exchange: https://www.mulesoft.com/exchange/
- Unix pipe design philosophy (McIlroy, 1978): Bell System Technical Journal
- APG codebase: `/Users/nyimbiodero/src/pjs/apg/` — capabilities/composition/, compiler/, COMPOSABILITY.md, examples/*.apg

**Open Questions:**
1. What is the current production status of `composition_events` (Bytewax) — is it deployed anywhere, or still in development?
2. Does `composition/studio` have a working UI prototype, or is it a placeholder?
3. Is there a target latency budget for inter-capability event delivery that would constrain the saga engine design?
4. For the marketplace, is the intent a hosted Datacraft registry or a self-hosted registry per customer deployment?
