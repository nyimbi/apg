# CKM Notification System Specification

## Purpose

The CKM Notification System (`ckm_not`) lets generated APG applications compose
tenant-scoped notification templates, campaigns, delivery governance,
recipient preferences, provider registration, engagement analytics, audit
evidence, and AI-agent review into larger collaboration and business workflows.

The package boundary is intentionally dependency-light. Live mail, SMS, push,
chat, voice, webhook, schedule, analytics, identity, secret, and stream workers
must be connected through adapters declared in the capability contract. The
package itself owns the executable lifecycle, deterministic rules, route
metadata, theming metadata, generated semantic evidence, and focused proof
commands.

## Users And Jobs

- Business operators create and approve templates, campaigns, and delivery
  windows.
- Application builders compose notification delivery and preference enforcement
  into ERP, CRM, CKM, workflow, and alerting applications.
- Compliance reviewers inspect consent, suppression, approval, and audit
  evidence.
- Platform engineers bind live providers, secret storage, scheduling,
  observability, and Bytewax stream workers.
- AI agents assist with template, audience, delivery, compliance, and
  escalation review under explicit registration and disclosure controls.

## Capability Boundary

`ckm_not` provides:

- notification delivery lifecycle governance;
- template management with locale, variable-schema, and channel-content gates;
- campaign orchestration with audience policy and bulk-send review;
- preference center enforcement for consent, channel opt-out, topic
  suppression, and quiet hours;
- channel provider registry metadata and secret-reference requirements;
- engagement analytics metadata and UI routes;
- AI notification-agent registration and policy enforcement.

`ckm_not` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `encr` for provider credential and message-protection adapter boundaries;
- `audl` for durable audit evidence.

## Lifecycle

Template lifecycle:

1. Draft template is created with tenant, locale, supported channels, channel
   content, and variable schema.
2. Activation requires variable-schema evidence and audit recording.
3. Approved templates can be used by delivery and campaign workflows.
4. Later template revisions must create a new lifecycle state transition with
   audit evidence.

Preference lifecycle:

1. Recipient preference is recorded with allowed channels, consent references,
   suppressed topics, and quiet-hour settings.
2. Delivery requests resolve preferences before provider dispatch.
3. Suppressed topics block delivery.
4. Quiet-hour delivery is deferred unless a permitted urgent override exists.

Delivery lifecycle:

1. Delivery request references an approved template, recipient, channels, and
   topic.
2. Rules validate tenant context, external-channel consent, suppression, quiet
   hours, and provider readiness.
3. Delivery becomes `queued`, `deferred`, or `blocked`.
4. Provider adapters record delivery results and engagement events.

Campaign lifecycle:

1. Campaign request references approved templates, audience policy, channels,
   delivery window, and owner.
2. Bulk campaigns require approval before execution.
3. Bytewax stream workers process batch lifecycle events and preserve delivery
   state.
4. Analytics views expose campaign, channel, template, and preference impact.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured notification review roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

The deterministic rules must deny or require review for:

- missing tenant context;
- incomplete channel content on template creation;
- missing variable schema on template activation;
- external delivery without consent evidence;
- delivery to a suppressed recipient/topic;
- quiet-hour sends without deferral or permitted urgent override;
- campaigns without audience policy;
- bulk campaigns without approval;
- provider registration without secret reference;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch notification mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes routes for dashboard, templates, campaigns,
deliveries, preferences, providers, agents, rules, analytics, audit, and
settings. The theme uses compact operational density with distinct visual
treatments for content approval, audience policy, delivery status, consent,
provider secret references, agent scope, stream health, and audit decisions.

## Streaming

Batch notification mutation is a Bytewax-governed lifecycle. The stream topic is
`apg.ckm_not.lifecycle`, and stream state covers templates, campaigns,
deliveries, preferences, providers, notification agents, and audit events. Live
Bytewax deployment is an adapter concern, but the package must declare and
enforce the stream guardrail.

## Adapter Boundaries

The package does not bind live provider SDKs. Adapters must handle:

- email, SMS, push, voice, webhook, chat, and web-push dispatch;
- provider secret storage and rotation;
- scheduler integration for send windows and quiet-hour deferral;
- durable audit sink;
- consent and privacy-policy synchronization;
- analytics ingestion and attribution;
- Bytewax stream topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes notification, preference, provider, agent,
  governance, observability, adapter, UI, and theme sections.
- Rules cover consent, suppression, templates, campaigns, providers, agents,
  audit, and Bytewax batch mutation.
- Lifecycle service can create and approve templates, register agents, enforce
  preferences, request deliveries, and summarize operational state.
- Semantic model exposes provides/requires, routes, rules, theme, and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
