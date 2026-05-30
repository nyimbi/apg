# NTFY Capability Specification

## Purpose

`ntfy` provides a composable Notifications and Alerts capability for APG
applications. It turns recipient preferences, channel providers, templates,
messages, campaigns, delivery guardrails, and audit events into executable
notification operations with deterministic rules and UI-ready view models.

The capability does not require live email, SMS, push, WebSocket, webhook,
Slack, Teams, analytics, personalization, or provider credentials for local
proof. Those systems remain adapter responsibilities.

## Scope

In scope:

- tenant-scoped channel provider records;
- recipient preferences, addresses, opt-in, unsubscribe, quiet hours, and
  channel choices;
- template registration, approval, versioning, locale, owner, and content;
- single-message delivery decisions;
- campaign creation, approval, batch review, and send lifecycle;
- idempotent send protection;
- provider health and fallback routing guardrails;
- sensitive-payload encryption guardrails;
- webhook signature guardrails;
- audit events for state changes and delivery decisions;
- route, permission, view-model, theme, and adapter metadata;
- package self-test, semantic model, manifest, release report, audit, and
  publish-plan evidence.

Out of scope for the local package:

- live provider delivery;
- live template rendering engines;
- live personalization models;
- live WebSocket servers;
- live analytics ingestion;
- persistent database migrations;
- live Bytewax execution.

## Users

- Application builders composing APG notification flows.
- Operations teams managing channel health and fallback routes.
- Marketing and product teams managing approved campaigns.
- Security and compliance teams enforcing consent, encryption, and audit.

## Domain Model

The lightweight runtime owns these records:

- `RecipientPreferenceRecord`
- `ChannelProviderRecord`
- `NotificationTemplateRecord`
- `DeliveryRecord`
- `CampaignRecord`
- `NotificationAuditEventRecord`

All records are stored under tenant-qualified internal keys.

## Lifecycle

### Channel

1. Register a channel provider with tenant, channel, provider, owner, health,
   and fallback channel.
2. Block missing provider or owner.
3. Use channel health in delivery decisions.

### Preference

1. Register recipient addresses and preferred channels.
2. Track opt-in, unsubscribe, and quiet-hour metadata.
3. Deny marketing messages without opt-in or to unsubscribed recipients.

### Template

1. Register template with tenant, owner, name, locale, channel content, and
   version.
2. Approve templates before sends or campaigns.
3. Block sends using unapproved templates.

### Message

1. Request send with tenant, template, recipient, channel, message class,
   priority, sensitivity, encryption, and idempotency key.
2. Deny hard guardrail failures.
3. Route quiet-hour or fallback gaps to review.
4. Record delivery and audit evidence.

### Campaign

1. Create campaign with owner, template, audience, channels, and message class.
2. Approve campaign before send.
3. Route large batches to review when review evidence is missing.
4. Record campaign send and audit evidence.

## Deterministic Rules

The contract currently exposes at least 30 rules covering:

- tenant context;
- recipient addresses and channel preferences;
- marketing consent and unsubscribe;
- quiet-hour review;
- template owner, name, locale, content, and approval;
- send template presence;
- campaign template approval and campaign approval;
- sensitive payload encryption;
- provider health;
- enabled channels;
- fallback routing;
- large batch review;
- campaign audience and owner;
- idempotency;
- webhook signatures;
- provider ownership;
- event bus evidence;
- delivery audit evidence;
- delivery TTL;
- tenant isolation;
- state-change audit evidence;
- Bytewax for batch notification mutation.

Rule decisions are one of:

- `allow`
- `require_review`
- `deny`

`deny` takes precedence over `require_review`.

## Configuration

Required configuration sections:

- `tenant_id`
- `channels`
- `delivery`
- `preferences`
- `templates`
- `campaigns`
- `security`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Key defaults:

- fallback routing enabled;
- provider health required;
- max batch size `5000`;
- quiet hours enforced;
- recipient opt-in required;
- template approval required;
- campaign approval required;
- sensitive payload encryption required;
- webhook signatures required;
- Bytewax event stream for batch mutations;
- delivery audit required.

## UI

Routes:

- `/ntfy/dashboard`
- `/ntfy/messages`
- `/ntfy/templates`
- `/ntfy/campaigns`
- `/ntfy/preferences`
- `/ntfy/suppression`
- `/ntfy/channels`
- `/ntfy/analytics`
- `/ntfy/audit`
- `/ntfy/settings`

View models must remain dependency-light data payloads. Browser rendering
belongs to generated applications.

## Theme

Theme name: `ntfy_notification_ops`.

Theme components:

- `channel_matrix`
- `delivery_timeline`
- `campaign_table`
- `preference_panel`
- `template_studio`
- `suppression_list`
- `audit_timeline`

## Adapter Boundaries

Adapter keys are declared in the capability contract:

- `message_bus`: `mqeb`
- `authentication`: `auth`
- `multi_tenancy`: `mten`
- `audit_sink`: `audl`
- `ai_orchestration`: `aicr`
- `collaboration`: `colb`
- `machine_channel`: `mchn`
- `security`: `secu`
- `event_stream`: `bytewax`

Adapters must not be required for local package self-tests.

## Acceptance Criteria

- Contract exposes configuration, schema, deterministic rules, UI routes,
  theme, and adapters.
- Rule count is at least 30.
- UI route count is at least 10.
- Bytewax is the event-stream adapter.
- Runtime executes channel, preference, template, message, campaign, and audit
  lifecycles.
- Marketing consent, unsubscribe, template approval, provider health, channel
  enablement, sensitive payload encryption, idempotency, webhook signature,
  campaign approval, and large-batch review guardrails are enforced.
- API helpers expose the lifecycle fields used by the runtime.
- View models expose all route families.
- `app.self_test()` passes.
- Focused package tests pass.
- Implementation audit and publish-plan pass.
