# Notification System

## Overview

The Notification System (`ckm_not`) is a multi-channel notification engine that manages the full lifecycle of notifications across email, SMS, push, in-app, voice, webhook, WhatsApp, Slack, Teams, and web push channels. It provides template-driven content authoring, campaign orchestration, recipient preference enforcement, and delivery governance — all within a tenant-scoped, consent-enforced, audit-trailed runtime.

Business value is delivered through three interlocking surfaces: a Template Studio that separates content from code with approval gates and A/B test support; a Campaign Console that coordinates multi-step drip and blast campaigns with audience policies; and a Preference Center that gives recipients sovereignty over channel opt-outs and quiet-hours windows. AI notification agents can participate in every stage — reviewing templates, auditing compliance, and escalating issues — subject to mandatory registration, scoped roles, and contribution disclosure.

## Capability ID

`ckm_not`  Version: 2.0.0

## Provides

| Service | Description |
|---------|-------------|
| notification_delivery | Multi-channel dispatch with fallback, retry, SLA tracking, and idempotent delivery |
| template_management | Versioned, locale-aware templates with Mustache/Jinja2/Handlebars rendering and A/B testing |
| campaign_orchestration | Multi-step drip/blast/triggered/A-B campaigns with audience policies and budget caps |
| preference_center | User-managed channel opt-outs, quiet hours, digest settings, and consent evidence |
| channel_provider_registry | Pluggable provider registry (SendGrid, Twilio, Firebase, etc.) with circuit breakers and health monitoring |
| engagement_analytics | Delivery rates, open/click/conversion tracking per notification and campaign |
| notification_agents | AI agent assist for template review, compliance, audience, delivery, and escalation |
| in_app_feed | Persistent notification inbox with unread counts and read-state management |
| digest_batching | Configurable rollup windows to reduce notification noise |
| consent_lifecycle | GDPR/TCPA evidence trail with per-channel consent records |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Identity context and RBAC permission checks |
| conf | Tenant-scoped configuration for channel and delivery settings |
| encr | Encryption of provider API keys and secrets at rest |
| audl | Audit log sink for all state-change and delivery events |

## Features

- **10 delivery channels**: email, SMS, push, in-app, voice, webhook, WhatsApp, Slack, Teams, web push
- **Idempotent delivery**: caller-supplied deduplication keys prevent duplicate sends on retry
- **Adaptive send-time optimisation**: per-recipient engagement histograms predict optimal delivery windows
- **Quiet-hours deferral**: timezone-aware enforcement of user-configured do-not-disturb windows
- **A/B template testing**: chi-squared significance gating auto-promotes winning variants at configurable confidence
- **Circuit breaker failover**: automatic provider rotation after consecutive failures, with CLOSED/OPEN/HALF_OPEN state
- **Signed webhook delivery**: HMAC-SHA256 payload signing and retry queue with exponential back-off
- **In-app notification feed**: persistent inbox with mark-read, mark-all-read, archiving, and unread counts
- **Digest batching**: buffer high-frequency notifications over configurable windows into single summaries
- **Consent lifecycle**: immutable per-channel consent records with revocation and audit trail
- **Cost-aware channel selection**: per-channel budget caps with automatic fallback to cheaper channels
- **Priority lanes**: four-tier fair-queue scheduler (CRITICAL/HIGH/NORMAL/LOW) prevents blast traffic from delaying OTPs
- **Template localisation**: locale fallback chain (`sw_KE → sw → en_KE → en`) per template
- **Real-time status webhooks**: push delivery events (delivered, opened, bounced) to subscriber URLs
- **Recipient rate limiting**: per-recipient hourly/daily frequency caps with `RateLimitExceeded` enforcement
- **Full-text notification archive**: filter by recipient, channel, status, template, date range, and search text

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scoping for all operations |
| channels.supported | list | all 10 channels | Active delivery channels |
| channels.provider_registry_required | bool | true | Providers must be registered before use |
| channels.delivery_fallback_required | bool | true | Fallback channel required per delivery |
| templates.approval_required | bool | true | Templates require approval before activation |
| templates.locale_required | bool | true | Locale must be declared on every template |
| templates.variable_schema_required | bool | true | Variable schema must be attached at activation |
| campaigns.audience_policy_required | bool | true | Audience policy required before send |
| campaigns.approval_required_for_bulk | bool | true | Bulk campaigns (>500 recipients) need approval |
| delivery.recipient_consent_required | bool | true | Consent evidence required for external channels |
| delivery.quiet_hours_deferral_required | bool | true | Quiet-hours sends must be deferred |
| preferences.consent_evidence_required | bool | true | Preference changes require consent trace |
| notification_agents.agent_registration_required | bool | true | Agents must be registered with runtime and scope |
| governance.batch_event_stream | string | "bytewax" | Batch mutations must route through Bytewax |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /ckm-not/dashboard | GET | ckm_not:view | Overview |
| templates | /ckm-not/templates | GET | ckm_not:manage_templates | Design |
| campaigns | /ckm-not/campaigns | GET | ckm_not:manage_campaigns | Campaigns |
| deliveries | /ckm-not/deliveries | GET | ckm_not:send | Operations |
| preferences | /ckm-not/preferences | GET | ckm_not:view_preferences | Governance |
| providers | /ckm-not/providers | GET | ckm_not:admin | Administration |
| agents | /ckm-not/agents | GET | ckm_not:govern | Governance |
| rules | /ckm-not/rules | GET | ckm_not:govern | Governance |
| analytics | /ckm-not/analytics | GET | ckm_not:view | Insights |
| audit | /ckm-not/audit | GET | ckm_not:view | Governance |
| settings | /ckm-not/settings | GET | ckm_not:admin | Administration |

API prefix: `/ckm-not/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| template_requires_channel_content | create_template with incomplete channel content | deny |
| template_requires_variable_schema | activate_template without variable schema | deny |
| external_delivery_requires_consent | External channel delivery without consent evidence | deny |
| delivery_channel_must_be_allowed | Channel blocked by recipient preferences | deny |
| suppressed_recipient_blocks_delivery | Recipient suppressed | deny |
| quiet_hours_requires_deferral | Delivery in quiet hours without deferral or urgent override | require_review |
| campaign_requires_audience_policy | Campaign operation without audience policy | deny |
| bulk_campaign_requires_approval | Campaign with >500 recipients without approval | require_review |
| provider_credentials_require_secret_reference | Register provider without managed secret reference | deny |
| notification_agent_requires_registration | Agent present but not registered | deny |
| notification_agent_runtime_supported | Agent uses unsupported runtime | deny |
| notification_agent_role_supported | Agent uses unsupported role | deny |
| notification_agent_requires_scope | Agent without explicit scope | deny |
| notification_agent_requires_disclosure | Agent contribution not disclosed | deny |
| notification_state_change_requires_audit | State change without audit event | deny |
| batch_notification_mutation_requires_bytewax | Batch mutation not using Bytewax | deny |

Supported agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

Supported agent roles: `template_reviewer`, `audience_reviewer`, `delivery_reviewer`, `compliance_reviewer`, `escalation_reviewer`

## Data Models

| Model | Key Fields |
|-------|-----------|
| NENotification | notification_id, tenant_id, title, message, recipient_id, channels, priority, status, delivery_attempts, delivered_at, read_at, clicked_at, campaign_id |
| NETemplate | template_id, tenant_id, code, version, locale, subject_template, html_template, sms_template, push_template, template_engine, variables_schema, supported_channels, is_active, ab_test_variant |
| NEDelivery | delivery_id, notification_id, channel, provider, recipient_address, status, attempt_number, sent_at, delivered_at, provider_id, provider_response, error_code, delivery_time_ms |
| NEInteraction | interaction_id, notification_id, interaction_type, channel, timestamp, user_agent, ip_address, click_url, device_info |
| NECampaign | campaign_id, tenant_id, name, campaign_type, trigger_event, target_audience, status, total_recipients, delivery_rate, open_rate, click_rate, conversion_rate |
| NECampaignStep | step_id, campaign_id, step_number, delay_minutes, template_id, channels, send_conditions, skip_conditions, engagement_score |
| NEUserPreference | preference_id, user_id, tenant_id, email_enabled, sms_enabled, push_enabled, in_app_enabled, quiet_hours_start, quiet_hours_end, is_subscribed, engagement_score |
| NEProvider | provider_id, tenant_id, name, provider_type, provider_key, is_enabled, is_primary, rate_limit_per_minute, health_status, consecutive_failures, success_rate |

## Streaming Events

Events emitted to the ckm event stream via Bytewax.

Topic: `apg.ckm_not.lifecycle`

| Event | Trigger |
|-------|---------|
| notification_template_created | New template record persisted |
| notification_template_approved | Template approval recorded |
| notification_campaign_requested | Campaign send initiated |
| notification_campaign_approved | Bulk campaign approval recorded |
| notification_delivery_requested | Delivery dispatched to channel provider |
| notification_delivery_deferred | Delivery deferred due to quiet hours |
| notification_delivery_recorded | Delivery outcome (success/fail) confirmed |
| notification_preference_updated | User preference or consent changed |
| notification_provider_registered | New channel provider added to registry |
| notification_agent_registered | AI notification agent registered |

Batch mutation guardrail: `batch_notification_mutation_requires_bytewax`

## World-Class Enhancements (v2.0)

Fifteen production-grade improvements aligned with best practices from Braze, Twilio, SendGrid, Stripe, and Intercom.

| # | Enhancement | Category | Summary |
|---|-------------|----------|---------|
| 1 | **Idempotent Delivery** | Reliability | `idempotency_key` on `send_notification` deduplicates within 24h; returns existing record on hit without re-sending |
| 2 | **Adaptive Send-Time Optimisation** | AI / Engagement | Per-recipient open-time histogram drives `optimal_send_window()`; `timezone_aware_send` can honour it |
| 3 | **Quiet-Hours Deferral** | Compliance / UX | Converts UTC send time to recipient `zoneinfo` timezone; defers to `quiet_hours_end` and emits `notification_delivery_deferred` event |
| 4 | **A/B Template Testing** | Optimisation | `create_ab_test()` / `evaluate_ab_test()` with chi-squared p-value gating; auto-promotes winner at configurable significance threshold (default 95%) and n >= 100 |
| 5 | **Circuit Breaker Failover** | Resilience | Per-provider CLOSED/OPEN/HALF_OPEN state machine; rotates to next provider after N failures; exposes `provider_circuit_status()` |
| 6 | **Signed Webhook Delivery** | Security | HMAC-SHA256 `X-APG-Signature-256` header; `_webhook_retry_queue` with exponential back-off up to 3 attempts; `list_webhook_failures()` |
| 7 | **In-App Notification Feed** | UX / Features | `get_inapp_feed()`, `mark_read()`, `mark_all_read()`, `get_unread_count()` — persistent inbox with per-recipient read state |
| 8 | **Digest Batching** | Volume Control | `add_to_digest()` / `flush_digest()` buffer notifications by `digest_key`; background task fires on configurable window expiry |
| 9 | **Consent Lifecycle** | Legal / GDPR | `record_consent()`, `revoke_consent()`, `verify_consent()` — immutable per-channel records with method, IP, timestamp; blocks delivery on missing consent |
| 10 | **Cost-Aware Channel Selection** | Cost Management | `set_channel_budget()` caps per-channel spend; `_channel_spend` checked before dispatch; `get_spend_summary()` reports actuals |
| 11 | **Priority Lanes** | Performance / SLA | Four `asyncio.PriorityQueue` instances (CRITICAL/HIGH/NORMAL/LOW) with weighted round-robin drain; `queue_depth()` surfaces per-lane backlog |
| 12 | **Template Localisation** | i18n / UX | `add_template_locale()` stores variants per `(template_id, locale)`; `render_template()` resolves `sw_KE → sw → en_KE → en` fallback chain |
| 13 | **Real-Time Status Webhooks** | Integration / DX | `subscribe_status_updates()` registers callback URLs per entity; status changes fire signed async POSTs without blocking the primary path |
| 14 | **Recipient Rate Limiting** | Anti-Spam | `set_recipient_rate_limit()` caps per-hour and per-day sends; `get_recipient_rate_status()` surfaces current counters; raises `RateLimitExceeded` |
| 15 | **Notification Archive / Search** | Observability | `notification_history()` accepts `channel`, `status`, `template_id`, `date_from`, `date_to`, `search_text` filters; `export_notifications()` emits JSON or CSV |

## New Methods

The following methods were added in v2.0. Import the service through `importlib` because `not` is a Python keyword.

```python
from importlib import import_module

not_pkg = import_module("capabilities.ckm.not")
svc = not_pkg.NotificationService(not_pkg.NotificationServiceConfig(tenant_id="acme"))
```

### Channel Registration and Health

```python
# Register an email channel
channel = await svc.register_channel(
    channel_type="email",
    config={
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "username": "notify@example.com",
        "password": "secret",
    },
)

# Verify connectivity without sending a live message
health = await svc.channel_health_check(channel["id"])
assert health["status"] == "healthy"

# Deactivate a channel with an audit trail
await svc.deactivate_channel(channel["id"], reason="provider contract expired")
```

### Idempotent Send + Track Delivery

```python
# First call dispatches; subsequent calls with the same key return the same record
notif = await svc.send_notification(
    recipient="user-42",
    template_id=template_id,
    variables={"invoice_id": "INV-001"},
    idempotency_key="billing-2026-06-01-user-42",
)

# Poll or webhook-push the delivery timeline
status = await svc.track_delivery(notif["id"])
# {"status": "DELIVERED", "timeline": {"sent_at": ..., "delivered_at": ...}}
```

### Delivery Report and Cost Summary

```python
period = {"start": "2026-06-01T00:00:00", "end": "2026-06-30T23:59:59"}

report = await svc.delivery_report(period, channel="sms")
# {"delivery_rate": 98.4, "bounce_rate": 0.6, "open_rate": 42.1, ...}

costs = await svc.cost_report(period)
# {"total_cost_usd": 1.42, "by_channel": {"sms": {"count": 189, "total_cost": 1.42}, ...}}
```

### Suppression Management

```python
# Global suppression — no channel will reach this recipient
await svc.add_suppression("user-99", reason="unsubscribed", channel=None)

# Per-channel suppression
await svc.add_suppression("user-55", reason="sms_bounce", channel="sms")

# Verify before sending
allowed = await svc.check_preference("user-55", channel="email", notification_type="transactional")
# True — only SMS is suppressed

# Remove all suppressions
await svc.remove_suppression("user-99")
```

### Recurring Schedule + Dashboard

```python
# Weekly digest every Monday at 08:00
schedule = await svc.recurring_notification(
    recipient="user-10",
    template_id=weekly_template_id,
    cron_expr="0 8 * * MON",
)

# Cancel when no longer needed
await svc.cancel_recurring(schedule["id"])

# Operational overview
summary = await svc.dashboard_summary()
# Keys: last_30_days, channel_performance_30d, active_channels,
#       active_templates, active_schedules, total_suppressed_recipients
```

## Edge Cases Handled

- Quiet hours spanning midnight are correctly detected by comparing start and end times and handling the wraparound case in `NEUserPreference.is_quiet_hours()` — times after midnight on the start side still fall within the window.
- Suppressed recipients are blocked at the rule layer before any provider call is made, preventing unintended delivery even if the caller bypasses preference checks upstream.
- A/B test template variants are linked via `parent_template_id` and `ab_test_variant` on `NETemplate`; metrics are tracked per variant independently so winning variants can be promoted without data loss.
- When a provider's `consecutive_failures >= 5` or `health_status != healthy`, `NEProvider.is_healthy()` returns false and the channel manager routes to the next eligible provider by priority order, enabling automatic failover without operator intervention.
- Bulk campaign approval threshold is 500 recipients. Campaigns at exactly 500 are not blocked; the rule fires at `recipient_count_gt: 500`, matching the intent that campaigns marginally under the threshold proceed without the overhead of formal approval.
- Provider credentials are never stored in plaintext. The rule `provider_credentials_require_secret_reference` enforces managed-secret references, with encryption handled by the `encr` adapter.
- Template rendering failures due to missing variables surface as `ValueError` from `validate_variables()` before dispatch, preventing malformed content from reaching recipients.
- `send_bulk` raises `ValueError` immediately if `recipients` and `variables_list` lengths differ, preventing silent per-index mismatches in large batches.

## Composability

- **Upstream**: `auth` provides identity context; `conf` supplies tenant-scoped channel configuration; `encr` encrypts provider secrets; `audl` receives all audit events.
- **Downstream**: `ckm_rtc` depends on `ckm_not` for session event notifications, participant join/leave alerts, and decision capture notifications. `ckm_wfa` depends on `ckm_not` for task assignment notifications, approval request routing, SLA breach alerts, and exception escalations.
- **Peer**: Commonly deployed alongside `ckm_rtc` and `ckm_wfa` as the notification backbone of the full CKM stack. The analytics surface aggregates delivery data from all upstream triggers.

## Development Notes

- All provider credentials flow through the `encr` adapter — never write raw keys to `NEProvider.api_key` / `api_secret` columns.
- `NETemplate.render()` supports three engines (Mustache via `pystache`, Jinja2 with autoescaping, plain `str.format()`). The engine is declared per template; mixing engines across variants of the same template is discouraged.
- `NEUserPreference.is_quiet_hours()` imports `pytz` at call time. Ensure `pytz` is in the dependency set or migrate to `zoneinfo` (Python 3.9+ stdlib).
- Batch notification mutations must use the Bytewax event stream rather than direct DB writes; the rule engine enforces this at the contract level.
- `NEProvider.can_send()` checks all rate-limit tiers (per-minute, per-hour, per-day, daily quota). Callers must pass current usage counters — the model does not query the DB internally.
- The package directory is named `not`, a Python keyword. Always import via `importlib.import_module("capabilities.ckm.not")`.
- `NotificationService` is instantiated directly (v2.0 API); `NotificationLifecycleService` is the contract-layer alias used by `NotificationLifecycleService` in `lifecycle.py`.

## Quick Use

Load the package through `importlib` because the directory name is `not`, which is a Python keyword:

```python
from importlib import import_module

not_pkg = import_module("capabilities.ckm.not")
service = not_pkg.NotificationLifecycleService("tenant-acme")

service.register_provider(
    provider_id="email-primary",
    name="Primary email",
    channel="email",
    secret_ref="secret/not/email-primary",
)

service.create_template(
    template_id="invoice-ready",
    name="Invoice ready",
    channels=["email", "in_app"],
    content={
        "email": "Invoice {{invoice_id}} is ready.",
        "in_app": "Invoice {{invoice_id}} is ready.",
    },
    variable_schema={"invoice_id": {"type": "string"}},
)
service.approve_template("invoice-ready", reviewer_id="user-finance-lead")

service.set_preference(
    recipient_id="customer-123",
    allowed_channels=["email", "in_app"],
    consent_refs={"email": "consent-2026-05-30"},
)

delivery = service.request_delivery(
    template_id="invoice-ready",
    recipient_id="customer-123",
    channels=["email"],
    topic="billing",
)
assert delivery["status"] == "queued"
```

## AI Agent Registration

AI agents are first-class contributors only after registration:

```python
agent = service.register_notification_agent(
    name="Template reviewer",
    runtime="codex",
    role="template_reviewer",
    scope="review invoice and account templates for policy gaps",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are `template_reviewer`, `audience_reviewer`, `delivery_reviewer`, `compliance_reviewer`, and `escalation_reviewer`.

## Bytewax Batch Mutation

Batch notification mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_notification_mutation("bytewax")
blocked = service.validate_batch_notification_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.ckm_not.lifecycle` and state for templates, campaigns, deliveries, preferences, providers, notification agents, and audit events.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/__init__.py capabilities/ckm/not/__init__.py capabilities/ckm/not/capability_contract.py capabilities/ckm/not/lifecycle.py capabilities/ckm/not/app.py capabilities/ckm/not/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/not/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.not'); service = pkg.NotificationLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/not --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/not --json
```

---

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke
