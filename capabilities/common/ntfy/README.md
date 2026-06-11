# Notifications and Alerts

`ntfy` is APG's package-backed Notifications and Alerts capability. It gives
generated applications a tenant-scoped notification runtime for recipient
preferences, channel providers, template governance, delivery decisions,
campaigns, governed AI agents, Bytewax lifecycle batches, audit events, UI
route metadata, and visual theme metadata.

The package keeps the large existing Flask/API, FAB view, channel, analytics,
security, geofencing, and personalization modules available as integration
surfaces. Generated APG applications should compose through the lightweight
packet surfaces declared in the capability contract: `notification_runtime.py`,
`package_api.py`, and `view_models.py`.

## What It Provides

- Multi-channel notification routing across email, SMS, push, WebSocket,
  webhook, Slack, and Teams style channels.
- Tenant-local recipient preferences with opt-in, unsubscribe, addresses,
  channel preferences, and quiet-hour metadata.
- Channel provider health, owner, fallback route, and provider metadata.
- Template registration, versioning, approval, locale, content, and ownership.
- Message delivery guardrails for template approval, consent, encryption,
  provider health, enabled channels, webhook signatures, event bus evidence,
  audit evidence, idempotency, and quiet-hour review.
- Campaign creation, approval, batch review, and send lifecycle.
- First-class provider-neutral notification agents for `codex`, `claude_code`,
  `opencode`, and `pi`.
- Human-review guardrails for privileged notification roles such as campaign
  reviewers, delivery reviewers, suppression reviewers, provider-health
  reviewers, alert-routing reviewers, lifecycle reviewers, and notification
  stewards.
- Bytewax-only lifecycle batch validation for channel, preference, template,
  message, delivery, campaign, suppression, provider-health, and
  notification-agent mutations.
- Hashed deterministic IDs and tenant-local record isolation.
- UI view models for dashboard, messages, templates, campaigns, preferences,
  channels, analytics, agents, lifecycle batches, audit, and settings.
- Contract-derived semantic model, package manifest, release report, and
  publish-plan support.

## Main Files

| File | Purpose |
| --- | --- |
| `SPECIFICATION.md` | Functional, lifecycle, rule, UI, adapter, and acceptance specification. |
| `PLAN.md` | Implementation and review plan for this capability packet. |
| `capability_contract.py` | Executable configuration, rule engine, UI routes, theme, and adapter contract. |
| `notification_runtime.py` | Dependency-light runtime for generated applications. |
| `package_api.py` | Dependency-light helper API for generated applications. |
| `view_models.py` | Data-only view models for generated APG UIs. |
| `app.py` | Package entrypoint, semantic model, component manifest, and self-test. |
| `service.py`, `api.py`, `views.py` | Existing production integration surfaces. |

## Runtime Flow

1. Register a channel provider and fallback route.
2. Register recipient preferences with addresses, opt-in state, and channel
   choices.
3. Register and approve a template.
4. Send a message through an enabled healthy channel.
5. Record a deterministic delivery record and audit event.
6. Create and approve campaigns for larger audiences.
7. Route large batches or quiet-hour sends to review when required.
8. Register scoped AI agents for delivery governance, alert routing, and
   notification operations.
9. Validate bulk lifecycle mutations through Bytewax before composing larger
   applications.

## Python Usage

```python
from capabilities.common.ntfy.notification_runtime import NotificationRuntime

runtime = NotificationRuntime()

runtime.register_channel("tenant-a", "email", "ses-primary", "ops", fallback_channel="sms")
runtime.register_preference(
	"tenant-a",
	"user-1",
	addresses={"email": "user@example.com"},
	preferred_channels=["email"],
	opted_in=True,
)
template = runtime.register_template(
	"tenant-a",
	"welcome",
	"Welcome",
	"marketing-owner",
	"en",
	["email"],
	{"email": "Hello {{name}}"},
	approved=True,
)
delivery = runtime.send_message(
	"tenant-a",
	template["id"],
	"user-1",
	"email",
	message_class="marketing",
	idempotency_key="welcome:user-1",
)
```

## Campaigns

```python
campaign = runtime.create_campaign(
	"tenant-a",
	"spring-launch",
	"Spring Launch",
	"marketing-owner",
	"welcome",
	["user-1", "user-2"],
	["email"],
)
runtime.approve_campaign("tenant-a", campaign["id"], "compliance-reviewer")
result = runtime.send_campaign("tenant-a", campaign["id"])
```

## AI Agent Composition

Notification agents are first-class records. They are not bound to one model,
CLI, or provider. The contract accepts provider-neutral runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Every agent must declare a runtime, supported role, explicit scope, accountable
owner, purpose, and machine-contribution disclosure. Privileged roles enter
`pending_review` unless human approval evidence is recorded.

```python
agent = runtime.register_notification_agent(
	"agent-steward",
	"tenant-a",
	"Notification Steward",
	"codex",
	"notification_steward",
	"campaign:spring-launch",
	"marketing-owner",
	"review campaign delivery before send",
	human_approval_required=True,
)
```

## Bytewax Lifecycle Batches

Batch mutations must use Bytewax. The local package validates stream metadata
and records accepted or denied lifecycle-batch evidence without starting a live
Bytewax topology.

```python
batch = runtime.validate_ntfy_lifecycle_batch(
	"tenant-a",
	"bytewax",
	2,
	"notification_agent_batch",
	"batch-agent-001",
)
```

## Contract And Composition

```python
from capabilities.common.ntfy.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-a")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapters:

- `generated_app_runtime`: `notification_runtime.NotificationRuntime`
- `helper_runtime`: `notification_runtime.py`
- `api_helpers`: `package_api.py`
- `view_models`: `view_models.py`
- `event_stream`: `bytewax`
- `agent_adapter`: `aicr_provider_neutral_notification_agent_adapter`
- `message_bus`: `mqeb`
- `authentication`: `auth`
- `multi_tenancy`: `mten`
- `audit_sink`: `audl`
- `ai_orchestration`: `aicr`

## UI Surfaces

The contract exposes these route names:

- `dashboard`
- `messages`
- `templates`
- `campaigns`
- `preferences`
- `suppression`
- `channels`
- `analytics`
- `agents`
- `lifecycle`
- `audit`
- `settings`

## New Service Features (service.py)

The `NotificationService` class in `service.py` now provides the following additional
async methods beyond the core runtime and package-API surfaces:

### Idempotent Delivery

```python
# Re-callable from retry loops — returns the original record on duplicate key.
notif = await svc.send_idempotent(
    recipient="user@example.com",
    template_id=tid,
    variables={"name": "Alice"},
    idempotency_key="welcome:user-42",
)
print(notif["idempotent_hit"])  # False on first call, True on replay
```

### Dead-Letter Queue Re-Drive

```python
# Retry all failed/bounced notifications created in the last 6 hours.
summary = await svc.requeue_dead_letters(max_age_hours=6, limit=200)
print(summary["requeued"])

# Check per-channel DLQ depth before deciding to re-drive.
depths = await svc.dlq_depth()
# {"email": 12, "sms": 3}
```

### Consent & Regulatory Compliance

```python
# GDPR opt-in — immutable append-only record.
consent = await svc.record_consent(
    recipient_id="user-42",
    channel="email",
    legal_basis="opt_in",
    evidence_ref="form-event-uuid7",
)

# Hard opt-out — adds a suppression automatically.
revocation = await svc.revoke_consent("user-42", "email")

# Inspect current status + full history.
status = await svc.get_consent_status("user-42", "email")
print(status["has_consent"])  # False after revocation

# Gate a campaign send list in one call.
consent_map = await svc.bulk_consent_check(["user-1", "user-2", "user-3"], "email")
eligible = [uid for uid, ok in consent_map.items() if ok]
```

### Notification Digest / Fatigue Prevention

```python
# Accumulate multiple notifications into a single digest within a 30-minute window.
await svc.send_digested("user-42", tid, {"msg": "Event A"}, digest_window_minutes=30)
await svc.send_digested("user-42", tid, {"msg": "Event B"}, digest_window_minutes=30)

# Force-flush the window early (e.g. on logout or end of session).
result = await svc.flush_digest("user-42")
print(result["flushed"])  # 2
```

### Predictive Send-Time Optimisation

```python
# Recommend the hour with historically highest open rate for this recipient.
prediction = await svc.predict_optimal_send_time("user-42", "email")
print(prediction["predicted_hour"])  # e.g. 9
print(prediction["next_send_at"])    # ISO datetime of next occurrence
print(prediction["confidence"])      # fraction of opens at that hour
print(prediction["basis"])           # "historical" or "fallback"
```

### Delivery Latency Percentiles

```python
# p50/p95/p99 latency across a time window, optionally per channel.
percentiles = await svc.delivery_latency_percentiles(
    {"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
    channel="email",
)
print(percentiles["p99_ms"])  # e.g. 142.5
```

### Template Cloning

```python
# Create an independent copy of a template under a new name.
clone = await svc.clone_template(source_template_id=tid, new_name="Welcome (B variant)")
print(clone["cloned_from"])   # original template ID
print(clone["id"])            # new independent template ID
```

## Improvement Roadmap

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 detailed proposals covering idempotency,
dead-letter queues, adaptive rate limiting, webhook HMAC signing, priority queues,
A/B testing, quiet-hour enforcement, consent compliance, latency percentiles,
digest batching, immutable audit sinks, circuit breakers, tenant isolation,
streaming delivery events, and predictive send-time optimisation.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/ntfy/__init__.py capabilities/common/ntfy/capability_contract.py capabilities/common/ntfy/notification_runtime.py capabilities/common/ntfy/package_api.py capabilities/common/ntfy/view_models.py capabilities/common/ntfy/app.py capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/python capabilities/common/ntfy/app.py
./.venv/bin/apg capabilities inspect ntfy --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ntfy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ntfy --json
```
