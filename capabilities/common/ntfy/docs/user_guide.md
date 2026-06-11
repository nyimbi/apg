# APG Notifications (ntfy) — User Guide

Copyright © 2025 Datacraft — Author: Nyimbi Odero

---

## Overview

`ntfy` is APG's Notifications and Alerts capability.  It provides a tenant-scoped,
multi-channel notification runtime for enterprise applications built on the APG
platform.  The service layer in `service.py` exposes clean async Python methods
for every notification lifecycle concern: template management, channel registration,
delivery, tracking, scheduling, analytics, consent compliance, and fatigue prevention.

This guide covers the `NotificationService` API in depth.  For the lightweight
generated-app runtime (`NotificationRuntime`) see `README.md`.

---

## Installation and Initialisation

```python
from capabilities.common.ntfy.service import create_notification_service

svc = create_notification_service(
    tenant_id="acme-corp",
    max_concurrent_deliveries=200,
    batch_size=500,
    enable_personalization=True,
    enable_analytics=True,
)
```

Use the async context manager for proper lifecycle management:

```python
from capabilities.common.ntfy.service import notification_service_context

async with notification_service_context("acme-corp") as svc:
    await svc.send_notification(...)
```

---

## Channel Management

Channels must be registered before sending.  Each channel type has required config keys.

| Channel type | Required config keys |
|---|---|
| `email` | `smtp_host`, `smtp_port`, `username`, `password` |
| `sms` | `provider`, `api_key`, `from_number` |
| `push` | `provider`, `app_id`, `api_key` |
| `webhook` | `url` |
| `slack` | `webhook_url` |
| `teams` | `webhook_url` |

```python
# Register an email channel
ch = await svc.register_channel(
    channel_type="email",
    config={
        "smtp_host": "smtp.sendgrid.net",
        "smtp_port": 587,
        "username": "apikey",
        "password": "SG.xxx",
    },
)
channel_id = ch["id"]

# Verify connectivity without sending a real message
health = await svc.channel_health_check(channel_id)
print(health["status"])  # "healthy" or "degraded"

# Send a test message through the channel
test = await svc.test_channel(channel_id)
print(test["success"])

# Update credentials without re-registering
await svc.update_channel_config(channel_id, {"password": "SG.new-key"})

# Disable the channel with an audit trail
await svc.deactivate_channel(channel_id, reason="provider_contract_expired")

# List active channels for the tenant
channels = await svc.list_channels(active_only=True)
```

---

## Template Management

Templates use `{{ variable }}` Jinja2-style placeholders.

```python
tmpl = await svc.create_template(
    name="Welcome Email",
    channel="email",
    subject="Welcome to {{ company }}, {{ name }}!",
    body="Hi {{ name }},\n\nThanks for joining {{ company }}.\n\n{{ cta_url }}",
    variables=["name", "company", "cta_url"],
)
template_id = tmpl["id"]

# Preview rendered output before any live send
preview = await svc.test_template(
    template_id,
    sample_vars={"name": "Alice", "company": "Acme", "cta_url": "https://acme.com/start"},
)
print(preview["rendered_subject"])
print(preview["rendered_body"])

# Create a new version (preserves history)
version_info = await svc.version_template(template_id)
print(version_info["new_version"])  # 2

# Clone a template for A/B variant work
variant = await svc.clone_template(template_id, new_name="Welcome Email (B variant)")
print(variant["cloned_from"])

# List templates, optionally filtered by channel
templates = await svc.list_templates(channel="email", active_only=True)

# Soft-delete (preserves version history)
await svc.delete_template(template_id)
```

---

## Sending Notifications

### Single Send

```python
notif = await svc.send_notification(
    recipient="alice@example.com",
    template_id=template_id,
    variables={"name": "Alice", "company": "Acme", "cta_url": "https://acme.com/start"},
    priority="high",
)
print(notif["id"], notif["status"])  # UUID7, "delivered"
```

### Idempotent Send

Safe to call multiple times from retry loops.  Returns the original record on duplicate key.

```python
notif = await svc.send_idempotent(
    recipient="alice@example.com",
    template_id=template_id,
    variables={"name": "Alice", "company": "Acme", "cta_url": "https://acme.com"},
    idempotency_key="welcome:alice@example.com",
)
print(notif["idempotent_hit"])  # False on first call, True on replay
```

### Scheduled Send

```python
from datetime import datetime, timedelta

send_at = datetime.utcnow() + timedelta(hours=24)
notif = await svc.send_notification(
    recipient="alice@example.com",
    template_id=template_id,
    variables={"name": "Alice"},
    scheduled_at=send_at,
)
print(notif["status"])  # "scheduled"

# Cancel before dispatch
await svc.cancel_scheduled(notif["id"])
```

### Bulk Send

```python
recipients = ["alice@example.com", "bob@example.com", "carol@example.com"]
variables_list = [
    {"name": "Alice", "company": "Acme", "cta_url": "https://acme.com"},
    {"name": "Bob",   "company": "Acme", "cta_url": "https://acme.com"},
    {"name": "Carol", "company": "Acme", "cta_url": "https://acme.com"},
]
results = await svc.send_bulk(recipients, template_id, variables_list)
success_count = sum(1 for r in results if r.get("status") == "delivered")
print(f"{success_count}/{len(results)} delivered")
```

### Timezone-Aware Send

Records the recipient's local timezone for downstream scheduling analysis.

```python
notif = await svc.timezone_aware_send(
    recipient="alice@example.com",
    template_id=template_id,
    recipient_timezone="Africa/Nairobi",
    variables={"name": "Alice"},
)
print(notif["recipient_timezone"])  # "Africa/Nairobi"
```

### Digest Send (Fatigue Prevention)

Accumulate notifications within a window and deliver as a single message.

```python
# These three calls within the window produce one outgoing notification.
await svc.send_digested("alice@example.com", template_id, {"msg": "Event A"}, digest_window_minutes=30)
await svc.send_digested("alice@example.com", template_id, {"msg": "Event B"}, digest_window_minutes=30)
await svc.send_digested("alice@example.com", template_id, {"msg": "Event C"}, digest_window_minutes=30)

# Force-flush the window (e.g. on session end)
result = await svc.flush_digest("alice@example.com")
print(result["flushed"])  # 3
```

---

## Delivery Tracking

```python
# Full status timeline: SENT → DELIVERED → OPENED → CLICKED / BOUNCED
tracking = await svc.track_delivery(notif["id"])
print(tracking["status"])    # "DELIVERED"
print(tracking["timeline"])  # {"sent_at": "...", "delivered_at": "...", ...}

# Retry a failed or bounced notification (exponential back-off metadata recorded)
retry = await svc.retry_failed(notif["id"])
print(retry["retry_count"], retry["backoff_seconds"])

# Per-recipient history (newest first)
history = await svc.notification_history("alice@example.com", limit=20)
```

---

## Dead-Letter Queue

```python
# View per-channel DLQ depth before deciding to re-drive
depths = await svc.dlq_depth()
# {"email": 5, "sms": 1}

# Re-drive up to 100 failed/bounced notifications from the last 24 hours
summary = await svc.requeue_dead_letters(max_age_hours=24, limit=100)
print(f"{summary['requeued']} requeued, {summary['skipped']} skipped, {summary['failed']} errored")
```

---

## Scheduling and Recurring Notifications

```python
# One-time future delivery
schedule = await svc.schedule_notification(
    recipient="alice@example.com",
    template_id=template_id,
    send_at=datetime(2025, 12, 25, 9, 0, 0),
    timezone="Africa/Nairobi",
)

# Recurring delivery driven by cron expression
recurring = await svc.recurring_notification(
    recipient="alice@example.com",
    template_id=template_id,
    cron_expr="0 9 * * MON",  # Every Monday at 9 AM
    end_date=datetime(2025, 12, 31),
)

# Cancel recurring schedule
await svc.cancel_recurring(recurring["id"])

# List all pending/active schedules for the tenant
schedules = await svc.list_scheduled(recipient_id="alice@example.com")
```

---

## Preferences and Suppression

```python
# Set per-channel opt-in/out preferences
prefs = await svc.set_preferences(
    "alice@example.com",
    preferences={"email": True, "sms": False, "push": True},
)

# Check if a recipient can receive a notification type on a channel
allowed = await svc.check_preference("alice@example.com", channel="email", notification_type="marketing")

# Global suppression (no sends on any channel)
await svc.add_suppression("alice@example.com", reason="user_requested_pause")

# Per-channel suppression
await svc.add_suppression("alice@example.com", reason="sms_opt_out", channel="sms")

# Lift all suppressions
await svc.remove_suppression("alice@example.com")

# List all suppressed recipients for the tenant
suppressed = await svc.suppression_list(channel="email")
```

---

## Consent and Compliance (GDPR / CCPA)

```python
# Record explicit opt-in with legal basis and evidence reference
consent = await svc.record_consent(
    recipient_id="alice@example.com",
    channel="email",
    legal_basis="opt_in",
    evidence_ref="signup-form-event-uuid7-abc123",
)
print(consent["id"])  # Immutable UUID7 consent record

# Hard revocation — automatically adds a channel suppression
revocation = await svc.revoke_consent("alice@example.com", "email")

# Inspect current consent status and full audit history
status = await svc.get_consent_status("alice@example.com", "email")
print(status["has_consent"])  # False after revocation
print(status["history"])      # Full append-only event chain

# Gate a campaign send list before dispatching
consent_map = await svc.bulk_consent_check(
    ["alice@example.com", "bob@example.com", "carol@example.com"],
    channel="email",
)
eligible = [uid for uid, ok in consent_map.items() if ok]
print(f"{len(eligible)}/{len(consent_map)} recipients have valid email consent")
```

---

## Analytics and Reporting

### Delivery Report

```python
report = await svc.delivery_report(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
    channel="email",
)
print(report["delivery_rate"])   # e.g. 98.5
print(report["open_rate"])       # e.g. 23.4
print(report["bounce_rate"])     # e.g. 0.8
```

### Engagement Report (per Template)

```python
engagement = await svc.engagement_report(
    template_id=template_id,
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
)
print(engagement["open_rate"])        # open / delivered * 100
print(engagement["click_to_open_rate"])
```

### Channel Performance Comparison

```python
perf = await svc.channel_performance(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
)
for channel, stats in perf["channels"].items():
    print(f"{channel}: delivery={stats['delivery_rate']}%, open={stats['open_rate']}%")
```

### Delivery Latency Percentiles

```python
latency = await svc.delivery_latency_percentiles(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
    channel="email",
)
print(f"p50={latency['p50_ms']}ms  p95={latency['p95_ms']}ms  p99={latency['p99_ms']}ms")
```

### Notification Volume Trend

```python
volume = await svc.notification_volume(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
    group_by="day",  # "day" | "week" | "month"
)
print(volume["trend"])   # {"2025-01-01": 42, "2025-01-02": 67, ...}
print(volume["total"])
```

### Cost Report

```python
costs = await svc.cost_report(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
)
print(f"Total cost: ${costs['total_cost_usd']:.4f}")
for ch, breakdown in costs["by_channel"].items():
    print(f"  {ch}: {breakdown['count']} messages @ ${breakdown['rate_per_message']} = ${breakdown['total_cost']}")
```

### Suppression Analytics

```python
supp_stats = await svc.suppression_analytics(
    period={"start": "2025-01-01T00:00:00", "end": "2025-01-31T23:59:59"},
)
print(supp_stats["total_suppressions"])
print(supp_stats["by_reason"])
```

### Dashboard Summary

```python
summary = await svc.dashboard_summary()
print(summary["last_30_days"]["total_sent"])
print(summary["active_channels"])
print(summary["active_schedules"])
```

---

## Predictive Send-Time Optimisation

Analyse a recipient's historical open timestamps to recommend the best hour to send.

```python
prediction = await svc.predict_optimal_send_time(
    recipient_id="alice@example.com",
    channel="email",
    fallback_hour=9,  # Used when fewer than 5 historical opens exist
)
print(prediction["predicted_hour"])  # e.g. 8 (8 AM UTC)
print(prediction["confidence"])      # e.g. 0.38 — 38% of opens at that hour
print(prediction["data_points"])     # e.g. 21 — opens analysed
print(prediction["next_send_at"])    # ISO datetime of next occurrence of that hour
print(prediction["basis"])           # "historical" or "fallback"

# Schedule using the predicted time
send_at = datetime.fromisoformat(prediction["next_send_at"])
await svc.schedule_notification("alice@example.com", template_id, send_at)
```

---

## Service Health

```python
health = await svc.health_check()
# {
#   "status": "healthy",
#   "stores": {"channels": 3, "templates": 12, "notifications": 8541, ...},
#   "channel_health_summary": {"healthy": 2, "degraded": 1, "unknown": 0},
#   "delivery_stats": {"total_sent": 8541, "total_delivered": 8400, ...},
#   "checked_at": "..."
# }
```

---

## Campaign Execution

The lower-level campaign API operates via `AdvancedCampaign` Pydantic models.

```python
from capabilities.common.ntfy.api_models import AdvancedCampaign, DeliveryChannel, NotificationPriority

campaign = AdvancedCampaign(
    name="Spring Launch",
    template_ids=[template_id],
    channels=[DeliveryChannel.EMAIL],
    priority=NotificationPriority.NORMAL,
    audience_segments=[
        {"all_registered": True},
    ],
    tracking_enabled=True,
)

# Register audience members first
svc.register_audience_members([
    {"user_id": "alice", "email": "alice@example.com"},
    {"user_id": "bob", "email": "bob@example.com"},
])

results = await svc.execute_campaign(campaign, execute_immediately=True)
print(f"Success rate: {results['success_rate']:.1f}%")
```

---

## Best Practices

**Idempotency keys** — Always supply `idempotency_key` for transactional notifications
(welcome, password reset, purchase receipt).  Use a deterministic key such as
`event_type:recipient_id:event_id`.

**Consent before send** — Call `bulk_consent_check` before dispatching any marketing
campaign.  Do not send to recipients with `has_consent=False`.

**Digest for high-volume events** — Use `send_digested` for event streams that may
generate many notifications for a single recipient within minutes (e.g. monitoring
alerts, batch job completions).  Set `digest_window_minutes` to match your SLA.

**DLQ monitoring** — Poll `dlq_depth()` as part of your operational health check.
Trigger `requeue_dead_letters` with conservative `max_age_hours` to avoid re-driving
stale records.

**Latency baselines** — Call `delivery_latency_percentiles` weekly per channel.
Alert when `p99_ms` exceeds 2× the established baseline for that channel.

**Predictive scheduling** — Use `predict_optimal_send_time` for non-urgent marketing
sends where timing flexibility exists.  Requires at least 5 historical opens per
recipient/channel pair to produce a data-driven result.

**Template versioning** — Call `version_template` before modifying template content.
Use `clone_template` for A/B variant work so both variants are tracked independently.

---

## Troubleshooting

**`KeyError: Template <id> not found`** — The template was soft-deleted or the ID is
from a different tenant.  Use `list_templates(active_only=False)` to inspect all
records.

**`ValueError: Notification is not in a retryable state`** — `retry_failed` only
accepts `failed` or `bounced` notifications.  Check `track_delivery(id)["status"]`
first.

**`ValueError: recipients and variables_list must have equal length`** — `send_bulk`
requires a 1:1 correspondence between recipients and variables.  Validate list
lengths before calling.

**Consent gate blocking sends** — After `revoke_consent`, the method automatically
calls `add_suppression` for that channel.  Call `remove_suppression` followed by
`record_consent` to reinstate the recipient.

**Digest not flushing** — Digests flush lazily on the next `send_digested` call after
window expiry.  For deterministic flushing, call `flush_digest(recipient_id)` at a
known point (session end, job completion).

---

## Reference

| Method | Section |
|---|---|
| `register_channel`, `test_channel`, `channel_health_check`, `list_channels`, `update_channel_config`, `deactivate_channel` | Channel Management |
| `create_template`, `render_template`, `test_template`, `version_template`, `clone_template`, `list_templates`, `delete_template` | Template Management |
| `send_notification`, `send_idempotent`, `send_bulk`, `timezone_aware_send`, `send_digested`, `flush_digest` | Sending |
| `track_delivery`, `retry_failed`, `cancel_scheduled`, `notification_history` | Tracking |
| `requeue_dead_letters`, `dlq_depth` | Dead-Letter Queue |
| `schedule_notification`, `recurring_notification`, `cancel_recurring`, `list_scheduled` | Scheduling |
| `set_preferences`, `check_preference`, `add_suppression`, `remove_suppression`, `suppression_list` | Preferences & Suppression |
| `record_consent`, `revoke_consent`, `get_consent_status`, `bulk_consent_check` | Consent & Compliance |
| `delivery_report`, `engagement_report`, `channel_performance`, `delivery_latency_percentiles`, `notification_volume`, `cost_report`, `suppression_analytics` | Analytics |
| `predict_optimal_send_time` | Predictive Optimisation |
| `health_check`, `dashboard_summary` | Health & Dashboard |
