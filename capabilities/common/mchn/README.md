# MCHN Multi-Channel Output Capability

**Version**: 2.0.0 | **Package**: `capabilities.common.mchn` | **Prefix**: `mchn_`

MCHN provides APG applications with a tenant-scoped output runtime: channels,
approved templates, delivery policies, delivery routes, rendered messages,
delivery batches, provider receipts, output agents, UI metadata, theme tokens,
audit evidence, and Bytewax-backed lifecycle events.

The package stays dependency-light. Production notification providers,
document renderers, print systems, audit sinks, localization services, theme
services, workflow engines, and Bytewax workers are represented as APG
adapters in the executable contract and are bound by the host application.

---

## What It Provides

- Output channels for email, SMS, push, PDF, web, API, and print.
- Template publication with approval, approver identity, content, channel,
  locale, and theme policy.
- Delivery policy with recipient limits, throttling, encryption posture,
  compliance reference, and retry configuration.
- Delivery routes that bind templates, primary channels, fallbacks, and policy
  with assignable priority ordering.
- Rendered output with recipient personalization, variable substitution, format
  validation, sensitive output encryption, and channel selection.
- Delivery batches with requester identity, rendered output references, large
  delivery review, and Bytewax stream enforcement.
- Provider receipts with delivery state filtering, confirmation tracking, and
  cost estimation.
- Suppression list management (per channel type) with check and bulk-add.
- Channel health probing, fallback resolution, and analytics aggregation.
- Output archiving for compliance retention.
- First-class AI output agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- Dashboard summary covering counts, unhealthy channels, failed receipts,
  and streaming metadata.

---

## Main Files

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | Normative capability behavior |
| `PLAN.md` | Implementation packet plan |
| `capability_contract.py` | Executable configuration, rules, routes, theme, adapters, provides/requires, Bytewax stream metadata |
| `models.py` | Tenant-scoped channels, templates, policies, routes, rendered output, batches, receipts, audit events, agents |
| `output_runtime.py` | Deterministic IDs, channel/format/health state, template rendering, channel selection, delivery-state helpers |
| `service.py` | Runtime facade — 37 public methods |
| `api.py` | Package-safe helper functions |
| `views.py` | UI view models |
| `test_capability_contract.py` | Lifecycle behavior and generated evidence |

---

## Quick Start

### Minimal pipeline: channel → template → policy → route → render → batch

```python
from capabilities.common.mchn import MchnService

svc = MchnService()

svc.create_channel(
    channel_id="ch-email",
    tenant_id="tenant-demo",
    name="Primary email",
    channel_type="email",
    owner="output-team",
    provider_ref="provider://sendgrid",
)

svc.publish_template(
    template_id="tmpl-notice",
    tenant_id="tenant-demo",
    name="Account notice",
    channel_types=("email",),
    subject_template="Notice $case_id",
    body_template="Hello $name, your case $case_id has been updated.",
    locale="en",
    theme_ref="mchn_omnichannel_output",
    approved=True,
    approved_by="content-owner",
)

svc.create_delivery_policy(
    policy_id="pol-default",
    tenant_id="tenant-demo",
    name="Default policy",
    max_recipients=50_000,
    throttle_per_minute=1_000,
    requires_encryption_for_sensitive=True,
    compliance_ref="compliance://gdpr-art25",
)

svc.create_route(
    route_id="route-notice",
    tenant_id="tenant-demo",
    name="Notice route",
    template_id="tmpl-notice",
    primary_channel_id="ch-email",
    fallback_channel_ids=[],
    policy_id="pol-default",
)

output = svc.render_output(
    output_id="out-001",
    tenant_id="tenant-demo",
    route_id="route-notice",
    recipient_ref="user:42",
    variables={"name": "Alice", "case_id": "C-1234"},
    output_format="html",
)

batch = svc.deliver_batch(
    batch_id="batch-001",
    tenant_id="tenant-demo",
    route_id="route-notice",
    requested_by="ops-team",
    rendered_output_ids=["out-001"],
    recipient_count=1,
)
```

### AI Output Agents

```python
agent = svc.register_mchn_agent(
    tenant_id="tenant-demo",
    name="Delivery reviewer",
    runtime="codex",
    role="delivery_reviewer",
    scope="Review large delivery batches and channel routing before release",
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: route, template, delivery, channel, compliance, accessibility review.

---

## API Reference

### Core write methods (synchronous)

| Method | Description |
|--------|-------------|
| `create_channel(...)` | Register an output channel (email/SMS/push/PDF/web/API/print) |
| `publish_template(...)` | Publish an approved message template |
| `create_delivery_policy(...)` | Define recipient limits, throttle, encryption, and compliance ref |
| `create_route(...)` | Bind template + primary channel + fallbacks + policy into a named route |
| `render_output(...)` | Render a template for a single recipient and persist as `RenderedOutput` |
| `deliver_batch(...)` | Queue a batch of rendered outputs for delivery |
| `record_receipt(...)` | Record a provider delivery receipt |
| `create_record(...)` | Convenience: scaffold channel/template/policy/route then render in one call |
| `register_mchn_agent(...)` | Register an AI output agent with role/runtime/scope/disclosure |
| `validate_batch_output_mutation(event_stream)` | Assert batch mutations flow through Bytewax |
| `dashboard_summary(tenant_id)` | Aggregate counts, health, failures, and streaming metadata |

### List methods

`list_channels`, `list_templates`, `list_policies`, `list_routes`,
`list_rendered_outputs`, `list_batches`, `list_receipts`,
`list_audit_events`, `list_mchn_agents`

All accept an optional `tenant_id` filter.

### New async methods (v2.0)

| Method | Description |
|--------|-------------|
| `channel_register(...)` | Async alias for `create_channel` |
| `channel_route(tenant_id, channel_type, event_type)` | Resolve best route+channel for an event type |
| `channel_health(tenant_id, channel_id)` | Probe and return current channel health with timestamp |
| `channel_fallback(tenant_id, primary_channel_id, reason)` | Resolve and log fallback channel for a primary |
| `channel_analytics(tenant_id, channel_id?)` | Aggregate delivery rates and unhealthy channel counts |
| `channel_cost_report(tenant_id, channel_id?, cost_per_message)` | Estimate delivery costs per channel |
| `message_format(tenant_id, template_id, variables, output_format)` | Render subject/body preview without delivery |
| `template_apply(application_id, tenant_id, template_id, recipient_refs, variables)` | Preview a template against multiple recipients in one call |
| `batch_send(batch_id, tenant_id, route_id, recipients, requested_by, ...)` | Render and queue a full recipient list in one call |
| `batch_status(tenant_id, batch_id)` | Return batch status and per-state receipt summary |
| `delivery_confirm(tenant_id, receipt_id, confirmed_by, confirmation_ref)` | Mark a receipt as confirmed by sender or webhook |
| `priority_route(tenant_id, route_id, priority, actor)` | Assign queue priority 1–10 to a route |
| `retry_policy(tenant_id, policy_id, max_retries, backoff_seconds, retry_on_states)` | Configure retry behavior for a delivery policy |
| `suppression_check(tenant_id, recipient_ref, channel_type)` | Check whether a recipient is suppressed |
| `suppression_add(tenant_id, channel_type, recipient_refs)` | Add recipients to the suppression list |
| `personalise_output(output_id, tenant_id, base_output_id, personalisation)` | Apply per-recipient variable overrides to a rendered output |
| `output_archive(tenant_id, output_id, archive_ref, actor)` | Archive a delivered output for compliance retention |
| `receipt_search(tenant_id, channel_id?, delivery_state?)` | Filter receipts by channel and/or delivery state |

---

## New Methods — Usage Examples

### 1. `batch_send` — render and queue in one shot

```python
import asyncio

result = asyncio.run(svc.batch_send(
    batch_id="batch-campaign-01",
    tenant_id="tenant-demo",
    route_id="route-notice",
    recipients=[
        {"ref": "user:1", "name": "Alice", "case_id": "C-1001"},
        {"ref": "user:2", "name": "Bob",   "case_id": "C-1002"},
    ],
    requested_by="campaigns-team",
))
# result["output_ids"] lists every rendered output ID
# result["recipient_count"] == 2
```

### 2. `channel_analytics` — per-channel delivery metrics

```python
metrics = asyncio.run(svc.channel_analytics(
    tenant_id="tenant-demo",
    channel_id="ch-email",
))
# {delivery_rate: 0.9843, delivered_count: 982, failed_count: 18, ...}
```

### 3. `retry_policy` — configure retry behavior

```python
asyncio.run(svc.retry_policy(
    tenant_id="tenant-demo",
    policy_id="pol-default",
    max_retries=5,
    backoff_seconds=120,
    retry_on_states=["failed", "bounced"],
))
```

### 4. `suppression_add` + `suppression_check`

```python
asyncio.run(svc.suppression_add(
    tenant_id="tenant-demo",
    channel_type="email",
    recipient_refs=["user:99", "user:100"],
))

check = asyncio.run(svc.suppression_check(
    tenant_id="tenant-demo",
    recipient_ref="user:99",
    channel_type="email",
))
# check["suppressed"] == True
```

### 5. `personalise_output` — per-recipient variable overrides

```python
personalised = asyncio.run(svc.personalise_output(
    output_id="out-001-personalised",
    tenant_id="tenant-demo",
    base_output_id="out-001",
    personalisation={"name": "Alice B.", "case_id": "C-9999"},
))
# Returns full RenderedOutput dict with overridden subject/body
```

---

## World-Class Enhancements (v2.0)

The 15 planned improvements that make `mchn` production-grade:

| # | Name | Tier | Summary |
|---|------|------|---------|
| 1 | Async-First Service Core | Infra | Promote all write methods to `async`; eliminate GIL contention under ASGI |
| 2 | PostgreSQL-Backed Persistent Store | Infra | Replace in-memory dicts with async SQLAlchemy + `alembic` migrations; survive restarts |
| 3 | Event Streaming via Bytewax | Infra | Emit CloudEvent envelopes for every state transition to downstream audit/analytics workers |
| 4 | Rate-Limiter Enforcement | Product | Token-bucket rate limiter in `output_runtime.py`; `rate_limited` status on excess batches |
| 5 | Webhook Inbound Receipt Ingestion | Product | `ingest_webhook(tenant_id, provider_id, payload)` with HMAC verification per provider |
| 6 | Multi-Locale Template Variants | Product | `publish_template_variant(template_id, locale, ...)` with BCP-47 fallback chain in `render_output` |
| 7 | A/B Test Routing | Product | `create_ab_test(...)` with deterministic `recipient_ref` hash; `ab_test_results` reporting |
| 8 | Delivery SLA Tracking | Observability | `sla_window_minutes` on policy; `sla_report(tenant_id, batch_id)` with breach fraction |
| 9 | Content Security Scanning | Product | `scan_template(template_id, tenant_id)` with configurable regex/keyword rules; blocks render on fail |
| 10 | Idempotency Keys on Write Operations | Infra | Optional `idempotency_key` on all create/publish/deliver; TTL-based cached response on retry |
| 11 | OpenTelemetry Observability | Observability | OTLP spans and `mchn.delivery.*` metrics on every public method; no-op when OTEL absent |
| 12 | Channel Circuit Breaker | Infra | Half-open breaker per `(tenant_id, channel_id)`; auto-mark unhealthy after N failures |
| 13 | Rendered Output Diff & Version History | Product | `output_diff(tenant_id, output_id_a, output_id_b)` with Myers-diff; versioned output store |
| 14 | Bulk Suppression Import | Product | `suppression_import(tenant_id, channel_type, source_uri, format, actor)` streaming CSV/JSONL |
| 15 | Delivery Cost Budget Enforcement | Product | `monthly_budget_usd` on policy; pre-delivery budget gate blocks `deliver_batch` on excess spend |

---

## Composition

MCHN composes with:

| Capability | Purpose |
|-----------|---------|
| `ntfy` | Provider-backed notification delivery |
| `auth` | Identity, permissions, and output RBAC |
| `conf` | Tenant output policy |
| `audl` | Durable audit evidence |
| `i18n` | Localization and template locale variants |
| `them` | Tenant theme policy |
| `wflo` | Output approval and delivery workflows |
| `comp` | Compliance and content policy |

Batch output mutation and delivery lifecycle events must use the `bytewax`
event-stream adapter.

---

## Verification

```bash
# Syntax check
./.venv/bin/python -m py_compile \
    capabilities/common/mchn/__init__.py \
    capabilities/common/mchn/capability_contract.py \
    capabilities/common/mchn/models.py \
    capabilities/common/mchn/output_runtime.py \
    capabilities/common/mchn/service.py \
    capabilities/common/mchn/api.py \
    capabilities/common/mchn/views.py \
    capabilities/common/mchn/app.py \
    capabilities/common/mchn/test_capability_contract.py

# Unit tests
./.venv/bin/pytest -q capabilities/common/mchn/test_capability_contract.py

# Capability audit
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mchn --json

# Publish plan
./.venv/bin/apg capabilities publish-plan capabilities/common/mchn --json
```

Live notification providers, document renderers, print systems, durable audit
stores, rendered UI, and Bytewax workers are integration concerns outside the
package proof.

---

*© 2025 Datacraft — www.datacraft.co.ke*
