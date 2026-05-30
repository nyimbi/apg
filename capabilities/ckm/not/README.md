# CKM Notification System

`ckm_not` is the APG Collaboration and Knowledge Management notification
capability. It lets generated applications compose notification templates,
campaigns, deliveries, recipient preferences, provider registration, audit
evidence, analytics metadata, and AI-agent review guardrails.

The package is deliberately dependency-light. It defines the executable
lifecycle, rules, UI route metadata, theme metadata, Bytewax stream declaration,
and semantic evidence. Live provider SDKs, durable stores, stream workers,
schedulers, and observability pipelines are adapter responsibilities.

## What It Provides

- Template studio for tenant-scoped notification content, locales, channel
  coverage, approval, and variable-schema evidence.
- Campaign console for audience policy, send-window control, approval, and
  batch execution governance.
- Delivery workbench for approved-template delivery requests across email, SMS,
  push, in-app, voice, webhook, WhatsApp, Slack, Teams, and web push.
- Preference center for consent references, channel choices, topic suppression,
  and quiet-hour behavior.
- Provider registry contract requiring managed secret references.
- AI notification-agent registration for Codex, Claude Code, OpenCode, Pi, and
  future runtimes behind the same contract.
- Bytewax stream guardrail for batch notification mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Use

Load the package through `importlib` because the directory name is `not`, which
is a Python keyword:

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

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `template_reviewer`, `audience_reviewer`, `delivery_reviewer`,
`compliance_reviewer`, and `escalation_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- template channel content is incomplete;
- template activation lacks variable schema;
- an external delivery lacks consent evidence;
- a recipient or topic is suppressed;
- quiet-hour delivery is not deferred and no permitted urgent override exists;
- a campaign lacks audience policy;
- a bulk campaign lacks approval;
- provider credentials do not use managed secret references;
- an AI notification agent is unregistered, unsupported, unscoped, or
  undisclosed;
- lifecycle state changes lack audit evidence;
- batch notification mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch notification mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_notification_mutation("bytewax")
blocked = service.validate_batch_notification_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.ckm_not.lifecycle` and state for templates,
campaigns, deliveries, preferences, providers, notification agents, and audit
events.

## Composition

Generated APG applications should compose `ckm_not` through:

- capability ID: `ckm_not`;
- provided services: notification delivery, template management, campaign
  orchestration, preference center, channel provider registry, engagement
  analytics, and notification agents;
- required services: `auth`, `conf`, `encr`, and `audl`;
- API prefix: `/ckm-not/api/v1`;
- UI routes: dashboard, templates, campaigns, deliveries, preferences,
  providers, agents, rules, analytics, audit, and settings;
- theme: `ckm_not_notification_ops`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/ckm/__init__.py capabilities/ckm/not/__init__.py capabilities/ckm/not/capability_contract.py capabilities/ckm/not/lifecycle.py capabilities/ckm/not/app.py capabilities/ckm/not/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/not/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.not'); service = pkg.NotificationLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/not --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/not --json
```
