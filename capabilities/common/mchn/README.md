# MCHN Multi-Channel Output Capability

MCHN provides APG applications with a tenant-scoped output runtime: output
channels, approved templates, delivery policies, delivery routes, rendered
messages and documents, delivery batches, provider receipts, output agents, UI
metadata, theme tokens, audit evidence, and Bytewax-backed lifecycle events.

The package stays dependency-light. Production notification providers,
document renderers, print systems, audit sinks, localization services, theme
services, workflow engines, and Bytewax workers are represented as APG
adapters in the executable contract and are bound by the host application.

## What It Provides

- Output channels for email, SMS, push, PDF, web, API, and print.
- Template publication with approval, approver identity, content, channel,
  locale, and theme policy.
- Delivery policy with recipient limits, throttling, encryption posture, and
  compliance reference.
- Delivery routes that bind templates, primary channels, fallbacks, and policy.
- Rendered output with recipient, variables, format validation, sensitive
  output encryption, and channel selection.
- Delivery batches with requester identity, rendered output references, large
  delivery review, and Bytewax stream enforcement.
- Provider receipts with delivery state and provider message reference.
- First-class AI output agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped channels, templates, policies, routes,
  rendered output, batches, receipts, audit events, and agents.
- `output_runtime.py` contains deterministic IDs, channel/format/health state,
  template rendering, channel selection, and delivery-state helpers.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.mchn import MchnService

service = MchnService()
service.create_channel(
    channel_id="channel-email",
    tenant_id="tenant-demo",
    name="Email primary",
    channel_type="email",
    owner="output-team",
    provider_ref="provider://email",
)
service.publish_template(
    template_id="template-notice",
    tenant_id="tenant-demo",
    name="Notice",
    channel_types=("email",),
    subject_template="Notice $case_id",
    body_template="Hello $name",
    locale="en",
    theme_ref="mchn_omnichannel_output",
    approved=True,
    approved_by="content-owner",
)
```

## AI Output Agents

Register AI agents before they assist with output operations:

```python
agent = service.register_mchn_agent(
    tenant_id="tenant-demo",
    name="Delivery reviewer",
    runtime="codex",
    role="delivery_reviewer",
    scope="Review large delivery batches and channel routing before release",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
Supported roles cover route, template, delivery, channel, compliance, and
accessibility review.

## Composition

MCHN composes with:

- `ntfy` for provider-backed notification delivery.
- `auth` for identity, permissions, and output RBAC.
- `conf` for tenant output policy.
- `audl` for durable audit evidence.
- `i18n` for localization.
- `them` for tenant theme policy.
- `wflo` for output approval and delivery workflows.
- `comp` for compliance and content policy.

Batch output mutation and delivery lifecycle events must use the `bytewax`
event-stream adapter.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/mchn/__init__.py capabilities/common/mchn/capability_contract.py capabilities/common/mchn/models.py capabilities/common/mchn/output_runtime.py capabilities/common/mchn/service.py capabilities/common/mchn/api.py capabilities/common/mchn/views.py capabilities/common/mchn/app.py capabilities/common/mchn/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/mchn/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mchn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mchn --json
```

Live notification providers, document renderers, print systems, durable audit
stores, rendered UI, and Bytewax workers are integration concerns outside the
package proof.
