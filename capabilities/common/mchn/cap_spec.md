# Multi-Channel Output Capability Specification

- **Capability Name**: Multi-Channel Output
- **Capability ID**: `mchn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

MCHN is APG's package-backed omnichannel output runtime for tenant-scoped
channels, templates, delivery policies, routes, rendered outputs, delivery
batches, delivery receipts, audit events, UI route metadata, theme metadata,
rule evaluation, semantic-model publication, and publish-plan evidence.

The package is dependency-light and deterministic. Live email gateways, SMS
providers, push providers, print services, PDF renderers, notification buses,
compliance archives, and delivery analytics systems remain adapter boundaries
until a future slice wires and verifies them directly.

## Provided Services

- `channel_routing`
- `format_rendering`
- `output_templates`
- `delivery_policy`
- `omnichannel_analytics`
- `mchn_operations`

## Required Services

- `tenant_context`
- `ntfy` for notification-channel integration
- `auth` for actor and permission context
- `conf` for tenant delivery and rendering policy
- optional `i18n`, `them`, `audl`, and `wflo` adapters for localization,
  theming, audit, and workflow approvals

## Runtime Surfaces

| File | Runtime responsibility |
| --- | --- |
| `models.py` | Channel, template, policy, route, rendered output, delivery batch, receipt, and audit dataclasses |
| `output_runtime.py` | Deterministic IDs, channel/format/health validation, template rendering, channel selection, batch status, and receipt-state helpers |
| `service.py` | Tenant-aware channel creation, template publication, policy creation, route creation, rendering, delivery, receipts, summaries, compatibility records, and guardrails |
| `api.py` | Dependency-light API helper functions over the service |
| `views.py` | Dashboard, render console, template manager, route console, channel monitor, analytics, and policy view models |
| `app.py` | Publishable APG package entrypoint and semantic-model evidence |

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Runtime operations require tenant context.
Output behavior is governed by the `channels`, `rendering`, `delivery`,
`governance`, `ui`, and `theme` configuration sections.

## Rules

MCHN evaluates the deterministic rules from the capability contract:

- `tenant_context_required`
- `channel_requires_owner`
- `template_requires_approval`
- `sensitive_output_requires_encryption`
- `unhealthy_channel_blocks_delivery`
- `large_delivery_requires_review`

The service enforces these rules directly. Missing tenant context, missing
channel owners, unapproved templates, unencrypted sensitive output, unhealthy
delivery channels, and unreviewed large deliveries are blocked or require
review.

## UI And Theme

The package exposes eight APG Python UI routes:

- `/mchn/dashboard`
- `/mchn/render`
- `/mchn/templates`
- `/mchn/routes`
- `/mchn/channels`
- `/mchn/analytics`
- `/mchn/policies`
- `/mchn/settings`

View helpers expose summaries, channels, templates, policies, routes, rendered
outputs, delivery batches, receipts, audit events, failed receipts, unhealthy
channels, rules, and theme metadata. The package uses the
`mchn_omnichannel_output` theme contract with route-console, template-manager,
channel-monitor, and render-preview component tokens.

## Adapter Boundaries

This package intentionally does not open network connections or require live
delivery providers. Production deployments should attach adapters for:

- email gateway delivery;
- SMS provider delivery;
- push notification delivery;
- PDF and document rendering;
- web/API output transport;
- print queue integration;
- notification-event bus integration;
- compliance archiving;
- audit-log persistence;
- delivery analytics and bounce processing.

The in-process service remains the executable APG behavior used by generated
apps, tests, publish-plan checks, and local capacity slices.

## Focused Verification

Use battery-conscious verification for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/mchn/__init__.py capabilities/common/mchn/models.py capabilities/common/mchn/output_runtime.py capabilities/common/mchn/service.py capabilities/common/mchn/api.py capabilities/common/mchn/views.py capabilities/common/mchn/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/mchn/test_capability_contract.py capabilities/common/mchn/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mchn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mchn --json
```
