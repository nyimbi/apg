# MCHN Multi-Channel Output Specification

## Purpose

MCHN is APG's common multi-channel output capability. It lets generated and
composed applications define output channels, approve templates, configure
delivery policy, route rendered output, queue delivery batches, record
provider receipts, and operate output workflows through APG UI and API
surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real notification providers, renderers, print services, audit
stores, localization services, theme services, workflow engines, and Bytewax
workers later.

## Capability Identity

- Capability id: `mchn`
- Display name: `Multi-Channel Output`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.MchnService`
- UI prefix: `/mchn`
- API prefix: `/mchn/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `channel_routing`
- `format_rendering`
- `output_templates`
- `delivery_policy`
- `delivery_receipts`
- `omnichannel_analytics`
- `mchn_agents`

## Required Capabilities

- `ntfy` for notification and output-provider delivery.
- `auth` for identity, permissions, and RBAC.
- `conf` for tenant output configuration.
- `audl` for durable audit evidence.

Optional adapters include `i18n`, `them`, `wflo`, and `comp`.

## Domain Model

`OutputChannel`

- Tenant-local channel id, channel type, owner, provider reference, health,
  fallback channel, lifecycle status, and creation time.

`OutputTemplate`

- Approved tenant template with channel types, subject, body, locale, theme,
  approver, lifecycle status, and creation time.

`DeliveryPolicy`

- Tenant policy with recipient limits, throttle limits, encryption posture,
  compliance reference, lifecycle status, and creation time.

`DeliveryRoute`

- Route tying template, primary channel, fallback channels, and delivery
  policy.

`RenderedOutput`

- Rendered output payload with route, template, selected channel, recipient,
  subject, body, output format, sensitivity, encryption state, and status.

`DeliveryBatch`

- Delivery batch with route, requester, recipient count, rendered output ids,
  review status, lifecycle status, and creation time.

`DeliveryReceipt`

- Provider receipt with batch, rendered output, channel, recipient, delivery
  state, provider message id, and creation time.

`MchnAuditEvent`

- Governance record for output lifecycle actions.

`MchnAgent`

- Registered AI output agent with tenant, runtime, role, explicit scope,
  registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every output operation;
- accountable channel owner;
- channel provider reference;
- approved output templates;
- approver identity for approved templates;
- template subject or body content;
- template channel type;
- valid recipient limits;
- valid throttle limits;
- compliance policy reference;
- encryption for sensitive output;
- recipient reference on rendered output;
- no delivery through unhealthy primary channels;
- requester identity on delivery batches;
- rendered outputs on delivery batches;
- positive recipient count;
- Bytewax event stream for delivery lifecycle events;
- review for large delivery batches;
- provider message reference for receipts;
- registered AI output agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch output mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/mchn/dashboard`
- `/mchn/render`
- `/mchn/templates`
- `/mchn/routes`
- `/mchn/channels`
- `/mchn/agents`
- `/mchn/analytics`
- `/mchn/policies`
- `/mchn/audit`
- `/mchn/settings`

View models must expose output summaries, channels, templates, policies,
routes, rendered outputs, delivery batches, receipts, output agents, rules,
audit events, theme data, and Bytewax stream metadata.

## Theme

The default theme is `mchn_omnichannel_output`. Theme components cover route
consoles, template managers, channel monitors, render previews, agent panels,
and audit timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.mchn.lifecycle`
- state: channels, templates, policies, routes, rendered outputs, batches,
  receipts, MCHN agents, audit events
- events: channel created, template published, policy created, route created,
  output rendered, delivery queued, receipt recorded, agent registered
- guardrail: `batch_output_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports channels, templates, policies, routes, rendering,
  delivery batches, receipts, AI-agent registration, audit events,
  tenant-local IDs, and Bytewax batch mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
