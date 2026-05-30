# Fintech Gateway Capability Specification

## Purpose

`fintech_gateway` makes payment orchestration executable as an APG composition unit. The capability owns merchant onboarding, provider connections, payment method tokenization, payment intent creation, routing, risk review, authorization, capture, refunds, webhooks, settlements, disputes, event emission, and AI-assisted operational review.

## Users

- Merchant operations teams onboard and govern merchants.
- Payment operations teams manage provider connectivity, routing, authorizations, captures, and refunds.
- Fraud teams review risky payments and blocked activity.
- Finance teams reconcile settlements and variances.
- Support teams manage disputes and webhook-driven payment changes.
- AI agents prepare operational recommendations under human approval constraints.
- Application builders compose gateway behavior into commerce, ERP, CRM, treasury, and analytics applications.

## Scope

In scope:

- Tenant-scoped merchant records.
- Provider connection records and credential references.
- Payment method token references.
- Payment intents and lifecycle state.
- Risk assessment and review gates.
- Authorization, capture, and refund controls.
- Webhook ingestion with event identity, signature, and idempotency.
- Settlement recording and variance review.
- Payment dispute opening and reviewed resolution.
- Gateway agent registration, runtime validation, role validation, and privileged-action review.
- Contract metadata for configuration, rules, UI routes, theme, streaming, dependencies, and provided services.
- Dependency-light service, API helpers, view models, semantic app entrypoint, package manifest, and tests.

Out of scope for this packet:

- Durable provider credential vault integrations.
- Live processor calls.
- Durable payment stores.
- Provider sandbox tests.
- Durable Bytewax deployment topology.
- Rendered browser UI.
- Large-scale performance and failover verification.

## Lifecycle

1. A merchant is onboarded with tenant context, code, legal name, country, and risk classification.
2. A provider connection is registered with supported provider, supported type, and credential reference.
3. A payment method is tokenized for a merchant and customer reference.
4. A payment intent is created with merchant, payment method, positive amount, and supported currency.
5. Payment risk is assessed and reviewed when required.
6. A payment is authorized through a provider when risk allows it.
7. A payment is captured without exceeding the authorized balance.
8. Refunds are recorded without exceeding the captured balance and with review where required.
9. Webhooks are ingested only with provider, event ID, signature, and idempotency key.
10. Settlements are recorded and variances require review.
11. Disputes are opened with supported reasons and resolved after review.
12. Gateway agents are registered and constrained by runtime, role, and human approval rules.
13. Lifecycle events are emitted with `processor = "bytewax"`.

## Configuration

The capability contract exposes configuration sections for:

- `merchants`
- `provider_connections`
- `payment_methods`
- `payment_intents`
- `authorization`
- `capture`
- `refunds`
- `webhooks`
- `settlements`
- `disputes`
- `gateway_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Each section is tenant-scoped and can be overridden through APG composition configuration.

## Rules

The deterministic rule engine must return `allow`, `require_review`, or `deny`.

Required rule groups:

- Governance: tenant context and write policy.
- Merchant: code, legal name, country, high-risk review.
- Provider: supported provider, supported provider type, credential reference.
- Payment method: merchant, customer reference, supported method type, token reference.
- Payment intent: merchant, method, positive amount, supported currency.
- Risk: payment intent, high-risk review, and blocked-risk authorization denial.
- Authorization: payment intent, provider selection, and high-value approval.
- Capture: authorized payment, positive capture amount, overcapture blocking.
- Refund: captured payment, positive refund amount, overrefund blocking, large-refund review.
- Webhook: provider, event ID, signature, idempotency.
- Settlement: provider, reference, nonnegative amount, variance review.
- Dispute: payment, supported reason, owner, resolution review.
- Streaming: gateway batch and event routing through Bytewax.
- Agents: supported runtimes, supported roles, human approval for privileged scope.

## UI And Theme

The capability must expose screen metadata for:

- Dashboard
- Merchants
- Providers
- Payment Methods
- Payments
- Routing
- Risk
- Webhooks
- Settlements
- Disputes
- Agents
- Settings

The theme name is `fintech_gateway_control`. Screens use compact payment-operation layouts with status chips, routing lanes, risk queues, event inboxes, settlement grids, dispute boards, and agent review lanes.

## Agent Composition

Gateway agents are first-class records. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles cover merchant underwriting, routing review, fraud review, settlement review, dispute review, and provider operations review. Agents can prepare and review work, but privileged actions require human approval.

## Adapter Boundaries

The capability declares dependencies on authorization, audit, notification, key management, encryption, cash management, accounts receivable, CRM, BI, and composition services. Payment processors remain live adapters outside the dependency-light package boundary.

## Acceptance Gates

- Contract validates through `validate_contract_shape`.
- Package imports without optional provider dependencies.
- Service executes the merchant-to-settlement lifecycle.
- Negative guardrail tests prove denials and review gates.
- Semantic model reports Bytewax streaming and gateway agent support.
- APG inspect, publish-plan, and implementation-audit commands run for the package.
- Documentation matches executable code.
