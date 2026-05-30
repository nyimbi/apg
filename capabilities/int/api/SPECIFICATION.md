# Integration API Management Specification

## Purpose

`int_api` gives APG applications a governed API management capability for API
registry, endpoint, policy, consumer, key, subscription, deployment,
gateway-route, analytics, and API-agent review lifecycles. The package must run
without external services and make every production integration boundary
explicit.

## Capability Identity

- Capability id: `int_api`
- Display name: `Integration API Management`
- Version: `2.1.0`
- Target: `python`
- Profile: `capability`
- Event stream: `apg.int.api.lifecycle`
- Stream processor: `bytewax`
- Theme: `int_api_control`

## Domain Records

### API

Fields:

- `id`
- `tenant_id`
- `name`
- `title`
- `base_path`
- `upstream_url`
- `owner_id`
- `version`
- `protocol`
- `auth_type`
- `rate_limit_per_minute`
- `reviewed_by`
- `approved_by`
- `metadata`
- `status`
- `created_at`
- `updated_at`

Supported protocols are REST, GraphQL, gRPC, and webhook. Supported auth types
are API key, OAuth2, JWT, mTLS, and none.

### Endpoint

Fields:

- `id`
- `tenant_id`
- `api_id`
- `path`
- `method`
- `auth_required`
- `rate_limit_override`
- `status`
- `created_at`

### Policy

Fields:

- `id`
- `tenant_id`
- `api_id`
- `policy_type`
- `name`
- `config`
- `execution_order`
- `status`
- `created_at`

Supported policy types are rate limit, quota, auth, transform, CORS, IP filter,
and circuit breaker.

### Consumer

Fields:

- `id`
- `tenant_id`
- `name`
- `contact_email`
- `owner_id`
- `external`
- `reviewed_by`
- `status`
- `created_at`

### API Key

Fields:

- `id`
- `tenant_id`
- `consumer_id`
- `name`
- `key_prefix`
- `scopes`
- `expires_on`
- `status`
- `created_at`

### Subscription

Fields:

- `id`
- `tenant_id`
- `consumer_id`
- `api_id`
- `plan`
- `approved_by`
- `status`
- `created_at`

Supported plans are sandbox, standard, premium, and internal.

### Deployment

Fields:

- `id`
- `tenant_id`
- `api_id`
- `environment`
- `gateway_route`
- `deployed_by`
- `approved_by`
- `status`
- `created_at`

Supported environments are dev, test, stage, and prod.

### Usage Record

Fields:

- `id`
- `tenant_id`
- `api_id`
- `consumer_id`
- `endpoint_id`
- `status_code`
- `latency_ms`
- `reviewed_by`
- `status`
- `created_at`

### API Agent

Fields:

- `id`
- `tenant_id`
- `name`
- `runtime`
- `role`
- `scope`
- `status`
- `created_at`

Supported runtimes are Codex, Claude Code, OpenCode, and Pi. Supported roles are
API designer, policy reviewer, security reviewer, consumer reviewer, deployment
reviewer, and analytics reviewer.

## Lifecycle Workflows

### API Design And Governance

1. Register API with owner, protocol, auth type, base path, upstream, and rate
   limit.
2. Require review evidence for external upstreams.
3. Register endpoints under same-tenant APIs.
4. Attach policies in explicit execution order.
5. Approve APIs before production deployment.

### Consumer And Access Management

1. Register consumers with owner and contact email.
2. Require review for external consumers.
3. Issue scoped API keys with expiration.
4. Create approved subscriptions to APIs.

### Deployment And Analytics

1. Deploy APIs to supported environments.
2. Require explicit approval for production deployments.
3. Require deployer identity for every deployment.
4. Record usage analytics with status and latency.
5. Require review for slow requests.

### AI-Agent Composition

1. Register API agents with supported runtime and role.
2. Limit agent scope to review, preparation, validation, and recommendation.
3. Require human approval for privileged actions.
4. Emit lifecycle evidence for agent registration and approved actions.

## Rule Engine

The deterministic rule engine returns:

- `decision`: allow, deny, or require_review;
- `matched_rules`: ordered matching rule names;
- `effects`: rule effects with reason and required action.

Rules cover tenant context, write policy attachment, API completeness, protocol
and auth support, external-upstream review, endpoint completeness, policy
configuration, consumer validation, key expiration, subscription approval,
API approval, deployment identity, production deployment approval, usage
analytics, Bytewax routing, agent runtime and role support, and
privileged-agent approval.

## UI Contract

Routes:

- `/int-api/dashboard`
- `/int-api/apis`
- `/int-api/endpoints`
- `/int-api/policies`
- `/int-api/consumers`
- `/int-api/keys`
- `/int-api/subscriptions`
- `/int-api/deployments`
- `/int-api/analytics`
- `/int-api/agents`
- `/int-api/settings`

Screen models are framework-neutral dictionaries that generated applications
can render through the selected APG Python UI target.

## Events

Lifecycle events include API registered, endpoint registered, policy attached,
consumer registered, API key issued, subscription created, API approved, API
deployed, usage recorded, and API agent registered.

Each event records tenant, event type, record id, record type, status, stream,
processor, and timestamp.

## Production Adapters

The package keeps these concerns behind adapters:

- authorization;
- audit vault;
- notification;
- live gateway;
- developer portal;
- service discovery;
- policy management;
- analytics sink;
- durable Bytewax topology;
- theme application;
- AI-agent runtime orchestration.

## Acceptance Criteria

- Contract shape validates through APG capability tooling.
- Service executes the full API management lifecycle in memory.
- Guardrails reject unsafe, incomplete, unsupported, or cross-tenant actions.
- UI routes and screen models cover all primary records.
- Semantic model includes provides, requires, rules, theme, screens, agents, and
  Bytewax streaming metadata.
- Package self-test passes.
- APG inspect, publish-plan, and implementation-audit pass for this capability.
