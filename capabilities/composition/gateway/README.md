# API Service Mesh

## Overview

The API Service Mesh provides service discovery, intelligent routing, traffic management, TLS certificate lifecycle, and policy enforcement for all services exposed within the APG composition layer. It acts as the single ingress and inter-service control plane, ensuring that every public route is protected by a policy, a rate limiter, a circuit breaker, and a valid TLS certificate before traffic is allowed.

The business value is a unified governance surface for API exposure: teams register services once and the mesh handles health checking, load balancing, canary deployments, and certificate rotation. All mesh state changes are emitted to the Bytewax event stream, giving operations teams a complete, replayable audit trail of every routing and policy change.

## Capability ID

`composition_gateway`  Version: see `package_manifest.json`

## Provides

| Service | Description |
|---------|-------------|
| service_mesh_registry | Register services with endpoints, health checks, and capability bindings |
| gateway_route_lifecycle | Create, approve, and manage routing rules with match conditions |
| traffic_management | Canary splits, weighted routing, rate limiting, and circuit breakers |
| gateway_policy_enforcement | Attach and enforce traffic, security, and rate-limit policies per route |
| certificate_lifecycle | Register, store (via secret reference), and manage TLS certificates |
| mesh_health_observability | Continuous health checks, distributed traces, topology mapping, and metrics |
| gateway_agents | AI agent workbench for mesh architecture and traffic review |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate operators managing mesh configuration |
| audl | Persist immutable mesh change audit records |
| ntfy | Send route approval and health degradation notifications |
| registry | Register this capability in the global catalog |
| composition_access | Enforce policy on all gateway write operations |
| composition_events | Receive and emit mesh lifecycle events via Bytewax |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| services.owner_required | bool | true | Services must have an accountable owner |
| services.health_check_required | bool | true | Services must define a health check |
| services.capability_binding_required | bool | true | Services must bind to an APG capability |
| routes.policy_required_for_public | bool | true | Public routes must have an attached policy |
| routes.approval_required_for_public | bool | true | Public routes require explicit approval |
| traffic.circuit_breaker_required | bool | true | Public services require circuit breakers |
| traffic.rate_limit_required_for_public | bool | true | Public services require rate limits |
| security.tls_required_for_public | bool | true | Public routes require TLS |
| security.secret_reference_required | bool | true | Certificate private keys must use vault references |
| gateway_agents.max_autonomous_scope | string | "recommend_and_validate" | Ceiling on autonomous agent actions |
| observability.event_stream | string | "apg.composition.gateway.lifecycle" | Bytewax stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-gateway/dashboard | GET | composition_gateway:view | Overview |
| services | /composition-gateway/services | GET/POST | composition_gateway:manage_services | Services |
| routes | /composition-gateway/routes | GET/POST | composition_gateway:manage_routes | Routes |
| policies | /composition-gateway/policies | GET/POST | composition_gateway:govern | Governance |
| traffic | /composition-gateway/traffic | GET/POST | composition_gateway:operate | Operations |
| certificates | /composition-gateway/certificates | GET/POST | composition_gateway:admin | Security |
| agents | /composition-gateway/agents | GET/POST | composition_gateway:admin | Automation |
| settings | /composition-gateway/settings | GET/PUT | composition_gateway:admin | Administration |

REST API prefix: `/composition-gateway/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| gateway_write_requires_policy | write operation without policy attached | deny |
| service_requires_owner | register_service without owner | deny |
| service_requires_endpoint | register_service without endpoint | deny |
| service_requires_health_check | register_service without health check | deny |
| public_route_requires_policy | create public route without policy | deny |
| public_route_requires_approval | create public route without approval | deny |
| public_route_requires_tls | create public route without TLS | deny |
| route_requires_bytewax_stream | create_route not via bytewax | deny |
| traffic_shift_requires_canary_evidence | shift canary traffic without evidence | require_review |
| traffic_shift_requires_bytewax_stream | shift_traffic not via bytewax | deny |
| public_service_requires_rate_limit | attach_policy to public service without rate limit | deny |
| public_service_requires_circuit_breaker | attach_policy to public service without circuit breaker | deny |
| certificate_requires_owner | register_certificate without owner | deny |
| certificate_requires_secret_reference | register_certificate without secret reference | deny |
| batch_route_change_requires_bytewax | batch_route_change not via bytewax | deny |
| gateway_agent_runtime_supported | register_gateway_agent with unsupported runtime | deny |
| gateway_agent_role_supported | register_gateway_agent with unsupported role | deny |
| privileged_agent_gateway_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| SMService | service_id, service_name, service_version, namespace, status, health_status, endpoints, tenant_id |
| SMEndpoint | endpoint_id, service_id, host, port, protocol, path, weight, tls_enabled, certificate_id, health_check_path |
| SMRoute | route_id, route_name, service_id, match_type, match_value, destination_services, timeout_ms, retry_attempts, priority |
| SMLoadBalancer | load_balancer_id, algorithm, session_affinity, circuit_breaker_enabled, failure_threshold, max_connections |
| SMPolicy | policy_id, policy_name, policy_type, route_id, configuration, rate_limit_requests, rate_limit_window_seconds |
| SMCertificate | certificate_id, certificate_name, common_name, not_before, not_after, status, auto_renew, renewal_days_before |
| SMSecurityPolicy | security_policy_id, policy_type, rules, enforcement_mode, allowed_sources, authentication_methods |
| SMRateLimiter | rate_limiter_id, requests_per_second, burst_size, scope, enforcement_mode |
| SMHealthCheck | health_check_id, service_id, status, response_time_ms, consecutive_successes, consecutive_failures |
| SMMetrics | metric_id, service_id, metric_name, metric_type, value, request_count, error_count, response_time_ms |
| SMTrace | trace_id, span_id, service_name, operation_name, start_time, duration_ms, status |
| SMTopology | topology_id, source_service_id, target_service_id, relationship_type, avg_response_time_ms |
| SMAlert | alert_id, condition, severity, is_active, notification_channels, trigger_count |

AI-powered models also present: `SMNaturalLanguagePolicy`, `SMIntelligentTopology`, `SMAutonomousMeshDecision`, `SMPredictiveAlert`, `SMCollaborativeSession`.

Pydantic API models: `ServiceConfig`, `EndpointConfig`, `RouteConfig`, `LoadBalancerConfig`, `PolicyConfig`.

## Streaming Events

Events emitted to the composition event stream via Bytewax (`apg.composition.gateway.lifecycle`).

| Event | Trigger |
|-------|---------|
| service_registered | New service added to the mesh registry |
| route_created | Routing rule created and approved |
| policy_attached | Policy attached to a route or service |
| traffic_shifted | Canary or weighted traffic split changed |
| certificate_registered | TLS certificate registered with vault reference |
| health_recorded | Health check result recorded |
| gateway_agent_registered | New gateway agent registered |

Stream states: `draft → active → healthy → degraded → canary → blocked → retired`

## Edge Cases Handled

- Public routes are blocked at creation if TLS, policy, and approval are not all present simultaneously; partial compliance is not accepted, preventing routes from going live in an insecure intermediate state.
- Canary traffic shifts that lack evidence produce `require_review` rather than `deny`, preserving the ability to execute emergency traffic shifts with an explicit human review record.
- Certificate private keys must reference a vault secret rather than being stored in `private_key_pem`; the model column exists for completeness but the `certificate_requires_secret_reference` rule enforces the vault path at registration time.
- `SMService.metadata` is exposed as a Python property over the underlying `metadata_json` column to avoid collision with SQLAlchemy's reserved `metadata` attribute; direct column access uses `metadata_json`.
- Circuit breakers are required at policy-attachment time for public services, not at service registration time, because the same service may be exposed through both internal and public routes with different policies.
- The `SMIntelligentTopology` and `SMAutonomousMeshDecision` models capture AI-generated predictions and autonomous self-healing decisions for audit and rollback purposes.

## Composability

- **Upstream**: `composition_access` (policy enforcement on all writes), `composition_events` (receives mesh lifecycle events), `auth` (operator identity)
- **Downstream**: All domain capabilities that expose HTTP APIs register their services here; the mesh handles routing, rate limiting, and TLS termination for their endpoints
- **Peer**: `audl` (long-term route and policy change audit), `ntfy` (health degradation and approval notifications), `composition_config` (reads route timeout and rate-limit config values)

## Development Notes

- `SMService`, `SMEndpoint`, `SMPolicy`, `SMMetrics`, `SMAlert`, `SMTopology`, `SMConfiguration`, `SMCertificate`, `SMSecurityPolicy`, and `SMRateLimiter` all have their `metadata` attribute patched as a property at module load time via the `_get_model_metadata` / `_set_model_metadata` pattern; avoid adding a mapped `metadata` column to any new models in this module.
- The `SMLoadBalancer` model is not directly bound to a route; it is associated with a service. Apply circuit breaker and health check settings at the load balancer level, not per-endpoint.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (SQLAlchemy + Pydantic models), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers).
