# API Gateway — composition_gateway

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>

## Overview

The API Gateway provides service discovery, intelligent routing, traffic management, TLS certificate lifecycle, and policy enforcement for all services exposed within the APG composition layer. It acts as the single ingress and inter-service control plane, ensuring that every public route is protected by a policy, a rate limiter, a circuit breaker, and a valid TLS certificate before traffic is allowed.

All mesh state changes are emitted to the Bytewax+NATS event stream (`apg.composition.gateway.lifecycle`), giving operations teams a complete, replayable audit trail of every routing and policy change.

## Capability ID

`composition_gateway`  — Version: see `package_manifest.json`

## New Features (2026)

| Feature | Method | Description |
|---------|--------|-------------|
| Dark traffic shadowing | `shadow_request()` | Copy live traffic to a shadow service via NATS without affecting primary response; returns divergence report |
| Predictive auto-scaling signals | `emit_scaling_signals()` | Pre-aggregate EW-forecasted scaling events to `apg.gateway.autoscale.<service_id>` for Bytewax consumption |
| JSON Schema payload validation | `validate_request_payload()` | Validate inbound payloads against JSON Schema (draft-2020-12); return full violation list |
| Adaptive timeout budgeting | `compute_request_budget()` | Compute remaining time budget from `X-Request-Deadline` header; prevent zombie work |
| Deadline propagation | `propagate_deadline()` | Inject `X-Request-Deadline` into upstream calls so the budget cascades through the entire call chain |
| API deprecation management | `deprecate_route()` | Schedule deprecation + sunset dates; auto-inject `Deprecation`/`Sunset` headers; return 410 after sunset |
| Deprecation header injection | `get_deprecation_headers()` | Compute headers to inject per-response for deprecated routes; return 410 sentinel when past sunset |
| Locality-aware load balancing | `check_locality_affinity()` | Score endpoints by zone/region affinity (zone_penalty × p50_latency / weight); return sorted list |

## Provides

| Service | Description |
|---------|-------------|
| service_mesh_registry | Register services with endpoints, health checks, and capability bindings |
| gateway_route_lifecycle | Create, approve, and manage routing rules with match conditions |
| traffic_management | Canary splits, weighted routing, shadow traffic, rate limiting, and circuit breakers |
| gateway_policy_enforcement | Attach and enforce traffic, security, and rate-limit policies per route |
| certificate_lifecycle | Register, store (via secret reference), and manage TLS certificates |
| mesh_health_observability | Continuous health checks, distributed traces, topology mapping, and metrics |
| gateway_agents | AI agent workbench for mesh architecture and traffic review |
| api_deprecation | Deprecation scheduling, Sunset header injection, and HTTP 410 enforcement |
| payload_validation | Per-route JSON Schema validation with full violation reporting |
| locality_load_balancing | Zone/region-aware endpoint scoring for latency-optimal routing |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate operators managing mesh configuration |
| audl | Persist immutable mesh change audit records |
| ntfy | Send route approval, health degradation, and sunset warning notifications |
| registry | Register this capability in the global catalog |
| composition_access | Enforce policy on all gateway write operations |
| composition_events | Receive and emit mesh lifecycle events via Bytewax+NATS |

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
| observability.event_stream | string | "apg.composition.gateway.lifecycle" | NATS/Bytewax stream name |
| shadow.nats_subject_prefix | string | "apg.gateway.shadow" | NATS subject prefix for shadow traffic |
| autoscale.nats_subject_prefix | string | "apg.gateway.autoscale" | NATS subject prefix for scaling signals |
| timeout.min_upstream_ms | int | 50 | Minimum remaining budget before short-circuiting |
| deprecation.warning_days | list[int] | [30, 7, 1] | Days before sunset to emit deprecation warnings |

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
| shadow | /composition-gateway/shadow | POST | composition_gateway:operate | Traffic Testing |
| deprecation | /composition-gateway/routes/{id}/deprecate | POST | composition_gateway:admin | Lifecycle |

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
| deprecated_route_returns_sunset_headers | route has deprecation_status=deprecated | inject Deprecation/Sunset headers |
| sunset_route_returns_410 | route has sunset_at in the past | return HTTP 410 Gone |

## Data Models

| Model | Key Fields |
|-------|-----------|
| SMService | service_id, service_name, service_version, namespace, status, health_status, endpoints, tenant_id |
| SMEndpoint | endpoint_id, service_id, host, port, protocol, path, weight, tls_enabled, certificate_id, health_check_path, zone, region |
| SMRoute | route_id, route_name, service_id, match_type, match_value, destination_services, timeout_ms, retry_attempts, priority, deprecated_at, sunset_at, migration_guide_url |
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

AI-powered models: `SMNaturalLanguagePolicy`, `SMIntelligentTopology`, `SMAutonomousMeshDecision`, `SMPredictiveAlert`, `SMCollaborativeSession`.

Pydantic API models: `ServiceConfig`, `EndpointConfig`, `RouteConfig`, `LoadBalancerConfig`, `PolicyConfig`.

## Streaming Events

Events emitted to the composition event stream via Bytewax+NATS (`apg.composition.gateway.lifecycle`).

| Event | Trigger |
|-------|---------|
| service_registered | New service added to the mesh registry |
| route_created | Routing rule created and approved |
| policy_attached | Policy attached to a route or service |
| traffic_shifted | Canary or weighted traffic split changed |
| certificate_registered | TLS certificate registered with vault reference |
| health_recorded | Health check result recorded |
| gateway_agent_registered | New gateway agent registered |
| shadow_request_completed | Shadow traffic divergence report ready |
| scaling_signal_emitted | Predictive auto-scaling signal computed |
| payload_validation_failed | Request payload failed JSON Schema validation |
| route_deprecated | Route marked deprecated with sunset date |
| route_sunset | Route past its sunset date (HTTP 410 active) |

Stream states: `draft → active → healthy → degraded → canary → blocked → retired → sunset`

## Edge Cases Handled

- Public routes are blocked at creation if TLS, policy, and approval are not all present simultaneously; partial compliance is not accepted.
- Canary traffic shifts that lack evidence produce `require_review` rather than `deny`, preserving the ability to execute emergency traffic shifts with an explicit human review record.
- Certificate private keys must reference a vault secret rather than being stored in `private_key_pem`; the `certificate_requires_secret_reference` rule enforces this at registration time.
- `SMService.metadata` is exposed as a Python property over the underlying `metadata_json` column to avoid collision with SQLAlchemy's reserved `metadata` attribute.
- Circuit breakers are required at policy-attachment time for public services, not at service registration time, because the same service may be exposed through both internal and public routes.
- Shadow requests use a 5-second hard timeout and never block the primary response path.
- Deadline propagation uses `time.time()` (wall clock) for the deadline epoch and `time.monotonic()` for elapsed measurement to avoid clock-skew errors.
- When `compute_request_budget()` returns `should_short_circuit: true`, the caller must return HTTP 504 before forwarding to prevent zombie work.
- `get_deprecation_headers()` returns the `X-Gateway-Gone: 410` sentinel as a dict key rather than raising an exception, so callers can inspect it before deciding whether to forward the request.
- Locality-aware load balancing falls back to `zone_penalty=3.0` (cross-region cost) when endpoint zone/region metadata is absent.

## Composability

- **Upstream**: `composition_access` (policy enforcement on all writes), `composition_events` (receives mesh lifecycle events), `auth` (operator identity)
- **Downstream**: All domain capabilities that expose HTTP APIs register their services here; the mesh handles routing, rate limiting, and TLS termination for their endpoints
- **Peer**: `audl` (long-term route and policy change audit), `ntfy` (health degradation, approval, and sunset warning notifications), `composition_config` (reads route timeout and rate-limit config values)
- **Streaming**: Bytewax+NATS (`apg.composition.gateway.lifecycle`) for all lifecycle events; NATS subjects `apg.gateway.shadow.*` for shadow traffic, `apg.gateway.autoscale.*` for scaling signals

## Development Notes

- `SMService`, `SMEndpoint`, `SMPolicy`, `SMMetrics`, `SMAlert`, `SMTopology`, `SMConfiguration`, `SMCertificate`, `SMSecurityPolicy`, and `SMRateLimiter` all have their `metadata` attribute patched as a property at module load time via the `_get_model_metadata` / `_set_model_metadata` pattern; avoid adding a mapped `metadata` column to any new models in this module.
- The `SMLoadBalancer` model is not directly bound to a route; it is associated with a service. Apply circuit breaker and health check settings at the load balancer level, not per-endpoint.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (SQLAlchemy + Pydantic models), `service.py` (lifecycle operations + 8 new async methods), `api.py` (API helpers), `views.py` (UI model helpers).
- `WORLD_CLASS_IMPROVEMENTS.md` documents 15 architectural improvements with competitor references.
