# API/Service Registry Capability Specification

- **Capability Name**: API/Service Registry
- **Capability ID**: `regy`
- **Category**: common
- **Version**: 1.0.0
- **Theme**: `regy_service_catalog`

## Purpose

REGY is the APG API and service registry. It gives generated applications and
capability packages a tenant-scoped catalog for service registration, discovery,
health posture, circuit-breaker metadata, API version governance, event capture,
metrics, gateway synchronization, UI route composition, and deterministic
registry policy evaluation.

The package is executable today through the Python package runtime:

- `models.py` defines service endpoints, health checks, circuit breakers,
  versions, instances, registrations, discovery queries/results, health
  status, events, and metrics.
- `service.py` provides `ServiceRegistryService` with async registration,
  deregistration, tenant-scoped discovery, health updates, health aggregation,
  metrics lookup, event capture, discovery caching, and APG adapter hooks.
- `api.py` exposes a Flask-compatible REST surface with local fallback classes
  for dependency-light testing when Flask-RESTX is unavailable.
- `views.py` exposes Flask-AppBuilder service catalog, registration,
  discovery, health, and dashboard view contracts.
- `capability_contract.py` publishes configuration defaults, deterministic
  rule definitions, UI routes, and theme tokens for APG composition tooling.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` provide side-effect-free publish and release evidence.

## Provided Services

- `service_registration`
- `service_discovery`
- `health_monitoring`
- `load_balancing`
- `circuit_breaking`
- `api_versioning`
- `regy_operations`

## Required Services And Adapter Boundaries

- `tenant_context` scopes every executable registry operation.
- `auth` is an optional adapter for service registration and deregistration
  permissions.
- `conf` is an optional adapter for dynamic registry configuration.
- `moni` is an optional adapter for metrics publication.
- `audl` is an optional adapter for audit event export.
- `apig` is the gateway synchronization boundary for routing and discovery
  updates.

The package keeps those integrations behind adapter imports and local fallbacks
so the registry lifecycle can be tested without live platform services.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant-specific overrides are merged into the
default configuration.

Important configuration groups:

- `registration`: owner, health endpoint, API version, and contract schema
  requirements.
- `discovery`: service discovery enablement, cache TTL, healthy-instance
  preference, and cross-tenant discovery policy.
- `health`: active health checks, default interval, failure threshold, and
  degraded-service gateway behavior.
- `governance`: tenant context, registration auditing, breaking-change review,
  and duplicate-name policy.
- `routing`: gateway synchronization, load-balancing metadata, and circuit
  breaker enablement.
- `ui`: service catalog, discovery console, health dashboard, and version
  manager enablement.
- `theme`: default theme and tenant override policy.

## Rules

REGY evaluates deterministic policy rules through `CapabilityRuleEngine`.

| Rule | Decision | Runtime intent |
| --- | --- | --- |
| `tenant_context_required` | deny | every registry operation must carry tenant context |
| `service_registration_requires_owner` | deny | registration requires an owner for audit and support |
| `service_registration_requires_health_endpoint` | deny | registration must include health endpoint metadata |
| `duplicate_service_name_blocked` | deny | duplicate service names are blocked within tenant and namespace |
| `breaking_change_requires_review` | require_review | breaking API changes require compatibility review |
| `cross_tenant_discovery_denied` | deny | cross-tenant discovery is blocked by default |

The service implementation also enforces duplicate service names, tenant
isolation in discovery, instance tenant matching, port validation, health-check
shape validation, and APG permission checks when the auth adapter is available.

## Runtime Lifecycle

The core lifecycle is:

1. Initialize `ServiceRegistryService` for a tenant.
2. Register a service with name, display name, service type, namespace,
   environment, optional versions, tags, endpoints, and instances.
3. Validate duplicate service names, instance tenant ownership, ports, and
   health-check configuration.
4. Store the service and initialize health posture for registered instances.
5. Capture a `service_registered` event and optional monitoring metric.
6. Discover services through tenant-scoped filters such as type, namespace,
   environment, status, health, tags, labels, pagination, and optional
   intelligent ranking.
7. Update and aggregate health posture for service instances.
8. Capture health transition events when status changes.
9. Deregister services and clean registry, health, metrics, and cache state.

This lifecycle is local and deterministic. Production integrations can attach
through the auth, configuration, monitoring, audit, and gateway adapter
boundaries without changing APG source or package contracts.

## UI

The package exposes these APG Python UI routes through the contract and semantic
model:

| Route | Path | Component | Permission |
| --- | --- | --- | --- |
| `dashboard` | `/regy/dashboard` | `RegistryDashboard` | `regy:view` |
| `services` | `/regy/services` | `ServiceCatalog` | `regy:view_services` |
| `register` | `/regy/register` | `ServiceRegistration` | `regy:register_service` |
| `discovery` | `/regy/discovery` | `DiscoveryConsole` | `regy:discover` |
| `health` | `/regy/health` | `ServiceHealthDashboard` | `regy:view_health` |
| `versions` | `/regy/versions` | `ServiceVersionManager` | `regy:manage_versions` |
| `gateway_sync` | `/regy/gateway-sync` | `GatewaySyncView` | `regy:sync_gateway` |
| `settings` | `/regy/settings` | `RegistrySettings` | `regy:admin` |

`views.py` adds Flask-AppBuilder-facing forms and screens for service
registration, service instances, discovery, health checks, circuit breakers,
service lists, service detail, and registry dashboard workflows.

## Theme

REGY uses `regy_service_catalog` with compact operational density:

- service catalog rows use network icons, health pills, and version bands;
- discovery results use instance stacks and endpoint chips;
- health timelines use probe timelines and failure-threshold indicators;
- version compatibility panels use matrix layouts and breaking-change chips.

Theme tokens are published by `CapabilityTheme` so composed applications can
apply consistent status, warning, danger, surface, text, radius, and density
values.

## Publish And Composition Evidence

The package publishes:

- semantic model: `semantic_model.json` and `app.semantic_model()`;
- component manifest: `app.component_manifest()`;
- self-test: `app.self_test()`;
- package metadata: `package_manifest.json`;
- release evidence: `release_report.json`;
- capability contract: `get_capability_contract()`;
- composition registration: `register_capability()`.

## Focused Verification

Use battery-conscious package verification first:

```bash
./.venv/bin/python -m py_compile capabilities/common/regy/__init__.py capabilities/common/regy/models.py capabilities/common/regy/service.py capabilities/common/regy/api.py capabilities/common/regy/views.py capabilities/common/regy/capability_contract.py capabilities/common/regy/app.py capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
```

Run broader REGY tests when changing service behavior, API endpoints,
Flask-AppBuilder views, or enhancement modules:

```bash
./.venv/bin/pytest -q capabilities/common/regy/tests
```

## Known Non-Goals

- Live service mesh registration, gateway writes, health probes, metrics
  export, audit export, and APG auth enforcement remain adapter integrations.
- AI-assisted ranking, predictive scaling, and anomaly detection metadata are
  modeled locally; production model execution belongs behind explicit
  enhancement or monitoring adapters.
- The package does not require a live database for focused package proof.
