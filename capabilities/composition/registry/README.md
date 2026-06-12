# Capability Registry

## Overview

The Capability Registry is the authoritative catalog and governance service for all APG capabilities. It stores capability metadata, manages dependency graphs with cycle detection, validates composition blueprints, governs version releases with compatibility evidence, and coordinates marketplace publication — all within the multi-tenant APG composition layer.

The business value is a single place to discover, compose, and govern APG capabilities. Dependency cycle detection prevents invalid compositions from being deployed. Version compatibility governance ensures that releases include evidence of backward compatibility before existing consumers are affected. Marketplace publication governance ensures that shared capabilities are documented and reviewed before being made available to other tenants.

## Capability ID

`composition_registry`  Version: see `package_manifest.json`

## Features

- Multi-tenant capability catalog with lifecycle governance (`discovered → registered → validated → active → deprecated → retired`)
- Dependency graph with cycle detection, transitive resolution, and version constraint tracking
- Composition blueprint validation and publication workflow
- Version compatibility governance with evidence requirements and deprecation migration plans
- Marketplace publication with review-gated workflow
- AI-enhanced registry agents for catalog curation, dependency review, and security review
- Bytewax-backed streaming events for all registry lifecycle changes
- Async bulk operations: `bulk_register_capabilities`, `bulk_create`, `bulk_delete`
- Registry analytics, compliance checks, and capability usage statistics
- Filesystem auto-discovery of capability `__init__.py` manifests
- Intent-ranked capability search with `_generate_capability_recommendations`
- Export to JSON or CSV via `export_registry` / `export_records`
- Audit event log with per-tenant query and replay

## Provides

| Service | Description |
|---------|-------------|
| capability_catalog_lifecycle | Register, update, validate, and retire capability metadata records |
| dependency_graph_management | Add dependency edges with version constraints and detect cycles |
| composition_blueprint_validation | Create and publish composition blueprints with validation evidence |
| version_compatibility_governance | Release versions with compatibility evidence and deprecation migration plans |
| marketplace_publication_governance | Review-gated publication of capabilities to the shared marketplace |
| registry_discovery | Search and discover capabilities by category, keyword, and composition keyword |
| registry_agents | AI agent workbench for catalog curation, dependency review, and security review |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate registry operators and curators |
| audl | Persist immutable registry change audit records |
| ntfy | Send deprecation, publication review, and security review notifications |
| composition_events | Receive and emit registry lifecycle events via Bytewax |
| composition_config | Read registry scan frequency and validation threshold configuration |
| composition_access | Enforce policy on all registry write operations |

## Quick Start

```python
from capabilities.composition.registry.service import CompositionRegistryService

svc = CompositionRegistryService()

# Register a capability
cap = svc.register_capability(
    capability_id="payments_v2",
    tenant_id="acme",
    name="Payments v2",
    domain="fintech",
    version="2.0.0",
    description="PCI-compliant payment processing",
    owner="payments-team",
    contract={"provides": ["process_payment", "refund"]},
)

# Discover capabilities
results = svc.discover_capabilities(tenant_id="acme", category="fintech")

# Compose and validate
comp = svc.create_composition(
    tenant_id="acme",
    name="Checkout Flow",
    composition_type="microservice",
    capability_ids=["payments_v2", "kyc_check"],
    owner="platform-team",
)
validation = svc.validate_composition(tenant_id="acme", capability_ids=["payments_v2", "kyc_check"])
```

## New Methods

### Bulk Register Capabilities

```python
result = await svc.bulk_register_capabilities(
    tenant_id="acme",
    capability_specs=[
        {"capability_id": "cap_a", "name": "Cap A", "domain": "finance", "version": "1.0.0",
         "description": "...", "owner": "team-a", "contract": {}},
        {"capability_id": "cap_b", "name": "Cap B", "domain": "ops", "version": "1.0.0",
         "description": "...", "owner": "team-b", "contract": {}},
    ],
)
# {"created_count": 2, "error_count": 0, "capabilities": [...], "errors": []}
```

### Registry Compliance Check

```python
report = await svc.registry_compliance_check(tenant_id="acme")
# {"total_capabilities": 42, "no_owner_count": 1, "no_contract_count": 0,
#  "compliance_rate_pct": 97.62, "checked_at": "2026-06-12T..."}
```

### Capability Usage Statistics

```python
stats = await svc.capability_usage_stats(tenant_id="acme")
# {"total_audit_events": 310, "unique_capabilities": 28,
#  "top_capabilities": [{"capability_id": "payments_v2", "event_count": 44}, ...]}
```

### Export Registry

```python
# JSON export
export = await svc.export_registry(tenant_id="acme", format="json")

# CSV export
export = await svc.export_registry(tenant_id="acme", format="csv")
# {"format": "csv", "record_count": 42, "content": "capability_id,name,...\n..."}
```

### Registry Analytics

```python
analytics = await svc.registry_analytics(tenant_id="acme", period="monthly")
# {"total_capabilities": 42, "by_domain": {"fintech": 12, "ops": 8, ...},
#  "by_status": {"active": 35, "deprecated": 4, ...}, "by_version": {...}}
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| catalog.owner_required | bool | true | Capabilities must have an accountable owner |
| catalog.category_required | bool | true | Capabilities must declare a category |
| catalog.contract_required | bool | true | Capabilities must reference an executable contract |
| catalog.provides_required | bool | true | Capabilities must declare at least one provided surface |
| dependencies.cycle_detection_enabled | bool | true | Detect dependency cycles at edge-add time |
| dependencies.version_constraint_required | bool | true | Dependency edges must specify a version constraint |
| composition_blueprints.validation_required | bool | true | Compositions require validation before publication |
| versions.compatibility_evidence_required | bool | true | Version releases require compatibility evidence |
| versions.migration_plan_required_for_deprecation | bool | true | Deprecations require a migration plan |
| marketplace.publication_review_required | bool | true | Marketplace publications require review |
| marketplace.documentation_required | bool | true | Marketplace publications require documentation |
| registry_agents.max_autonomous_scope | string | "recommend_validate_and_prepare" | Ceiling on autonomous agent actions |
| observability.event_stream | string | "apg.composition.registry.lifecycle" | Bytewax stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-registry/dashboard | GET | composition_registry:view | Overview |
| catalog | /composition-registry/catalog | GET/POST | composition_registry:manage_catalog | Catalog |
| dependencies | /composition-registry/dependencies | GET/POST | composition_registry:manage_dependencies | Graph |
| compositions | /composition-registry/compositions | GET/POST | composition_registry:compose | Compositions |
| versions | /composition-registry/versions | GET/POST | composition_registry:release | Release |
| marketplace | /composition-registry/marketplace | GET/POST | composition_registry:publish | Marketplace |
| rules | /composition-registry/rules | GET | composition_registry:govern | Governance |
| agents | /composition-registry/agents | GET/POST | composition_registry:admin | Automation |
| settings | /composition-registry/settings | GET/PUT | composition_registry:admin | Administration |

REST API prefix: `/composition-registry/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| registry_write_requires_policy | write operation without policy attached | deny |
| capability_requires_owner | register_capability without owner | deny |
| capability_requires_category | register_capability without category | deny |
| capability_requires_version | register_capability without version | deny |
| capability_requires_provides | register_capability without provided surfaces | deny |
| capability_requires_contract | register_capability without executable contract reference | deny |
| dependency_requires_target | add_dependency without target capability | deny |
| dependency_requires_type | add_dependency without dependency type | deny |
| dependency_requires_version_constraint | add_dependency without version constraint | deny |
| composition_requires_owner | create_composition without owner | deny |
| composition_requires_capabilities | create_composition without capabilities | deny |
| composition_publish_requires_validation | publish_composition without validation evidence | deny |
| version_release_requires_compatibility | release_version without compatibility evidence | deny |
| deprecation_requires_migration_plan | deprecate_capability without migration plan | deny |
| marketplace_publish_requires_review | publish_marketplace without review | require_review |
| marketplace_publish_requires_documentation | publish_marketplace without documentation | deny |
| registry_import_requires_bytewax | registry_import not via bytewax | deny |
| registry_event_requires_bytewax | registry_event not via bytewax | deny |
| registry_agent_runtime_supported | register_registry_agent with unsupported runtime | deny |
| registry_agent_role_supported | register_registry_agent with unsupported role | deny |
| privileged_agent_registry_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| CRCapability | capability_id, tenant_id, capability_code, capability_name, version, category, status, provides_services, composition_keywords, multi_tenant, ai_enhanced |
| CRDependency | dependency_id, capability_id, depends_on_id, dependency_type, version_constraint, version_min, version_max, load_priority, alternative_capabilities |
| CRComposition | composition_id, tenant_id, name, composition_type, version, validation_status, validation_results, deployment_config, is_template, is_public |
| CRCompositionCapability | comp_cap_id, composition_id, capability_id, version_constraint, required, load_order |
| CRVersion | version_id, capability_id, version_number, major_version, minor_version, patch_version, breaking_changes, compatible_versions, backward_compatible, security_audit_passed |
| CRMetadata | metadata_id, capability_id, metadata_type, metadata_key, metadata_value, is_searchable |
| CRRegistry | registry_id, tenant_id, auto_discovery_enabled, discovery_paths, scan_frequency_hours, validation_rules, max_composition_size |
| CRUsageAnalytics | usage_id, capability_id, usage_date, usage_count, composition_count, avg_response_time_ms |
| CRHealthMetrics | metric_id, capability_id, health_score, availability_pct, dependency_health_score, missing_dependencies, security_score |

Dependency types: `required`, `optional`, `recommended`, `conflicting`, `enhancing`.
Composition types: `erp_enterprise`, `industry_vertical`, `departmental`, `microservice`, `hybrid`, `custom`.
Capability statuses: `discovered → registered → validated → active → deprecated → retired`.

## Streaming Events

Events emitted to the composition event stream via Bytewax (`apg.composition.registry.lifecycle`).

| Event | Trigger |
|-------|---------|
| capability_registered | New capability added to the catalog |
| dependency_added | Dependency edge added between capabilities |
| composition_created | New composition blueprint created |
| composition_validated | Composition passes all validation checks |
| version_released | New capability version published with compatibility evidence |
| capability_deprecated | Capability marked deprecated with migration plan |
| marketplace_publication_prepared | Capability prepared for marketplace publication |
| registry_agent_registered | New registry agent registered |
| bulk_capabilities_registered | Bulk registration batch completed |
| registry_exported | Registry exported to JSON or CSV |
| registry_compliance_check_run | Compliance audit run |

Stream states: `draft → registered → validated → released → published → deprecated → retired`

## World-Class Enhancements (v2.0)

Fifteen targeted improvements drawn from production-grade registries (Consul, Netflix Eureka, AWS Service Catalog, Backstage, Apigee):

1. **Distributed Lease-Based Health Heartbeat** — TTL-lease health tracking per capability; stale leases auto-downgrade to `failing` without a full polling scan. (cf. Consul TTL checks, Kubernetes liveness probes)

2. **Semantic Versioning Constraint Solver** — `resolve_version_constraints` evaluates transitive `version_constraint` strings via `packaging.specifiers.SpecifierSet`, reporting satisfiability and conflict ranges. (cf. npm semver, Poetry resolver)

3. **Lazy-Loaded Filesystem Auto-Discovery** — `auto_discover_capabilities` walks `root_path` for `__init__.py` files, extracts metadata, and bulk-registers results. (cf. Backstage catalog auto-discovery, AWS Service Catalog portfolio import)

4. **Capability Scoring and Quality Gate** — `score_capability` evaluates owner, contract, provides, display_name, manifest_path, and health status; returns grade and pass/fail checklist. (cf. Spotify Backstage Scorecards)

5. **Transitive Dependency Impact Analysis** — `impact_analysis` runs reverse BFS from a capability to surface all direct/transitive dependents, affected compositions, and a risk level. (cf. Maven `dependency:tree`, Snyk dependency graph)

6. **Event Replay and Audit Reconstruction** — `replay_audit_to_snapshot` replays audit events up to a timestamp, reconstructing a point-in-time capability state snapshot. (cf. Kafka consumer replay, EventStore projections)

7. **Composition Diff and Migration Plan Generator** — `diff_compositions` compares two composition blueprints and returns added/removed/version-changed capabilities plus auto-generated migration notes. (cf. Terraform `plan`, `kubectl diff`)

8. **Multi-Tenant Capability Sharing and Visibility Control** — `grant_capability_access` creates cross-tenant sharing grants; `discover_capabilities` gains `include_shared=True`. (cf. AWS RAM, Azure Service Catalog shared galleries)

9. **Circuit-Breaker State Tracking per Capability** — `record_capability_failure` / `record_capability_success` drive closed/open/half-open state transitions; `health_check_all` includes `circuit_state`. (cf. Netflix Hystrix, Resilience4j)

10. **Signed Capability Manifests with Integrity Verification** — `sign_capability_manifest` computes SHA-256 of canonical JSON; `verify_capability_manifest` re-checks the hash. (cf. sigstore/cosign, SLSA provenance)

11. **Canary and Staged Rollout Tracking** — `create_rollout_plan` / `promote_rollout_stage` model progressive delivery stages per version record; `list_versions` gains `rollout_status`. (cf. Argo Rollouts, Flagger, LaunchDarkly)

12. **Capability Deprecation Notification Workflow** — `notify_deprecation_consumers` queries all downstream dependents and emits structured `deprecation_notice_sent` events with sunset date and migration guide. (cf. AWS deprecation notices, GCP API sunset headers)

13. **Composition Execution Dry-Run Validation** — `dry_run_composition` runs topological sort, contract surface matching, and conflict edge detection; returns `{executable, blockers, warnings, simulated_order}`. (cf. Terraform `-plan`, AWS CloudFormation change sets)

14. **Registry Federation and Peer Sync** — `sync_from_peer` pulls capability manifests from a peer registry endpoint, tagging records with `{source_registry, synced_at}`; `discover_capabilities` gains `include_federated=True`. (cf. Netflix Eureka federation, Consul WAN gossip)

15. **Capability Contract Test Runner** — `run_contract_tests` dynamically imports a capability's `contract_ref`, runs it against synthetic test contexts, and updates `health_status` based on outcome. (cf. Pact broker, Spring Cloud Contract)

## Edge Cases Handled

- The `composition_registry` capability registers itself in its own catalog at startup; this creates a circular bootstrap dependency that is resolved by initializing the registry record with a pre-computed contract reference before the rule engine is active.
- `CRCapability.metadata` is exposed as a Python property over `metadata_json` to avoid SQLAlchemy's reserved attribute name; the same pattern is applied to `CRDependency`, `CRComposition`, `CRVersion`, `CRRegistry`, `CRUsageAnalytics`, and `CRHealthMetrics`.
- `marketplace_publish_requires_review` produces `require_review` rather than `deny`; this allows the publication workflow to proceed if an explicit review record is provided, rather than requiring a separate pre-approval step.
- `conflicting` dependency type edges are valid records in the graph; they signal incompatibility between capabilities and are used by the composition engine to reject compositions that include both ends of a conflicting edge.
- `CRVersion.backward_compatible` and `forward_compatible` are separate boolean fields because a version can be backward-compatible without being forward-compatible; both are evaluated by downstream consumers when determining upgrade safety.
- `CRRegistry.max_dependency_depth` (default 10) limits the depth of transitive dependency resolution; compositions that require deeper graphs are rejected at validation time to prevent unbounded recursion in the dependency resolver.

## Composability

- **Upstream**: `composition_access` (policy enforcement on all writes), `composition_events` (receives registry lifecycle events via Bytewax), `auth` (operator identity)
- **Downstream**: All capabilities are registered here; `composition_orchestration` consults the registry to resolve cross-capability task contract references; `composition_gateway` reads capability binding metadata for service registration
- **Peer**: `audl` (long-term catalog change audit), `ntfy` (deprecation and publication review notifications), `composition_config` (registry scan and validation threshold configuration)

## Development Notes

- `CRCapability.composition_keywords` is a JSON list of trigger keywords used by the APG composition engine to match user intent to capabilities; keep this field populated for all capabilities intended for AI-assisted composition.
- `CRRegistry.auto_discovery_enabled` triggers periodic filesystem scans of `discovery_paths` to find new `capability_contract.py` files; the scan respects `excluded_paths` and runs at `scan_frequency_hours` intervals.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (SQLAlchemy models and enums), `service.py` (lifecycle operations), `registry.py` (discovery and search), `composer.py` (blueprint validation), `version_manager.py` (version governance), `marketplace.py` (publication governance).
- Injected async methods (`bulk_register_capabilities`, `registry_analytics`, `export_registry`, `health_check`, `registry_compliance_check`, `deprecate_capability`, `capability_usage_stats`) are attached to `CompositionRegistryService` at module load via direct class attribute assignment.
