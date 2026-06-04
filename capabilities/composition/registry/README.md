# Capability Registry

## Overview

The Capability Registry is the authoritative catalog and governance service for all APG capabilities. It stores capability metadata, manages dependency graphs with cycle detection, validates composition blueprints, governs version releases with compatibility evidence, and coordinates marketplace publication — all within the multi-tenant APG composition layer.

The business value is a single place to discover, compose, and govern APG capabilities. Dependency cycle detection prevents invalid compositions from being deployed. Version compatibility governance ensures that releases include evidence of backward compatibility before existing consumers are affected. Marketplace publication governance ensures that shared capabilities are documented and reviewed before being made available to other tenants.

## Capability ID

`composition_registry`  Version: see `package_manifest.json`

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

Stream states: `draft → registered → validated → released → published → deprecated → retired`

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
