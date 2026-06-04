# Central Configuration Management

## Overview

Central Configuration Management is APG's shared configuration plane for the composition layer. It provides tenant-aware namespaces, versioned configuration values with schema validation, production deployment approval workflows, reusable template libraries, and continuous drift detection across all environments.

The business value is a single source of truth for configuration across all composed capabilities. Every configuration change goes through a structured lifecycle — draft, validate, activate, deploy — with canary evidence required for high-impact changes and mandatory approval for production. Secret values are never stored in plaintext; they are replaced with vault references at creation time, and redacted on read.

## Capability ID

`composition_config`  Version: see `package_manifest.json`

## Provides

| Service | Description |
|---------|-------------|
| configuration_namespace_registry | Define and own tenant/environment/capability-scoped namespaces |
| configuration_value_lifecycle | Create, validate, activate, update, and rollback versioned config values |
| configuration_schema_validation | Enforce JSON schemas on restricted configuration values |
| configuration_release_workflows | Approval, canary evidence, and Bytewax-coordinated deployment |
| configuration_template_library | Reusable configuration bundles with variable schemas |
| configuration_drift_monitoring | Detect and surface expected-vs-observed version mismatches |
| config_agents | AI agent workbench with approved runtimes and human approval gates |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authenticate operators managing configuration |
| audl | Persist immutable change audit records |
| ntfy | Send deployment approval and drift alert notifications |
| registry | Register this capability in the global catalog |
| composition_access | Enforce policy on all configuration write operations |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant scope for all operations |
| namespaces.owner_required | bool | true | Namespace must have an accountable owner |
| namespaces.environment_required | bool | true | Namespace must declare an environment |
| configurations.schema_required_for_restricted | bool | true | Restricted configs must have a JSON schema |
| configurations.secret_reference_required | bool | true | Secret configs must use a vault reference |
| configurations.versioning_enabled | bool | true | Track version history for all config values |
| configurations.drift_detection_enabled | bool | true | Enable expected-vs-observed diff monitoring |
| deployments.approval_required_for_production | bool | true | Production deployments require approval |
| deployments.canary_required_for_high_impact | bool | true | High-impact changes require canary evidence |
| config_agents.max_autonomous_scope | string | "recommend_and_validate" | Ceiling on autonomous agent actions |
| observability.event_stream | string | "apg.composition.config.lifecycle" | Bytewax stream name |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /composition-config/dashboard | GET | composition_config:view | Overview |
| namespaces | /composition-config/namespaces | GET/POST | composition_config:admin | Namespaces |
| configurations | /composition-config/configurations | GET/POST | composition_config:edit | Configuration |
| releases | /composition-config/releases | GET/POST | composition_config:release | Release |
| templates | /composition-config/templates | GET/POST | composition_config:edit | Configuration |
| drift | /composition-config/drift | GET | composition_config:operate | Operations |
| agents | /composition-config/agents | GET/POST | composition_config:admin | Automation |
| settings | /composition-config/settings | GET/PUT | composition_config:admin | Administration |

REST API prefix: `/composition-config/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| namespace_requires_owner | register_namespace without owner | deny |
| namespace_requires_environment | register_namespace without environment | deny |
| configuration_requires_policy | write operation without policy attached | deny |
| restricted_configuration_requires_schema | create_configuration on restricted config without schema | deny |
| secret_configuration_requires_reference | create_configuration for secret without vault ref | deny |
| activation_requires_validation | activate_configuration without validation evidence | deny |
| production_deployment_requires_approval | deploy to production without approval | deny |
| high_impact_deployment_requires_canary | deploy high-impact config without canary evidence | require_review |
| deployment_requires_bytewax_stream | deploy_configuration not via bytewax | deny |
| rollback_requires_reason | rollback_configuration without reason | deny |
| rollback_requires_bytewax_stream | rollback not via bytewax | deny |
| shared_template_requires_review | create shared template without review | require_review |
| batch_change_requires_bytewax | batch_configuration_change not via bytewax | deny |
| config_agent_runtime_supported | register_config_agent with unsupported runtime | deny |
| config_agent_role_supported | register_config_agent with unsupported role | deny |
| privileged_agent_config_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| ConfigNamespaceRecord | id, tenant_id, name, environment, owner_id, path_prefix, capability_id, status |
| ConfigurationRecord | id, tenant_id, namespace_id, key_path, value, version, owner_id, restricted, secret, schema, secret_reference, status, validation_evidence |
| ConfigDeploymentRecord | id, tenant_id, configuration_id, environment, impact_level, status, approved_by, canary_evidence, event_stream |
| ConfigTemplateRecord | id, tenant_id, name, owner_id, values, variable_schema, shared, reviewed_by, status |
| ConfigDriftRecord | id, tenant_id, configuration_id, expected_version, observed_version, severity, status |
| ConfigAgentRecord | id, tenant_id, name, runtime, role, instructions, status |
| ConfigAuditEventRecord | id, tenant_id, event_type, entity_id, actor_id, created_at |

`ConfigurationRecord.to_dict()` automatically redacts `value` for secret configurations, replacing it with `{"redacted": True, "secret_reference": ...}`.

## Streaming Events

Events emitted to the composition event stream via Bytewax (`apg.composition.config.lifecycle`).

| Event | Trigger |
|-------|---------|
| namespace_registered | Configuration namespace created |
| configuration_created | New configuration value record created |
| configuration_validated | Validation evidence attached and accepted |
| configuration_activated | Configuration moves from draft to active |
| configuration_deployed | Deployment record committed to an environment |
| configuration_rolled_back | Rollback operation completed |
| template_created | Reusable template added to library |
| drift_detected | Expected-vs-observed version mismatch found |
| config_agent_registered | New configuration agent registered |

Stream states: `draft → validated → active → release_pending → deployed → rolled_back → drifted → blocked`

## Edge Cases Handled

- `ConfigurationRecord.to_dict()` silently redacts secret values at serialization time; callers cannot accidentally log or return plaintext secrets through the standard dict conversion path.
- High-impact deployments that lack canary evidence produce `require_review` (not `deny`), allowing an emergency bypass path via explicit human review recording, while still creating an audit trail.
- Rollback events are also required to route through Bytewax, ensuring that roll-forwards and roll-backs are symmetric in the audit trail and cannot silently diverge.
- Shared templates require review before publication, preventing unreviewed configuration bundles from propagating across tenants via the template library.
- Drift records carry both `expected_version` and `observed_version`, enabling root-cause analysis without re-querying the configuration history.

## Composability

- **Upstream**: `composition_access` (policy enforcement on all writes), `auth` (operator identity), `conf` (self-bootstrapping secret references)
- **Downstream**: `composition_orchestration` (reads environment config during workflow execution), `composition_gateway` (reads route and traffic policy config), all domain capabilities that consume environment-specific settings
- **Peer**: `audl` (long-term change audit), `ntfy` (approval and drift alert notifications), `composition_events` (receives lifecycle events via Bytewax)

## Development Notes

- Record models are dataclasses with no ORM dependency, allowing the service layer to run in environments where SQLAlchemy is not available.
- The `stable_id` function produces deterministic IDs from a SHA-256 hash of the logical key parts; this supports idempotent replay of configuration creation events.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (dataclass records), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers), `app.py` (package self-test).
