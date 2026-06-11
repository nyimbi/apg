# Central Configuration Management

© 2025 Datacraft — Nyimbi Odero

## Overview

Central Configuration Management is APG's shared configuration plane for the composition layer. It provides tenant-aware namespaces, versioned configuration values with schema validation, production deployment approval workflows, reusable template libraries, continuous drift detection, and a suite of advanced async operations including config scheduling, lint enforcement, ancestry-chain resolution, and cryptographic audit chain verification.

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
| config_scheduling | Schedule config changes to activate at a future UTC timestamp |
| config_lint | Best-practice lint enforcement before deployment |
| config_hierarchy_resolution | Walk ancestor namespace chains to resolve inherited values |
| config_audit_verification | Structural integrity check of the immutable audit event chain |
| config_compliance | Automated compliance posture check (secret refs, schema coverage) |

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
| observability.event_stream | string | "apg.composition.config.lifecycle" | Bytewax/NATS stream name |

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
| schedule | /composition-config/schedule | POST | composition_config:edit | Scheduling |
| lint | /composition-config/lint | POST | composition_config:view | Quality |
| compliance | /composition-config/compliance | GET | composition_config:admin | Compliance |

REST API prefix: `/composition-config/api/v1`

## Service Methods

### Core CRUD

| Method | Description |
|--------|-------------|
| `get_config(namespace, key, tenant_id, version?)` | Retrieve config value; optionally fetch a specific historical version |
| `set_config(namespace, key, value, tenant_id, ...)` | Create or update a config key; auto-increments version and snapshots history |
| `delete_config(namespace, key, tenant_id, ...)` | Soft-delete a config key; retained in audit history |
| `list_configs(namespace, tenant_id, filters?, ...)` | List all configs in a namespace with optional status/restricted/secret filters |
| `bulk_config_import(namespace, config_map, tenant_id, ...)` | Atomically import many keys; returns created/updated/failed counts |

### Lifecycle

| Method | Description |
|--------|-------------|
| `validate_configuration(configuration_id, actor_id, evidence)` | Attach validation evidence; transitions status to validated |
| `activate_configuration(configuration_id, actor_id, ...)` | Transition validated config to active; required before deployment |
| `update_configuration(configuration_id, actor_id, value, ...)` | Update value; resets to draft and snapshots previous version |
| `deploy_configuration(deployment_key, tenant_id, config_id, environment, ...)` | Deploy an active config to an environment; enforces approval and canary gates |
| `rollback_configuration(deployment_id, actor_id, reason, ...)` | Roll back a deployment; requires reason and Bytewax stream routing |
| `rollback_config(namespace, key, version, tenant_id, reason, ...)` | Roll back a config value to a specific previous version |

### Versioning and History

| Method | Description |
|--------|-------------|
| `config_version_history(namespace, key, tenant_id)` | Full chronological version history including current version |
| `config_diff(namespace, tenant_a, tenant_b)` | Compare a namespace across two tenants; returns only-in-a, only-in-b, and differing keys |
| `validate_schema(namespace, schema_definition, tenant_id, ...)` | Register and structurally validate a JSON Schema for a namespace |

### Async Operations (New)

| Method | Description |
|--------|-------------|
| `async_get_config(namespace, key, tenant_id, version?)` | Async wrapper; use with `asyncio.gather` for parallel reads |
| `async_set_config(namespace, key, value, tenant_id, ...)` | Async wrapper; safe for concurrent write workflows |
| `async_bulk_import(namespace, config_map, tenant_id, ...)` | Async sequential bulk import preserving audit ordering |
| `async_config_diff(namespace, tenant_a, tenant_b)` | Async diff; compose with other async calls in fan-out patterns |
| `resolve_config(namespace, key, tenant_id, ancestor_namespaces?)` | Walk ancestor chain; returns first matching value with provenance |
| `schedule_config_change(namespace, key, value, ..., effective_at)` | Create a scheduled config; marks status as 'scheduled' until activation time |
| `activate_scheduled_configs(tenant_id, reference_time?)` | Activate all scheduled configs whose effective_at <= reference_time |
| `lint_config(namespace, key, tenant_id, environment?)` | Lint a config against built-in best-practice rules; returns findings |
| `verify_audit_chain(tenant_id)` | Structural integrity check of the audit event chain |
| `export_records(tenant_id, format?)` | Export all configs as JSON or env-style flat dict |
| `health_check(tenant_id?)` | Service health with storage counts |
| `compliance_check(tenant_id?)` | Compliance posture: secret refs, schema coverage, audit chain |

### Templates, Drift, and Agents

| Method | Description |
|--------|-------------|
| `create_template(template_key, tenant_id, name, ...)` | Create reusable config template; shared templates require review |
| `record_drift(tenant_id, configuration_id, expected_version, ...)` | Record a drift event with severity classification |
| `register_config_agent(tenant_id, name, runtime, role, ...)` | Register an AI config agent; validated against supported runtimes and roles |
| `validate_agent_config_action(tenant_id, agent_id, action, ...)` | Gate privileged agent actions on human approval |
| `validate_batch_configuration_change(tenant_id, change_count, ...)` | Validate a batch change meets Bytewax/NATS stream requirements |
| `config_analytics(period, tenant_id?)` | Aggregated namespace, config, deployment, drift, and version statistics |

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
| deployment_requires_bytewax_stream | deploy_configuration not via bytewax/NATS | deny |
| rollback_requires_reason | rollback_configuration without reason | deny |
| rollback_requires_bytewax_stream | rollback not via bytewax/NATS | deny |
| shared_template_requires_review | create shared template without review | require_review |
| batch_change_requires_bytewax | batch_configuration_change not via bytewax/NATS | deny |
| config_agent_runtime_supported | register_config_agent with unsupported runtime | deny |
| config_agent_role_supported | register_config_agent with unsupported role | deny |
| privileged_agent_config_action_requires_human_approval | agent proposes privileged action without human approval | deny |

## Lint Rules

Built-in rules applied by `lint_config`:

| Rule | Severity | Condition |
|------|----------|-----------|
| no_debug_in_production | warning | log_level is DEBUG/TRACE in production environment |
| positive_timeout_required | error | timeout/ttl/interval key has value <= 0 |
| sensitive_key_must_be_secret | error | key name contains password/secret/token/api_key but secret=False |
| no_null_in_production | error | config value is null/None in production environment |

Additional rules can be added via `register_lint_rule` (planned, see `WORLD_CLASS_IMPROVEMENTS.md` I14).

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

Events emitted to the composition event stream via Bytewax + NATS (`apg.composition.config.lifecycle`).

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
| config_change_scheduled | Config change scheduled for future activation |
| scheduled_config_activated | Scheduled config activated by the scheduler |
| async_bulk_config_imported | Async bulk import completed |
| config_linted | Lint check completed; includes finding count |

Stream states: `draft → scheduled → validated → active → release_pending → deployed → rolled_back → drifted → blocked`

## Config Scheduling Workflow

```
schedule_config_change(...)    # status → "scheduled"
         ↓
activate_scheduled_configs(...)  # called by bytewax dataflow at effective_at
         ↓
validate_configuration(...)    # attach evidence
         ↓
activate_configuration(...)    # status → "active"
         ↓
deploy_configuration(...)      # deploy to environment
```

## Edge Cases Handled

- `ConfigurationRecord.to_dict()` silently redacts secret values at serialization time; callers cannot accidentally log or return plaintext secrets through the standard dict conversion path.
- High-impact deployments that lack canary evidence produce `require_review` (not `deny`), allowing an emergency bypass path via explicit human review recording, while still creating an audit trail.
- Rollback events are also required to route through Bytewax/NATS, ensuring that roll-forwards and roll-backs are symmetric in the audit trail and cannot silently diverge.
- Shared templates require review before publication, preventing unreviewed configuration bundles from propagating across tenants via the template library.
- Drift records carry both `expected_version` and `observed_version`, enabling root-cause analysis without re-querying the configuration history.
- `resolve_config` walks ancestor namespaces in order and returns the first match; the `resolved_from` field in the result enables provenance tracing for inherited values.
- `lint_config` returns `passed=False` only when at least one `error`-severity finding is present; `warning`-only results return `passed=True` to avoid blocking deployments for non-critical issues.

## Composability

- **Upstream**: `composition_access` (policy enforcement on all writes), `auth` (operator identity), `conf` (self-bootstrapping secret references)
- **Downstream**: `composition_orchestration` (reads environment config during workflow execution), `composition_gateway` (reads route and traffic policy config), all domain capabilities that consume environment-specific settings
- **Peer**: `audl` (long-term change audit), `ntfy` (approval and drift alert notifications), `composition_events` (receives lifecycle events via Bytewax+NATS)
- **Streaming**: All deployment and rollback operations route through Bytewax dataflows publishing to NATS JetStream `apg.composition.config.*` subjects

## Development Notes

- Record models are dataclasses with no ORM dependency, allowing the service layer to run in environments where SQLAlchemy is not available.
- The `stable_id` function produces deterministic IDs from a SHA-256 hash of the logical key parts; this supports idempotent replay of configuration creation events.
- Async methods are thin wrappers over synchronous service logic; they are provided to enable `asyncio.gather` fan-out in composition contexts without introducing thread-pool complexity.
- Key files: `capability_contract.py` (executable contract and rule engine), `models.py` (dataclass records), `service.py` (lifecycle operations), `api.py` (API helpers), `views.py` (UI model helpers), `app.py` (package self-test).
- `WORLD_CLASS_IMPROVEMENTS.md` documents 15 concrete improvement paths toward production-grade capability, each with implementation strategy and competitor reference.
