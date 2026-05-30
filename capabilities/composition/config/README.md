# Central Configuration Management

Central Configuration Management is APG's shared configuration plane. It lets composed applications define tenant-aware namespaces, configuration values, schemas, release workflows, templates, drift evidence, AI-agent review lanes, UI surfaces, theme tokens, and Bytewax lifecycle events.

## What It Provides

- Namespace registry for tenants, environments, capability boundaries, owners, and path prefixes.
- Configuration lifecycle for create, validate, activate, update, deploy, rollback, and drift capture.
- Guardrails for restricted values, secret references, validation evidence, production approval, canary evidence, and Bytewax stream routing.
- Template library for reusable configuration bundles.
- First-class configuration agents for Codex, Claude Code, OpenCode, and Pi.
- APG UI contracts for dashboard, namespaces, configurations, releases, templates, drift, agents, and settings.
- Compact operational theme contracts for configuration editors and release boards.

## Basic Usage

```python
from capabilities.composition.config import CompositionConfigService

service = CompositionConfigService()

namespace = service.register_namespace(
	namespace_key="orders-prod",
	tenant_id="tenant-a",
	name="Orders Production",
	environment="production",
	owner_id="orders-owner",
	path_prefix="/orders/prod",
	capability_id="orders",
)

config = service.create_configuration(
	config_key="database",
	tenant_id="tenant-a",
	namespace_id=namespace["id"],
	key_path="/orders/prod/database",
	value={"pool_size": 20},
	owner_id="orders-owner",
	restricted=True,
	schema={"type": "object", "required": ["pool_size"]},
)

service.validate_configuration(config["id"], "orders-owner", "schema-check-1")
service.activate_configuration(config["id"], "orders-owner")

deployment = service.deploy_configuration(
	deployment_key="orders-db-release",
	tenant_id="tenant-a",
	configuration_id=config["id"],
	environment="production",
	impact_level="high",
	actor_id="release-manager",
	approved_by="security-owner",
	canary_evidence="canary-healthy",
	event_stream="bytewax",
)
```

## AI Agent Review

```python
agent = service.register_config_agent(
	tenant_id="tenant-a",
	name="Config Release Agent",
	runtime="codex",
	role="release_reviewer",
	instructions="Review production configuration releases.",
)

service.validate_agent_config_action(
	tenant_id="tenant-a",
	agent_id=agent["id"],
	action="recommend_production_release",
	privileged_scope=True,
	human_approval_recorded=True,
)
```

## UI Models

Use `views.py` helpers to drive APG screens:

- `dashboard_model()`
- `namespace_console_model()`
- `config_editor_model()`
- `release_board_model()`
- `template_library_model()`
- `drift_monitor_model()`
- `agent_workbench_model()`

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/composition/config/__init__.py capabilities/composition/config/capability_contract.py capabilities/composition/config/models.py capabilities/composition/config/service.py capabilities/composition/config/api.py capabilities/composition/config/views.py capabilities/composition/config/app.py capabilities/composition/config/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/composition/config/tests/test_package_contract.py
```
