# NCOD - No-Code/Low-Code Builder

NCOD is APG's governed no-code and low-code application composition capability.
It gives tenants a deterministic app library, screen composer, component
catalog, data modeler, workflow binding surface, script and connector extension
model, AI builder-agent roster, validation gate, publish center, deployment
center, audit stream, UI route model, and theme contract.

The package is executable without external builder infrastructure. Production
systems attach through explicit APG adapters while this package keeps the
contract, local runtime behavior, and generated application evidence testable.

## What NCOD Provides

- Tenant-scoped app/project records with owners, RBAC policy, data-residency
  policy, theme, lifecycle state, versioning, and audit.
- Screen definitions with stable routes, layouts, composition metadata, and
  component relationships.
- Component placement with governed component types, bindings, order, and
  accessibility enforcement for interactive components.
- Business data models with fields and governance policy.
- Data source bindings with schema validation.
- WFLO workflow bindings with required trigger, workflow reference, and
  automation policy.
- SCPT script extension references with required script policy.
- External connector bindings with required connector policy.
- First-class AI builder agents for runtimes such as Codex, Claude Code,
  OpenCode, and Pi, with role, scope, registration, policy, and contribution
  disclosure.
- Validation, publishing, and deployment workflows with approval, production
  review, target runtime, and rollback-plan evidence.
- Bytewax lifecycle stream contract for batch and runtime builder mutations.

### New in v1.1 — Async-First Enhancements

- `async_create_app` — coroutine-native app creation for concurrent build pipelines.
- `infer_form_from_data_model` — scaffolds a complete form page from a `DataModelDefinition`.
- `clone_app` — deep-clones an app and all sub-resources to another tenant namespace.
- `validate_app_incremental` — content-hash-cached validation; only changed domains re-evaluated.
- `snapshot_app` / `restore_snapshot` — full app-graph serialization and atomic rollback.
- `preview_data_binding` — validates sample rows against a binding schema with field scores.
- `accessibility_audit` — automated WCAG 2.1 Level AA heuristic scan; replaces the boolean flag.
- `enforce_performance_budget` — per-page component and global binding quota enforcement.
- `app_diff` — logical diff between two snapshots (added / removed / modified per category).

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `capability_contract.py` | Configuration, deterministic rules, UI routes, theme, Bytewax stream contract, and adapter map. |
| `models.py` | Dataclasses for apps, screens, components, data models, data bindings, workflows, themes, scripts, connectors, AI agents, validations, releases, deployments, and audit events. |
| `builder_runtime.py` | Deterministic IDs, type normalization, accessibility checks, field/schema/theme validation, versioning, validation checks, and publish posture helpers. |
| `service.py` | In-process builder service that enforces tenant, owner, policy, AI-agent, publish, deployment, and state-change guardrails. Includes 10 async methods added in v1.1. |
| `api.py` | Payload-oriented helpers for app builder, modeler, workflows, agents, validation, publishing, deployment, and compatibility calls. |
| `views.py` | UI model helpers for dashboards, app library, builder, data modeler, workflow designer, publish/deploy centers, agents, audit, analytics, and settings. |
| `app.py` | Publishable APG package entrypoint generated from the capability contract. |

## Minimal Usage

```python
from capabilities.common.ncod.service import NcodService

service = NcodService()
tenant_id = "tenant-builder"

app = service.create_app(
	"field-service",
	tenant_id,
	"Field Service",
	"ops-platform",
	rbac_policy_ref="rbac:field-service",
	data_residency_policy_ref="residency:ke",
	accessibility_checked=True,
)
page = service.add_page("work-orders", tenant_id, app["id"], "Work Orders", "/work-orders")
service.add_component("orders-table", tenant_id, page["id"], "table", "Orders", accessibility_label="Orders table")
service.define_data_model(
	"work-order",
	tenant_id,
	app["id"],
	"Work Order",
	[{"name": "id", "type": "text"}, {"name": "status", "type": "text"}],
	"data-policy:work-order",
)
service.attach_workflow("dispatch", tenant_id, app["id"], "on_dispatch", "wflo:dispatch", "workflow-policy:dispatch")
service.register_builder_agent("codex-builder", tenant_id, app["id"], "Codex Builder", "codex", "app_architect", "screens,workflows", True)
validation = service.validate_app("validate-field-service", tenant_id, app["id"])
release = service.publish_app("release-field-service", tenant_id, app["id"], "production", True, "approval:release", True)
deployment = service.deploy_release("deploy-field-service", tenant_id, release["id"], "python", "apg://apps/field-service", True, "rollback:field-service")
```

## Key Service Methods

**Sync (existing)**: `describe`, `evaluate`, `create_app`, `add_page`, `add_component`,
`define_data_model`, `bind_data_source`, `attach_workflow`, `create_theme_variant`,
`add_script_extension`, `add_connector_binding`, `register_builder_agent`,
`validate_app`, `publish_app`, `deploy_release`, `change_app_state`, `app_template`,
`widget_library`, `data_connector`, `trigger_define`, `action_block`,
`condition_builder`, `preview_deploy`, `version_control_app`, `ncod_analytics`,
`dashboard_summary`

**Async (v1.1)**: `async_create_app`, `infer_form_from_data_model`, `clone_app`,
`validate_app_incremental`, `snapshot_app`, `restore_snapshot`, `preview_data_binding`,
`accessibility_audit`, `enforce_performance_budget`, `app_diff`

_(See `service.py` for complete signatures and docstrings.)_

## Guardrail Summary

NCOD denies operations that lack tenant context, app ownership, app name, theme,
RBAC policy, data-residency policy, screen routes, component screen attachment,
interactive accessibility labels, valid data model fields, data model policy,
valid binding schema, workflow trigger/reference/policy, publish approval,
passing validation, script policy, connector policy, deployment target,
deployment approval, rollback plan, AI builder-agent registration/runtime/scope/
disclosure, state-change reason, state-change audit, tenant isolation, or
Bytewax event-stream evidence for batch mutations.

Some rules return `require_review` for cases that should be routed to human or
policy review, including production changes and incomplete screen relationship
metadata.

## Focused Verification

Battery-conscious checks for this capability:

```bash
./.venv/bin/python -m py_compile capabilities/common/ncod/__init__.py capabilities/common/ncod/models.py capabilities/common/ncod/builder_runtime.py capabilities/common/ncod/service.py capabilities/common/ncod/api.py capabilities/common/ncod/views.py capabilities/common/ncod/capability_contract.py capabilities/common/ncod/app.py capabilities/common/ncod/test_capability_contract.py capabilities/common/ncod/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ncod/test_capability_contract.py capabilities/common/ncod/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.ncod import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ncod --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ncod --json
```
