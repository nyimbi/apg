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
- Snapshot/rollback, incremental validation, app diffing, accessibility audit,
  performance budget enforcement, fine-grained page RBAC, webhook dispatch,
  micro-app federation, bulk component operations, cost estimation, and portable
  schema export/import.

## World-Class Enhancements (v2.0)

The following 15 improvements were implemented and are available in v2.0:

| # | Feature | Key Method | Benefit |
|---|---------|------------|---------|
| 1 | Async-First Service Layer | `async_create_app` | Coroutine-native app creation; zero-overhead for FastAPI/Bytewax callers |
| 2 | Persistent Backend via Async SQLAlchemy | _(adapter interface)_ | `NcodStore` interface; `MemoryNcodStore` for tests; asyncpg for production |
| 3 | Event Streaming via Domain Events | _(event bus adapter)_ | Typed `NcodDomainEvent` queue; enables real-time dashboards and WFLO triggers |
| 4 | AI-Assisted Component Generation | _(Ollama adapter)_ | `generate_component_from_prompt` delegates to local LLM; returns validated `BuilderComponent` |
| 5 | Form Schema Inference from Data Model | `infer_form_from_data_model` | Scaffolds typed form components from a `DataModelDefinition`; eliminates manual form/model sync |
| 6 | Multi-Tenant App Cloning / Templating | `clone_app` | Deep-copies pages, components, models, workflows to a target tenant with deterministic IDs |
| 7 | Incremental Validation with Per-Check Caching | `validate_app_incremental` | Content-hash caching per domain; only changed domains are re-evaluated |
| 8 | Visual Diff Between Versions | `app_diff` | Structured added/removed/modified change-set across pages, components, models, and bindings |
| 9 | Role-Based Builder Permissions per Page | `set_page_permissions` | Fine-grained `roles_allowed`/`roles_denied` + conditions at the page level |
| 10 | Webhook / Notification Dispatch | `register_webhook` | HMAC-signed delivery to external systems on lifecycle events; wildcard event matching |
| 11 | Snapshot / Rollback of App State | `snapshot_app` / `restore_snapshot` | Full app-graph serialization; atomic rollback to any named snapshot |
| 12 | Data Pipeline Preview / Sample Data Injection | `preview_data_binding` | Validates sample rows against binding schema; returns per-field conformance scores |
| 13 | Accessibility Audit Report | `accessibility_audit` | WCAG 2.1 Level AA heuristics across all interactive components; produces a compliance score |
| 14 | App Performance Budget Enforcement | `enforce_performance_budget` | Per-page component and global binding quota enforcement; audit events for violations |
| 15 | Composable Micro-App Federation | `federate_app` | Embeds a remote app's route tree under a mount point in the host app; independent lifecycle |

Additional async methods added alongside these improvements:

- `bulk_add_components` — atomically add multiple components to a page with pre-validation
- `compute_app_cost_estimate` — Decimal-precise monthly cost estimate across all resource types
- `export_app_schema` / `import_app_schema` — portable JSON bundle with content-hash integrity
- `get_app_tab_summary` — seven-tab domain breakdown for Flask-AppBuilder detail views

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `capability_contract.py` | Configuration, deterministic rules, UI routes, theme, Bytewax stream contract, and adapter map. |
| `models.py` | Dataclasses for apps, screens, components, data models, data bindings, workflows, themes, scripts, connectors, AI agents, validations, releases, deployments, and audit events. |
| `builder_runtime.py` | Deterministic IDs, type normalization, accessibility checks, field/schema/theme validation, versioning, validation checks, and publish posture helpers. |
| `service.py` | In-process builder service enforcing tenant, owner, policy, AI-agent, publish, deployment, and state-change guardrails. Includes all sync methods and 19 async methods (v1.1 + v2.0). |
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

## New Methods

### `infer_form_from_data_model` — Auto-scaffold a form from a data model

```python
import asyncio
from capabilities.common.ncod.service import NcodService

service = NcodService()
service.define_data_model(
    "dm-invoice", "tenant-a", "app-1", "Invoice",
    [{"name": "amount", "type": "number"}, {"name": "status", "type": "enum", "options": ["draft", "sent"]}],
    "policy://invoice",
)
page = service.add_page("pg-invoice-form", "tenant-a", "app-1", "Invoice Form", "/invoices/new",
                        metadata={"relationships": True})

result = asyncio.run(service.infer_form_from_data_model(
    tenant_id="tenant-a",
    app_id="app-1",
    model_id="dm-invoice",
    page_id="pg-invoice-form",
))
# result["components_created"] == 2  (input for amount, select for status)
# result["binding_id"] is the auto-created DataBinding ID
```

### `clone_app` — Multi-tenant app cloning

```python
cloned = asyncio.run(service.clone_app(
    source_app_id="app-crm-prod",
    source_tenant_id="tenant-platform",
    target_tenant_id="tenant-customer-xyz",
    new_app_name="CRM for XYZ",
    new_owner="onboarding-team",
    deep=True,   # copies pages, components, models, workflows; skips builder agents
))
# cloned["clone_counts"] == {"pages": 4, "components": 22, "data_models": 3, "workflow_bindings": 5}
```

### `snapshot_app` / `restore_snapshot` — Safe experimentation with rollback

```python
snap = asyncio.run(service.snapshot_app(
    tenant_id="tenant-a",
    app_id="app-1",
    snapshot_id="snap-before-restructure",
    label="before dashboard redesign",
))

# ... destructive restructuring ...

restored = asyncio.run(service.restore_snapshot(
    tenant_id="tenant-a",
    snapshot_id="snap-before-restructure",
    restore_reason="Rolled back dashboard redesign — performance regression",
))
# restored["restore_counts"] shows how many records were reloaded per category
```

### `accessibility_audit` — Automated WCAG 2.1 Level AA scan

```python
report = asyncio.run(service.accessibility_audit(
    tenant_id="tenant-a",
    app_id="app-1",
))
# report["compliance_score"] == 0.92  (0.0–1.0)
# report["findings"]  — per-component issues with severity (error/warning/info)
# report["recommend_accessibility_checked"] == True  if score >= 0.9 and no errors
```

### `federate_app` — Micro-app federation (Module Federation pattern)

```python
# Both apps must be in validated/published/deployed status
mount = asyncio.run(service.federate_app(
    tenant_id="tenant-platform",
    host_app_id="app-portal",
    remote_app_id="app-reporting-module",
    mount_route="/reports",
    remote_tenant_id="tenant-reporting-team",
    policy_ref="policy://federation-reporting",
))
# mount["mount_id"] identifies the FederatedMount record
# mount["mount_route"] == "/reports"
```

## Key Service Methods

**Sync (v1.0)**: `describe`, `evaluate`, `create_app`, `add_page`, `add_component`,
`define_data_model`, `bind_data_source`, `attach_workflow`, `create_theme_variant`,
`add_script_extension`, `add_connector_binding`, `register_builder_agent`,
`validate_app`, `publish_app`, `deploy_release`, `change_app_state`, `app_template`,
`widget_library`, `data_connector`, `trigger_define`, `action_block`,
`condition_builder`, `preview_deploy`, `version_control_app`, `ncod_analytics`,
`dashboard_summary`

**Async (v1.1)**: `async_create_app`, `infer_form_from_data_model`, `clone_app`,
`validate_app_incremental`, `snapshot_app`, `restore_snapshot`, `preview_data_binding`,
`accessibility_audit`, `enforce_performance_budget`, `app_diff`

**Async (v2.0)**: `set_page_permissions`, `register_webhook`, `federate_app`,
`bulk_add_components`, `compute_app_cost_estimate`, `export_app_schema`,
`import_app_schema`, `get_app_tab_summary`

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

Additional v2.0 guardrails: `set_page_permissions` rejects overlapping
allowed/denied role sets; `restore_snapshot` requires a non-empty reason;
`federate_app` requires both apps to be in a validated-or-later status;
`import_app_schema` verifies `content_hash` before writing any records;
`register_webhook` enforces a non-empty `event_types` list and a `retry_limit`
of 1–10.

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
