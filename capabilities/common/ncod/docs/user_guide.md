# No-Code/Low-Code Builder — User Guide

**Capability ID**: `ncod` | **Domain**: `common` | **Version**: `1.1.0`

## Overview

NCOD is APG's governed no-code and low-code application composition capability.
It provides a deterministic app library, screen composer, component catalog,
data modeler, workflow binding surface, script and connector extension model,
AI builder-agent roster, validation gate, publish center, deployment center,
audit stream, UI route model, and theme contract — all as a standalone Python
package that runs without external builder infrastructure.

## Installation

```bash
pip install apg-common-ncod
```

## Provides

- `app_builder`
- `page_composer`
- `data_modeler`
- `workflow_binding`
- `script_extensions`
- `snapshot_restore` (v1.1)
- `form_inference` (v1.1)
- `accessibility_audit` (v1.1)
- `performance_budget` (v1.1)

## Requires

- `wflo`
- `scpt`
- `auth`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ncod/dashboard` | `ncod:view` | Overview |
| `/ncod/apps` | `ncod:manage_apps` | Apps |
| `/ncod/builder` | `ncod:build` | Build |
| `/ncod/pages` | `ncod:build` | Build |
| `/ncod/data-models` | `ncod:build` | Build |
| `/ncod/components` | `ncod:build` | Build |
| `/ncod/workflows` | `ncod:build` | Automation |
| `/ncod/publishing` | `ncod:publish` | Release |

---

## Core Workflow

### 1. Create an App

Every resource in NCOD is scoped to a `tenant_id`. Apps require an owner,
RBAC policy ref, data-residency policy ref, and a theme.

```python
import asyncio
from capabilities.common.ncod.service import NcodService

service = NcodService()
tenant_id = "acme-corp"

app = service.create_app(
    app_id="crm-app",
    tenant_id=tenant_id,
    name="CRM",
    owner="platform-team",
    rbac_policy_ref="rbac:crm",
    data_residency_policy_ref="residency:ke",
    accessibility_checked=False,
)
```

### 2. Add Pages and Components

```python
page = service.add_page(
    "contacts-page", tenant_id, app["id"],
    "Contacts", "/contacts",
    metadata={"relationships": True},
)
service.add_component(
    "contacts-table", tenant_id, page["id"],
    "table", "Contacts",
    accessibility_label="Contacts table",
)
service.add_component(
    "add-contact-btn", tenant_id, page["id"],
    "button", "Add Contact",
    props={"action_type": "create"},
    accessibility_label="Add Contact",
)
```

### 3. Define a Data Model

```python
service.define_data_model(
    "contact-model", tenant_id, app["id"],
    "Contact",
    fields=[
        {"name": "id", "type": "text"},
        {"name": "name", "type": "text"},
        {"name": "email", "type": "text"},
        {"name": "status", "type": "enum", "options": ["active", "inactive"]},
    ],
    policy_ref="data-policy:contact",
)
```

### 4. Auto-Scaffold a Form from a Data Model (v1.1)

`infer_form_from_data_model` inspects the model's fields and emits typed
`BuilderComponent` records (input, select) plus a `DataBinding` in one call —
no manual field-by-field recreation.

```python
form_page = service.add_page(
    "contact-form", tenant_id, app["id"],
    "Contact Form", "/contacts/new",
    layout="form",
    metadata={"relationships": True},
)
result = asyncio.run(service.infer_form_from_data_model(
    tenant_id=tenant_id,
    app_id=app["id"],
    model_id="contact-model",
    page_id=form_page["id"],
))
print(f"Created {result['components_created']} components + binding {result['binding_id']}")
```

### 5. Bind a Data Source

```python
service.bind_data_source(
    "contacts-binding", tenant_id, app["id"],
    "Contacts DB",
    source_type="entity",
    source_ref="entity://contacts",
    schema={"fields": ["id", "name", "email", "status"]},
    policy_ref="data-policy:contact",
)
```

### 6. Attach a Workflow

```python
service.attach_workflow(
    "on-save-workflow", tenant_id, app["id"],
    trigger="on_submit",
    workflow_ref="wflo:save-contact",
    policy_ref="workflow-policy:crm",
)
```

### 7. Validate and Publish

```python
validation = service.validate_app("v1-validate", tenant_id, app["id"])
assert validation["passed"], validation["issues"]

release = service.publish_app(
    "v1-release", tenant_id, app["id"],
    target_environment="production",
    approval_recorded=True,
    approval_ref="approval:ticket-123",
    change_review_recorded=True,
)
deployment = service.deploy_release(
    "v1-deploy", tenant_id, release["id"],
    target_runtime="python",
    target_ref="apg://apps/crm",
    approval_recorded=True,
    rollback_plan_ref="rollback:crm-v0",
)
```

---

## Async Methods (v1.1)

All async methods are `async def` coroutines. Drive them with `asyncio.run()`
or `await` inside an existing event loop (FastAPI, asyncio).

### async_create_app

Coroutine-native `create_app`. Yields to the event loop between policy check
and write — safe for concurrent build sessions.

```python
app = asyncio.run(service.async_create_app(
    "my-app", tenant_id, "My App", "me",
    rbac_policy_ref="rbac:me",
    data_residency_policy_ref="residency:ke",
))
```

### infer_form_from_data_model

See section 4 above. Automatically creates the `DataBinding` as well.

### clone_app

Deep-clones an app (pages, components, data models, workflow bindings) to a
new tenant namespace. Useful for white-labelling and tenant onboarding.

```python
cloned = asyncio.run(service.clone_app(
    source_app_id="crm-app",
    source_tenant_id="acme-corp",
    target_tenant_id="new-tenant",
    new_app_name="CRM (New Tenant)",
    new_owner="new-team",
))
print(cloned["clone_counts"])  # {"pages": 2, "components": 4, ...}
```

### validate_app_incremental

Same result as `validate_app` but skips domains whose content hash matches
the previous validation. Critical for interactive builder UIs.

```python
result = asyncio.run(service.validate_app_incremental(
    "v2-validate", tenant_id, app["id"]
))
print(result["cache_hit_domains"])   # ["pages", "components", ...]
print(result["evaluated_domains"])   # domains that changed
```

### snapshot_app / restore_snapshot

Take a named snapshot before destructive edits; restore atomically if needed.

```python
asyncio.run(service.snapshot_app(
    tenant_id, app["id"],
    snapshot_id="snap-before-refactor",
    label="Before component restructure",
    tagged_by="alice",
))

# ... make changes ...

asyncio.run(service.restore_snapshot(
    tenant_id,
    snapshot_id="snap-before-refactor",
    restore_reason="Component restructure broke layout — rolling back",
))
```

### preview_data_binding

Test sample rows against a binding schema before deploying. Returns per-field
conformance scores (fraction of rows where the field is non-null) and
per-row violation details.

```python
report = asyncio.run(service.preview_data_binding(
    binding_id="contacts-binding",
    tenant_id=tenant_id,
    sample_rows=[
        {"id": "1", "name": "Alice", "email": "a@example.com", "status": "active"},
        {"id": "2", "name": "Bob", "email": None, "status": "inactive"},
    ],
))
print(report["field_scores"])    # {"id": 1.0, "name": 1.0, "email": 0.5, ...}
print(report["violations"])      # row-level violation details
```

### accessibility_audit

Run automated WCAG 2.1 Level AA heuristics across all interactive components.
Checks: missing accessibility labels, chart/table aria attributes, button
`action_type`, select `options`, app-level `accessibility_checked` flag, page
layout coverage, and theme contrast tokens.

```python
audit = asyncio.run(service.accessibility_audit(tenant_id, app["id"]))
print(f"Compliance: {audit['compliance_score']:.0%}")
print(f"Findings: {len(audit['findings'])}")
if audit["recommend_accessibility_checked"]:
    service.change_app_state(tenant_id, app["id"], "validated", "Accessibility confirmed")
```

### enforce_performance_budget

Checks component-per-page limits and global binding/connector/script quotas.
Emits graded audit events (`warning` / `error`) for each violation.

```python
budget = asyncio.run(service.enforce_performance_budget(
    tenant_id, app["id"],
    max_components_per_page=30,
    max_data_bindings=15,
))
print(budget["within_budget"])   # True / False
print(budget["violations"])      # list of violations with severity and actual/limit
```

### app_diff

Compare two snapshots to see what changed before approving a production publish.

```python
asyncio.run(service.snapshot_app(tenant_id, app["id"], "snap-v1", "v1"))
# ... make changes ...
asyncio.run(service.snapshot_app(tenant_id, app["id"], "snap-v2", "v2"))

diff = asyncio.run(service.app_diff(tenant_id, app["id"], "snap-v1", "snap-v2"))
print(f"Total changes: {diff['total_changes']}")
# diff["diff"]["added"]["components"]    — new components
# diff["diff"]["removed"]["pages"]       — removed pages
# diff["diff"]["modified"]["data_models"] — before/after pairs
```

---

## Interoperability

`ncod` integrates with other APG capabilities through the composition engine:

```apg
use ncod;
```

Workflow bindings reference `wflo` capability refs. Script extensions
reference `scpt` capability refs. All policy refs are resolved at deploy time
through the `auth` capability.

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or
environment variables prefixed with `NCOD_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `NCOD_DEFAULT_THEME` | `ncod_app_builder` | Default app theme |
| `NCOD_MAX_COMPONENTS_PER_PAGE` | `50` | Performance budget default |
| `NCOD_MAX_DATA_BINDINGS` | `20` | Performance budget default |

## Guardrail Summary

NCOD denies operations missing: tenant context, app ownership, app name, theme,
RBAC policy, data-residency policy, screen routes, component accessibility
labels (interactive types), valid data model fields, data model policy, valid
binding schema, workflow trigger/reference/policy, publish approval, passing
validation, script policy, connector policy, deployment target/approval/rollback
plan, AI agent registration/runtime/scope/disclosure, state-change reason, or
state-change audit evidence.

`require_review` is returned for production publishes and pages lacking
element relationship metadata.

## Further Reading

- `service.py` — Business logic (sync + async methods)
- `models.py` — Data models
- `builder_runtime.py` — Normalization and validation helpers
- `api.py` — REST API payload helpers
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap and design rationale
- `README.md` — Quick reference
