# No-Code/Low-Code Builder Capability Specification

- **Capability Name**: No-Code/Low-Code Builder
- **Capability ID**: `ncod`
- **Category**: common
- **Version**: 1.0.0

## Purpose

NCOD is APG's package-backed no-code and low-code application builder. It gives
composed applications a deterministic app library, page composer, component
catalog, data-binding registry, workflow binding surface, governed script and
connector extension model, validation gate, publishing center, audit stream, UI
route model, and theme contract.

The package runs without external builder infrastructure. Production workflow,
script, connector, component-library, accessibility, audit, and deployment
systems should be attached through explicit adapters while this package keeps
the APG contract and local builder behavior executable.

## Provided Services

- `app_builder`
- `page_composer`
- `workflow_binding`
- `script_extensions`
- `app_publishing`
- `component_catalog`
- `connector_bindings`
- `ncod_operations`

## Required Services

- `tenant_context`
- `wflo` for workflow execution and orchestration
- `scpt` for governed low-code script execution
- `auth` for RBAC policy checks
- Optional `audl`, `conn`, `them`, and `accs` integration from registration metadata

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `capability_contract.py` | Configuration schema, deterministic rule engine, UI routes, and theme. |
| `models.py` | Domain dataclasses for apps, pages, components, data bindings, workflow bindings, script extensions, connector bindings, validation results, releases, and audit events. |
| `builder_runtime.py` | Deterministic IDs, route/layout/type normalization, version bumping, accessibility checks, data-schema validation, readiness checks, and publish posture helpers. |
| `service.py` | In-process app-builder service enforcing tenant, owner, script policy, connector policy, validation, approval, and production-review guardrails. |
| `api.py` | Thin payload helpers for apps, pages, components, data bindings, workflows, scripts, connectors, validation, publishing, status, and compatibility calls. |
| `views.py` | Dashboard, app library, builder, page composer, component catalog, publish center, connector binding, and settings view models. |
| `app.py` | Package entrypoint, manifest, semantic model, and self-test surface. |

## Builder Behavior

1. Create a tenant-scoped app with an accountable owner, theme, RBAC policy,
   data-residency policy, and accessibility-check state.
2. Add pages and components with normalized routes, supported layouts, governed
   component types, bindings, and accessibility labels.
3. Bind data sources with schemas that can be validated locally before external
   connectors are introduced.
4. Attach workflow bindings for low-code automation.
5. Add script extensions only when an approved script policy is attached.
6. Add external connector bindings only when a connector policy is attached.
7. Validate the app before publishing. Readiness checks cover owner, pages,
   components, theme, accessibility, RBAC, data residency, data bindings, script
   policies, and connector policies.
8. Publish only after approval. Production publishes also require production
   change review.

## Rules

- `tenant_context_required`
- `app_requires_owner`
- `publish_requires_approval`
- `script_extension_requires_policy`
- `external_connector_requires_policy`
- `production_change_requires_review`

## UI

The package exposes 8 APG Python UI routes through `views.py` and the package
semantic model:

- `/ncod/dashboard`
- `/ncod/apps`
- `/ncod/builder`
- `/ncod/pages`
- `/ncod/components`
- `/ncod/publishing`
- `/ncod/connectors`
- `/ncod/settings`

## Theme

The package uses the `ncod_app_builder` APG theme contract. The theme is
optimized for compact app-building work: app library rows, component canvas,
component catalog grids, and release checklists.

## Adapter Boundaries

The executable package does not call external systems directly. Production
integrations should be introduced through adapters for:

- Workflow execution and task orchestration through WFLO.
- Low-code scripts, validation hooks, and automation through SCPT.
- External connectors, credentials, APIs, event buses, and data stores through
  CONN or connector-specific packages.
- RBAC, tenant policy, data residency, and approval systems through AUTH and
  governance services.
- Accessibility auditing, theming, component catalogs, static asset pipelines,
  deployment targets, audit stores, and app marketplace publication.

## Focused Verification

Use focused checks while battery-constrained:

```bash
./.venv/bin/python -m py_compile capabilities/common/ncod/__init__.py capabilities/common/ncod/models.py capabilities/common/ncod/builder_runtime.py capabilities/common/ncod/service.py capabilities/common/ncod/api.py capabilities/common/ncod/views.py capabilities/common/ncod/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/ncod/test_capability_contract.py capabilities/common/ncod/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ncod --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ncod --json
```
