# Intelligence Dashboard

`intel_dashboard` is an executable APG capability package for building governed
intelligence-dashboard applications. It gives generated APG apps a concrete
runtime for lawful authority, dashboard workspaces, dashboards, data sources,
metrics, widgets, filters, views, shares, reviews, Bytewax lifecycle checks,
UI models, and provider-neutral AI-agent support.

## What It Provides

- `dashboard_authority_workflow`
- `dashboard_workspace_workflow`
- `dashboard_composition_workflow`
- `dashboard_source_workflow`
- `dashboard_metric_workflow`
- `dashboard_widget_workflow`
- `dashboard_filter_workflow`
- `dashboard_view_workflow`
- `dashboard_share_workflow`
- `dashboard_review_workflow`
- `dashboard_agent_workflow`

## Using The Service

```python
from capabilities.intel.dashboard import IntelligenceDashboardService

service = IntelligenceDashboardService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "mission_order",
    "scope-ref",
    "confidential",
    "approver-1",
    "2026-12-31",
    "authority-evidence",
)
workspace = service.record_workspace(
    "workspace-1",
    "tenant-a",
    "operations_center",
    "Operations Center",
    "confidential",
    authority["id"],
    "workspace-evidence",
)
```

All write operations evaluate deterministic rules before mutation. Invalid
authority, missing evidence, missing owners, missing source custodians, missing
share approval, unsupported taxonomies, non-Bytewax lifecycle routing, and
unsafe AI-agent scopes raise `PermissionError`.

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies uncited metrics, classification leaks, source tampering,
privacy bypasses, autonomous shares, unapproved public views, and privileged
agent actions without human approval. AI agents are first-class but bounded:
supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## Verification

Focused verification for this package covers Python compilation, app self-test,
manifest JSON validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

