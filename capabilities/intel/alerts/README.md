# Alert Management

`intel_alerts` is an executable APG capability package for building governed
alert-management applications. It gives generated APG apps a concrete runtime
for lawful authority, alert workspaces, rules, signals, alerts, escalations,
notifications, assignments, resolutions, reviews, Bytewax lifecycle checks, UI
models, and provider-neutral AI-agent support.

## What It Provides

- `alert_authority_workflow`
- `alert_workspace_workflow`
- `alert_rule_workflow`
- `alert_signal_workflow`
- `alert_record_workflow`
- `alert_escalation_workflow`
- `alert_notification_workflow`
- `alert_assignment_workflow`
- `alert_resolution_workflow`
- `alert_review_workflow`
- `alert_agent_workflow`

## Using The Service

```python
from capabilities.intel.alerts import AlertManagementService

service = AlertManagementService()
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
    "watch_center",
    "Watch Center",
    "confidential",
    authority["id"],
    "workspace-evidence",
)
```

All write operations evaluate deterministic rules before mutation. Invalid
authority, missing evidence, missing owners, invalid confidence, missing
approval, unsupported taxonomies, non-Bytewax lifecycle routing, and unsafe
AI-agent scopes raise `PermissionError`.

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies unapproved escalation, unapproved notification, alert
suppression, evidence fabrication, privacy bypass, autonomous closure, severity
downgrade, and privileged agent actions without human approval. AI agents are
first-class but bounded: supported runtimes are `codex`, `claude_code`,
`opencode`, and `pi`.

## Verification

Focused verification for this package covers Python compilation, app self-test,
manifest JSON validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

