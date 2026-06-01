# Predictive Intelligence

`intel_prediction` is an executable APG capability package for building
governed predictive-intelligence applications. It gives generated APG apps a
concrete runtime for lawful authority, analytical workspaces, scenarios,
signals, validated models, forecasts, projections, early warnings,
recommendations, reviews, Bytewax lifecycle checks, UI models, and
provider-neutral AI-agent support.

## What It Provides

- `prediction_authority_workflow`
- `prediction_workspace_workflow`
- `prediction_scenario_workflow`
- `prediction_indicator_workflow`
- `prediction_model_workflow`
- `prediction_forecast_workflow`
- `prediction_projection_workflow`
- `prediction_warning_workflow`
- `prediction_recommendation_workflow`
- `prediction_review_workflow`
- `prediction_agent_workflow`

## Using The Service

```python
from capabilities.intel.prediction import PredictiveIntelligenceService

service = PredictiveIntelligenceService()
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
    "threat_prediction",
    "Threat Forecasts",
    "confidential",
    authority["id"],
    "workspace-evidence",
)
```

All write operations evaluate deterministic rules before mutation. Invalid
authority, missing evidence, missing validation, missing approval, unsupported
taxonomies, non-Bytewax lifecycle routing, and unsafe AI-agent scopes raise
`PermissionError`.

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies unsupported automated decisions, hallucinated forecasts,
privacy bypasses, unapproved model deployment, autonomous warnings, autonomous
recommendations, and privileged agent actions without human approval. AI agents
are first-class but bounded: supported runtimes are `codex`, `claude_code`,
`opencode`, and `pi`.

## Verification

Focused verification for this package covers Python compilation, app self-test,
manifest JSON validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

