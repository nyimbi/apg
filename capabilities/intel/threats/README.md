# Threat Intelligence

`intel_threats` is an executable APG capability package for building governed
threat-intelligence applications. It gives generated APG apps a concrete
runtime for lawful authority, threat workspaces, source lineage, indicators,
actors, campaigns, assessments, reports, mitigations, reviews, Bytewax
lifecycle checks, UI models, and provider-neutral AI-agent support.

## What It Provides

- `threat_authority_workflow`
- `threat_workspace_workflow`
- `threat_source_workflow`
- `threat_indicator_workflow`
- `threat_actor_workflow`
- `threat_campaign_workflow`
- `threat_assessment_workflow`
- `threat_report_workflow`
- `threat_mitigation_workflow`
- `threat_review_workflow`
- `threat_agent_workflow`

## Using The Service

```python
from capabilities.intel.threats import ThreatIntelligenceService

service = ThreatIntelligenceService()
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
    "cyber_threat",
    "Cyber Threats",
    "confidential",
    authority["id"],
    "workspace-evidence",
)
```

All write operations evaluate deterministic rules before mutation. Invalid
authority, missing evidence, missing lineage, missing approval, unsupported
taxonomies, non-Bytewax lifecycle routing, and unsafe AI-agent scopes raise
`PermissionError`.

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies unsupported attribution, fabricated indicators, source
tampering, privacy bypasses, autonomous mitigation, unapproved publication, and
privileged agent actions without human approval. AI agents are first-class but
bounded: supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## Verification

Focused verification for this package covers Python compilation, app self-test,
manifest JSON validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

