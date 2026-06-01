# Intelligence Reporting

`intel_reporting` is an executable APG capability package for building governed
intelligence-reporting applications. It gives generated APG apps a concrete
runtime for lawful authority, reporting workspaces, templates, products,
sections, citations, approvals, distributions, publications, reviews, Bytewax
lifecycle checks, UI models, and provider-neutral AI-agent support.

## What It Provides

- `reporting_authority_workflow`
- `reporting_workspace_workflow`
- `reporting_template_workflow`
- `reporting_product_workflow`
- `reporting_section_workflow`
- `reporting_citation_workflow`
- `reporting_approval_workflow`
- `reporting_distribution_workflow`
- `reporting_publication_workflow`
- `reporting_review_workflow`
- `reporting_agent_workflow`

## Using The Service

```python
from capabilities.intel.reporting import IntelligenceReportingService

service = IntelligenceReportingService()
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
    "threat_reporting",
    "Threat Reporting",
    "confidential",
    authority["id"],
    "workspace-evidence",
)
```

All write operations evaluate deterministic rules before mutation. Invalid
authority, missing evidence, missing citations, missing approvals, unsupported
taxonomies, non-Bytewax lifecycle routing, and unsafe AI-agent scopes raise
`PermissionError`.

## Generated Application Surfaces

- `app.semantic_model()` returns an APG semantic model for compiler output.
- `app.component_manifest()` returns a publishable component manifest.
- `app.self_test()` verifies the package entrypoint and key invariants.
- `api.py` exposes process-local helpers for generated applications.
- `views.py` exposes dashboard, console, and agent-workbench view models.

## Guardrails

The capability denies uncited claims, classification downgrades, source
fabrication, privacy bypasses, autonomous publication, unapproved
distribution, and privileged agent actions without human approval. AI agents
are first-class but bounded: supported runtimes are `codex`, `claude_code`,
`opencode`, and `pi`.

## Verification

Focused verification for this package covers Python compilation, app self-test,
manifest JSON validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

