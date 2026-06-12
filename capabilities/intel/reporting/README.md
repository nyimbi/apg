# Intelligence Reporting

`intel_reporting` is an executable APG capability package for building governed
intelligence-reporting applications. It gives generated APG apps a concrete
runtime for lawful authority, reporting workspaces, templates, products,
sections, citations, approvals, distributions, publications, reviews, Bytewax
lifecycle checks, UI models, and provider-neutral AI-agent support.

## What It Provides

### Core Workflows
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

### Async Lifecycle Methods
- `create_report` / `add_section` / `add_intelligence_item`
- `peer_review` / `approve_report` / `disseminate_report`
- `archive_report` / `report_archive_batch`
- `report_feedback` / `analytic_judgment` / `key_judgment`
- `caveat_add`
- `report_search` / `report_search_advanced`
- `reporting_analytics` / `report_analytics_extended`
- `get_report_state` / `report_index` / `pending_approvals`
- `citation_integrity_check` / `intelligence_score`
- `dissemination_track` / `template_usage_report`
- `report_workflow` (end-to-end orchestration)

### Advanced Methods (v1.2)
- `version_report` — immutable snapshot of product + sections + citations
- `diff_versions` — structured diff between two version snapshots
- `register_kiq` / `answer_kiq` / `kiq_coverage_report` — Key Intelligence Question tracking
- `redact_report` — produce a sanitised lower-classification copy
- `subscription_register` / `subscription_events` — push/poll lifecycle change notifications
- `report_classification_audit` — scan for classification mismatches
- `review_sla_check` — identify peer-review items exceeding SLA threshold

## Quick Start

```python
from capabilities.intel.reporting import IntelligenceReportingService

service = IntelligenceReportingService(tenant_id="tenant-a", actor_id="analyst-1")

authority = service.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "scope-ref", "confidential",
    "approver-1", "2026-12-31", "authority-evidence",
)
workspace = service.record_workspace(
    "ws-1", "tenant-a", "threat_reporting",
    "Threat Reporting", "confidential",
    authority["id"], "ws-evidence",
)
template = service.record_template(
    "tmpl-1", "tenant-a", "ws-1",
    "threat_report", "templates/threat.json",
    "confidential", "tmpl-evidence",
)

import asyncio

async def run():
    # End-to-end workflow
    result = await service.report_workflow(
        "threat_report", "confidential",
        "Q3 Threat Assessment", "analyst-1",
        ["consumer-a", "consumer-b"],
    )
    print(result)

asyncio.run(run())
```

## Classification Hierarchy

```
unclassified < restricted < confidential < secret < top_secret
```

Child records (sections, redacted copies) may not exceed their parent product's
classification without a recorded authority of type `classification_upgrade`.
`redact_report` enforces downgrade via a recorded `classification_downgrade` authority.

## Key Intelligence Questions (KIQ)

Register formal requirements, tag reports against them, and track coverage:

```python
async def kiq_demo(svc):
    await svc.register_kiq("kiq-1", "What is the threat actor's TTPs?", priority=1)
    await svc.answer_kiq("kiq-1", "rpt_analyst-1_threat_report_20260601T120000")
    coverage = await svc.kiq_coverage_report()
    print(coverage["coverage_ratio"])
```

## Report Versioning

```python
snap = await service.version_report("rpt-abc")
# ... make edits ...
snap2 = await service.version_report("rpt-abc")
diff = await service.diff_versions("rpt-abc", snap["version"], snap2["version"])
```

## Subscription Notifications

```python
sub = await service.subscription_register(
    "consumer-a",
    filters={"classification": "confidential", "product_type": "threat_report"},
)
# ... later ...
events = await service.subscription_events("consumer-a")
```

## Guardrails

The capability denies:
- Uncited claims
- Classification downgrades without a recorded authority
- Source fabrication
- Privacy bypasses
- Autonomous publication without human approval
- Unapproved distribution
- Privileged agent actions without human approval

AI agents are first-class but bounded. Supported runtimes: `codex`,
`claude_code`, `opencode`, `pi`.

## Generated Application Surfaces

- `app.semantic_model()` — APG semantic model for compiler output
- `app.component_manifest()` — publishable component manifest
- `app.self_test()` — verifies package entrypoint and key invariants
- `api.py` — process-local helpers for generated applications
- `views.py` — dashboard, console, and agent-workbench view models

## Verification

Focused verification covers Python compilation, app self-test, manifest JSON
validation, package tests, APG inspect, APG publish-plan, package
implementation audit, lifecycle audit, global implementation audit, strict
package-artifact audit, stale-marker scan, disallowed messaging scan, and
`git diff --check`.

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements — Intelligence Reporting Capability
- **I2.** Overview
- **I3.** Event-Sourced Audit Trail with Tamper-Evidence
- **I4.** Classification-Label Enforcement at Every Write Boundary
- **I5.** Pluggable Persistence Backend (Repository Pattern)
- **I6.** Report Versioning and Diff Engine
- **I7.** Recipient Need-to-Know Validation Before Distribution
- **I8.** Structured Key Intelligence Questions (KIQ) Lifecycle
- **I9.** Source Reliability and Information Credibility (SRIC) Scoring
- **I10.** Parallel Dissemination with Structured Failure Handling
- **I11.** Redaction Engine for Downgraded Copies
- **I12.** Machine-Readable Report Schema Registry
- **I13.** Metrics Emission via OpenTelemetry
- **I14.** Async Background Classification Review Scheduler
- **I15.** Compartment and Codeword Management

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
