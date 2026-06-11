# Intelligence Reporting — User Guide

**Capability ID**: `intel_reporting` | **Domain**: `intel` | **Version**: `1.2.0`

## Description

`intel_reporting` is an executable APG capability package for building governed
intelligence-reporting applications. It provides a complete runtime for lawful
authority management, reporting workspaces, templates, products, sections,
citations, approvals, distributions, publications, reviews, and subscription
notifications — all with tenant-scoped enforcement of classification rules and
AI-agent guardrails.

---

## Installation

```bash
pip install apg-intel-reporting
```

---

## Core Concepts

### Entities and Their Relationships

```
ReportingAuthority
  └─ ReportingWorkspace
       └─ ReportingTemplate
            └─ ReportingProduct  (a "report")
                 ├─ ReportingSection (1..N)
                 │    └─ ReportingCitation (0..N per section)
                 ├─ ReportingApproval
                 ├─ ReportingDistribution (1..N recipients)
                 │    └─ ReportingPublication
                 └─ ReportingReview
```

### Report Lifecycle

```
draft → peer_review → approved → disseminated → archived
```

State transitions are enforced. Attempting to approve a `draft` or disseminate
an unapproved report raises `PermissionError`.

### Classification Hierarchy

```
unclassified < restricted < confidential < secret < top_secret
```

All writes enforce that child records do not exceed the classification of their
parent context.  Downgrades require a recorded authority of type
`classification_downgrade`.

---

## Getting Started

### 1. Initialise the Service

```python
from capabilities.intel.reporting import IntelligenceReportingService

service = IntelligenceReportingService(
    tenant_id="tenant-a",
    actor_id="analyst-1",
    # Optional adapters:
    # auth=my_auth_adapter,
    # audit=my_audit_adapter,
    # notify=my_notify_adapter,
)
```

### 2. Establish Governance Chain

```python
authority = service.record_authority(
    authority_id="auth-1",
    tenant_id="tenant-a",
    authority_type="mission_order",
    scope_reference="DIA-ORD-2026-001",
    classification="confidential",
    approver_id="director-1",
    expires_at="2026-12-31T00:00:00Z",
    evidence_reference="signed-order-2026-001.pdf",
)

workspace = service.record_workspace(
    workspace_id="ws-1",
    tenant_id="tenant-a",
    workspace_type="threat_reporting",
    name="Threat Analysis Cell",
    classification="confidential",
    authority_id=authority["id"],
    evidence_reference="ws-charter-001.pdf",
)

template = service.record_template(
    template_id="tmpl-1",
    tenant_id="tenant-a",
    workspace_id="ws-1",
    template_type="threat_report",
    template_reference="templates/threat_report_v2.json",
    classification="confidential",
    evidence_reference="template-baseline-001.pdf",
)
```

### 3. End-to-End Report Workflow (automated path)

```python
import asyncio

async def main():
    result = await service.report_workflow(
        report_type="threat_report",
        classification="confidential",
        title="Q3 2026 Threat Assessment",
        author_id="analyst-1",
        distribution_list=["commander-1", "j2-staff", "partner-agency-a"],
    )
    print(result)

asyncio.run(main())
```

### 4. Step-by-Step Report Production (manual path)

```python
async def manual_workflow():
    # Create report
    report = await service.create_report(
        report_type="threat_report",
        classification="confidential",
        title="Q3 2026 Threat Assessment",
        author_id="analyst-1",
    )
    product_id = report["product_id"]

    # Add sections
    await service.add_section(product_id, "executive_summary", "Key findings...")
    await service.add_section(product_id, "body", "Detailed analysis of actor TTPs...")
    await service.add_section(product_id, "key_judgments", "With high confidence, actor X...")

    # Attach intelligence items as citations
    await service.add_intelligence_item(product_id, ["HUMINT-001", "SIGINT-042"])

    # Peer review
    await service.peer_review(product_id, "reviewer-1", "Assess source reliability for HUMINT-001.")

    # Approve
    await service.approve_report(product_id, "director-1")

    # Disseminate
    dist = await service.disseminate_report(product_id, ["j2-staff", "partner-agency-a"])
    print(dist)
```

---

## Advanced Features

### Key Intelligence Questions (KIQ)

KIQs formalise intelligence requirements.  Tagging reports against them enables
measurable requirements coverage.

```python
async def kiq_workflow():
    await service.register_kiq("kiq-1", "What TTPs is actor X employing?", priority=1)
    await service.register_kiq("kiq-2", "What is actor X's targeting intent?", priority=1)

    # After a report answers kiq-1:
    await service.answer_kiq("kiq-1", product_id)

    coverage = await service.kiq_coverage_report()
    print(f"Requirements coverage: {coverage['coverage_ratio']:.0%}")
    print(f"High-priority open: {len(coverage['high_priority_open'])}")
```

### Report Versioning and Diff

Snapshot immutable versions before and after editorial cycles:

```python
async def version_demo():
    v1 = await service.version_report(product_id)

    # Analyst makes revisions via add_section / add_intelligence_item
    await service.add_section(product_id, "body", "Revised analysis...")

    v2 = await service.version_report(product_id)
    diff = await service.diff_versions(product_id, v1["version"], v2["version"])

    print(f"Sections added: {len(diff['sections_added'])}")
    print(f"Mean confidence delta: {diff['mean_confidence_delta']}")
```

### Redaction (Tearline / Write-for-Release)

Produce a sanitised copy at a lower classification level:

```python
async def redact_demo():
    redacted = await service.redact_report(
        source_product_id=product_id,
        target_classification="restricted",
        redaction_authority_id="auth-downgrade-1",
    )
    print(f"Sections redacted: {redacted['sections_redacted']}")
    print(f"Redacted product ID: {redacted['product_id']}")
```

Sections carrying a classification level above `target_classification` are
replaced with `[REDACTED]` placeholders.  The redacted product is linked to the
source via `parent_product_id`.

### Analytic Judgments and Caveats

Structured assessments beyond raw section content:

```python
async def judgment_demo():
    await service.analytic_judgment(
        product_id=product_id,
        judgment="With high confidence, actor X will conduct cyber operations against financial sector within 90 days.",
        analyst_id="analyst-1",
    )
    await service.caveat_add(
        product_id=product_id,
        caveat="HUMINT-001 is a single source; corroborate before executive dissemination.",
        analyst_id="analyst-1",
    )

    judgments = await service.key_judgment(product_id)
    analytics = await service.report_analytics_extended("30d")
    print(f"Total judgments: {analytics['total_judgments']}")
```

### Subscriptions and Lifecycle Notifications

Subscribe consumers to report lifecycle events:

```python
async def subscription_demo():
    sub = await service.subscription_register(
        "consumer-a",
        filters={"classification": "confidential", "product_type": "threat_report"},
    )

    # After lifecycle transitions occur...
    events = await service.subscription_events("consumer-a")
    for event in events:
        print(f"Event: {event['event_type']} on {event['product_id']}")
```

When the `notify` adapter is configured, events are pushed immediately. Without
it, `subscription_events` acts as a polling endpoint.

### Classification Audit

Detect sections whose classification exceeds their parent product:

```python
async def audit_demo():
    report = await service.report_classification_audit()
    if report["risk_level"] == "high":
        for mismatch in report["mismatches"]:
            print(f"Mismatch: {mismatch['section_id']} ({mismatch['section_classification']}) "
                  f"under product {mismatch['product_id']} ({mismatch['product_classification']})")
```

### Review SLA Enforcement

Identify reports overdue for approval:

```python
async def sla_demo():
    overdue = await service.review_sla_check(sla_hours=24)
    print(f"Overdue for review: {overdue['overdue_count']}")
    for record in overdue["overdue"]:
        print(f"  {record['product_id']}: {record['age_hours']:.1f}h old (reviewer: {record['last_reviewer']})")
```

Configure SLA hours to match operational tempo (e.g., 4h for tactical reporting,
48h for strategic assessments).

### Citation Integrity

Verify every section carries at least one citation before approval:

```python
async def citation_demo():
    check = await service.citation_integrity_check(product_id)
    if check["uncited_sections"]:
        print(f"Uncited: {check['uncited_sections']}")
    else:
        print(f"Coverage: {check['citation_coverage']:.0%}")
```

### Intelligence Value Score

Score a product on citation coverage and section confidence:

```python
async def score_demo():
    score = await service.intelligence_score(product_id)
    print(f"Grade: {score['grade']} (score={score['intelligence_score']})")
```

---

## Analytics

```python
async def analytics_demo():
    summary = await service.reporting_analytics("30d")
    extended = await service.report_analytics_extended("30d")

    print(f"Products: {summary['product_count']}")
    print(f"Avg section confidence: {summary['avg_section_confidence']}")
    print(f"By lifecycle: {summary['by_lifecycle_status']}")
    print(f"Total judgments: {extended['total_judgments']}")
    print(f"Total caveats: {extended['total_caveats']}")
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-reporting/dashboard` | `intel_reporting:view` | Overview |
| `/intel-reporting/authorities` | `intel_reporting:authorities` | Governance |
| `/intel-reporting/workspaces` | `intel_reporting:workspaces` | Planning |
| `/intel-reporting/templates` | `intel_reporting:templates` | Products |
| `/intel-reporting/products` | `intel_reporting:products` | Products |
| `/intel-reporting/sections` | `intel_reporting:sections` | Products |
| `/intel-reporting/citations` | `intel_reporting:citations` | Evidence |
| `/intel-reporting/approvals` | `intel_reporting:approvals` | Governance |
| `/intel-reporting/distributions` | `intel_reporting:distributions` | Dissemination |
| `/intel-reporting/subscriptions` | `intel_reporting:subscriptions` | Dissemination |
| `/intel-reporting/kiqs` | `intel_reporting:kiqs` | Requirements |
| `/intel-reporting/audit` | `intel_reporting:audit` | Governance |

---

## Service Method Reference

### Synchronous (CRUD)
| Method | Description |
|--------|-------------|
| `record_authority` | Register a lawful dissemination authority |
| `record_workspace` | Create a classification-bounded workspace |
| `record_template` | Register a report template |
| `record_product` | Create a report product |
| `record_section` | Add a section to a product |
| `record_citation` | Attach a citation to a section |
| `record_approval` | Record an approval decision |
| `record_distribution` | Create a distribution record |
| `record_publication` | Record a publication event |
| `record_review` | Submit a review |
| `register_reporting_agent` | Register an AI agent |
| `validate_agent_action` | Gate AI agent operations |
| `validate_batch` | Gate batch processing |
| `dashboard_summary` | Tenant-level entity counts |

### Async (Lifecycle & Analytics)
| Method | Description |
|--------|-------------|
| `create_report` | Create a new draft product |
| `add_section` | Append a section |
| `add_intelligence_item` | Attach intel items as citations |
| `peer_review` | Submit peer review |
| `approve_report` | Approve for dissemination |
| `disseminate_report` | Distribute to recipient list |
| `archive_report` | Move to archived state |
| `report_archive_batch` | Archive multiple reports |
| `report_feedback` | Record recipient feedback |
| `analytic_judgment` | Record a structured judgment |
| `key_judgment` | Retrieve judgments for a product |
| `caveat_add` | Add a source or time caveat |
| `citation_integrity_check` | Verify citation coverage |
| `intelligence_score` | Compute intelligence value grade |
| `report_search` | Full-scan title search |
| `report_search_advanced` | Search with classification/status filters |
| `reporting_analytics` | Aggregate tenant statistics |
| `report_analytics_extended` | Analytics plus judgment/caveat counts |
| `get_report_state` | Retrieve lifecycle state |
| `report_index` | List all reports with status |
| `pending_approvals` | List peer-review items |
| `dissemination_track` | Track distribution records |
| `template_usage_report` | Count products per template |
| `report_workflow` | End-to-end automation |
| `version_report` | Snapshot immutable version |
| `diff_versions` | Diff two version snapshots |
| `register_kiq` | Register a Key Intelligence Question |
| `answer_kiq` | Link a product as KIQ answer |
| `kiq_coverage_report` | Requirements coverage statistics |
| `redact_report` | Produce a sanitised lower-classification copy |
| `subscription_register` | Subscribe to lifecycle events |
| `subscription_events` | Poll pending events |
| `report_classification_audit` | Detect classification mismatches |
| `review_sla_check` | Flag peer-review SLA breaches |

---

## Guardrails

All write operations evaluate deterministic policy rules before mutation:

| Denied Action | Rule |
|---------------|------|
| Uncited claim | `uncited_claim_scope` must be `False` |
| Classification downgrade | requires authority + `classification_downgrade_scope` |
| Source fabrication | `source_fabrication_scope` must be `False` |
| Privacy bypass | `privacy_bypass_scope` must be `False` |
| Autonomous publication | `autonomous_publication_scope` must be `False` |
| Unapproved distribution | `unapproved_distribution_scope` must be `False` |
| Privileged agent action | requires `human_approval_recorded=True` |

---

## Interoperability

```apg
use intel_reporting;
```

Integrates with: `auth`, `audl`, `ntfy`, `nlpc`, `grph`, `intel_alerts`,
`intel_threats`, `intel_correlation`, `intel_prediction`.

---

## Configuration

All configuration is tenant-scoped. Set via the `conf` capability or environment
variables prefixed with `INTEL_REPORTING_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `INTEL_REPORTING_DB_URL` | `None` | PostgreSQL URL for persistent store |
| `INTEL_REPORTING_SLA_HOURS` | `24` | Default review SLA threshold |
| `INTEL_REPORTING_NOTIFY_URL` | `None` | Webhook URL for subscription events |

---

## Further Reading

- `service.py` — Complete business logic implementation
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `capability_contract.py` — Policy rules and supported taxonomies
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 production improvements
- `SPECIFICATION.md` — Full capability specification
- `README.md` — Quick reference
