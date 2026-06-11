# Intelligence Fusion — User Guide

**Capability ID**: `intel_fusion` | **Domain**: `intel` | **Version**: `1.2.0`

---

## Description

`intel_fusion` is an executable APG capability for lawful, evidence-led intelligence fusion. It implements the full all-source analysis lifecycle from raw item ingestion through multi-source correlation, structured analytic techniques, assessment production, and dissemination-controlled product release.

---

## Installation

```bash
pip install apg-intel-fusion
```

---

## Architecture Overview

```
IntelligenceItem (raw source data)
  └─ FusionWorkspace (analytical container)
       ├─ CorrelationSet (cross-source links)
       ├─ HypothesisTest (SAT-based testing)
       ├─ AssessmentPicture (synthesised picture)
       ├─ IntelligenceProduct (finished product)
       │    └─ DisseminationRecord (TLP-controlled release)
       ├─ AnalyticalJudgement (calibrated estimates)
       ├─ Evidence (provenance-tracked support)
       └─ IntelligenceGap (collection shortfalls)
```

---

## Provided Workflows

| Workflow | Description |
|----------|-------------|
| `fusion_authority_workflow` | Record legal/mission authority for fusion operations |
| `fusion_workspace_workflow` | Create and manage analytical workspaces |
| `fusion_source_workflow` | Register and rate intelligence sources |
| `fusion_artifact_workflow` | Record and fingerprint source artifacts |
| `fusion_correlation_workflow` | Cross-source correlation with ACH support |
| `fusion_hypothesis_workflow` | Structured analytic technique lifecycle |
| `fusion_assessment_workflow` | Assessment picture production |
| `fusion_product_workflow` | Draft → review → approve → release |
| `fusion_gap_workflow` | Collection gap tracking and PIR management |

---

## Requires

- `auth` — tenant authentication
- `audl` — audit logging
- `ntfy` — notifications
- `nlpc` — NLP composition (optional, semantic dedup)
- `grph` — graph composition (optional, network correlation)

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-fusion/dashboard` | `intel_fusion:view` | Overview |
| `/intel-fusion/authorities` | `intel_fusion:authorities` | Governance |
| `/intel-fusion/workspaces` | `intel_fusion:workspaces` | Planning |
| `/intel-fusion/sources` | `intel_fusion:sources` | Sources |
| `/intel-fusion/artifacts` | `intel_fusion:artifacts` | Evidence |
| `/intel-fusion/correlations` | `intel_fusion:correlations` | Analysis |
| `/intel-fusion/hypotheses` | `intel_fusion:hypotheses` | Analysis |
| `/intel-fusion/assessments` | `intel_fusion:assessments` | Analysis |
| `/intel-fusion/gaps` | `intel_fusion:gaps` | Collection |
| `/intel-fusion/products` | `intel_fusion:products` | Products |

---

## Core Service Methods

### IntelligenceItem

| Method | Description |
|--------|-------------|
| `create_intel_item(payload)` | Ingest a raw item from any source discipline |
| `get_intel_item(item_id)` | Retrieve by ID |
| `list_intel_items(workspace_id, source_type, status)` | Filtered list |
| `update_intel_item(item_id, patch)` | Partial update |
| `validate_intel_item(item_id)` | Mark as custodian-validated |
| `reject_intel_item(item_id)` | Remove from fusion pipeline |
| `delete_intel_item(item_id)` | Soft-delete |
| `batch_ingest_items(payloads)` | Bulk ingest with fingerprint deduplication |
| `admiralty_coded_confidence(item_id, source_reliability, information_credibility)` | NATO Admiralty Code composite scoring |

### FusionWorkspace

| Method | Description |
|--------|-------------|
| `create_workspace(payload)` | Create new analytical workspace |
| `get_workspace(workspace_id)` | Retrieve by ID |
| `list_workspaces(status, workspace_type)` | Filtered list |
| `update_workspace(workspace_id, patch)` | Partial update |
| `suspend_workspace(workspace_id)` | Suspend active workspace |
| `close_workspace(workspace_id)` | Close permanently |
| `workspace_summary(workspace_id)` | Item/correlation/product counts |
| `delete_workspace(workspace_id)` | Soft-delete |

### CorrelationSet

| Method | Description |
|--------|-------------|
| `create_correlation(payload)` | Link multiple items into a correlation |
| `get_correlation(correlation_id)` | Retrieve by ID |
| `list_correlations(workspace_id, status, correlation_type)` | Filtered list |
| `update_correlation(correlation_id, patch)` | Partial update |
| `confirm_correlation(correlation_id)` | Mark confirmed |
| `dispute_correlation(correlation_id)` | Mark disputed |
| `delete_correlation(correlation_id)` | Soft-delete |

### AssessmentPicture

| Method | Description |
|--------|-------------|
| `create_assessment(payload)` | Synthesise assessment from hypotheses + correlations |
| `get_assessment(assessment_id)` | Retrieve by ID |
| `list_assessments(workspace_id, risk_level, assessment_type)` | Filtered list |
| `update_assessment(assessment_id, patch)` | Partial update |
| `approve_assessment(assessment_id, approver_id)` | Senior analyst approval |
| `delete_assessment(assessment_id)` | Soft-delete |

### IntelligenceProduct

| Method | Description |
|--------|-------------|
| `create_product(payload)` | Create draft finished intelligence product |
| `get_product(product_id)` | Retrieve by ID |
| `list_products(workspace_id, status, product_type, tlp)` | Filtered list |
| `update_product(product_id, patch)` | Update draft/review product |
| `submit_product_for_review(product_id, reviewer_id)` | Advance to review state |
| `approve_product(product_id, approver_id)` | Approve reviewed product |
| `release_product(product_id, approval_reference)` | Release for dissemination |
| `recall_product(product_id)` | Recall a released product |
| `generate_finished_intelligence(workspace_id, product_id)` | Quality-gated product generation |
| `dissemination_with_tlp(product_id, audience, ...)` | TLP-controlled dissemination |
| `delete_product(product_id)` | Soft-delete |

### HypothesisTest

| Method | Description |
|--------|-------------|
| `create_hypothesis(payload)` | Create SAT-backed hypothesis test |
| `get_hypothesis(hypothesis_id)` | Retrieve by ID |
| `list_hypotheses(workspace_id, status, sat_method)` | Filtered list |
| `update_hypothesis(hypothesis_id, patch)` | Update with new evidence/conclusion |
| `delete_hypothesis(hypothesis_id)` | Soft-delete |
| `detect_hypothesis_conflicts(workspace_id)` | Scan for mutually-exclusive supported pairs |
| `register_conflict_resolution(workspace_id, preferred_id, deferred_id, rationale)` | Document analyst resolution |

### AnalyticalJudgement

| Method | Description |
|--------|-------------|
| `create_judgement(payload)` | Record calibrated analytical judgement |
| `get_judgement(judgement_id)` | Retrieve by ID |
| `list_judgements(workspace_id, judgement_type)` | Filtered list |
| `update_judgement(judgement_id, patch)` | Update confidence or assumptions |
| `challenge_judgement(judgement_id, challenger_id)` | Register adversarial challenge |
| `delete_judgement(judgement_id)` | Soft-delete |

### Evidence

| Method | Description |
|--------|-------------|
| `create_evidence(payload)` | Record provenance-tracked evidence |
| `get_evidence(evidence_id)` | Retrieve by ID |
| `list_evidence(workspace_id, evidence_type, status)` | Filtered list |
| `update_evidence(evidence_id, patch)` | Update status or custody chain |
| `verify_evidence(evidence_id)` | Mark verified |
| `challenge_evidence(evidence_id)` | Mark challenged |
| `discredit_evidence(evidence_id)` | Discredit — removed from hypothesis use |
| `delete_evidence(evidence_id)` | Soft-delete |

### Intelligence Gaps

| Method | Description |
|--------|-------------|
| `create_intelligence_gap(workspace_id, gap_description, priority, ...)` | Record a collection shortfall |
| `list_intelligence_gaps(workspace_id, status, priority)` | Filtered list |
| `close_intelligence_gap(gap_id, resolution_notes)` | Satisfy a gap |
| `gap_coverage_report(workspace_id)` | Open/closed counts, blocked hypotheses |

### Structured Analytic Techniques

| Method | Description |
|--------|-------------|
| `apply_structured_analytic_techniques(workspace_id, method, ...)` | Dispatch to named SAT |
| `analysis_of_competing_hypotheses(workspace_id, hypotheses, evidence_items)` | Full ACH matrix |
| `ace_method(workspace_id, analysis_statement, confidence_score, evidence_ids)` | ACE structured output |
| `key_assumptions_check(workspace_id, assumptions, confidence_scores)` | KAC robustness report |
| `confidence_calibration(prior, likelihood_given_true, likelihood_given_false)` | Bayesian calibration |

### Fusion Operations

| Method | Description |
|--------|-------------|
| `fuse_intelligence(workspace_id, source_ids, time_window)` | Standard multi-source fusion |
| `staleness_weighted_fusion(workspace_id, half_life_hours)` | Temporal decay-weighted fusion |
| `correlate_across_domains(workspace_id, osint_ids, sigint_ids, humint_ids, ...)` | Cross-domain correlation |

### Audit and Analytics

| Method | Description |
|--------|-------------|
| `dashboard_report()` | Tenant-level dashboard report |
| `list_workspace_events(workspace_id, event_type)` | Persisted event audit trail |
| `audit_trail_for_product(product_id)` | Full decision-chain timeline |
| `analyst_performance_report(analyst_id, workspace_id)` | Per-analyst metrics |

---

## Typical Workflow

```python
from capabilities.intel.fusion import IntelligenceFusionService
from capabilities.intel.fusion.models import (
    FusionWorkspaceCreate, IntelligenceItemCreate, CorrelationSetCreate,
    HypothesisTestCreate, AssessmentPictureCreate, IntelligenceProductCreate,
    WorkspaceType, SourceType, CorrelationType, SATMethod, AssessmentType,
    RiskLevel, ProductType, TLPLevel, ClassificationLevel,
)

svc = IntelligenceFusionService(tenant_id="acme", actor_id="analyst-1")

# 1. Create workspace
ws = await svc.create_workspace(FusionWorkspaceCreate(
    tenant_id="acme",
    name="Operation Alpha",
    workspace_type=WorkspaceType.THREAT_FUSION,
    classification=ClassificationLevel.CONFIDENTIAL,
    lead_analyst_id="analyst-1",
))

# 2. Register a collection gap before ingestion
gap = await svc.create_intelligence_gap(
    workspace_id=ws.id,
    gap_description="No SIGINT coverage on target comms node",
    priority="high",
)

# 3. Batch-ingest items from multiple disciplines
report = await svc.batch_ingest_items([
    IntelligenceItemCreate(
        tenant_id="acme", workspace_id=ws.id,
        source_type=SourceType.OSINT, content_fingerprint="fp-001",
        custodian_id="analyst-1", confidence_score=0.75,
    ),
    IntelligenceItemCreate(
        tenant_id="acme", workspace_id=ws.id,
        source_type=SourceType.HUMINT, content_fingerprint="fp-002",
        custodian_id="analyst-1", confidence_score=0.85,
    ),
])
# report["accepted"] == 2

# 4. Apply Admiralty Code to a critical HUMINT item
item_id = report["accepted_ids"][1]
rating = await svc.admiralty_coded_confidence(item_id, "B", "2")
# rating["composite_code"] == "B2"

# 5. Stale-weighted fusion
fusion = await svc.staleness_weighted_fusion(ws.id, half_life_hours={"humint": 96.0})

# 6. Standard fusion
fusion_result = await svc.fuse_intelligence(ws.id)

# 7. Create correlation
corr = await svc.create_correlation(CorrelationSetCreate(
    tenant_id="acme", workspace_id=ws.id,
    correlation_type=CorrelationType.CROSS_SOURCE_CONFIRM,
    item_ids=report["accepted_ids"],
    analyst_id="analyst-1", confidence_score=0.80,
))

# 8. Hypothesis test via ACH
hyp = await svc.create_hypothesis(HypothesisTestCreate(
    tenant_id="acme", workspace_id=ws.id,
    claim="Actor X is planning an attack within 72 hours",
    analyst_id="analyst-1",
    sat_method=SATMethod.ACH,
    initial_confidence=0.70,
    alternative_hypotheses=["Actor X is conducting reconnaissance only"],
))

# 9. Check for conflicting hypotheses
conflicts = await svc.detect_hypothesis_conflicts(ws.id)

# 10. Assessment
assessment = await svc.create_assessment(AssessmentPictureCreate(
    tenant_id="acme", workspace_id=ws.id,
    assessment_type=AssessmentType.THREAT,
    risk_level=RiskLevel.HIGH,
    analyst_id="analyst-1",
    confidence_score=0.78,
    hypothesis_ids=[hyp.id],
    correlation_ids=[corr.id],
))

# 11. Approve and release product
product = await svc.create_product(IntelligenceProductCreate(
    tenant_id="acme", workspace_id=ws.id,
    title="Threat Assessment — Actor X",
    product_type=ProductType.THREAT_ASSESSMENT,
    classification=ClassificationLevel.CONFIDENTIAL,
    tlp=TLPLevel.AMBER,
    assessment_ids=[assessment.id],
))
await svc.submit_product_for_review(product.id, reviewer_id="analyst-2")
await svc.approve_product(product.id, approver_id="analyst-2")
await svc.release_product(product.id, approval_reference="AUTH-2026-001")

# 12. Audit the full decision chain
trail = await svc.audit_trail_for_product(product.id)

# 13. Analyst performance
perf = await svc.analyst_performance_report("analyst-1", workspace_id=ws.id)
```

---

## Intelligence Gap Workflow

Intelligence gaps track what is **unknown** and which hypotheses are blocked by missing collection. Gaps should be created before or during workspace setup and closed when the collection shortfall is satisfied.

```python
# Create gap
gap = await svc.create_intelligence_gap(
    workspace_id=ws.id,
    gap_description="No financial transaction data for entity B",
    priority="critical",
    linked_hypothesis_ids=["hyp-finance-1"],
    collection_requirement="Request FININT tasking on entity B accounts",
)

# Close when satisfied
await svc.close_intelligence_gap(
    gap_id=gap["id"],
    resolution_notes="FININT data received via partner exchange 2026-06-10",
)

# Coverage report
report = await svc.gap_coverage_report(workspace_id=ws.id)
# report["blocked_hypothesis_ids"] shows which hypotheses remain at risk
```

---

## Admralty Code Confidence Scoring

The NATO/Admiralty Code separates **source reliability** (track record) from **information credibility** (this specific report's corroboration). This is distinct from the flat `confidence_score` field.

| Code | Source Reliability | Code | Information Credibility |
|------|--------------------|------|------------------------|
| A | Completely reliable | 1 | Confirmed by other sources |
| B | Usually reliable | 2 | Probably true |
| C | Fairly reliable | 3 | Possibly true |
| D | Not usually reliable | 4 | Doubtful |
| E | Unreliable | 5 | Improbable |
| F | Cannot be judged | 6 | Cannot be judged |

```python
rating = await svc.admiralty_coded_confidence(
    item_id="item-1",
    source_reliability="B",
    information_credibility="2",
)
# composite_code: "B2"
# composite_score: 0.80
# icd203_word_equivalent: "highly_likely"
```

---

## Hypothesis Conflict Detection

When two hypotheses are both well-supported (confidence ≥ 0.55) in the same workspace, a logical conflict may exist. The service detects these automatically.

```python
conflicts = await svc.detect_hypothesis_conflicts(workspace_id=ws.id)
# conflicts["conflicts"] -> list of conflict pairs with severity and recommended action

# Resolve by designating the preferred hypothesis
await svc.register_conflict_resolution(
    workspace_id=ws.id,
    hypothesis_id_preferred="hyp-attack",
    hypothesis_id_deferred="hyp-recon",
    analyst_rationale="Signal intercepts corroborate attack timeline. Recon-only "
                      "hypothesis does not account for logistics activity at target site.",
    supporting_evidence_ids=["ev-signal-1", "ev-humint-2"],
)
```

---

## Temporal Decay Fusion

Standard fusion treats all items as equally current. Staleness-weighted fusion applies per-source half-life decay so recent signals dominate.

```python
result = await svc.staleness_weighted_fusion(
    workspace_id=ws.id,
    half_life_hours={
        "sigint": 4.0,    # degrades rapidly
        "humint": 96.0,   # strategic reports remain valid longer
        "osint": 18.0,
    },
)
# result["weighted_avg_confidence"] is the decay-adjusted fused estimate
# result["source_decay_stats"] shows per-source average decay weight
```

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_FUSION_`.

| Key | Description | Default |
|-----|-------------|---------|
| `INTEL_FUSION_DB_URL` | Database URL | `sqlite+aiosqlite:///fusion.db` |
| `INTEL_FUSION_MIN_SOURCES` | Minimum sources required for fusion | `2` |
| `INTEL_FUSION_QUALITY_THRESHOLD` | Minimum quality score for product release | `0.55` |

---

## Guardrails

The capability enforces the following at every operation:

- Tenant context must be present on all writes.
- Cross-tenant access is denied unconditionally.
- Source types, correlation types, assessment types, and risk levels must be in the supported contract lists.
- Evidence must have a content fingerprint and chain of custody.
- Products cannot be updated after recall.
- TLP level must be compatible with recipient clearance before dissemination.
- AI-agent actions with prohibited scopes (fabrication, tampering, privacy bypass, autonomous dissemination, unapproved attribution) are rejected by the rule engine.

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `domain/rules.py` — Domain rule assertions
- `domain/calculations.py` — Fusion calculations (ACH, KAC, corroboration)
- `WORLD_CLASS_IMPROVEMENTS.md` — Prioritised improvement roadmap
- `README.md` — Quick reference

---

*© 2025 Datacraft — Nyimbi Odero*
