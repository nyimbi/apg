# APG Human Intelligence

`intel_humint` is the APG package-backed capability for governed
human-intelligence applications. It composes authorities, human sources,
contact plans, contact reports, debriefings, reliability assessments, leads,
dissemination, reviews, Bytewax lifecycle metadata, UI/view models, visual
theming, and provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification,
  and evidence. Automated expiry scanning flags expired authorities and
  linked sources via `authority_expiry_check`.
- Human source registry with handling status, risk level, owner, protection
  reference, authority, and evidence.
- Contact planning with objective, safety plan, approval, source-authority
  matching, and evidence. Optimal handler assignment via `assign_handler`.
- Contact reports, debriefings, reliability assessments, leads, dissemination,
  and review workflows. Dissemination compliance enforced via
  `dissemination_compliance_check`.
- Welfare trend monitoring with alert thresholds via `welfare_trend_analysis`.
- Temporal credibility decay (exponential model) via `intel_credibility_decay`.
- Operational security holistic assessment via `operational_security_assessment`.
- Source compartment compliance checking via `source_compartment_check`.
- Collection cycle feedback loop closing via `collection_cycle_feedback`.
- Bulk intelligence validation (up to 200 items) via `bulk_validate_intelligence`.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.humint.lifecycle`.

## Use The Service

```python
from capabilities.intel.humint import HumanIntelligenceService

service = HumanIntelligenceService("tenant-a", actor_id="analyst-1")

# Record lawful authority
authority = service.record_authority(
    "auth-1", "tenant-a", "mission_order", "scope://mission",
    "secret", "approver-1", "2027-12-31", "evidence://authority",
)

# Register source
source = service.register_source(
    "source-1", "tenant-a", "voluntary_source", "active", "medium",
    "owner-1", authority["id"], "protection://source-1", "evidence://source",
)

# Collect intelligence (async)
import asyncio

async def run():
    intel = await service.collect_intelligence(
        source_id="source-1",
        subject="target_org_finances",
        content="...",
        confidence=0.75,
    )
    validated = await service.validate_intelligence(
        intel_id=intel["intel_id"],
        validation_method="CORROBORATION",
    )
    report = await service.humint_report(classification="secret")
    return report

asyncio.run(run())
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`source_authority_mismatch`, `safety_plan_required`,
`coercive_humint_action_denied`, or `bytewax_event_stream_required`.

## Service Method Reference

### Sync CRUD (governance workflow)

| Method | Description |
|---|---|
| `record_authority(...)` | Register a lawful authority with classification and expiry |
| `register_source(...)` | Register a human source under an authority |
| `record_contact_plan(...)` | Plan a contact with source and safety constraints |
| `record_contact_report(...)` | Record outcome of a contact meeting |
| `record_debriefing(...)` | Record a debriefing session with credibility score |
| `record_reliability(...)` | Record a NATO admiralty reliability assessment |
| `record_lead(...)` | Record an intelligence lead from a debriefing |
| `record_dissemination(...)` | Record intelligence dissemination with release marking |
| `record_review(...)` | Record a governance review decision |
| `register_humint_agent(...)` | Register an AI agent for HUMINT automation |
| `validate_agent_action(...)` | Validate an agent action against policy rules |
| `validate_batch(...)` | Validate a Bytewax batch operation |
| `dashboard_summary(...)` | Return a summary of all tenant-scoped state |

### Async Operational (phase 1)

| Method | Description |
|---|---|
| `source_meeting(...)` | Record a source meeting with composite risk scoring |
| `collect_intelligence(...)` | Collect intel with NATO admiralty weighting |
| `validate_intelligence(...)` | Apply validation method and uplift credibility |
| `source_protection(...)` | Assess and record source protection requirements |
| `false_flag_detection(...)` | Detect false flag indicators for a source/intel pair |
| `source_reliability_assessment(...)` | Period-based reliability assessment with trend |
| `cross_reference_human_intel(...)` | Cross-reference intel against other collections |
| `humint_report(...)` | Generate a tenant-scoped HUMINT intelligence report |
| `source_lifecycle_management(...)` | Activate, suspend, reactivate, terminate, or archive a source |
| `osint_collection(...)` | Collect OSINT profile for a target |
| `debrief_batch(...)` | Batch-process debriefings into a consolidated summary |
| `intelligence_sharing(...)` | Share intel with partner agencies under TLP markings |
| `source_risk_scoring(...)` | Composite risk score from risk level, false flag, and credibility |
| `bulk_register_sources(...)` | Bulk-register up to 100 sources in one call |
| `analytical_assessment(...)` | Weighted credibility composite for a subject |
| `export_sources(...)` | Export source registry (json or csv) |
| `health_check()` | Service health and key operational metrics |
| `contact_deconfliction(...)` | Detect dual coverage and handler overload |
| `source_network_analysis(...)` | Graph-style relationship analysis among sources |
| `counter_humint_assessment(...)` | Counter-HUMINT risk for an operation |
| `collection_requirements(...)` | Generate intelligence collection requirements |
| `reporting_cycle(...)` | Execute a complete HUMINT reporting cycle |
| `handler_performance(...)` | Evaluate handler performance metrics |
| `source_vetting(...)` | Formal vetting with structured check results |
| `intelligence_gap_analysis()` | Identify intelligence gaps across collections |

### Async Operational (phase 2)

| Method | Description |
|---|---|
| `authority_expiry_check()` | Scan authorities for expiry; flag sources at risk |
| `welfare_trend_analysis(...)` | Rolling welfare trend with alert threshold detection |
| `assign_handler(...)` | Recommend optimal handler via workload and welfare scoring |
| `dissemination_compliance_check(...)` | Full compliance verification before release |
| `intel_credibility_decay(...)` | Apply exponential temporal decay to credibility |
| `collection_cycle_feedback(...)` | Close the intelligence cycle; auto-escalate gaps |
| `source_compartment_check(...)` | Verify compartment assignments against authorities |
| `bulk_validate_intelligence(...)` | Batch validate up to 200 intelligence items |
| `operational_security_assessment(...)` | Holistic OPSEC score combining all risk dimensions |

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/humint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/humint/app.py
./.venv/bin/apg capabilities inspect intel_humint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/humint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/humint --json
```

## Production Boundaries

Field operations, source recruitment, coercive operations, covert
communications, payment handling, physical security, identity protection
infrastructure, partner case systems, storage backends, GraphRAG projections,
dissemination delivery, and durable Bytewax topology execution stay behind
adapters.
