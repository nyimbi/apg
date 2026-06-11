# Human Intelligence – User Guide

**Capability ID**: `intel_humint` | **Domain**: `intel` | **Version**: `1.1.0`

---

## Overview

`intel_humint` provides a fully governed human intelligence (HUMINT) workflow
engine for APG-generated applications. Every operation is governed by a
deterministic rule engine, tenant-scoped, and emits audit events to the
Bytewax lifecycle stream. The service layer (`service.py`) is the single
point of truth for business logic.

---

## Installation

```bash
pip install apg-intel-humint
```

Or in development:

```bash
cd capabilities/intel/humint
pip install -e .
```

---

## Concepts

### Tenant Scope

All records are scoped to a `tenant_id`. Cross-tenant reads and writes are
denied by the rule engine. The `HUMINTService` constructor accepts
`tenant_id` as its first argument and all subsequent operations use it.

### NATO Admiralty Scale

Reliability grades (`A`–`F`) and credibility scores (1–6) follow the NATO
admiralty scale. The service maps these to numeric weights and computes
`adjusted_credibility = raw_confidence × reliability_weight` at collection
time.

### Intelligence Cycle

The full Plan-Collect-Process-Produce-Disseminate (PCPD) cycle is supported:

1. **Plan**: `record_authority` → `register_source` → `record_contact_plan`
2. **Collect**: `source_meeting` → `record_contact_report` → `collect_intelligence`
3. **Process**: `record_debriefing` → `validate_intelligence` → `cross_reference_human_intel` → `source_reliability_assessment`
4. **Produce**: `record_lead` → `analytical_assessment` → `humint_report`
5. **Disseminate**: `record_dissemination` → `dissemination_compliance_check` → `intelligence_sharing`

The `collection_cycle_feedback` method closes the loop by comparing actual
collections against declared priorities and escalating gaps.

---

## Quick Start

```python
import asyncio
from capabilities.intel.humint import HumanIntelligenceService

service = HumanIntelligenceService("tenant-a", actor_id="analyst-1")

# 1. Record lawful authority
auth = service.record_authority(
    "auth-1", "tenant-a", "mission_order",
    "scope://mission/alpha", "secret",
    "approver-chief", "2027-06-30", "evidence://auth/001",
)

# 2. Register source under that authority
src = service.register_source(
    "src-1", "tenant-a", "voluntary_source", "active", "medium",
    "handler-jones", auth["id"],
    "protection://src/001", "evidence://src/001",
)

# 3. Record contact plan
plan = service.record_contact_plan(
    "plan-1", "tenant-a", auth["id"], src["id"],
    "in_person", "objective://target_org",
    "safety://plan-1", "approval://plan-1", "evidence://plan-1",
)

async def operational_flow():
    # 4. Record the meeting
    meeting = await service.source_meeting(
        source_id="src-1",
        location="nairobi_safe_house_7",
        date="2026-06-15",
        handler_id="handler-jones",
    )

    # 5. Record contact report
    report = service.record_contact_report(
        "rpt-1", "tenant-a", "plan-1",
        "report://contact/001", "handler-jones", 0.82, "evidence://rpt/001",
    )

    # 6. Collect intelligence
    intel = await service.collect_intelligence(
        source_id="src-1",
        subject="target_org_leadership",
        content="Source confirmed succession conflict at board level...",
        confidence=0.78,
    )

    # 7. Validate
    validated = await service.validate_intelligence(
        intel_id=intel["intel_id"],
        validation_method="CORROBORATION",
    )
    print(f"Validated credibility: {validated['validated_credibility']}")

    # 8. Apply temporal decay (90 days old)
    decayed = await service.intel_credibility_decay(
        intel_id=intel["intel_id"],
        age_days=90,
    )
    print(f"Decayed credibility: {decayed['decayed_credibility']}")

    # 9. Generate report
    report_doc = await service.humint_report(classification="secret")
    print(report_doc["summary"])

asyncio.run(operational_flow())
```

---

## Governance Workflows

### Authority Management

Authorities define the legal mandate for source handling. They must include
an approver, expiry date, classification, and evidence reference.

```python
# Check for expired or soon-to-expire authorities
expiry_report = await service.authority_expiry_check()
for expired_auth in expiry_report["expired"]:
    print(f"EXPIRED: {expired_auth['authority_id']} — {len(expired_auth['linked_sources'])} sources at risk")
```

Authorities expiring within 30 days appear in `expiring_soon`.
Linked sources for expired authorities are returned in `linked_sources`
and should be moved to `SUSPENDED` via `source_lifecycle_management`.

### Source Lifecycle

```python
# Suspend a source (e.g. while renewing expired authority)
action = await service.source_lifecycle_management(
    source_id="src-1",
    action="SUSPEND",
)

# Reactivate once authority is renewed
action = await service.source_lifecycle_management(
    source_id="src-1",
    action="REACTIVATE",
)
```

Valid actions: `ACTIVATE`, `SUSPEND`, `REACTIVATE`, `TERMINATE`, `ARCHIVE`.

### Source Vetting

```python
vetting = await service.source_vetting(
    source_id="src-1",
    vetter_id="security-officer-1",
)
print(vetting["outcome"])  # APPROVED | CONDITIONAL | REJECTED
```

---

## Source Protection

### Threat Assessment

```python
protection = await service.source_protection(
    source_id="src-1",
    threat_level="HIGH",
)
# Returns recommended measures: STERILE_COMMS_ONLY, EXFILTRATION_PLAN_ACTIVATED, ...
```

### Welfare Monitoring

Welfare trend analysis uses a rolling 3-report moving average and triggers
an alert if the average drops below 0.4.

```python
welfare = await service.welfare_trend_analysis(source_id="src-1")
if welfare["welfare_alert"]:
    print(f"ALERT: {welfare['alert_reason']} — current avg {welfare['current_3report_avg']}")
```

### Compartment Compliance

```python
compartment_check = await service.source_compartment_check(
    source_id="src-1",
    compartment_ids=["COMPARTMENT_ALPHA", "COMPARTMENT_BRAVO"],
)
if not compartment_check["all_compliant"]:
    print("Issues:", compartment_check["issues"])
```

---

## Intelligence Analysis

### Credibility Decay

All intelligence items degrade in credibility over time. Apply temporal
decay before using old items in assessments.

```python
decayed = await service.intel_credibility_decay(
    intel_id="some-intel-id",
    age_days=180,          # 6 months old
    decay_lambda=0.005,    # ~50% at 139 days
)
# decayed["decayed_credibility"] is the effective credibility now
```

### Cross-Reference

```python
xref = await service.cross_reference_human_intel(
    intel_id="primary-intel-id",
    other_sources=["intel-2", "intel-3", "intel-4"],
)
print(f"Corroboration score: {xref['corroboration_score']}")
```

### Analytical Assessment

```python
assessment = await service.analytical_assessment(
    subject="target_org_leadership",
    time_window_days=90,
)
print(f"Coverage: {assessment['collection_coverage']}, Gaps: {assessment['knowledge_gaps']}")
```

### Bulk Validation

```python
results = await service.bulk_validate_intelligence([
    {"intel_id": "intel-1", "validation_method": "CORROBORATION"},
    {"intel_id": "intel-2", "validation_method": "SIGINT_CROSSREF"},
    {"intel_id": "intel-3", "validation_method": "ANALYST_REVIEW"},
])
print(f"Validated {results['succeeded']}/{results['submitted']}")
```

---

## Handler Management

### Assign Optimal Handler

```python
assignment = await service.assign_handler(
    source_id="src-1",
    candidate_handler_ids=["handler-jones", "handler-smith", "handler-odera"],
)
print(f"Recommended: {assignment['recommended_handler_id']}")
# Includes full ranked list with composite scores
```

### Handler Performance

```python
perf = await service.handler_performance(
    handler_id="handler-jones",
    period="Q2-2026",
)
print(f"Band: {perf['performance_band']}, Welfare avg: {perf['mean_welfare_score']}")
```

---

## Network Analysis

### Source Network

```python
network = await service.source_network_analysis(
    source_ids=["src-1", "src-2", "src-3"],
)
if network["potential_network_compromise"]:
    print(f"Warning: {network['common_subject_count']} shared subjects detected")
```

### OPSEC Assessment

Holistic assessment combining counter-HUMINT indicators, per-source risk
scores, and welfare trends.

```python
opsec = await service.operational_security_assessment(
    operation_name="OP_GRANITE",
    source_ids=["src-1", "src-2"],
)
print(f"OPSEC score: {opsec['aggregate_opsec_score']}, Posture: {opsec['recommended_posture']}")
# Posture: NORMAL | ELEVATED | HEIGHTENED | SUSPEND
```

---

## Intelligence Cycle Closure

### Collection Cycle Feedback

```python
feedback = await service.collection_cycle_feedback(
    cycle_id="cycle-q2-2026",
    priorities=["target_org_finances", "target_network_expansion", "key_personnel"],
)
for p in feedback["priority_feedback"]:
    if p["coverage_gap"]:
        print(f"GAP: {p['priority']} — coverage {p['coverage_ratio']:.0%}, escalated to {p['escalated_urgency']}")
```

### Full Reporting Cycle

```python
cycle = await service.reporting_cycle(cycle="Q2-2026")
print(f"Unvalidated items: {cycle['unvalidated_collections']}")
print(f"Cycle complete: {cycle['cycle_complete']}")
```

---

## Dissemination

### Compliance Check

Always run a compliance check before releasing intelligence.

```python
check = await service.dissemination_compliance_check(
    dissemination_id="dis-1",
)
if not check["compliant"]:
    print("Release blocked:", check["failures"])
```

### Intelligence Sharing

```python
sharing = await service.intelligence_sharing(
    intel_id="intel-1",
    recipient_agencies=["PARTNER_AGENCY_A", "PARTNER_AGENCY_B"],
    classification="confidential",
)
```

---

## AI Agent Automation

Register AI agents for HUMINT automation tasks:

```python
agent = service.register_humint_agent(
    "agent-1", "tenant-a", "DebriefingAnalyst",
    "claude_code", "debriefing_analyst", "scope://debriefings",
)

# Validate any privileged action before execution
service.validate_agent_action(
    tenant_id="tenant-a",
    privileged_scope=True,
    human_approval_recorded=True,  # must be True for privileged actions
)
```

Coercive scope, cross-tenant scope, privilege escalation, autonomous
dissemination, and source identity disclosure are unconditionally denied.

---

## Dashboard and Reporting

```python
# Operational summary
summary = service.dashboard_summary("tenant-a")

# Gap analysis
gaps = await service.intelligence_gap_analysis()
print(gaps["gaps"])

# Health check
health = await service.health_check()
print(health["status"], health["pending_validations"])
```

---

## Rule Denials Reference

| Rule | Trigger |
|---|---|
| `tenant_context_required` | Missing or empty `tenant_id` |
| `humint_policy_required` | Write without `policy_attached=True` |
| `lawful_authority_required` | Source or plan without a valid authority |
| `source_authority_mismatch` | Plan source does not belong to stated authority |
| `safety_plan_required` | Contact plan missing safety plan reference |
| `coercive_humint_action_denied` | Agent action with `coercive_scope=True` |
| `human_approval_required` | Privileged agent action without recorded approval |
| `cross_tenant_humint_scope_denied` | Agent attempting cross-tenant scope |
| `source_identity_disclosure_scope_denied` | Agent attempting identity disclosure |
| `autonomous_dissemination_scope_denied` | Agent attempting autonomous release |
| `bytewax_event_stream_required` | Batch operation without Bytewax stream routing |

---

## UI Routes

| Path | Permission | Purpose |
|---|---|---|
| `/intel-humint/dashboard` | `intel_humint:view` | Operational overview |
| `/intel-humint/authorities` | `intel_humint:authorities` | Authority governance |
| `/intel-humint/sources` | `intel_humint:sources` | Source registry |
| `/intel-humint/contact-plans` | `intel_humint:contacts` | Contact planning |
| `/intel-humint/contact-reports` | `intel_humint:reports` | Contact outcomes |
| `/intel-humint/debriefings` | `intel_humint:analysis` | Debriefing workbench |
| `/intel-humint/reliability` | `intel_humint:analysis` | Reliability assessment |
| `/intel-humint/leads` | `intel_humint:leads` | Lead management |
| `/intel-humint/dissemination` | `intel_humint:dissemination` | Release management |
| `/intel-humint/reviews` | `intel_humint:reviews` | Governance reviews |
| `/intel-humint/agents` | `intel_humint:admin` | AI agent management |

---

## Further Reading

- `service.py` — Complete method implementations
- `models.py` — Data model definitions
- `capability_contract.py` — Rule engine and contract definition
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap for production-grade enhancements
- `SPECIFICATION.md` — Full capability specification
