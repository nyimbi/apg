# User Guide — Incident & Crisis Management (grc_icm)

**Capability ID**: `grc_icm` | **Domain**: `grc` | **Version**: 1.1.0  
© 2025 Datacraft — Author: Nyimbi Odero

---

## Contents

1. [Introduction](#1-introduction)
2. [Installation & Configuration](#2-installation--configuration)
3. [Incident Lifecycle Walkthrough](#3-incident-lifecycle-walkthrough)
4. [Playbook-Driven Response](#4-playbook-driven-response)
5. [Evidence Management & Chain of Custody](#5-evidence-management--chain-of-custody)
6. [Business Impact Assessment](#6-business-impact-assessment)
7. [Crisis War Room](#7-crisis-war-room)
8. [Regulatory Notifications](#8-regulatory-notifications)
9. [Third-Party / Vendor Notifications](#9-third-party--vendor-notifications)
10. [SLA Tracking](#10-sla-tracking)
11. [Compliance Testing & Deficiency Management](#11-compliance-testing--deficiency-management)
12. [Reporting & Analytics](#12-reporting--analytics)
13. [Executive Briefing Generator](#13-executive-briefing-generator)
14. [Incident Similarity Search](#14-incident-similarity-search)
15. [Business Continuity (BCMS)](#15-business-continuity-bcms)
16. [Composability](#16-composability)
17. [Permissions Reference](#17-permissions-reference)
18. [Configuration Reference](#18-configuration-reference)

---

## 1. Introduction

`grc_icm` provides end-to-end incident response and crisis management for APG. Covers:

- **Incident lifecycle**: report → triage → investigate → root cause → close
- **Playbook automation**: DAG-based response task management
- **Evidence management**: cryptographic chain-of-custody (SHA-256 hash chain)
- **Business impact assessment**: RTO/RPO tracking and BCP activation
- **Crisis war rooms**: audit-logged communication channels per incident
- **Regulatory notifications**: GDPR (72h), PCI-DSS (24h), CBK (24h), custom windows
- **Third-party coordination**: vendor and data-processor notification tracking
- **Compliance testing**: design / operating-effectiveness / walkthrough / inquiry
- **SLA monitoring**: automatic breach detection with owner notification
- **Analytics**: MTTR, SLA breach rate, resolution rate, executive briefings

---

## 2. Installation & Configuration

```bash
pip install apg-grc-icm
```

**Environment variables** (prefixed `GRC_ICM_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GRC_ICM_DB_URL` | in-memory | PostgreSQL async URL |
| `GRC_ICM_TENANT_ID` | `default` | Active tenant |
| `OLLAMA_BASE_URL` | — | Enables ML severity classification |

```python
from apg_grc_icm import IncidentComplianceService

svc = IncidentComplianceService(
    db_url="postgresql+asyncpg://user:pass@localhost/icm",
    tenant_id="acme_corp",
)
```

---

## 3. Incident Lifecycle Walkthrough

### Severity and Priority

| Severity | Auto-notification |
|----------|-------------------|
| critical | CISO + IR team immediate |
| high | Compliance team |
| medium / low | None |

| Priority | SLA |
|----------|-----|
| P1 | 4h |
| P2 | 8h |
| P3 | 24h |
| P4 | 72h |

### Status flow

`new → triaged → in_investigation → pending_closure → closed`

### Step-by-step

```python
# 1. Report
inc = await svc.report_incident(
    entity_id="ENT-001",
    incident_type="security_breach",
    description="Credential stuffing on API gateway",
    severity="high",
    affected_systems=["api-gateway"],
    reported_by="alice",
)

# 2. Triage
await svc.incident_triage(inc["id"], incident_commander_id="bob",
    priority="P2", initial_response="Rate limiting applied")

# 3. Assign investigator
await svc.investigation_assign(inc["id"], investigator_id="carol", scope="API logs June 2026")

# 4. Record investigation
await svc.incident_investigation(inc["id"], findings="Credential list from prior breach",
    evidence=[{"type": "log", "description": "WAF logs", "hash": "sha256:abc"}],
    investigator_id="carol")

# 5. RCA
rca = await svc.root_cause_analysis(inc["id"], rca_method="5_whys",
    root_causes=["No stuffing detection rule"],
    contributing_factors=["Shared passwords from prior breach"])
await svc.root_cause_confirm(inc["id"], root_cause="No stuffing detection rule", confirmed_by="bob")

# 6. Corrective action
action = await svc.corrective_action(inc["id"], action_type="corrective",
    description="Deploy stuffing detection", owner_id="carol", deadline="2026-06-25")
await svc.corrective_action_update(action["id"], progress_pct=100, notes="Deployed", updated_by="carol")
await svc.corrective_action_verify(action["id"], verified_by="bob", verification_notes="Confirmed")

# 7. Post-incident review (required for high/critical)
await svc.post_incident_review(inc["id"], review_date="2026-06-15",
    reviewers=["bob", "carol"],
    actions=[{"description": "Phishing training", "owner_id": "hr", "deadline": "2026-09-01"}])

# 8. Close
await svc.close_incident(inc["id"],
    resolution="Stuffing rule deployed; no accounts compromised",
    lessons_learned="API gateways must have stuffing detection from day one",
    closed_by="bob")
```

---

## 4. Playbook-Driven Response

```python
# Activate a response playbook
run = await svc.activate_playbook(inc["id"], playbook_id="PB-CRED-STUFF", activated_by="bob")

# Advance a task (unlocks dependent tasks automatically)
await svc.advance_playbook_task(
    run_id=run["id"],
    task_id=run["tasks"][0]["task_id"],
    status="completed",   # completed|blocked|skipped
    completed_by="ops",
    notes="300 IPs blocked",
)
```

Run auto-closes when all tasks reach a terminal state.

---

## 5. Evidence Management & Chain of Custody

Evidence is stored with a SHA-256 hash-chained custody log. Verify at any time:

```python
result = await svc.verify_evidence_chain(evidence_id="ev-uuid")
# {"valid": True, "custody_entries": 3}
# On tampering: {"valid": False, "failed_at_index": 1, "expected_hash": "..."}
```

---

## 6. Business Impact Assessment

```python
bia = await svc.business_impact_assessment(
    incident_id=inc["id"],
    impacted_processes=[
        {"process_name": "Online Banking", "rto_hours": 2.0,
         "hourly_revenue_impact": 500_000, "hours_down": 3.5,
         "current_recovery_state": "degraded"},
    ],
    assessed_by="risk-manager",
)
print(bia["total_financial_exposure"])       # 1_750_000.0
print(bia["bcp_activation_recommended"])     # True (RTO breached)
```

---

## 7. Crisis War Room

```python
room = await svc.create_war_room(inc["id"],
    participants=["bob", "ciso", "legal"],
    channel_type="teams",   # matrix|slack|teams|email
    created_by="bob")

await svc.war_room_post(room["id"], message="Containment confirmed.",
    posted_by="bob", audience="internal")

# External/press posts trigger compliance review notification
await svc.war_room_post(room["id"], message="Public statement draft.",
    posted_by="comms", audience="press")

await svc.close_war_room(room["id"], closed_by="bob", summary="Incident contained")
```

---

## 8. Regulatory Notifications

```python
notif = await svc.regulatory_notification(inc["id"],
    regulator="gdpr",          # gdpr|pci_dss|cbk|default
    notification_type="initial",  # initial|update|final
    deadline="2026-06-14T10:00:00Z")

print(notif["window_exceeded"])                  # False (within 72h)
print(notif["hours_elapsed_since_detection"])    # e.g. 18.5
```

Windows: GDPR 72h, PCI-DSS 24h, CBK 24h, default 72h.

---

## 9. Third-Party / Vendor Notifications

```python
vn = await svc.third_party_incident_notify(inc["id"],
    vendor_id="VENDOR-AWS", contact_email="security@aws.com",
    notification_scope="data_breach",   # data_breach|service_disruption|security_event|supply_chain
    notification_type="initial")

await svc.vendor_acknowledgement_record(vn["id"],
    acknowledged_by="aws-security",
    acknowledgement_notes="No exfiltration on our side.")
```

---

## 10. SLA Tracking

```python
status = await svc.get_sla_status(inc["id"])
# {"priority": "P2", "sla_hours": 8.0, "elapsed_hours": 5.2,
#  "remaining_hours": 2.8, "sla_breached": false}
```

When `elapsed_hours > sla_hours`, incident is auto-flagged `sla_breached=True`
and the owner is notified.

---

## 11. Compliance Testing & Deficiency Management

```python
test = await svc.compliance_test(entity_id="ENT-001", control_id="CTRL-IAM-01",
    test_type="operating_effectiveness", test_date="2026-06-01",
    result="fail", tester_id="auditor")

deficiency = await svc.compliance_deficiency(control_id="CTRL-IAM-01",
    deficiency_type="operating_ineffectiveness",
    severity="significant", identified_by="auditor")

plan = await svc.remediation_plan(deficiency["id"],
    remediation_actions=[{"step": "Enforce MFA", "owner": "iam-team"}],
    deadline="2026-07-31", owner_id="iam-team")

score = await svc.compliance_score(entity_id="ENT-001",
    framework="ISO27001", period="2026-Q2")
print(score["score"], score["rating"])  # 87.5, "satisfactory"
```

---

## 12. Reporting & Analytics

```python
# Incident analytics for a period
analytics = await svc.incident_analytics(entity_id="ENT-001", period="2026-Q2")

# KPI card
kpi = await svc.incident_kpi_summary(entity_id="ENT-001", period="2026-06")

# Compliance dashboard
dashboard = await svc.compliance_dashboard(entity_id="ENT-001")

# Regulatory report
report = await svc.regulatory_reporting_icm(period="2026-Q2", jurisdiction="KE-CBK")
```

---

## 13. Executive Briefing Generator

```python
briefing = await svc.generate_executive_briefing(inc["id"],
    generated_by="ciso", version=1)
print(briefing["content_md"])
# Structured Markdown: situation, impact, root cause,
# regulatory exposure, containment status, next actions.
```

Each call creates a versioned record in `icm_exec_briefings`.

---

## 14. Incident Similarity Search

```python
similar = await svc.find_similar_incidents(inc["id"], top_k=5)
for match in similar["similar_incidents"]:
    print(match["title"], match["similarity_score"], match["lessons_learned"])
```

Pure-Python TF-IDF cosine similarity — no external ML dependency.

---

## 15. Business Continuity (BCMS)

```python
# Activate BCP
activation = await svc.business_continuity_activation(inc["id"],
    bcp_plan_id="BCP-CYBER-001", activator_id="ciso")

# Capture lessons
await svc.lessons_learned_capture(inc["id"],
    lessons=["API gateways need stuffing detection baseline"],
    captured_by="carol")

# Preventive action plan
await svc.preventive_action_plan(inc["id"],
    actions=[{"description": "Deploy OWASP CRS", "owner": "security-arch"}],
    owner_id="security-arch", deadline="2026-08-01")
```

---

## 16. Composability

| Downstream Capability | Trigger |
|----------------------|---------|
| `intel_correlation` | `incident_reported` with MITRE annotations |
| `intel_threats` | Critical severity incidents |
| `grc_risk` | `corrective_action_created`, `compliance_deficiency_identified` |
| `grc_policy` | Compliance test failures |
| `fin_reporting` | `insurance_claim_triggered` |

```apg
use grc_icm;
use intel_correlation;

on grc_icm.incident_reported where severity = "critical" {
    intel_correlation.correlate_event(incident_id, source="icm");
}
```

---

## 17. Permissions Reference

| Permission | Grants |
|-----------|--------|
| `grc_icm:view` | Read incidents, dashboard, reports |
| `grc_icm:manage_incidents` | Create, triage, investigate, close |
| `grc_icm:manage_cases` | Case management |
| `grc_icm:manage_evidence` | Evidence collection, chain-of-custody |
| `grc_icm:manage_compliance` | Tests, deficiencies, remediation |
| `grc_icm:regulatory` | Regulatory and vendor notifications |
| `grc_icm:admin` | All permissions + playbook management |

---

## 18. Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_url` | str | in-memory | PostgreSQL async connection string |
| `tenant_id` | str | `default` | Active tenant ID |
| `notification_email_ir` | str | `incident-response@datacraft.co.ke` | IR team email |
| `notification_email_ciso` | str | `ciso@datacraft.co.ke` | CISO email |
| `ollama_base_url` | str | — | Enables ML severity classification |
| `sla_p1_hours` | float | 4.0 | P1 SLA threshold in hours |
| `sla_p2_hours` | float | 8.0 | P2 SLA threshold in hours |
| `sla_p3_hours` | float | 24.0 | P3 SLA threshold in hours |
| `sla_p4_hours` | float | 72.0 | P4 SLA threshold in hours |

---

_Further reading: `service.py` (business logic), `models.py` (Pydantic v2 models),
`api.py` (REST endpoints), `views.py` (Flask-AppBuilder views)._
