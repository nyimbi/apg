# Risk and Security Assessment — User Guide

**Capability ID**: `grc_rsa` | **Domain**: `grc` | **Version**: `1.1.0`

## Description

Risk and Security Assessment provides a world-class, standalone-deployable implementation of enterprise risk and security assessment for the APG platform. Version 1.1.0 adds a native CVSS v3.1 scoring engine, full penetration testing engagement lifecycle, vulnerability register with SLA enforcement, formal risk acceptance workflow with expiry tracking, vendor risk tiering, and cross-entity portfolio aggregation.

## Installation

```bash
pip install apg-grc-rsa
```

---

## Core Concepts

### Risk Scoring

Inherent risk = likelihood (1–5) × impact (1–5). Rating bands:

| Score | Rating |
|-------|--------|
| ≥ 20 | critical |
| ≥ 12 | high |
| ≥ 6  | medium |
| ≥ 2  | low |
| < 2  | negligible |

Residual risk = inherent × (1 − control_effectiveness_pct / 100).

### CVSS v3.1 Scoring

The built-in CVSS engine accepts standard vector strings and returns the numeric base score (0.0–10.0), severity label, and exploitability/impact sub-scores.

| Score | Severity |
|-------|----------|
| 9.0–10.0 | Critical |
| 7.0–8.9  | High |
| 4.0–6.9  | Medium |
| 0.1–3.9  | Low |
| 0.0      | None |

### SLA Deadlines (CISA KEV-aligned)

| Severity | Remediation Deadline |
|----------|----------------------|
| Critical | 24 hours |
| High | 7 days |
| Medium | 30 days |
| Low | 90 days |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-rsa/dashboard` | `grc_rsa:view` | Overview |
| `/grc-rsa/assessments` | `grc_rsa:manage_assessments` | Assessments |
| `/grc-rsa/assessments/:id` | `grc_rsa:view` | Assessments |
| `/grc-rsa/findings` | `grc_rsa:manage_findings` | Findings |
| `/grc-rsa/findings/:id` | `grc_rsa:view` | Findings |
| `/grc-rsa/remediation` | `grc_rsa:manage_remediation` | Remediation |
| `/grc-rsa/vendor-risk` | `grc_rsa:manage_vendor_risk` | Vendor Risk |
| `/grc-rsa/threat-model` | `grc_rsa:view` | Threat Intelligence |
| `/grc-rsa/pentest` | `grc_rsa:manage_pentest` | Pen Testing |
| `/grc-rsa/vulnerabilities` | `grc_rsa:manage_findings` | Vulnerabilities |
| `/grc-rsa/portfolio` | `grc_rsa:view` | Portfolio |

---

## Workflow Guides

### 1. Risk Register Lifecycle

```python
from apg_grc_rsa.service import RiskAssessmentService

svc = RiskAssessmentService(tenant_id="acme")

# Step 1 — Register
risk = await svc.risk_register_entry(
    "ENT-1", "Data Breach via API", "technology",
    "Sensitive PII exposed through unauthenticated endpoints", "alice"
)

# Step 2 — Assess
assessment = await svc.risk_assessment(risk["id"], 4, 5, "high", "alice")
# inherent_score = 20.0, inherent_rating = "critical"

# Step 3 — Assign owner
await svc.risk_owner_assign(risk["id"], "bob", "alice")

# Step 4 — Create treatment plan
plan = await svc.risk_treatment_plan(
    risk["id"], "mitigate",
    [
        {"description": "Implement API authentication", "action_owner": "bob", "due_date": "2025-08-01"},
        {"description": "Add rate limiting", "action_owner": "charlie", "due_date": "2025-08-15"},
    ],
    "bob", "2025-08-31"
)

# Step 5 — Update progress
await svc.risk_treatment_update(plan["id"], 50.0, "API auth implemented; rate limiting in progress", "bob")

# Step 6 — Approve plan
await svc.risk_treatment_approve(plan["id"], "ciso", "Adequate mitigations planned")
```

### 2. CVSS Scoring

```python
# Score from a vector string
result = await svc.cvss_score(
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    risk_id=risk["id"]  # optionally link to a risk
)
print(result["base_score"])   # 9.8
print(result["severity"])     # critical
print(result["exploitability_score"])  # 3.9
print(result["impact_score"])          # 5.9
```

### 3. Vulnerability Register with SLA

```python
# Register a vulnerability — SLA deadline auto-computed
vuln = await svc.vulnerability_register(
    "ENT-1",
    "SQL Injection in /login endpoint",
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    "api.acme.com/login",
    "alice",
    cve_id="CVE-2024-12345",
    source="pentest",
)
# remediation_deadline is set to today + 1 day (Critical severity)

# Check SLA compliance across all open vulnerabilities for an entity
sla_report = await svc.vulnerability_sla_check("ENT-1")
print(sla_report["sla_breached"])     # count of breached items
print(sla_report["sla_at_risk_3d"])   # items breaching within 3 days

# Update patch status
await svc.vulnerability_patch_update(
    vuln["id"], "patched",
    "Applied parameterised queries in commit abc123", "bob"
)
# status is automatically set to "closed"
```

### 4. Penetration Testing Lifecycle

```python
# Step 1 — Create engagement
engagement = await svc.pentest_engagement_create(
    "ENT-1",
    "Q3 2025 External Infrastructure Pentest",
    ["192.168.1.0/24", "api.acme.com", "admin.acme.com"],
    "grey_box",
    "RedTeam-Alpha",
    "2025-09-01",
    "2025-09-14",
)

# Step 2 — Record findings (CVSS scored automatically)
finding = await svc.pentest_finding_record(
    engagement["id"],
    "Unauthenticated Remote Code Execution",
    "The /upload endpoint processes unsanitised filenames allowing RCE via path traversal.",
    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
    "api.acme.com/upload",
    "redteam-bob",
    cve_id="CVE-2024-56789",
    proof_of_concept="curl -X POST 'https://api.acme.com/upload' -F 'file=@shell.php;filename=../../../../shell.php'",
    remediation_advice="Validate and sanitise all user-supplied filenames; store uploads outside webroot.",
)

# Step 3 — Schedule retest after remediation
retest = await svc.pentest_retest_schedule(
    finding["id"], "2025-09-28", "redteam-alice"
)

# Step 4 — Generate report
report = await svc.pentest_report_generate(
    engagement["id"],
    "The Q3 external pentest identified 3 critical and 2 high findings. "
    "Immediate remediation of the RCE vulnerability is required.",
    "redteam-lead",
)
print(f"Report generated: {report['id']}, total findings: {report['total_findings']}")
```

### 5. Risk Acceptance Workflow

```python
# Step 1 — Submit acceptance request
acceptance = await svc.risk_acceptance_request(
    risk["id"],
    "alice",
    "Residual risk is below board-approved tolerance; mitigating controls are operational.",
    "2026-06-30",   # expiry date — acceptance must be renewed annually
)

# Step 2 — Approve (governance / CISO)
approved = await svc.risk_acceptance_approve(
    acceptance["id"],
    "ciso",
    "Approved subject to quarterly KRI review.",
)

# Step 3 — Run expiry check periodically (e.g. via cron)
expired = await svc.risk_acceptance_expiry_check("ENT-1")
print(f"Expired and reverted: {expired['expired_and_reverted']} risks")
```

### 6. Vendor Risk Assessment

```python
vendor_result = await svc.vendor_risk_assess(
    "VND-007",
    "CloudStorage Inc.",
    "confidential",         # data_sensitivity
    "read_write",           # access_level
    {
        "information_security": 75.0,
        "business_continuity": 60.0,
        "data_privacy": 82.0,
        "incident_response": 70.0,
    },
    "alice",
)
print(f"Tier {vendor_result['tier']} — {vendor_result['risk_rating']}")
# Tier 2 — high (confidential data + read_write access)
# review_due is set to today + 180 days for Tier 1/2
```

### 7. Portfolio Risk Aggregation

```python
# Aggregate risk posture across all subsidiaries
portfolio = await svc.portfolio_risk_aggregate(
    ["ACME-UK", "ACME-KE", "ACME-NG", "ACME-ZA"],
    "2025",
)
print(portfolio["total_risks"])
print(portfolio["concentration_by_category"])  # e.g. {"technology": 0.42, "operational": 0.28, ...}

# Top-10 highest risks across the group
for r in portfolio["top_10_portfolio_risks"]:
    print(f"{r['entity_id']:12} {r['inherent_score']:5}  {r['risk_name']}")
```

### 8. KRI Monitoring

```python
# Define a KRI
kri = await svc.key_risk_indicator(
    "Mean Time to Patch (days)",
    threshold_amber=15.0,
    threshold_red=30.0,
    current_value=22.5,
    period="2025-07",
    entity_id="ENT-1",
    unit="days",
)
# status = "amber"; breach alert fired automatically

# Define without initial measurement
await svc.kri_define("Patch SLA Breach Rate (%)", 5.0, 15.0, "ENT-1", unit="%")
```

---

## Key Service Methods

### Risk Register & Assessment
- `risk_register_entry(entity_id, risk_name, category, description, owner_id)`
- `risk_assessment(risk_id, likelihood_1_5, impact_1_5, velocity, assessor_id)`
- `update_residual_score(risk_id, control_effectiveness_pct)`
- `risk_heat_map(entity_id, as_of_date)`
- `risk_escalate(risk_id, escalated_to, reason)`
- `risk_owner_assign(risk_id, owner_id, assigned_by)`

### CVSS & Vulnerabilities (v1.1.0)
- `cvss_score(vector_string, *, vulnerability_id, risk_id)`
- `vulnerability_register(entity_id, title, cvss_vector, affected_asset, discovered_by)`
- `vulnerability_sla_check(entity_id)`
- `vulnerability_patch_update(vulnerability_id, patch_status, notes, updated_by)`

### Penetration Testing (v1.1.0)
- `pentest_engagement_create(entity_id, name, scope, methodology, tester_team, start_date, end_date)`
- `pentest_finding_record(engagement_id, title, description, cvss_vector, affected_asset, tester_id)`
- `pentest_retest_schedule(finding_id, retest_date, assigned_tester)`
- `pentest_report_generate(engagement_id, executive_summary, generated_by)`

### Controls & Treatment
- `control_assessment(control_id, effectiveness_rating, evidence, assessed_by)`
- `control_gap(risk_id)`
- `risk_treatment_plan(risk_id, treatment_type, actions, owner_id, deadline)`
- `risk_treatment_update(treatment_id, progress_pct, notes, updated_by)`
- `risk_treatment_approve(treatment_id, approver_id, comments)`

### Risk Acceptance (v1.1.0)
- `risk_acceptance_request(risk_id, requestor_id, justification, expiry_date)`
- `risk_acceptance_approve(acceptance_id, approver_id, comments)`
- `risk_acceptance_expiry_check(entity_id)`

### KRI & Appetite
- `key_risk_indicator(kri_name, threshold_amber, threshold_red, current_value, period)`
- `kri_breach_alert(kri_id, breach_level, current_value)`
- `risk_appetite_statement(entity_id, risk_category, tolerance_level)`
- `kri_define(kri_name, threshold_amber, threshold_red, entity_id)`

### Vendor Risk (v1.1.0)
- `vendor_risk_assess(vendor_id, vendor_name, data_sensitivity, access_level, questionnaire_scores, assessed_by)`

### Portfolio & Reporting (v1.1.0)
- `portfolio_risk_aggregate(entity_ids, period)`
- `board_risk_report(entity_id, period)`
- `risk_reporting(entity_id, report_type, period)`
- `risk_analytics(entity_id, period)`
- `risk_kpi_summary(entity_id, period)`
- `risk_dashboard(entity_id)`

_(See `service.py` for complete API.)_

---

## Interoperability

`grc_rsa` integrates with other APG capabilities through the composition engine:

```apg
use grc_rsa;
```

The capability publishes events consumed by:
- `grc_audit` — risk and control audit trail
- `intel_alerts` — KRI breach and critical finding alerts
- `ntfy` — email/SMS notifications for critical findings and SLA breaches

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_RSA_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GRC_RSA_DB_URL` | in-memory | PostgreSQL connection URL |
| `GRC_RSA_NOTIFY_EMAIL` | `security@datacraft.co.ke` | Alert recipient email |
| `OLLAMA_BASE_URL` | (unset) | Enables AI risk narratives if set |

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned enhancements
- `SPECIFICATION.md` — Full capability specification
