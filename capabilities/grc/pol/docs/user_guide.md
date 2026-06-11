# Policy Management

**Capability ID**: `grc_pol` | **Domain**: `grc` | **Version**: `1.0.0`

## Description

Policy Management provides a world-class, standalone-deployable implementation of policy management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-grc-pol
```

## Provides

- `policy_lifecycle_management`
- `policy_acknowledgement_workflow`
- `policy_exception_workflow`
- `policy_review_workflow`
- `policy_publication_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-pol/dashboard` | `grc_pol:view` | Overview |
| `/grc-pol/policies` | `grc_pol:manage_policies` | Policies |
| `/grc-pol/policies/:id` | `grc_pol:view` | Policies |
| `/grc-pol/acknowledgements` | `grc_pol:manage_acknowledgements` | Compliance |
| `/grc-pol/exceptions` | `grc_pol:manage_exceptions` | Governance |
| `/grc-pol/reviews` | `grc_pol:review` | Governance |
| `/grc-pol/review-calendar` | `grc_pol:view` | Planning |
| `/grc-pol/gap-analysis` | `grc_pol:view` | Analysis |

## Key Service Methods

- `_audit_event()`
- `_get_policy()`
- `create_policy()`
- `draft_policy_content()`
- `policy_review()`
- `approve_policy()`
- `publish_policy()`
- `acknowledge_policy()`
- `policy_exception_request()`
- `approve_exception()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_pol` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_pol;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_POL_`.

## 3. Policy Lifecycle Walkthrough

### 3.1 Create a Draft Policy

```python
from apg_grc_pol import PolicyManagementService

svc = PolicyManagementService(tenant_id="acme")

policy = await svc.create_policy(
    title="Remote Working Policy",
    category="hr",
    policy_type="hr",
    owner_id="hr_manager@acme.com",
    effective_date="2026-07-01",
    review_cycle_months=12,
    scope="organization_wide",
)
```

### 3.2 Add Content Sections

```python
await svc.draft_policy_content(
    policy["id"],
    content_sections=[
        {"section_number": 1, "title": "Purpose", "body": "This policy establishes..."},
        {"section_number": 2, "title": "Scope", "body": "Applies to all employees..."},
    ],
    author_id="hr_manager@acme.com",
)
```

### 3.3 Submit for Review

```python
await svc.policy_review(
    policy["id"],
    reviewer_id="legal@acme.com",   # must differ from owner
    comments="Minor wording needed in section 2.",
    recommended_action="request_changes",
)
```

### 3.4 Approve and Publish

```python
await svc.approve_policy(policy["id"], "coo@acme.com", "2026-06-15")
await svc.publish_policy(policy["id"], ["emp1@acme.com", "emp2@acme.com"])
```

---

## 4. Attestation Campaigns

```python
campaign = await svc.create_attestation_campaign(
    policy["id"],
    campaign_name="Q3 2026 ISMS Attestation",
    target_employee_ids=["emp1@acme.com", "emp2@acme.com"],
    start_date="2026-07-01",
    end_date="2026-07-31",
    completion_sla_pct=95.0,
    created_by="compliance@acme.com",
    chase_interval_days=7,
)

# Chase overdue employees
await svc.acknowledgement_chase(policy["id"], chased_by="compliance@acme.com")
```

---

## 5. Exception Handling

```python
# Request
exception = await svc.policy_exception_request(
    policy["id"],
    requestor_id="team_lead@acme.com",
    reason="Legacy system limitation.",
    compensating_controls="Manual quarterly review.",
    risk_level="medium",
    duration_days=90,
)

# Approve (must differ from requestor)
await svc.approve_exception(
    exception["id"], approver_id="ciso@acme.com",
    approved_until="2026-09-30", conditions="Monthly status reports.",
)

# Monitor
monitor = await svc.policy_exception_monitor()
```

---

## 6. Policy Revision and Delta

```python
revision = await svc.policy_revision(
    policy["id"],
    revision_reason="CBK Circular 2026-03 update",
    revision_summary="Added data localisation requirements.",
    revised_by="compliance@acme.com",
)

delta = await svc.policy_delta_report(policy["id"], "1.0", "1.1")
print(delta["delta"]["changes_from_revision"])
```

---

## 7. Compliance and Gap Analysis

```python
# Per-policy compliance check
compliance = await svc.policy_compliance_check("acme", policy["id"])
print(compliance["acknowledgement_rate_pct"])
print(compliance["overall_compliant"])

# Framework gap analysis
gaps = await svc.policy_gap_analysis("acme", "iso_27001")
print(gaps["missing_types"])
print(gaps["coverage_pct"])

# Map to regulations
await svc.policy_mapping(policy["id"], ["GDPR_Art5"], ["ISO27001_A.5.1"])
```

---

## 8. Policy Hierarchy and Conflict Detection

```python
# Link child to parent
await svc.policy_set_parent(
    policy_id=child_policy["id"],
    parent_policy_id=isms_policy["id"],
    set_by="ciso@acme.com",
)

# Check for conflicts before review
conflicts = await svc.policy_conflict_check(draft_policy["id"])
for c in conflicts["conflicts"]:
    print(c["conflicting_title"], c["overlap_reasons"])
```

---

## 9. Templates and Bulk Import

```python
# Create template
template = await svc.policy_template(
    template_name="ISMS Policy Template v2",
    policy_type="information_security",
    scope="organization_wide",
    standard_sections=["Purpose", "Scope", "Policy Statement", "Compliance"],
)

# Instantiate from template
new_policy = await svc.create_policy_from_template(
    template_id=template["id"],
    title="Access Control Policy",
    owner_id="ciso@acme.com",
    effective_date="2026-08-01",
    review_cycle_months=12,
)

# Bulk import from migration
result = await svc.policy_bulk_import(
    records=[
        {
            "title": "BYOD Policy", "category": "it",
            "policy_type": "acceptable_use", "owner_id": "it_mgr@acme.com",
            "effective_date": "2025-01-01", "review_cycle_months": 12,
        },
    ],
    imported_by="compliance@acme.com",
)
print(result["created_count"], result["errors"])
```

---

## 10. Analytics and Reporting

```python
# Period analytics
analytics = await svc.policy_analytics("2026-06")

# Policies due for review
expiry = await svc.policy_expiry_report(days_ahead=60)
print(expiry["due_count"], expiry["overdue_count"])

# Dashboard
dashboard = await svc.policy_dashboard("acme")
print(dashboard["overall_acknowledgement_rate_pct"])

# KPI summary
kpi = await svc.policy_kpi_summary("acme", "2026-Q2")
```

---

## 11. Cross-Capability Events

```python
event = await svc.policy_event_publish(
    event_type="policy_published",
    policy_id=policy["id"],
    payload={"distribution_count": 10},
    actor_id="compliance@acme.com",
)
```

Supported: `policy_published`, `policy_archived`, `exception_approved`, `policy_revised`, `policy_retired`, `attestation_completed`, `policy_drafted`, `policy_approved`.

---

## 12. Search

```python
results = await svc.policy_search_advanced(
    "remote working", policy_type="hr", status="published"
)
```

---

## 13. Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GRC_POL_DB_URL` | in-memory | SQLAlchemy async database URL |
| `GRC_POL_ACK_DEADLINE_DAYS` | 30 | Default acknowledgement deadline (days) |
| `GRC_POL_EXCEPTION_MAX_DAYS` | 365 | Maximum exception duration (days) |
| `GRC_POL_COMPLIANCE_THRESHOLD_PCT` | 95.0 | Ack rate % for compliant status |
| `OLLAMA_BASE_URL` | (unset) | Enable AI-assisted compliance scoring |

---

## Further Reading

- `service.py` — All async business logic methods
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder blueprint views
- `capability_contract.py` — Governance rules
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned architectural improvements
- `SPECIFICATION.md` — Full functional specification
- `README.md` — Quick reference
