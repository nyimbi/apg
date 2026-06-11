# Multi-Country Operations — User Guide

**Capability ID**: `loc_mco` | **Domain**: `loc` | **Version**: `1.1.0`

---

## Description

Multi-Country Operations (MCO) provides country entity management, local regulatory compliance mapping, cross-border intercompany transaction governance, statutory reporting, and OECD BEPS Action 13 Country-by-Country Reporting for organisations operating across multiple jurisdictions.

It enforces arms-length transfer pricing, tenant-scoped entity isolation, and audit-trailed compliance workflows across all supported jurisdictions.

---

## Installation

```bash
pip install apg-loc-mco
```

---

## Quick Start

```python
import asyncio
from apg_loc_mco.service import MultiCountryOperationsService
from apg_loc_mco.models import CountryCreate, EntityCreate

svc = MultiCountryOperationsService()

async def main():
    # 1. Register a jurisdiction
    country = await svc.register_country(CountryCreate(
        tenant_id="acme",
        name="Kenya",
        jurisdiction="ke",
        functional_currency="KES",
        regulatory_framework="IFRS",
        tax_registration_required=True,
    ))

    # 2. Register a legal entity
    entity = await svc.register_entity(EntityCreate(
        tenant_id="acme",
        name="Acme Kenya Ltd",
        entity_type="subsidiary",
        country_id=country.id,
        registration_number="CPR/2024/001",
        functional_currency="KES",
    ))

    # 3. Dashboard
    dash = await svc.dashboard_summary("acme")
    print(dash)

asyncio.run(main())
```

---

## Provides

| Service | Description |
|---------|-------------|
| `country_entity_management` | Register and manage country records with jurisdiction, functional currency, and regulatory framework |
| `regulatory_compliance_mapping` | Map compliance domains (tax, AML, data protection) to entities with owner assignment and evidence tracking |
| `intercompany_transaction_workflow` | Create, approve, and settle intercompany transactions with transfer pricing validation |
| `statutory_reporting_workflow` | Draft, review, file, and track acceptance of statutory reports per entity and period |
| `transfer_pricing_validation` | Validate OECD-aligned transfer pricing methods and documentation per transaction |
| `cross_border_governance` | Enforce arms-length, approval, and jurisdiction rules across all cross-border operations |
| `multi_entity_consolidation_data` | IFRS 10-correct consolidation with intercompany elimination |
| `jurisdiction_registry` | Tenant-scoped registry of active jurisdictions and their regulatory profiles |
| `compliance_monitoring` | Surface overdue reviews, non-compliant entities, and overdue statutory filings |
| `cbcr_reporting` | OECD BEPS Action 13 Country-by-Country Report (Table I & II) |

---

## Requires

| Capability | Reason |
|-----------|--------|
| `auth` | User identity and permission enforcement |
| `audl` | Immutable audit trail for all write operations |
| `mten` | Tenant context isolation |
| `conf` | Configuration management |
| `ntfy` | Overdue filing and compliance alerts |
| `wflo` | Intercompany approval workflow state machine |
| `comp` | Regulatory compliance engine integration |
| `moni` | Operational monitoring of compliance and filing SLAs |
| `mqeb` | Event bus for streaming MCO lifecycle events via bytewax |

---

## Service Methods

### Country Management

| Method | Description |
|--------|-------------|
| `register_country(payload, actor_id)` | Register a country/jurisdiction |
| `get_country(tenant_id, country_id)` | Retrieve a country by ID |
| `list_countries(tenant_id, status)` | List countries, optionally filtered by status |
| `update_country(tenant_id, country_id, payload, actor_id)` | Update a country record |

### Entity Management

| Method | Description |
|--------|-------------|
| `register_entity(payload, actor_id)` | Register a single legal entity |
| `register_entities_bulk(payloads, actor_id)` | Register multiple entities concurrently (asyncio.gather) |
| `get_entity(tenant_id, entity_id)` | Retrieve an entity by ID |
| `list_entities(tenant_id, country_id, entity_type, is_active)` | List entities with optional filters |
| `update_entity(tenant_id, entity_id, payload, actor_id)` | Update entity record |
| `get_entity_hierarchy(tenant_id, root_entity_id)` | BFS ownership tree from a root entity |

### Compliance

| Method | Description |
|--------|-------------|
| `record_compliance_mapping(payload, actor_id)` | Record a regulatory compliance mapping for an entity |
| `get_compliance_mapping(tenant_id, mapping_id)` | Retrieve a compliance mapping |
| `list_compliance_mappings(tenant_id, entity_id, domain, status)` | List compliance mappings |
| `update_compliance_mapping(tenant_id, mapping_id, payload, actor_id)` | Update compliance mapping status |
| `compliance_review_alerts(tenant_id, lookahead_days)` | Surface mappings due for review within N days |
| `get_compliance_mapping_history(tenant_id, mapping_id)` | Ordered audit-event history for a mapping |

### Intercompany Transactions

| Method | Description |
|--------|-------------|
| `create_intercompany_transaction(payload, actor_id)` | Create a transaction with TP validation |
| `get_intercompany_transaction(tenant_id, txn_id)` | Retrieve a transaction |
| `list_intercompany_transactions(tenant_id, entity_id, txn_type, status)` | List transactions |
| `approve_intercompany_transaction(tenant_id, txn_id, approver_id, approval_reference)` | Approve a pending transaction |
| `settle_intercompany_transaction(tenant_id, txn_id, settlement_date, actor_id)` | Mark approved transaction settled |
| `validate_transfer_pricing(tenant_id, txn_id, tp_method, documentation_reference)` | Validate TP method and documentation |
| `intercompany_exposure_summary(tenant_id, reporting_currency, fx_rates)` | FX-normalised outstanding exposure |

### Statutory Reports

| Method | Description |
|--------|-------------|
| `create_statutory_report(payload, actor_id)` | Create a statutory report |
| `get_statutory_report(tenant_id, report_id)` | Retrieve a report |
| `list_statutory_reports(tenant_id, entity_id, report_type, status)` | List reports |
| `file_statutory_report(tenant_id, report_id, filer_id, filed_date)` | Mark a report as filed |
| `accept_statutory_report(tenant_id, report_id, acceptance_reference, actor_id)` | Record acceptance by authorities |
| `escalate_overdue_reports(tenant_id, escalation_owner_id, actor_id)` | Escalate all overdue reports |
| `statutory_report_schedule(tenant_id, entity_id, year)` | Return statutory deadline schedule for a year |

### Consolidation & Reporting

| Method | Description |
|--------|-------------|
| `holding_consolidation_with_elimination(tenant_id, parent_id, subsidiaries, period, reporting_currency, fx_rates, actor_id)` | IFRS 10-correct consolidation with intercompany elimination |
| `generate_cbcr_report(tenant_id, fiscal_year)` | OECD BEPS Action 13 CbCR (Table I & II) |
| `mco_analytics(tenant_id, period)` | MCO operational analytics for a period |
| `mco_kpi_dashboard(tenant_id, period)` | Concise KPI card for dashboard consumption |

### Agents & Audit

| Method | Description |
|--------|-------------|
| `register_agent(payload, actor_id)` | Register an MCO automation agent |
| `list_agents(tenant_id)` | List all agents for a tenant |
| `validate_agent_action(tenant_id, privileged_scope, human_approval_recorded)` | Validate agent action permissibility |
| `dashboard_summary(tenant_id)` | Aggregate counts and status breakdown |
| `list_audit_events(tenant_id, limit)` | Recent audit events, newest first |
| `describe(tenant_id)` | Full capability contract |
| `evaluate(context)` | Evaluate capability rules against a context dict |

---

## Bulk Entity Registration

```python
from apg_loc_mco.models import EntityCreate

payloads = [
    EntityCreate(tenant_id="acme", name=f"Sub {i}", entity_type="subsidiary",
                 country_id=country.id, registration_number=f"CPR/2024/{i:03}",
                 functional_currency="USD")
    for i in range(50)
]

result = await svc.register_entities_bulk(payloads, actor_id="onboarding-bot")
print(f"Registered: {len(result['succeeded'])}, Failed: {len(result['failed'])}")
```

---

## Compliance Review Alerts

```python
# Alert on items due within 30 days
alerts = await svc.compliance_review_alerts("acme", lookahead_days=30)
for alert in alerts:
    urgency = "OVERDUE" if alert["overdue"] else f"due in {alert['days_remaining']}d"
    print(f"{alert['domain']} / {alert['entity_id']} — {urgency}")
```

---

## Entity Hierarchy

```python
hierarchy = await svc.get_entity_hierarchy("acme", root_entity_id=parent.id)
# Returns nested tree: { id, name, entity_type, depth, children: [...], descendant_count }
```

---

## FX-Normalised Exposure

```python
exposure = await svc.intercompany_exposure_summary(
    "acme",
    reporting_currency="USD",
    fx_rates={"KES": 0.0077, "EUR": 1.09, "GBP": 1.27},
)
print(f"Gross exposure: {exposure['reporting_currency']} {exposure['gross_exposure']:,.2f}")
```

---

## IFRS 10 Consolidation with Elimination

```python
consol = await svc.holding_consolidation_with_elimination(
    tenant_id="acme",
    parent_id=holding.id,
    subsidiaries=[sub1.id, sub2.id, sub3.id],
    period="FY2025",
    reporting_currency="USD",
    fx_rates={"KES": 0.0077, "EUR": 1.09},
    actor_id="group_cfo",
)
print(f"Net consolidated revenue: {consol['net_consolidated_revenue']:,.2f}")
print(f"Eliminated intercompany: {consol['eliminated_intercompany_amount']:,.2f}")
```

---

## OECD CbCR Report

```python
cbcr = await svc.generate_cbcr_report(tenant_id="acme", fiscal_year=2025)
# cbcr["table_i"] — per-jurisdiction revenue + entity count
# cbcr["table_ii"] — entity roster with registration numbers
# cbcr["content_hash"] — SHA-256 prefix for audit trail
```

---

## Overdue Escalation

```python
result = await svc.escalate_overdue_reports(
    tenant_id="acme",
    escalation_owner_id="cfo@acme.com",
)
print(f"Escalated {result['escalated_count']} overdue reports")
```

---

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/loc-mco/api/v1/countries` | GET | List registered countries | `loc_mco:countries` |
| `/loc-mco/api/v1/countries` | POST | Register a country | `loc_mco:countries_write` |
| `/loc-mco/api/v1/countries/<id>` | GET | Get a country | `loc_mco:countries` |
| `/loc-mco/api/v1/countries/<id>` | PUT | Update a country | `loc_mco:countries_write` |
| `/loc-mco/api/v1/entities` | GET | List legal entities | `loc_mco:entities` |
| `/loc-mco/api/v1/entities` | POST | Register an entity | `loc_mco:entities_write` |
| `/loc-mco/api/v1/entities/bulk` | POST | Bulk register entities | `loc_mco:entities_write` |
| `/loc-mco/api/v1/entities/<id>` | GET | Get an entity | `loc_mco:entities` |
| `/loc-mco/api/v1/entities/<id>` | PUT | Update an entity | `loc_mco:entities_write` |
| `/loc-mco/api/v1/entities/<id>/hierarchy` | GET | Entity ownership hierarchy | `loc_mco:entities` |
| `/loc-mco/api/v1/compliance` | GET | List compliance mappings | `loc_mco:compliance` |
| `/loc-mco/api/v1/compliance` | POST | Record compliance mapping | `loc_mco:compliance_write` |
| `/loc-mco/api/v1/compliance/<id>` | GET | Get compliance mapping | `loc_mco:compliance` |
| `/loc-mco/api/v1/compliance/<id>` | PUT | Update compliance mapping | `loc_mco:compliance_write` |
| `/loc-mco/api/v1/compliance/<id>/history` | GET | Compliance mapping history | `loc_mco:compliance` |
| `/loc-mco/api/v1/compliance/alerts` | GET | Compliance review alerts | `loc_mco:compliance` |
| `/loc-mco/api/v1/intercompany` | GET | List intercompany transactions | `loc_mco:intercompany` |
| `/loc-mco/api/v1/intercompany` | POST | Create intercompany transaction | `loc_mco:intercompany_write` |
| `/loc-mco/api/v1/intercompany/<id>` | GET | Get transaction | `loc_mco:intercompany` |
| `/loc-mco/api/v1/intercompany/<id>/approve` | POST | Approve transaction | `loc_mco:intercompany_write` |
| `/loc-mco/api/v1/intercompany/<id>/settle` | POST | Settle transaction | `loc_mco:intercompany_write` |
| `/loc-mco/api/v1/intercompany/exposure` | GET | FX-normalised exposure summary | `loc_mco:intercompany` |
| `/loc-mco/api/v1/statutory-reports` | GET | List statutory reports | `loc_mco:statutory_reports` |
| `/loc-mco/api/v1/statutory-reports` | POST | Create statutory report | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/statutory-reports/<id>` | GET | Get report | `loc_mco:statutory_reports` |
| `/loc-mco/api/v1/statutory-reports/<id>/file` | POST | File a report | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/statutory-reports/<id>/accept` | POST | Accept a filed report | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/statutory-reports/escalate` | POST | Escalate overdue reports | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/consolidation` | POST | IFRS 10 consolidation with elimination | `loc_mco:consolidation` |
| `/loc-mco/api/v1/cbcr/<year>` | GET | OECD CbCR report | `loc_mco:cbcr` |
| `/loc-mco/api/v1/agents` | GET | List MCO agents | `loc_mco:admin` |
| `/loc-mco/api/v1/agents` | POST | Register agent | `loc_mco:admin` |
| `/loc-mco/api/v1/dashboard` | GET | Dashboard summary | `loc_mco:view` |
| `/loc-mco/api/v1/analytics` | GET | MCO analytics | `loc_mco:view` |
| `/loc-mco/api/v1/audit-events` | GET | Audit event log | `loc_mco:admin` |

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `tenant_context_required` | `tenant_context_present=False` | deny — attach tenant context |
| `write_requires_policy` | write + no policy | deny — attach policy |
| `cross_tenant_entity_denied` | cross-tenant operation | deny — use tenant-scoped op |
| `country_jurisdiction_supported` | unsupported jurisdiction code | deny — select supported jurisdiction |
| `country_currency_supported` | unsupported currency | deny — select supported currency |
| `entity_type_supported` | unsupported entity type | deny — select supported type |
| `entity_country_required` | no country assigned | deny — assign country |
| `entity_registration_number_required` | no reg number | deny — provide registration number |
| `compliance_domain_supported` | unsupported domain | deny — select supported domain |
| `compliance_evidence_required` | no evidence reference | deny — attach evidence |
| `intercompany_type_supported` | unsupported ICT type | deny — select supported type |
| `arms_length_bypass_denied` | bypass flag set | deny — apply transfer pricing method |
| `intercompany_approval_required` | approve without approver | deny — assign approver |
| `transfer_pricing_documentation_required` | no TP documentation | deny — attach documentation |
| `statutory_report_type_supported` | unsupported report type | deny — select supported type |
| `statutory_report_filer_required` | filing without filer | deny — assign filer |
| `overdue_report_filing_blocked` | existing overdue unfiled | deny — file overdue report first |
| `privileged_agent_action_requires_human_approval` | privileged + no approval | deny — record human approval |
| `escalation_owner_required` | escalation without owner_id | deny — provide escalation_owner_id |

---

## Streaming Events

| Event | Trigger |
|-------|---------|
| `country_registered` | New country registered |
| `country_updated` | Country record updated |
| `entity_registered` | New legal entity registered |
| `entity_updated` | Entity record updated |
| `entities_bulk_registered` | Bulk entity registration completed |
| `compliance_mapping_recorded` | New compliance mapping created |
| `compliance_status_updated` | Compliance status changed |
| `compliance_review_due` | Compliance mapping approaching review deadline |
| `intercompany_transaction_created` | New ICT created |
| `intercompany_transaction_approved` | ICT approved |
| `intercompany_transaction_settled` | ICT settled |
| `statutory_report_created` | Report created |
| `statutory_report_filed` | Report filed with authorities |
| `statutory_report_accepted` | Report accepted by authorities |
| `statutory_report_escalated` | Overdue report escalated to named owner |
| `transfer_pricing_validated` | TP method validated |
| `agent_registered` | MCO agent registered |
| `holding_consolidated` | Group consolidation run completed |
| `cbcr_report_generated` | OECD CbCR report generated (includes content hash) |

---

## Composability Notes

- `mcy` (Multi-Currency Management) consumes country functional currencies and entity currency assignments from MCO
- `mlg` (Multi-Language) uses entity country mappings to drive locale selection
- `fin` statutory reporting pulls report data from MCO's statutory report records
- `grc` compliance modules consume MCO compliance mapping status as evidence
- `rep` (Reporting) consumes `generate_cbcr_report` output for OECD XML filing
- MCO emits all lifecycle events to the `apg.loc.mco.lifecycle` bytewax stream

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `capability_contract.py` — Supported enumerations and rule engine
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 prioritised enhancements
- `README.md` — Quick reference
