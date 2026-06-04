# Multi-Country Operations

## Overview

Multi-Country Operations (MCO) provides country entity management, local regulatory compliance mapping, cross-border intercompany transaction governance, and statutory reporting for organisations operating across multiple jurisdictions. It enforces arms-length transfer pricing, tenant-scoped entity isolation, and audit-trailed compliance workflows across any combination of supported jurisdictions.

## Capability ID

`loc_mco`

## Provides

| Service | Description |
|---------|-------------|
| `country_entity_management` | Register and manage country records with jurisdiction, functional currency, and regulatory framework |
| `regulatory_compliance_mapping` | Map compliance domains (tax, AML, data protection, etc.) to entities with owner assignment and evidence tracking |
| `intercompany_transaction_workflow` | Create, approve, and settle intercompany transactions with transfer pricing validation |
| `statutory_reporting_workflow` | Draft, review, file, and track acceptance of statutory reports per entity and period |
| `transfer_pricing_validation` | Validate OECD-aligned transfer pricing methods and documentation per transaction |
| `cross_border_governance` | Enforce arms-length, approval, and jurisdiction rules across all cross-border operations |
| `multi_entity_consolidation_data` | Provide entity and country data to downstream consolidation and reporting capabilities |
| `jurisdiction_registry` | Maintain a tenant-scoped registry of active jurisdictions and their regulatory profiles |
| `compliance_monitoring` | Surface overdue reviews, non-compliant entities, and overdue statutory filings |

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

## Configuration

| Key | Type | Description |
|-----|------|-------------|
| `tenant_id` | string | Tenant identifier |
| `countries.supported_jurisdictions` | list | ISO 2-letter jurisdiction codes |
| `countries.supported_currencies` | list | ISO 4217 currency codes |
| `entities.registration_number_required` | bool | Enforce company registration numbers |
| `compliance.owner_required` | bool | Require named compliance owner per mapping |
| `intercompany.arms_length_validation` | bool | Enforce transfer pricing on all ICT |
| `statutory_reports.period_required` | bool | Require reporting period dates |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/loc-mco/api/v1/countries` | GET | List registered countries | `loc_mco:countries` |
| `/loc-mco/api/v1/countries` | POST | Register a country | `loc_mco:countries_write` |
| `/loc-mco/api/v1/countries/<id>` | GET | Get a country | `loc_mco:countries` |
| `/loc-mco/api/v1/countries/<id>` | PUT | Update a country | `loc_mco:countries_write` |
| `/loc-mco/api/v1/entities` | GET | List legal entities | `loc_mco:entities` |
| `/loc-mco/api/v1/entities` | POST | Register an entity | `loc_mco:entities_write` |
| `/loc-mco/api/v1/entities/<id>` | GET | Get an entity | `loc_mco:entities` |
| `/loc-mco/api/v1/entities/<id>` | PUT | Update an entity | `loc_mco:entities_write` |
| `/loc-mco/api/v1/compliance` | GET | List compliance mappings | `loc_mco:compliance` |
| `/loc-mco/api/v1/compliance` | POST | Record compliance mapping | `loc_mco:compliance_write` |
| `/loc-mco/api/v1/compliance/<id>` | GET | Get compliance mapping | `loc_mco:compliance` |
| `/loc-mco/api/v1/compliance/<id>` | PUT | Update compliance mapping | `loc_mco:compliance_write` |
| `/loc-mco/api/v1/intercompany` | GET | List intercompany transactions | `loc_mco:intercompany` |
| `/loc-mco/api/v1/intercompany` | POST | Create intercompany transaction | `loc_mco:intercompany_write` |
| `/loc-mco/api/v1/intercompany/<id>` | GET | Get transaction | `loc_mco:intercompany` |
| `/loc-mco/api/v1/intercompany/<id>/approve` | POST | Approve transaction | `loc_mco:intercompany_write` |
| `/loc-mco/api/v1/statutory-reports` | GET | List statutory reports | `loc_mco:statutory_reports` |
| `/loc-mco/api/v1/statutory-reports` | POST | Create statutory report | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/statutory-reports/<id>` | GET | Get report | `loc_mco:statutory_reports` |
| `/loc-mco/api/v1/statutory-reports/<id>/file` | POST | File a report | `loc_mco:statutory_reports_write` |
| `/loc-mco/api/v1/agents` | GET | List MCO agents | `loc_mco:admin` |
| `/loc-mco/api/v1/agents` | POST | Register agent | `loc_mco:admin` |
| `/loc-mco/api/v1/dashboard` | GET | Dashboard summary | `loc_mco:view` |
| `/loc-mco/api/v1/audit-events` | GET | Audit event log | `loc_mco:admin` |

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

## Data Models

| Model | Key Fields |
|-------|-----------|
| `CountryResponse` | id, tenant_id, name, jurisdiction, functional_currency, regulatory_framework, status |
| `EntityResponse` | id, tenant_id, name, entity_type, country_id, registration_number, functional_currency, parent_entity_id |
| `ComplianceMappingResponse` | id, tenant_id, entity_id, domain, framework, status, owner_id, next_review_date, evidence_reference |
| `IntercompanyTransactionResponse` | id, tenant_id, transaction_type, originator_entity_id, counterparty_entity_id, amount, currency, transfer_pricing_method, status |
| `StatutoryReportResponse` | id, tenant_id, entity_id, report_type, period_start, period_end, due_date, filer_id, status, filed_date |
| `McoAgentResponse` | id, tenant_id, name, runtime, role, scope |
| `McoAuditEvent` | id, tenant_id, event_type, reference_id, actor_id, processor, stream, occurred_at |

## Streaming Events

| Event | Trigger |
|-------|---------|
| `country_registered` | New country registered |
| `country_updated` | Country record updated |
| `entity_registered` | New legal entity registered |
| `entity_updated` | Entity record updated |
| `compliance_mapping_recorded` | New compliance mapping created |
| `compliance_status_updated` | Compliance status changed |
| `intercompany_transaction_created` | New ICT created |
| `intercompany_transaction_approved` | ICT approved |
| `intercompany_transaction_settled` | ICT settled |
| `statutory_report_created` | Report created |
| `statutory_report_filed` | Report filed with authorities |
| `statutory_report_accepted` | Report accepted by authorities |
| `transfer_pricing_validated` | TP method validated |
| `agent_registered` | MCO agent registered |

## Edge Cases Handled

- Functional currency normalised to uppercase ISO 4217 regardless of input case
- Jurisdiction codes normalised to lowercase
- Approval of an intercompany transaction requires the status to be `pending_approval` — direct approval of `draft` transactions is rejected
- Filing a statutory report when another report for the same entity is in `overdue` status is blocked until the overdue report is filed
- Entity registration is rejected if the referenced `country_id` does not exist in the tenant's country registry
- Cross-tenant entity lookups are hard-rejected by the rule engine before any store access
- Privileged agent actions (e.g. auto-filing) require recorded human approval

## Composability Notes

- `mcy` (Multi-Currency Management) consumes country functional currencies and entity currency assignments from MCO
- `mlg` (Multi-Language) uses entity country mappings to drive locale selection
- `fin` statutory reporting can pull report data from MCO's statutory report records
- `grc` compliance modules consume MCO compliance mapping status as evidence
- MCO emits all lifecycle events to the `apg.loc.mco.lifecycle` bytewax stream for downstream consumers
