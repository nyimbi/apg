# Regulatory Technology

## Overview
Regulatory Technology provides automated tracking and management of regulatory obligations: regulatory source registration, change intake (new rules, updates, guidance, enforcement actions, consultations), obligation mapping with policy references, impact assessment across APG capabilities, regulatory filing preparation and submission, regulatory inquiry management, and approved response recording. It is the regulatory horizon scanning and filing layer that feeds obligation evidence into `fintech_compliance`.

Every response to a regulatory inquiry requires an approval reference before being recorded. Submission acknowledgments are mandatory. Impact assessments require a reviewer. All RegTech lifecycle events stream to `apg.fintech.regtech.lifecycle` via Bytewax.

## Capability ID
`fintech_regtech`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| regulatory_source_workflow | Register regulatory sources by regulator, jurisdiction, owner, and evidence |
| regulatory_change_workflow | Record rule changes, guidance, and enforcement actions with effective dates |
| regulatory_obligation_mapping_workflow | Map obligations to specific regulatory changes with policy references |
| regulatory_policy_mapping_workflow | Link policy documents to regulatory obligations with owner and due dates |
| regulatory_impact_workflow | Assess the impact of regulatory changes on specific APG capabilities |
| regulatory_filing_workflow | Prepare regulatory returns, incident notices, and prudential reports |
| regulatory_submission_workflow | Record filing submissions with channel, submitter, timestamp, and acknowledgment |
| regulatory_inquiry_workflow | Open and track regulatory inquiries with severity and due dates |
| regulatory_response_workflow | Record approved responses to regulatory inquiries |
| regulatory_review_workflow | Governance reviews for filings, responses, and impact assessments |
| regulatory_agent_workflow | Register AI agents for horizon scanning, filing preparation, and response drafting |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Compliance officer notifications |
| nlpc | NLP for regulatory text analysis |
| keym | Key management |
| fintech_compliance | Compliance obligation catalog |
| fintech_risk | Risk assessment for regulatory impact |
| fintech_aml | AML regulatory context |
| fintech_kyc | KYC regulatory context |
| fin_rpt | Financial reporting for regulatory returns |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| sources.supported_regulators | list | central_bank, securities_regulator, data_protection_authority, financial_conduct_authority, payments_regulator, tax_authority | Supported regulatory bodies |
| sources.supported_jurisdictions | list | KE, US, GB, EU, NG, GH, ZA, SG, GLOBAL | Jurisdictions |
| changes.supported_types | list | new_rule, rule_update, guidance, enforcement_action, consultation, deadline_change | Change categories |
| filings.supported_types | list | regulatory_return, incident_notice, license_update, audit_response, prudential_report, conduct_report | Filing types |
| submissions.supported_channels | list | portal, api, email, sftp, manual | Submission channels |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-regtech/dashboard | GET | fintech_regtech:view | Overview |
| sources | /fintech-regtech/sources | GET/POST | fintech_regtech:sources | Sources |
| changes | /fintech-regtech/changes | GET/POST | fintech_regtech:changes | Horizon |
| obligations | /fintech-regtech/obligations | GET/POST | fintech_regtech:obligations | Obligations |
| impact | /fintech-regtech/impact | GET/POST | fintech_regtech:impact | Impact |
| filings | /fintech-regtech/filings | GET/POST | fintech_regtech:filings | Filings |
| submissions | /fintech-regtech/submissions | GET/POST | fintech_regtech:submissions | Filings |
| inquiries | /fintech-regtech/inquiries | GET/POST | fintech_regtech:inquiries | Inquiries |
| responses | /fintech-regtech/responses | GET/POST | fintech_regtech:responses | Inquiries |
| reviews | /fintech-regtech/reviews | GET/POST | fintech_regtech:reviews | Governance |
| agents | /fintech-regtech/agents | GET/POST | fintech_regtech:admin | Automation |
| settings | /fintech-regtech/settings | GET/POST | fintech_regtech:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| source_regulator_supported | Unsupported regulator type | deny |
| source_jurisdiction_supported | Unsupported jurisdiction | deny |
| change_effective_date_required | Change without effective date | deny |
| change_severity_supported | Unsupported severity level | deny |
| obligation_due_date_required | Obligation mapping without due date | deny |
| impact_reviewer_required | Impact assessment without reviewer | deny |
| impact_capability_required | Impact assessment without impacted capability | deny |
| filing_period_required | Filing without period | deny |
| submission_acknowledgment_required | Submission without acknowledgment | deny |
| submission_timestamp_required | Submission without timestamp | deny |
| response_approval_required | Regulatory response without approval | deny |
| regtech_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_regtech_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| RegulatorySource | id, regulator, jurisdiction, source_reference, owner_id, evidence_reference |
| RegulatoryChange | id, source_id, framework, change_type, title, effective_date, severity, evidence_reference |
| ObligationMapping | id, change_id, obligation_reference, policy_reference, owner_id, due_date |
| ImpactAssessment | id, change_id, impacted_capability, risk_rating, reviewer_id, evidence_reference |
| RegulatoryFiling | id, framework, filing_type, period, owner_id, evidence_references, status |
| RegulatorySubmission | id, filing_id, channel, submitted_by, submitted_at, acknowledgment_reference, status |
| RegulatoryInquiry | id, regulator, reference, severity, due_date, evidence_references, status |
| RegulatoryResponse | id, inquiry_id, responder_id, response_reference, approval_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| regulatory_source_registered | Source registered |
| regulatory_change_recorded | Change recorded |
| regulatory_obligation_mapped | Obligation mapped |
| regulatory_impact_assessed | Impact assessment recorded |
| regulatory_filing_prepared | Filing prepared |
| regulatory_submission_recorded | Filing submitted |
| regulatory_inquiry_opened | Inquiry opened |
| regulatory_response_recorded | Response recorded |
| regulatory_review_recorded | Review completed |
| regulatory_agent_registered | AI agent registered |

## Edge Cases Handled
- Regulatory responses require an approval reference — informal verbal responses cannot be recorded; every response must carry evidence of internal approval before it is submitted to the regulator
- Submission acknowledgments are mandatory — a submission without a regulator acknowledgment (portal confirmation, email receipt, API response) is not treated as filed; this prevents submissions from being marked complete without delivery confirmation
- Impact assessments reference a specific APG capability ID — generic "all capabilities" assessments are not supported; each impacted capability must have a separate assessment record
- Change effective date is required even for consultation documents (which may not yet have a binding date) — the field accepts future dates; the absence of an effective date implies an untracked change
- `GLOBAL` is a valid jurisdiction for multi-jurisdictional regulatory changes (e.g., FATF guidance)

## Composability
- **Upstream**: `fintech_compliance` is the primary consumer of obligation mappings; `fin_rpt` provides the financial data backing regulatory returns (prudential reports, conduct reports)
- **Downstream**: Impact assessments feed back into `fintech_compliance` for control mapping updates; filing submissions feed into audit evidence for `fintech_compliance` reports
- **Peer**: Deployed alongside `fintech_compliance` (internal control framework) and `fintech_risk` (risk appetite for regulatory change impact)

## Development Notes
- `enforcement_action` is a supported change type — recording regulator enforcement actions against the organization requires the same evidence and review controls as proactive regulatory changes
- `incident_notice` filing type maps to mandatory breach notifications (GDPR, PCI DSS incident reports); the filing period for these is typically 72 hours from discovery
- Regulatory agent roles include `regulatory_horizon_agent` for automated scanning of regulatory feeds; this role is read-only by design — the agent can suggest obligation mappings but cannot create them without human approval
- `SUPPORTED_JURISDICTIONS` is a closed list; organizations operating in unlisted jurisdictions must add the jurisdiction to the constant and redeploy before filings can be created
