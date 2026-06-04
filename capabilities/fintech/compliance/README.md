# FinTech Compliance Automation

## Overview
FinTech Compliance Automation provides a structured framework for managing regulatory obligations, control mappings, compliance checks, evidence collection, attestations, issues, remediation plans, reports, and governance reviews across all supported regulatory frameworks. It acts as the internal compliance layer that links every operational capability to its governing regulatory requirements.

The capability is framework-agnostic: it supports PCI DSS, PSD2, Open Banking, GDPR, SOX, Basel III, MiFID II, AML, KYC, and data privacy frameworks in a single obligation catalog. Failed checks require attached evidence. Reports require approver assignment. High-impact remediations require human approval. All compliance lifecycle events stream to `apg.fintech.compliance.lifecycle` via Bytewax.

## Capability ID
`fintech_compliance`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| compliance_obligation_workflow | Catalog regulatory obligations with framework, owner, effective date, and evidence |
| compliance_control_workflow | Map controls to obligations with type, owner, frequency, and evidence |
| compliance_check_workflow | Record compliance checks with result; failed checks require failure evidence |
| compliance_evidence_workflow | Maintain an evidence vault with type, source, and retention metadata |
| compliance_attestation_workflow | Capture attestations linking obligations, attestors, status, and evidence |
| compliance_issue_workflow | Open compliance issues with severity, owner, due date, and evidence |
| compliance_remediation_workflow | Record remediation plans for issues with approval gates for high-impact cases |
| compliance_report_workflow | Publish regulatory, board, audit, and exception reports with approver |
| compliance_review_workflow | Governance reviews for obligations, controls, and reports |
| compliance_agent_workflow | Register AI agents for obligation review, control testing, and issue remediation |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Compliance officer notifications |
| nlpc | NLP for report narrative |
| keym | Key management |
| fintech_payments | Payments compliance context |
| fintech_wallets | Wallets compliance context |
| fintech_kyc | KYC obligation evidence |
| fintech_aml | AML obligation evidence |
| fintech_fraud | Fraud control context |
| fintech_risk | Risk exposure and control linkage |
| fin_rpt | Financial reporting integration |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| obligations.supported_frameworks | list | pci_dss, psd2, open_banking, gdpr, sox, basel_iii, mifid_ii, aml, kyc, data_privacy | Regulatory frameworks |
| obligations.supported_types | list | policy, control, reporting, retention, disclosure, monitoring, approval, training | Obligation types |
| controls.supported_types | list | preventive, detective, corrective, automated, manual, compensating | Control types |
| issues.supported_severities | list | low, medium, high, critical | Issue severity levels |
| reports.supported_types | list | regulatory, board, audit, management, exception, incident | Report categories |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-compliance/dashboard | GET | fintech_compliance:view | Overview |
| obligations | /fintech-compliance/obligations | GET/POST | fintech_compliance:obligations | Obligations |
| controls | /fintech-compliance/controls | GET/POST | fintech_compliance:controls | Controls |
| checks | /fintech-compliance/checks | GET/POST | fintech_compliance:checks | Testing |
| evidence | /fintech-compliance/evidence | GET/POST | fintech_compliance:evidence | Evidence |
| attestations | /fintech-compliance/attestations | GET/POST | fintech_compliance:attestations | Governance |
| issues | /fintech-compliance/issues | GET/POST | fintech_compliance:issues | Issues |
| reports | /fintech-compliance/reports | GET/POST | fintech_compliance:reports | Reporting |
| reviews | /fintech-compliance/reviews | GET/POST | fintech_compliance:reviews | Governance |
| agents | /fintech-compliance/agents | GET/POST | fintech_compliance:admin | Automation |
| settings | /fintech-compliance/settings | GET/POST | fintech_compliance:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| obligation_effective_date_required | Obligation without effective date | deny |
| control_frequency_required | Control without test frequency | deny |
| failed_check_requires_evidence | Failed check without failure evidence | deny |
| evidence_retention_required | Evidence without retention period | deny |
| issue_due_date_required | Issue without due date | deny |
| high_impact_remediation_requires_approval | High-impact remediation without approval | deny |
| report_approver_required | Report without assigned approver | deny |
| compliance_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_compliance_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| ComplianceObligation | id, framework, obligation_type, title, owner_id, evidence_reference, effective_date, status |
| ComplianceControl | id, obligation_id, control_type, owner_id, evidence_reference, frequency |
| ComplianceCheck | id, obligation_id, control_id, check_type, subject_reference, result, evidence_reference |
| ComplianceEvidence | id, evidence_type, reference, source, retention_period |
| ComplianceAttestation | id, obligation_id, attestor_id, status, evidence_reference |
| ComplianceIssue | id, obligation_id, severity, owner_id, evidence_reference, due_date, status |
| ComplianceRemediation | id, issue_id, owner_id, plan_reference, approval_reference |
| ComplianceReport | id, report_type, framework, period, evidence_references, approver_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| compliance_obligation_registered | Obligation cataloged |
| compliance_control_mapped | Control mapped to obligation |
| compliance_check_recorded | Compliance check recorded |
| compliance_evidence_attached | Evidence attached to record |
| compliance_attestation_recorded | Attestation captured |
| compliance_issue_opened | Issue opened |
| compliance_remediation_recorded | Remediation plan recorded |
| compliance_report_published | Report published |
| compliance_review_recorded | Governance review completed |
| compliance_agent_registered | AI agent registered |

## Edge Cases Handled
- Failed checks require attached failure evidence — a check result of `non_compliant` without evidence is denied; passing checks do not require evidence
- Evidence retention period is mandatory: attaching evidence without specifying how long it must be retained violates data management policy and is denied
- High-impact remediations require human approval even when the owner is present; ownership alone is insufficient for high-impact cases
- Reports require an approver at publication time — draft reports without an approver are denied; the approver check fires on publish, not on creation
- Control frequency is required to distinguish one-time from continuous controls; the absence of frequency renders the control untestable and is rejected

## Composability
- **Upstream**: `fintech_risk` provides risk appetite and exposure data as compliance evidence; `fintech_aml` and `fintech_kyc` provide obligation fulfillment evidence for AML/KYC frameworks; `fintech_regtech` feeds regulatory change obligations into this catalog
- **Downstream**: `fintech_regtech` reads compliance reports as filing inputs; internal audit systems consume attestation and issue records
- **Peer**: Deployed alongside `fintech_risk` (risk control linkage) and `fintech_regtech` (regulatory change horizon scanning)

## Development Notes
- Obligation types (policy, control, reporting, retention, disclosure, monitoring, approval, training) are mutually exclusive categories; an obligation can only have one type
- Evidence types are validated against `SUPPORTED_EVIDENCE_TYPES`; custom evidence types require contract update
- The check types (`transaction`, `customer`, `account`, `merchant`, `policy`, `control`, `report`, `agent_action`) map to the operational capability domain being tested
- Framework validation fires on both obligation registration and control mapping; a control mapped to an obligation in an unsupported framework is rejected
