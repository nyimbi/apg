# FinTech Compliance Automation

## Overview
FinTech Compliance Automation provides a structured framework for managing regulatory obligations, control mappings, compliance checks, evidence collection, attestations, issues, remediation plans, reports, and governance reviews across all supported regulatory frameworks. It acts as the internal compliance layer that links every operational capability to its governing regulatory requirements.

The capability is framework-agnostic: it supports PCI DSS, PSD2, Open Banking, GDPR, SOX, Basel III, MiFID II, AML, KYC, and data privacy frameworks in a single obligation catalog. Failed checks require attached evidence. Reports require approver assignment. High-impact remediations require human approval. All compliance lifecycle events stream to `apg.fintech.compliance.lifecycle` via Bytewax.

Additional production-grade capabilities introduced in v2.0 include: automated SAR/CTR filing, FATF AML risk assessment, sanctions and PEP screening, multi-central-bank returns (CBK, CBN, RBZ, BoU, BoG), GDPR/PDPA data subject request handling, compliance programme setup, and a real-time health dashboard.

## Capability ID
`fintech_compliance`  Version: 2.0.0

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
| compliance_programme_setup | Set up multi-regulation compliance programmes for an entity |
| obligation_mapping | Bulk-map regulatory obligations to entities |
| control_assessment | Assess controls and record longitudinal outcomes |
| compliance_gap_report | Generate gap analysis with risk-scored remediation priorities |
| regulatory_alert | Record and broadcast regulatory change alerts |
| policy_management | Manage policy lifecycle with versioned approval history |
| training_completion_tracking | Track employee training scores and pass/fail outcomes |
| compliance_dashboard | Real-time health dashboard with obligation, issue, and training metrics |
| cbk_compliance_return | Generate CBK/CBN/RBZ/BoU/BoG regulatory compliance returns |
| compliance_analytics | Period analytics: pass rates, closure rates, framework coverage |
| fatf_aml_risk_assessment | FATF 40 Recommendations AML risk assessment by component |
| sanctions_screening | Screen subjects against OFAC, UN, EU, HMT, and CBK sanctions lists |
| pep_screening | Politically Exposed Person screening with EDD flag |
| transaction_monitoring_rule | Register AML transaction monitoring rules |
| sar_filing | File Suspicious Activity Reports with FRC Kenya |
| ctr_filing | File Currency Transaction Reports above CBK threshold (KES 1M) |
| gdpr_data_request | Handle GDPR/PDPA data subject access, erasure, portability requests |
| aml_risk_rating_update | Update AML risk ratings for customers or entities |
| bulk_register_obligations | Bulk-register obligations in a single call |
| export_compliance_data | Export compliance data in JSON, CSV, or Excel format |

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
| compliance_programme_setup | Compliance programme created |
| regulatory_alert_raised | Regulatory change alert broadcast |
| sar_filed | SAR submitted to FRC Kenya |
| ctr_filed | CTR submitted to CBK FCIU |
| fatf_aml_assessment_completed | FATF AML risk assessment run |
| sanctions_screening_completed | Sanctions list screening completed |
| cbk_compliance_return_submitted | Regulatory return filed |

## World-Class Enhancements (v2.0)

Fifteen targeted improvements elevating this from a solid foundation to a production-grade, regulator-ready compliance engine:

1. **Real-Time Obligation Deadline Tracker** — `obligation_deadline_monitor()` scans obligations for imminent deadlines (7/14/30-day horizon), emits severity-ranked alerts, integrates with `ntfy` for automated officer reminders.

2. **Automated Control Effectiveness Scoring** — `control_effectiveness_score()` aggregates historical check results per control with exponential decay weighting (recent failures penalised more), returns a 0–100 effectiveness index with trend direction.

3. **Regulatory Change Impact Analysis** — `regulatory_change_impact_analysis()` maps incoming changes to affected obligations, estimates remediation effort (low/medium/high), auto-opens high-severity issues, returns a ranked action plan.

4. **KYC Refresh Workflow with CDD Tiering** — `kyc_refresh_workflow()` drives CDD refresh cycles per CBK 2020 risk tiers (simplified/standard/enhanced), tracks document expiry, triggers re-verification, records outcome as attestation.

5. **Multi-Jurisdictional Reporting Calendar** — `compliance_calendar()` builds a time-ordered schedule of mandatory reporting deadlines across CBK, CMA, FRC, CRB, ODPC, and FATF; returns iCal-compatible dict.

6. **Automated Suspicious Transaction Detection** — `evaluate_transaction_for_aml()` evaluates transactions against all registered monitoring rules (threshold, velocity, geography, counterparty), returns a risk score, auto-files draft SAR above threshold.

7. **Evidence Chain-of-Custody Validation** — `validate_evidence_chain()` verifies source references, hash/checksums, retention windows, and linked reviews; returns a custody report highlighting regulatory defensibility gaps.

8. **Board-Level Compliance Pack Generator** — `generate_board_pack()` assembles the monthly/quarterly compliance pack: health score trend, top-5 issues, remediation progress, upcoming deadlines, regulatory changes, training stats.

9. **Continuous Control Monitoring (CCM) Engine** — `continuous_control_monitor()` accepts a schedule (daily/weekly/monthly), auto-evaluates all assigned controls, records results, escalates failures to issues — shifting from point-in-time audits to persistent assurance.

10. **Regulatory Filing Status Tracker** — `update_filing_status()` / `get_filing_pipeline()` manage the full submission lifecycle (draft → submitted → acknowledged → accepted/rejected), flag overdue acknowledgements, maintain transition audit trail.

11. **Employee Compliance Risk Profiling** — `employee_compliance_profile()` aggregates training scores, attestation completion, policy acknowledgements, and incident involvement into a low/medium/high risk score that triggers enhanced monitoring or mandatory retraining.

12. **Regulatory Obligation Versioning and Diff** — `version_obligation()` records obligation amendments with structured diffs (field, old value, new value), links to triggering regulatory change, maintains full amendment history for regulator inspections.

13. **Cross-Capability Compliance Posture Score** — `cross_capability_posture()` ingests compliance signals from `fintech_risk`, `fintech_aml`, `fintech_kyc`, and `fintech_fraud`, computes a weighted per-framework posture score, returns a composite enterprise-wide compliance rating.

14. **Consent and Data Lineage Registry (PDPA/GDPR)** — `register_consent()` / `get_data_lineage()` maintain a consent registry (purpose, scope, withdrawal history) and a data lineage map (origin, transformations, destinations) for ODPC/GDPR accountability and automated DSR fulfilment.

15. **Compliance Obligation Graph for Conflict Detection** — `build_obligation_graph()` / `detect_obligation_conflicts()` construct a directed obligation graph and detect circular dependencies and conflicting requirements (e.g., GDPR erasure vs. AML retention), returning a conflict report with recommended resolutions.

### Implementation Priority
| Priority | Improvements |
|----------|-------------|
| P1 | Compliance Calendar (#5), Deadline Tracker (#1), AML Transaction Eval (#6), Regulatory Change Impact (#3) |
| P2 | KYC Refresh Workflow (#4), CCM Engine (#9), Control Effectiveness (#2), Board Pack Generator (#8), Filing Status Tracker (#10), Evidence Chain Validation (#7) |
| P3 | Employee Risk Profile (#11), Consent / Data Lineage (#14), Obligation Versioning (#12), Cross-Capability Posture (#13), Obligation Graph (#15) |

## New Methods

### Compliance Programme Setup
```python
svc = FintechComplianceService()

programme = await svc.compliance_programme_setup(
    entity_id="bank-001",
    regulations=["cbk", "aml", "gdpr"],
    risk_appetite="medium",
    tenant_id="acme",
    programme_name="ACME Bank 2025 Compliance Programme",
)
# Returns: programme_id, obligation_ids, status, created_at
```

### Real-Time Compliance Dashboard
```python
dashboard = await svc.compliance_dashboard(
    entity_id="bank-001",
    tenant_id="acme",
)
# Returns: health_score (0-100), compliance_rate, open_issues_by_severity,
#          training_completion_rate, active_regulatory_alerts, snapshot_at
```

### Compliance Gap Report
```python
gap = await svc.compliance_gap_report(
    entity_id="bank-001",
    regulation="aml",
    tenant_id="acme",
)
# Returns: total_obligations, unchecked_controls, open_issues,
#          gap_score, risk_level (low/medium/high/critical)
```

### SAR Filing
```python
sar = await svc.sar_filing(
    entity_id="bank-001",
    subject_name="John Doe",
    suspicious_activity="Structuring: 9 x KES 99,000 deposits in 48 hours",
    amount=891_000.0,
    currency="KES",
    tenant_id="acme",
)
# Returns: sar_id, regulatory_body (FRC_KENYA), status (filed), filed_at
```

### FATF AML Risk Assessment
```python
assessment = await svc.fatf_aml_risk_assessment(
    entity_id="bank-001",
    tenant_id="acme",
)
# Returns: component_scores (customer/product/channel/geographic/transaction risk),
#          overall_risk_score, risk_rating (low/medium/high)
```

### Sanctions & PEP Screening
```python
sanction_result = await svc.sanctions_screening(
    subject_name="Acme Trading Ltd",
    subject_type="entity",
    tenant_id="acme",
)
# Checks: OFAC_SDN, UN_CONSOLIDATED, EU_CONSOLIDATED, HMT_UK, CBK_SANCTIONS
# Returns: hit (bool), match_details, risk_score, decision (clear/block)

pep_result = await svc.pep_screening(
    subject_name="Minister Jane Otieno",
    country="KE",
    tenant_id="acme",
)
# Returns: is_pep, pep_category, enhanced_due_diligence_required
```

## Edge Cases Handled
- Failed checks require attached failure evidence — a check result of `non_compliant` without evidence is denied; passing checks do not require evidence
- Evidence retention period is mandatory: attaching evidence without specifying how long it must be retained violates data management policy and is denied
- High-impact remediations require human approval even when the owner is present; ownership alone is insufficient for high-impact cases
- Reports require an approver at publication time — draft reports without an approver are denied; the approver check fires on publish, not on creation
- Control frequency is required to distinguish one-time from continuous controls; the absence of frequency renders the control untestable and is rejected
- CTR filing enforces the CBK threshold of KES 1,000,000; sub-threshold filings raise `ValueError`
- GDPR/PDPA data subject requests are restricted to `access | erasure | portability | rectification`

## Composability
- **Upstream**: `fintech_risk` provides risk appetite and exposure data as compliance evidence; `fintech_aml` and `fintech_kyc` provide obligation fulfillment evidence for AML/KYC frameworks; `fintech_regtech` feeds regulatory change obligations into this catalog
- **Downstream**: `fintech_regtech` reads compliance reports as filing inputs; internal audit systems consume attestation and issue records
- **Peer**: Deployed alongside `fintech_risk` (risk control linkage) and `fintech_regtech` (regulatory change horizon scanning)
- **v2.0 cross-capability**: `cross_capability_posture()` ingests signals from `fintech_risk`, `fintech_aml`, `fintech_kyc`, `fintech_fraud` for enterprise-wide posture scoring

## Development Notes
- Obligation types (policy, control, reporting, retention, disclosure, monitoring, approval, training) are mutually exclusive categories; an obligation can only have one type
- Evidence types are validated against `SUPPORTED_EVIDENCE_TYPES`; custom evidence types require contract update
- The check types (`transaction`, `customer`, `account`, `merchant`, `policy`, `control`, `report`, `agent_action`) map to the operational capability domain being tested
- Framework validation fires on both obligation registration and control mapping; a control mapped to an obligation in an unsupported framework is rejected
- `ComplianceAutomationService` is kept as a backward-compatible alias for `FintechComplianceService`

---

© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke
