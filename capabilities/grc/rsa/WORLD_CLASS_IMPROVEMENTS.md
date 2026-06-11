# Risk & Security Assessment (grc_rsa) — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

## Overview

Fifteen targeted improvements to elevate `grc_rsa` from a solid risk register to an enterprise-grade security assessment platform covering CVSS scoring, penetration testing lifecycle, vulnerability management, and threat intelligence integration.

---

## Improvement 1: Full CVSS v3.1 / v4.0 Scoring Engine

**Current state**: Risk scoring uses a simple L×I matrix (1–25 scale) with no alignment to industry standards.

**Improvement**: Implement a native CVSS v3.1 Base/Temporal/Environmental score calculator and a CVSS 4.0 calculator. The engine should accept all CVSS vector string components (AV, AC, PR, UI, S, C, I, A for v3.1; plus the new supplemental metrics for v4.0) and produce the numeric score, severity label, and exploitability/impact sub-scores.

**Value**: Enables direct NVD/CVE ingestion, standardises vulnerability severity across pen-test findings, and satisfies ISO 27001 A.12.6.1 and PCI DSS Req 6.3 evidence requirements.

---

## Improvement 2: Penetration Testing Engagement Lifecycle

**Current state**: No pen-test management; assessments are generic risk assessments.

**Improvement**: Add a full pen-test lifecycle: `pentest_engagement_create`, `pentest_scope_define`, `pentest_finding_record`, `pentest_finding_cvss_score`, `pentest_retest_schedule`, `pentest_report_generate`. Each engagement tracks scope, methodology (black/grey/white box), tester team, start/end dates, finding count by severity, and executive summary.

**Value**: Closes the gap between vulnerability discovery and risk treatment, providing an end-to-end audit trail required by ISO 27001 A.18.2.3 and SOC 2 CC7.1.

---

## Improvement 3: Vulnerability Lifecycle Management with CVE Correlation

**Current state**: No vulnerability tracking; control gaps are coarse-grained.

**Improvement**: Add `vulnerability_register_entry`, `vulnerability_cvss_update`, `vulnerability_patch_status`, `vulnerability_sla_check`, and `vulnerability_close` methods. Correlate findings against CVE IDs from NVD feeds; SLA enforcement (Critical: 24 h, High: 7 d, Medium: 30 d, Low: 90 d) with breach alerts.

**Value**: Implements a NIST SP 800-40 Rev 4 patch management programme and gives CISO dashboards a live exploitability posture.

---

## Improvement 4: Threat Intelligence Integration

**Current state**: Scenario analysis uses static multipliers with no live threat data.

**Improvement**: Add `threat_intelligence_ingest`, `threat_indicator_match`, and `threat_risk_amplify` methods. Consume STIX 2.1 bundles from MISP/OpenCTI feeds (via configurable adapters), match indicators against registered risks, and automatically elevate residual scores when active TTPs align with a risk's category.

**Value**: Converts the risk register from a snapshot to a live-updated posture, directly aligned with NIST Cybersecurity Framework Identify (ID.RA-3) and Detect (DE.AE-2) functions.

---

## Improvement 5: Attack Surface Management & Asset Inventory Link

**Current state**: Risks reference entities by string ID with no asset context.

**Improvement**: Add `attack_surface_define`, `asset_risk_map`, and `attack_path_analyse` methods. Link risks to specific assets (servers, APIs, domains); compute attack paths using a directed graph; surface top-three lateral movement vectors.

**Value**: Prioritises remediation by exploitability and business impact rather than raw CVSS score alone, embodying the MITRE ATT&CK framework's adversary-centric view.

---

## Improvement 6: Third-Party / Vendor Risk Assessment

**Current state**: No vendor risk capability despite `vendor_risk_assessment_workflow` listed in capability contract.

**Improvement**: Implement `vendor_risk_questionnaire_send`, `vendor_risk_response_ingest`, `vendor_risk_score_compute`, `vendor_risk_tier_classify` (Tier 1–4 by data sensitivity and access), and `vendor_risk_review_schedule`. Questionnaire templates aligned to SIG Lite and CAIQ.

**Value**: Satisfies ISO 27001 A.15 (Supplier relationships), DORA Article 28 (ICT third-party risk), and SOC 2 CC9.2.

---

## Improvement 7: Regulatory Compliance Overlay

**Current state**: Risk ratings and categories are freeform with no mapping to regulatory controls.

**Improvement**: Add `compliance_control_map`, `compliance_gap_assess`, and `compliance_evidence_attach` methods. Maintain a configurable mapping table from risk categories to control frameworks (ISO 27001, NIST CSF, CIS Controls, PCI DSS, GDPR). Auto-flag risks that breach regulatory thresholds and link them to required evidence artefacts.

**Value**: Reduces manual compliance work by 60–70%, enabling continuous compliance monitoring rather than point-in-time audits.

---

## Improvement 8: Automated Risk Quantification (Monte Carlo / FAIR)

**Current state**: Risk scoring is ordinal (L×I); no financial loss estimation.

**Improvement**: Add `risk_quantify_fair` (Factor Analysis of Information Risk) and `risk_monte_carlo_simulate` methods. Accept loss event frequency distributions and loss magnitude parameters; return expected loss (EL), Value at Risk (VaR 95%), and annualised loss expectancy (ALE) with confidence intervals.

**Value**: Enables board-level conversations in financial terms; supports cyber insurance premium optimisation and budget prioritisation.

---

## Improvement 9: AI-Driven Risk Narrative and Remediation Playbooks

**Current state**: Ollama integration exists but is limited to a score + rationale tuple in `risk_assessment`.

**Improvement**: Add `risk_narrative_generate`, `remediation_playbook_generate`, and `risk_summary_translate` (plain-language board summaries). Use structured prompts that incorporate CVSS vector, affected asset class, threat actor profile, and existing controls to produce context-aware, actionable remediation steps.

**Value**: Reduces mean-time-to-remediate (MTTR) by giving engineers specific, ordered steps rather than generic recommendations.

---

## Improvement 10: Real-Time Risk Posture Streaming (SSE / WebSocket)

**Current state**: All outputs are request-response; no event streaming.

**Improvement**: Add `risk_posture_stream` (Server-Sent Events endpoint) and `kri_live_feed` methods. Emit events when: new critical/high vulnerability registered, KRI breaches threshold, pen-test finding CVSS ≥ 7.0, treatment SLA breached. Consumers subscribe by entity ID and optionally filter by category.

**Value**: Enables SOC dashboards, SIEM integrations, and on-call alerting to receive posture changes in <1 s rather than polling every 5 minutes.

---

## Improvement 11: Risk Acceptance Workflow with Time-Bounded Expiry

**Current state**: Treatment type "accept" has no formal approval workflow or expiry tracking.

**Improvement**: Add `risk_acceptance_request`, `risk_acceptance_approve`, `risk_acceptance_expiry_check`, and `risk_acceptance_renew` methods. Accepted risks carry an expiry date; approaching expiry triggers owner notification; expired acceptances auto-revert status to "requires_review".

**Value**: Prevents indefinite risk deferrals — a common audit finding — and demonstrates formal governance required by ISO 27001 Clause 6.1.3(f).

---

## Improvement 12: Comprehensive Audit Trail with Immutable Ledger

**Current state**: Audit events are logged via `AuditAdapter` but are mutable store entries.

**Improvement**: Add `audit_ledger_append` (append-only log with cryptographic chaining — each entry hashes the previous), `audit_ledger_verify`, and `audit_trail_export` (signed PDF/JSON). Use SHA-256 hash chain; any tamper detected by `verify` raises `AuditIntegrityError`.

**Value**: Produces forensically sound evidence for regulatory investigations, e-discovery, and ISO 27001 A.16.1.7 (evidence collection).

---

## Improvement 13: Risk Aggregation and Portfolio View

**Current state**: Reporting is per-entity; no cross-entity or portfolio view.

**Improvement**: Add `portfolio_risk_aggregate`, `entity_risk_rank`, and `risk_concentration_analyse` methods. Aggregate risks across all entities, identify concentration (e.g., 40% of critical risks in technology category), and produce group-level heatmaps. Support hierarchical entities (subsidiary → division → group).

**Value**: Enables Group CRO and board-level oversight; satisfies Basel II/III operational risk aggregation requirements.

---

## Improvement 14: Structured Testing and Regression Suite

**Current state**: `tests/test_contract.py` exists but coverage of service methods is sparse.

**Improvement**: Add a comprehensive `tests/ci/` suite: unit tests for every service method (using in-memory store fixtures), integration tests for the full pen-test → vulnerability → treatment → closure lifecycle, CVSS calculation property tests using Hypothesis, and contract regression tests that pin the public API surface.

**Value**: Raises test coverage to >90%; prevents regression during future capability composition; enables CI/CD gating.

---

## Improvement 15: Exportable Risk Evidence Packs

**Current state**: Reports return JSON; no document generation.

**Improvement**: Add `evidence_pack_generate` method that assembles a ZIP archive containing: risk register (CSV + XLSX), heat map (PNG via matplotlib), CVSS scoring worksheet (XLSX), treatment plan tracker (PDF), and audit trail extract (signed JSON). Support templated PDF generation using WeasyPrint or ReportLab with Datacraft branding.

**Value**: Directly satisfies auditor and regulator requests for evidence packs; reduces manual reporting effort from days to seconds.
