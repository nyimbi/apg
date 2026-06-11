# leg_ctr — World-Class Improvements

Fifteen high-impact enhancements that elevate `leg_ctr` from a CRUD tracker to a competitive Contract Intelligence Platform.

---

### I1. AI-Powered Clause Risk Scoring
**Category**: AI/ML
**Justification**: Automated risk assessment surfaces liability concentration, ambiguous indemnification clauses, and SLA gaps before execution — reducing legal review time by 60% and eliminating post-signature surprises. Enterprise buyers rank this as the #1 CLM selection criterion.
**Implementation**: Score each contract 0–100 across six risk dimensions (liability, IP, termination, payment, data protection, jurisdiction) using a locally-hosted Ollama LLM against clause-level text; store scores with version snapshots for trend analysis and surface a composite `risk_score` on the contract record.
**Competitive reference**: ContractPodAi (Connie AI risk radar), Ironclad AI Assist

---

### I2. Playbook-Driven Deviation Detection
**Category**: Compliance
**Justification**: Every legal department maintains standard playbook clauses (fallback positions, must-have language). Detecting deviations automatically at redline time prevents playbook erosion and reduces escalation cycles — a primary selling point for modern CLMs vs. legacy document management tools.
**Implementation**: Store tenant-level playbook entries keyed by `(contract_type, clause_key)` with preferred/acceptable/last-resort tiers; at `create_redline` time, compare proposed text against playbook via fuzzy similarity and emit a `playbook_deviation_flag` with severity level on the redline record.
**Competitive reference**: Ironclad Playbook, Spotdraft Playbooks

---

### I3. Obligation Calendar with iCalendar Feed
**Category**: Feature
**Justification**: Post-execution obligation management is the most common source of contract value leakage. Missed SLA deadlines, notice periods, and payment milestones cost organisations an estimated 9% of annual contract value (World Commerce & Contracting, 2024).
**Implementation**: Generate RFC 5545 iCalendar VCALENDAR feeds per tenant from obligations and renewal schedules with VALARM triggers at configurable lead times (7/14/30 days); expose a `/calendar.ics` endpoint for direct calendar app subscription.
**Competitive reference**: Conga Contracts, Agiloft, DocuSign CLM

---

### I4. Counterparty Risk Intelligence Integration
**Category**: Integration
**Justification**: Executing a contract with a counterparty entering insolvency or under sanctions exposure creates immediate legal and financial risk. Embedded risk scores let procurement and legal teams gate approvals before execution rather than discovering problems post-signature.
**Implementation**: Accept external counterparty health signals (credit score, sanctions hit, litigation flag) via `update_counterparty_health`; surface a `counterparty_risk_flag` on contracts where health score drops below a configurable threshold; block execution if `sanctions_hit=True`.
**Competitive reference**: Icertis third-party risk management, ContractPodAi counterparty intelligence, Jaggaer

---

### I5. Templated Contract Generation from Structured Inputs
**Category**: Feature
**Justification**: 70% of contracts are variations of five master templates. Parameterised generation eliminates copy-paste errors, ensures playbook-compliant language from the first draft, and reduces drafting time from hours to minutes.
**Implementation**: Store Jinja2-style contract templates per `(tenant, contract_type)` with typed variable slots; `generate_from_template` merges input params, validates required fields, and creates a versioned draft with `template_id` provenance metadata.
**Competitive reference**: Juro template engine, Spotdraft smart templates, ContractExpress (Thomson Reuters)

---

### I6. Decimal-Accurate Financial Milestone Tracking
**Category**: Compliance
**Justification**: Float arithmetic silently corrupts contract values at scale (binary floating-point misrepresents KES 1,250,000.50). Regulatory audits and ERP reconciliation require exact decimal arithmetic — a correctness defect in the current implementation that must be closed.
**Implementation**: Replace all `float` monetary fields with `Decimal`; add a `ContractMilestone` record with `amount: Decimal`, `trigger_condition`, and `paid_at`; expose `record_milestone_payment` and `outstanding_milestone_summary` with Decimal-accurate running balance.
**Competitive reference**: Coupa CLM financial milestones, SAP Ariba, Oracle Procurement Cloud

---

### I7. E-Signature Audit Trail with SHA-256 Hash Chain
**Category**: Security
**Justification**: Legally admissible e-signatures require tamper-evident audit chains under Kenya ICT Act 2022 and eIDAS. Hash-chained signature records make post-fact forgery computationally infeasible and satisfy court evidence standards — a gap in the current `record_signature` implementation.
**Implementation**: On each `record_signature` call, compute `SHA-256(prev_chain_hash + signatory_id + signed_at + contract_content_hash)` and store the rolling chain; expose `verify_signature_chain` which re-walks and revalidates the entire chain and returns a tamper-detection report.
**Competitive reference**: DocuSign Certificate of Completion, Adobe Sign tamper-evident seal

---

### I8. LLM Metadata Extraction from Counterparty Paper
**Category**: AI/ML
**Justification**: Uploaded legacy contracts and counterparty-drafted documents contain all critical metadata locked in unstructured text. Auto-extracting parties, dates, payment terms, and termination rights into structured fields eliminates manual data entry — the primary cost driver in contract migration and M&A due diligence projects.
**Implementation**: `extract_contract_metadata` accepts raw contract text, sends to a local Ollama LLM with a structured extraction prompt, parses the JSON response into contract fields (`effective_date`, `expiry_date`, `value`, `governing_law`, `payment_terms`, `termination_notice_days`), and returns a populated draft record.
**Competitive reference**: Kira Systems, Luminance, Thoughtriver

---

### I9. Approval Delegation and Escalation Engine
**Category**: Compliance
**Justification**: Static single-approver workflows fail when approvers are on leave or contracts need board-level sign-off above a value threshold. A Delegation of Authority (DoA) matrix with auto-escalation on inactivity is a hard requirement for SOX-compliant organisations and audit-ready mid-market companies.
**Implementation**: `delegate_approval` transfers a pending approval to a named delegate with reason and audit trail; `escalate_approval` auto-promotes to the next approval level when an SLA threshold is breached; DoA rules configure escalation thresholds by `(contract_type, value_band)`.
**Competitive reference**: Coupa approval delegation, SAP Ariba approval escalation rules, Agiloft

---

### I10. Semantic Full-Text Contract Search
**Category**: Performance
**Justification**: Lawyers spend 40% of review time locating relevant precedent clauses across hundreds of executed contracts. Semantic search over clause and redline text cuts retrieval from hours to seconds and surfaces contracts a keyword search would miss (synonyms, paraphrases, related concepts).
**Implementation**: Build a per-tenant inverted index over contract title, description, tags, and redline text at write time; `search_contracts` accepts a query string, scores by keyword frequency + recency weight, and returns ranked results with matched field highlights.
**Competitive reference**: LinkSquares VISION AI search, Kira Systems, ContractPodAi search

---

### I11. Contract Performance Scorecard
**Category**: Feature
**Justification**: Active contracts generate ongoing performance data (SLA breaches, late payments, obligation misses, disputes) that CLM tools rarely surface. A per-contract scorecard turns the CLM from a repository into an active risk management tool, closing the loop between legal, operations, and finance.
**Implementation**: `record_performance_event` accepts `metric_type` (`sla_breach`, `late_payment`, `obligation_miss`, `dispute`), `severity` (1–5), and optional `amount: Decimal`; aggregates into a `performance_scorecard` with weighted overall score (0–100) and trend direction (`improving` | `stable` | `declining`).
**Competitive reference**: Icertis Contract Intelligence, Conga Contracts, SirionLabs

---

### I12. Jurisdiction-Aware Compliance Checklist
**Category**: Compliance
**Justification**: Contracts governed by Kenyan law, English law, or OHADA require different mandatory clauses, stamp duty rules, and filing obligations. A jurisdiction-aware checklist that validates contracts before execution prevents regulatory non-compliance that voids contracts or triggers penalties.
**Implementation**: Maintain a `JURISDICTION_RULES` registry mapping `(jurisdiction, contract_type)` to required clauses and filing obligations; `run_compliance_scan` checks the contract record against applicable rules and returns a structured report with `passed`, `warnings`, and `failures` lists.
**Competitive reference**: Practical Law (Thomson Reuters), LexisNexis Contract Compass, LexCheck

---

### I13. Automated Redline Conflict Detection
**Category**: AI/ML
**Justification**: When two reviewers independently redline the same clause, conflicts go undetected until a lawyer manually reconciles them. Automated conflict detection at redline-creation time prevents version divergence and reduces the cost of multi-party negotiation cycles.
**Implementation**: On `create_redline`, check for any open redline on the same `(contract_id, section_ref)`; if a conflict is detected, set `conflict_flag=True` on both records and populate a `conflicting_redline_ids` list; emit a `redline_conflict_detected` audit event.
**Competitive reference**: Ironclad concurrent edit detection, ContractPodAi redline conflict alerts

---

### I14. Webhook Notification Bus for Contract Lifecycle Events
**Category**: Integration
**Justification**: Downstream systems (ERP, CRM, AP, HR) need real-time contract-state signals to trigger purchase orders, revenue recognition, and onboarding workflows. Polling is brittle; webhooks with HMAC-SHA256 signatures and exponential-backoff retry are the enterprise integration standard.
**Implementation**: Maintain a `WebhookSubscription` registry per tenant mapping `event_pattern` globs to endpoint URLs; on `_emit`, fan-out matching events to registered endpoints with HMAC-SHA256 request signatures and retry tracking; expose `register_webhook` and `list_webhooks` methods.
**Competitive reference**: DocuSign Connect webhooks, Ironclad workflow triggers, Zapier integration

---

### I15. Renewal Forecast and Portfolio Value-at-Risk Dashboard
**Category**: Feature
**Justification**: Legal and finance teams need forward-looking portfolio analytics — renewal pipeline value, contracts expiring by quarter, projected obligation cash flows. This transforms the CLM from a records system into a strategic forecasting tool used by CFOs and General Counsel, and is the differentiator that justifies enterprise CLM pricing.
**Implementation**: `renewal_forecast` aggregates active contracts by expiry quarter, sums `Decimal` contract values, identifies auto-renewing vs. decision-required contracts, and computes `value_at_risk` (total value of contracts expiring with no renewal scheduled); returns a structured per-quarter breakdown.
**Competitive reference**: Icertis Contract Intelligence analytics, SirionLabs, ContractPodAi Portfolio Analytics
