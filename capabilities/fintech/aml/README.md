# Anti Money Laundering

## Overview
Anti Money Laundering provides real-time transaction monitoring, typology-driven alert generation, sanctions and PEP screening escalation, AML case investigation, and Suspicious Activity Report (SAR) drafting workflows. It acts as the AML control layer across all payment-generating capabilities, receiving transaction signals, applying velocity/structuring/sanctions rules, and routing findings to human analysts or AI-assisted reviewers.

Every monitored transaction must be linked to a KYC profile, ensuring AML decisions are grounded in verified customer identity. SAR filing is gated behind mandatory human approval. All alert, case, and SAR lifecycle events stream to `apg.fintech.aml.lifecycle` via Bytewax.

## Capability ID
`fintech_aml`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| transaction_monitoring | Score and flag transactions against large-transaction, velocity, structuring, and sanctions thresholds |
| aml_alert_triage | Create, triage, and close AML alerts with disposition and reviewer evidence |
| sanctions_pep_escalation | Escalate sanctions and PEP hits requiring immediate review |
| suspicious_activity_case_management | Open and manage AML investigation cases linked to alerts |
| sar_workflow | Draft, approve, and file Suspicious Activity Reports with mandatory human approval |
| typology_rule_engine | Define and evaluate AML typology rules (velocity windows, thresholds, pattern matching) |
| aml_agent_workflow | Register and govern AI agents acting in AML analyst and reviewer roles |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Immutable audit trail |
| ntfy | Analyst and compliance notifications |
| nlpc | NLP for SAR narrative generation |
| keym | Key management |
| fintech_payments | Payment transaction source |
| fintech_wallets | Wallet transfer source |
| fintech_kyc | KYC profile linking (mandatory per monitored transaction) |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| monitoring.large_transaction_threshold | number | 10000.0 | Single-transaction reporting threshold |
| monitoring.velocity_window_minutes | number | 60 | Rolling window for velocity checks |
| monitoring.velocity_count_threshold | number | 5 | Max transactions per window before flag |
| monitoring.velocity_amount_threshold | number | 25000.0 | Max cumulative amount per window |
| monitoring.structuring_threshold | number | 9500.0 | Per-transaction amount suggesting structuring |
| monitoring.structuring_count_threshold | number | 3 | Min transactions to trigger structuring alert |
| monitoring.high_risk_score_threshold | number | 75 | KYC risk score triggering enhanced monitoring |
| alerts.auto_close_allowed | bool | False | Auto-close disabled; human disposition required |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-aml/dashboard | GET | fintech_aml:view | Overview |
| alerts | /fintech-aml/alerts | GET/POST | fintech_aml:triage | Alerts |
| monitoring | /fintech-aml/monitoring | GET | fintech_aml:monitor | Monitoring |
| cases | /fintech-aml/cases | GET/POST | fintech_aml:investigate | Cases |
| sar | /fintech-aml/sar | GET/POST | fintech_aml:file_sar | Regulatory |
| typologies | /fintech-aml/typologies | GET/POST | fintech_aml:admin | Rules |
| agents | /fintech-aml/agents | GET/POST | fintech_aml:admin | Automation |
| settings | /fintech-aml/settings | GET/POST | fintech_aml:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| transaction_requires_kyc_link | Transaction without KYC profile | deny |
| large_transaction_requires_review | Amount > 10,000 without review | require_review |
| velocity_requires_review | Velocity pattern without review | require_review |
| structuring_requires_review | Structuring pattern without review | require_review |
| sanctions_requires_escalation | Sanctions hit without review | require_review |
| high_risk_kyc_requires_review | KYC risk score > 75 without review | require_review |
| alert_close_requires_disposition | Closing alert without disposition | deny |
| alert_escalation_requires_reviewer | Escalating alert without reviewer | deny |
| sar_human_approval_required | SAR without human approval | deny |
| aml_batch_requires_bytewax | Batch without Bytewax | deny |
| aml_event_requires_bytewax | Event without Bytewax | deny |
| privileged_aml_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| AmlTransaction | id, tenant_id, subject_reference, kyc_profile_id, amount, currency, source_capability, source_reference, risk_score, typology_flags, status |
| AmlAlert | id, alert_type, severity, subject_reference, evidence_references, status, disposition, reviewer_id |
| AmlCase | id, alert_id, case_type, investigator_id, subject_reference, status, evidence_references |
| AmlSarDraft | id, case_id, subject_reference, jurisdiction, narrative, evidence_references, human_approval_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| aml_transaction_monitored | Transaction passes through monitoring engine |
| aml_alert_created | New AML alert generated |
| aml_alert_triaged | Alert disposition recorded |
| aml_case_opened | Investigation case opened from alert |
| aml_sar_drafted | SAR draft created for case |
| aml_agent_registered | AI agent registered for AML role |

## Edge Cases Handled
- KYC link is mandatory for every monitored transaction — there is no pathway to record a transaction without a `kyc_profile_id`; anonymous AML monitoring is architecturally blocked
- Auto-close of alerts is explicitly disabled; every alert must have a human-recorded disposition before closure
- SAR drafts require all five fields (case, subject, jurisdiction, narrative, evidence) plus human approval; any missing field produces a deny
- Structuring detection is count-based: a single transaction below the threshold does not trigger the rule, only the multi-transaction pattern does
- Both batch operations and individual events require Bytewax routing — two separate guardrail rules cover each path

## Composability
- **Upstream**: `fintech_kyc` is a hard dependency — every transaction must have a linked KYC profile; `fintech_payments` and `fintech_wallets` are the primary transaction sources
- **Downstream**: `fintech_fraud` reads AML alert presence as an additional fraud signal; `fintech_compliance` ingests AML case outcomes as compliance evidence; `fintech_regtech` uses SAR filings as regulatory submissions
- **Peer**: Deployed alongside `fintech_kyc` (identity foundation), `fintech_fraud` (complementary signal scoring), and `fintech_compliance` (policy and control framework)

## Development Notes
- Typology rules are evaluated against a context dict; adding new typologies requires both a new entry in `SUPPORTED_ALERT_TYPES` and corresponding rule definitions
- The `high_risk_score_threshold` (75) gates enhanced monitoring for KYC profiles; transactions from customers above this score are automatically flagged
- Bytewax is mandatory for both individual events and batch operations — two separate `_ne` guard rules
- `source_reference_required` enforces provenance: every monitored transaction must carry a reference back to the originating capability and record ID
