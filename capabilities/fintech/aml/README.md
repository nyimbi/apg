# Anti Money Laundering

## Overview
Anti Money Laundering provides real-time transaction monitoring, typology-driven alert generation, sanctions and PEP screening escalation, AML case investigation, and Suspicious Activity Report (SAR) drafting workflows. It acts as the AML control layer across all payment-generating capabilities, receiving transaction signals, applying velocity/structuring/sanctions rules, and routing findings to human analysts or AI-assisted reviewers.

Every monitored transaction must be linked to a KYC profile, ensuring AML decisions are grounded in verified customer identity. SAR filing is gated behind mandatory human approval. All alert, case, and SAR lifecycle events stream to `apg.fintech.aml.lifecycle` via Bytewax.

New in v2.0: trade-based ML detection, NFT wash-trade analysis, crypto mixer routing detection, correspondent banking nesting assessment, terrorist financing indicators, and Ollama-powered ML risk scoring.

## Capability ID
`fintech_aml`  Version: 2.0.0

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
| trade_based_ml_detection | Detect over/under-invoicing, phantom shipments, and multiple-invoicing TBML patterns |
| nft_wash_trade_detection | Identify circular NFT transfers at artificially inflated prices |
| crypto_mixer_detection | Flag routing through known mixing/tumbling services (Tornado Cash, CoinJoin, etc.) |
| correspondent_banking_analysis | Assess nested account risk per FATF Recommendation 13 |
| terrorist_financing_detection | Detect hawala patterns, small-amount high-risk-jurisdiction flows, charity misuse |
| network_analysis | Graph-based round-trip and layering detection with Ollama-assisted risk scoring |

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

## Quick Start

```python
from capabilities.fintech.aml.service import AMLService

svc = AMLService(tenant_id="acme", actor_id="compliance-system")

# Monitor a transaction
result = await svc.monitor_transaction({
    "id": "txn-001",
    "subject_reference": "cust-42",
    "kyc_profile_id": "kyc-42",
    "amount": 9800.0,
    "currency": "USD",
    "source_capability": "fintech_payments",
    "source_reference": "pay-001",
    "sender_account": "acc-a",
    "receiver_account": "acc-b",
})
# {"risk_score": 45, "typology_flags": ["structuring"], "alerts_generated": ["alert-xyz"], ...}
```

## New Methods

### `detect_trade_based_ml` — Over/under-invoicing and phantom shipment detection
```python
result = await svc.detect_trade_based_ml(
    invoices=[
        {"commodity_code": "HS8471", "unit_price": 500.0, "quantity": 10, "amount": 5000.0},
        {"commodity_code": "HS8471", "unit_price": 50.0,  "quantity": 10, "amount": 500.0},
    ],
    market_value_lookup={"HS8471": 300.0},
    over_under_threshold=0.15,
)
# {"detected": True, "typologies": ["under_invoicing"], "flagged_invoices": [...], "risk_score": 72}
```

### `detect_nft_wash_trading` — Circular NFT transfer detection
```python
result = await svc.detect_nft_wash_trading(
    nft_transfers=[
        {"token_id": "nft-1", "from_wallet": "0xA", "to_wallet": "0xB", "price": 1.0, "created_at": "..."},
        {"token_id": "nft-1", "from_wallet": "0xB", "to_wallet": "0xA", "price": 8.0, "created_at": "..."},
    ],
    lookback_days=30,
    price_inflation_threshold=3.0,
)
# {"detected": True, "wash_trade_score": 0.87, "flagged_tokens": ["nft-1"], "patterns": [...]}
```

### `detect_crypto_mixer_routing` — Tumbler and CoinJoin detection
```python
result = await svc.detect_crypto_mixer_routing(
    crypto_transactions=[
        {"tx_hash": "0xabc", "from_address": "0x1", "to_address": "0xTornadoCash", "service_label": "tornado_cash"},
    ],
    known_mixer_addresses={"0xTornadoCash", "0xChipMixer"},
)
# {"detected": True, "mixer_indicators": ["known_mixer_address", "service_label_match"], "flagged_transactions": [...]}
```

### `correspondent_banking_analysis` — FATF Rec. 13 nesting risk
```python
result = await svc.correspondent_banking_analysis(
    correspondent_chain=[
        {"bic": "BARCGB22", "jurisdiction": "GB", "aml_rating": "good",  "kyb_verified": True},
        {"bic": "CBKEXXX0", "jurisdiction": "IR", "aml_rating": "poor",  "kyb_verified": False},
        {"bic": "LOCALXXX", "jurisdiction": "IR", "aml_rating": "unknown", "kyb_verified": False},
    ],
    max_nesting_depth=3,
)
# {"nesting_depth": 3, "risk_score": 85, "risk_factors": ["high_risk_jurisdiction", "poor_aml_rating"], ...}
```

### `detect_terrorist_financing` — FATF TF typology screening
```python
result = await svc.detect_terrorist_financing(
    customer_id="cust-42",
    lookback_days=90,
    customer_profile={"adverse_media_terrorism": True, "charity_sector": False},
)
# {"detected": True, "tf_indicators": ["adverse_media_terrorism", "high_risk_jurisdiction_small_amount"], "risk_score": 80, ...}
```

## World-Class Enhancements (v2.0)

| # | Enhancement | Benefit | Complexity | APG Integration |
|---|-------------|---------|-----------|-----------------|
| 1 | Behavioural Biometric Risk Scoring | -35% false positives via device/session signals | Medium | ai_orchestration |
| 2 | Federated GNN Network Analysis | +60% layering recall; cross-tenant learning without data sharing | High | federated_learning |
| 3 | LLM SAR Narrative Auto-Draft | -40% investigator time; Ollama drafts FinCEN/FCA-compliant narratives | Low | ai_orchestration/Ollama |
| 4 | Real-Time Bytewax Streaming | Sub-100ms alert generation; catches structuring within the window | Medium | Bytewax |
| 5 | SHAP Explainability per Alert | Per-alert feature attribution for regulatory AI explainability | Medium | ai_orchestration |
| 6 | Correspondent Banking Nesting | First-class FATF Rec. 13 nested-account chain reconstruction | Low | SWIFT integration |
| 7 | Predictive SAR Prioritisation | Gradient-boosted P(SAR) score; -25% time-to-SAR | High | ai_orchestration |
| 8 | Regulatory Change Intelligence | LLM-driven mapping of new FinCEN/FCA/FATF guidance to affected rules | Low | ai_orchestration/feeds |
| 9 | Cross-Tenant Typology Intelligence | Differential-privacy typology updates across tenants in real time | High | federated_learning |
| 10 | Automated E2E Regulatory Filing | FinCEN BSA XML generation, API submission, cryptographic audit trail | Medium | keym/audl/FinCEN API |

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
| correspondent_nesting_depth | Chain depth > max_nesting_depth | deny |
| crypto_mixer_detected | Routing through known mixer | deny |
| nft_wash_trade_detected | wash_trade_score above threshold | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| AmlTransaction | id, tenant_id, subject_reference, kyc_profile_id, amount, currency, source_capability, source_reference, risk_score, typology_flags, status |
| AmlAlert | id, alert_type, severity, subject_reference, evidence_references, status, disposition, reviewer_id |
| AmlCase | id, alert_id, case_type, investigator_id, subject_reference, status, evidence_references |
| AmlSarDraft | id, case_id, subject_reference, jurisdiction, narrative, evidence_references, human_approval_reference |
| NetworkAnalysisResult | subject_reference, round_trip_detected, layering_detected, network_risk_score, typology_flags |
| PatternDetectionResult | subject_reference, structuring_detected, velocity_anomaly, risk_delta, recommended_action |

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
| tbml_detected | Trade-based ML pattern identified |
| nft_wash_trade_detected | NFT wash-trading pattern identified |
| correspondent_banking_assessed | Correspondent chain risk scored |
| terrorist_financing_indicators_detected | TF indicators found for customer |
| network_analysis_complete | Network risk graph analysis complete |
| pattern_detection_complete | Pattern scan complete for customer |

## Edge Cases Handled
- KYC link is mandatory for every monitored transaction — anonymous AML monitoring is architecturally blocked
- Auto-close of alerts is explicitly disabled; every alert must have a human-recorded disposition
- SAR drafts require all five fields (case, subject, jurisdiction, narrative, evidence) plus human approval
- Structuring detection is count-based: a single sub-threshold transaction does not trigger the rule
- Both batch operations and individual events require Bytewax routing — two separate guardrail rules cover each path
- Correspondent banking chains exceeding `max_nesting_depth` raise a `RuleViolation` before scoring completes
- Crypto mixer detection raises `RuleViolation` on any positive detection — the method does not return a result object
- Ollama-based ML risk scoring in `pattern_detection` gracefully degrades to rule-based scoring when `OLLAMA_BASE_URL` is unset

## Composability
- **Upstream**: `fintech_kyc` is a hard dependency — every transaction must have a linked KYC profile; `fintech_payments` and `fintech_wallets` are the primary transaction sources
- **Downstream**: `fintech_fraud` reads AML alert presence as an additional fraud signal; `fintech_compliance` ingests AML case outcomes as compliance evidence; `fintech_regtech` uses SAR filings as regulatory submissions
- **Peer**: Deployed alongside `fintech_kyc` (identity foundation), `fintech_fraud` (complementary signal scoring), and `fintech_compliance` (policy and control framework)
- **AI**: Integrates with `ai_orchestration` for Ollama-based risk scoring and SAR narrative generation; `federated_learning` for cross-tenant GNN training and differential-privacy typology sharing

## Development Notes
- Typology rules are evaluated against a context dict; adding new typologies requires both a new entry in `SUPPORTED_ALERT_TYPES` and corresponding rule definitions
- The `high_risk_score_threshold` (75) gates enhanced monitoring; transactions from customers above this score are automatically flagged
- Bytewax is mandatory for both individual events and batch operations — two separate `_ne` guard rules
- `source_reference_required` enforces provenance: every monitored transaction must carry a reference back to the originating capability and record ID
- `AMLService` is async-native; `AntiMoneyLaunderingService` (alias `FintechAmlService`) is a backward-compatible sync wrapper for legacy tests
