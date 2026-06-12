# Know Your Customer (fintech_kyc)

## Overview
`fintech_kyc` provides the customer identity foundation for the entire APG fintech platform: tenant-scoped identity profiles, consent-backed onboarding, document verification with minimum confidence thresholds, sanctions/PEP/adverse-media/watchlist screening, KYC risk scoring, customer due diligence, enhanced due diligence for high-risk profiles, and AI-assisted review workflows. It is a hard dependency for every capability that onboards customers.

Business invariants enforced at every layer:
- Profiles require consent evidence; documents require a tokenized storage reference and an extracted subject record
- Screening hits require review before a verification decision can be recorded
- High-risk profiles (score > 75) require EDD; open review flags block verification
- All KYC lifecycle events stream to `apg.fintech.kyc.lifecycle` via Bytewax

**Version**: 2.0.0 | **Capability ID**: `fintech_kyc`

---

## Quick Start

```python
from capabilities.fintech.kyc.service import KYCService

svc = KYCService(
    tenant_id="acme",
    actor_id="ops@acme.co",
    db_url="postgresql+asyncpg://user:pass@localhost/apg",
)

# Open an application
app = await svc.start_kyc_application(
    customer_id="cust_001",
    customer_type="individual",
    jurisdiction="KE",
    legal_name="Jane Kamau",
    consent_reference="cons_xyz",
    kyc_tier="standard",
)

# Upload and verify a document
doc = await svc.upload_document(
    application_id=app["id"],
    doc_type="national_id",
    file_metadata={"token_reference": "vault://docs/abc", "filename": "id.pdf"},
    uploaded_by="ops@acme.co",
)
await svc.verify_document_authenticity(doc["id"])

# Screen and score
await svc.watchlist_screening("Jane Kamau", application_id=app["id"])
await svc.calculate_risk_score(app["id"])

# Approve
result = await svc.approve_application(app["id"], reviewer_id="reviewer_001")
```

---

## Provides

| Service | Description |
|---------|-------------|
| customer_identity_lifecycle | Open and maintain tenant-scoped KYC profiles with consent and country |
| document_verification_workflow | Register documents with tokenized reference, extracted subject, and confidence threshold |
| sanctions_pep_screening | Screen profiles against sanctions, PEP, watchlist, and adverse media sources |
| kyc_risk_scoring | Score profiles 0–100 with high-risk EDD gating |
| customer_due_diligence | Record standard CDD verification decisions with required evidence chain |
| enhanced_due_diligence | Record EDD reviews for high-risk profiles |
| kyc_agent_workflow | Register AI agents for document review, sanctions review, and onboarding |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| cons | Consent management |
| ntfy | KYC officer notifications |
| biop | Biometrics for liveness and face match |
| cvsn | Computer vision for document extraction |
| nlpc | NLP for document analysis |
| keym | Key management |
| fintech_payments | Payment account linkage |
| fintech_wallets | Wallet account linkage |

---

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-kyc/dashboard | GET | fintech_kyc:view | Overview |
| health | /fintech-kyc/health | GET | fintech_kyc:view | Operations |
| profiles | /fintech-kyc/profiles | GET/POST | fintech_kyc:manage_profiles | Profiles |
| documents | /fintech-kyc/documents | GET/POST | fintech_kyc:manage_documents | Evidence |
| screening | /fintech-kyc/screening | GET/POST | fintech_kyc:screen | Screening |
| risk | /fintech-kyc/risk | GET/POST | fintech_kyc:review_risk | Risk |
| reviews | /fintech-kyc/reviews | GET/POST | fintech_kyc:review | Reviews |
| agents | /fintech-kyc/agents | GET/POST | fintech_kyc:admin | Automation |
| settings | /fintech-kyc/settings | GET/POST | fintech_kyc:admin | Administration |

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| documents.minimum_confidence | number | 0.75 | Minimum document confidence score |
| risk.high_risk_threshold | number | 75 | Score triggering EDD requirement |
| risk.medium_risk_threshold | number | 45 | Score triggering enhanced monitoring |
| decisions.expiry_days | number | 365 | KYC decision validity period |
| customers.supported_types | list | individual, sole_proprietor, business, nonprofit, government | Customer types |
| documents.supported_types | list | passport, national_id, driver_license, resident_permit, business_registration, tax_id, utility_bill, bank_statement | Document types |

---

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| profile_consent_required | Profile without consent evidence | deny |
| profile_legal_name_required | Profile without legal name | deny |
| document_token_required | Document without tokenized reference | deny |
| document_subject_required | Document without extracted subject | deny |
| document_confidence_minimum | Document confidence < 0.75 | deny |
| screening_hits_require_review | Sanctions/PEP/watchlist/adverse-media hit without review | require_review |
| risk_score_range | Score outside 0–100 | deny |
| high_risk_requires_edd | Score > 75 without EDD review | require_review |
| decision_identity_document_required | Verification without identity document | deny |
| decision_address_document_required | Verification without address document | deny |
| decision_blocks_open_reviews | Verification with unresolved review flags | deny |
| kyc_batch_requires_bytewax | Batch without Bytewax | deny |
| kyc_event_requires_bytewax | Event without Bytewax | deny |

---

## Data Models

| Model | Key Fields |
|-------|-----------|
| KycProfile | id, tenant_id, subject_reference, legal_name, customer_type, country, consent_reference, status |
| KycDocument | id, profile_id, document_type, token_reference, extracted_subject, confidence, status |
| KycScreening | id, profile_id, sanctions_hit, pep_hit, adverse_media_hit, watchlist_hit, review_reference |
| KycRiskScore | id, profile_id, score, risk_band, source_reference, score_breakdown |
| KycDecision | id, profile_id, decision, identity_doc_reference, address_doc_reference, screening_reference, risk_reference, expiry_date |

---

## Streaming Events

| Event | Trigger |
|-------|---------|
| kyc_profile_opened | Identity profile created |
| kyc_document_registered | Document registered and confidence-checked |
| kyc_screening_recorded | Sanctions/PEP/watchlist screening recorded |
| kyc_risk_scored | Risk score recorded |
| kyc_decision_recorded | CDD/EDD verification decision recorded |
| kyc_agent_registered | AI agent registered |
| kyc_behavioral_risk_updated | Rolling 90-day behavioural re-score complete |
| kyc_tier_upgrade_recommended | ML model recommends tier upgrade |
| kyc_bulk_ocr_progress | Bulk OCR batch progress event |
| kyc_sanctions_list_synced | Sanctions list delta sync complete |

---

## World-Class Enhancements (v2.0)

All 15 improvements are targeting Africa-ready, FATF-compliant, production-grade KYC:

1. **AI Document Fraud Detection** — `ai_document_fraud_analysis(document_id, model="llava")` routes to a locally-hosted Ollama vision model (llava:13b / moondream) for hologram, microprint, font, and edge-artifact analysis. Returns per-feature `FraudSignal` scores. No cloud dependency.

2. **Continuous Behavioural Risk Re-Scoring** — `recompute_behavioral_risk(customer_id, transaction_summary)` ingests a rolling 90-day summary from `fintech_aml`, recalculates risk, and emits `kyc_behavioral_risk_updated`. Adds a `behavioral` sub-score to `RiskProfile.score_breakdown`.

3. **Federated Identity / Verifiable Credentials** — `verify_verifiable_credential(vc_jwt, issuer_did)` validates W3C VC format, resolves issuer DID, verifies JWT signature, and asserts schema compliance. Enables wallet-based identity presentation for Kenya NIIMS, Nigeria NIMC e-ID, GhanaCard digital.

4. **Tamper-Evident Audit Trail** — Each audit event is hash-chained (SHA-256 of prior event ID). `verify_audit_integrity(from_dt, to_dt)` re-derives the chain and flags gaps or mutations. Satisfies ISAE 3402 / SOC 2 Type II.

5. **Multi-Jurisdiction Rule Engine** — `JurisdictionRuleSet` registry keyed by `country_code`. `load_jurisdiction_rules(country_code)` and `evaluate_jurisdiction_compliance(application_id)` apply correct thresholds for KE, NG, GH, ZA, TZ concurrently (e.g. CBK EDD gate at score > 60 vs FATF default 75).

6. **Async Batch Document OCR** — `bulk_document_ocr(application_ids, concurrency=10)` uses `asyncio.Semaphore`-bounded concurrent extraction, persists batch job progress, emits `kyc_bulk_ocr_progress`. Returns `job_id` for `get_batch_job_status(job_id)`.

7. **Real-Time Sanctions List Delta Sync** — `sync_sanctions_lists(lists)` pulls delta updates from OFAC/UN/EU mirrors, diffs against prior snapshot, and re-screens only affected customers. O(delta × customers) vs O(full_list × customers). Emits counts of additions, removals, and rescreened.

8. **Multi-Language Name Transliteration** — `transliterate_and_screen(name, source_script, aliases)` produces Latin-script variants of Ge'ez/Amharic, Arabic, and Swahili-diacritic names before PEP/sanctions screening. Eliminates false negatives from script mismatch.

9. **Active Liveness Challenge-Response** — `active_liveness_challenge(session_id)` issues a randomised challenge (blink twice, turn left, smile) returning a `ChallengeToken`. `verify_liveness_challenge_response(session_id, challenge_token, response_frames)` validates against the issued challenge. Reaches ISO 30107-3 PAD Level 2.

10. **Sub-National Geographic Risk Layering** — `geo_risk_score(latitude, longitude, radius_km)` queries a local risk raster (ACLED conflict events, FSI fragility index, UNODC drug-route proximity) returning a 0.0–1.0 score. Feeds `calculate_risk_score` as a `geographic_risk` factor.

11. **Customer-Controlled KYC Export / Erasure** — `export_customer_kyc_data(customer_id, format)` packages all KYC data as structured JSON or encrypted PDF. `delete_customer_kyc_data(customer_id, reason)` supports right-to-erasure under Kenya DPA 2019, NDPR, GDPR Art. 20, POPIA.

12. **Automated KYC Tier Upgrade via ML** — `evaluate_tier_upgrade(customer_id)` applies a gradient boosting model (local scikit-learn or Ollama) over transaction history, document quality, screening streak, and account tenure. Emits `kyc_tier_upgrade_recommended` for ops approval.

13. **Typed Event Bus for Cross-Capability Composability** — `KYCLifecycleEvent` Pydantic model replaces raw `_emit` calls with typed Bytewax publishes. `subscribe_to_kyc_events(event_types, handler)` allows peer capabilities (`fintech_aml`, `fintech_fraud`) to register interest without polling the KYC store.

14. **UBO Corporate Ownership Tree Traversal** — `resolve_ownership_tree(entity_id, max_depth=5)` recursively resolves corporate shareholders, builds a directed ownership graph, flags circular patterns (shell structures), and identifies UBOs at any depth exceeding the threshold. Reference: FATF R.24, EU 6AMLD.

15. **KYC Health Dashboard with SLA Metrics** — `kyc_health_metrics(period)` computes p50/p95/p99 processing times by step, `document_verification_pass_rate`, `screening_hit_rate`, `edd_escalation_rate`, `sla_breach_count` (configurable per tier), and `onboarding_funnel_conversion`. Exposed at `/fintech-kyc/health`.

---

## New Methods

### `recompute_behavioral_risk`
```python
result = await svc.recompute_behavioral_risk(
    customer_id="cust_001",
    transaction_summary={
        "window_days": 90,
        "tx_count": 143,
        "avg_amount_usd": 420.0,
        "high_risk_counterparty_count": 2,
        "channel_changes": 1,
    },
)
# result["score"] — updated composite score
# result["score_breakdown"]["behavioral"] — behavioural sub-score
```

### `ai_document_fraud_analysis`
```python
fraud = await svc.ai_document_fraud_analysis(
    document_id="doc_abc",
    model="llava",  # local Ollama vision model
)
# fraud["fraud_signal"]["hologram_score"]
# fraud["fraud_signal"]["microprint_score"]
# fraud["authentic"] — bool
```

### `verify_verifiable_credential`
```python
vc_result = await svc.verify_verifiable_credential(
    vc_jwt="eyJ...",
    issuer_did="did:web:niims.go.ke",
)
# vc_result["valid"] — bool
# vc_result["credential_subject"] — extracted claims
# vc_result["schema_compliant"] — bool
```

### `resolve_ownership_tree`
```python
tree = await svc.resolve_ownership_tree(
    entity_id="entity_corp_001",
    max_depth=5,
)
# tree["ubos"] — list of UBOs at any depth ≥ threshold
# tree["circular_patterns"] — list of detected shell cycles
# tree["graph_edges"] — ownership graph for visualisation
```

### `kyc_health_metrics`
```python
metrics = await svc.kyc_health_metrics(period="2026-05")
# metrics["p95_processing_seconds"]["document_verification"]
# metrics["sla_breach_count"]
# metrics["onboarding_funnel_conversion"]
# Also available at GET /fintech-kyc/health
```

---

## Composability

- **Upstream**: `cons` (consent), `biop` (liveness/face match), `cvsn` (document OCR), `keym` (tokenized vault)
- **Downstream**: Every fintech capability that onboards customers reads KYC profile references — `fintech_aml`, `fintech_fraud`, `fintech_payments`, `fintech_wallets`, `fintech_cards`, `fintech_lending`, `fintech_neobanking`, `fintech_agency`, `fintech_remittance`, `fintech_mobile`, `fintech_bnpl`
- **Peer**: `fintech_aml` — every AML-monitored transaction requires a linked KYC profile; the two capabilities are deployed as a pair
- **Event consumers**: peer capabilities subscribe via `subscribe_to_kyc_events` rather than polling the KYC store directly

---

## Edge Cases Handled

- Documents with confidence < 0.75 are rejected at registration — prevents weak identity proofs anchoring a verification decision
- Open review flags block verification decisions — all must be resolved before approval
- High-risk profiles (score > 75) gate on EDD completion before approval
- Screening records all four dimensions (sanctions, PEP, adverse media, watchlist) in a single record; any one hit triggers the `screening_hit` flag
- KYC decisions carry an expiry date (default 365 days); expired decisions require re-verification
- Refugee customers and informal sector workers receive relaxed document requirements flagged at `start_kyc_application`
- `sole_proprietor` sits between individual and business — individual-style identity documents apply; service does not differentiate rule requirements by type
- Raw document bytes never flow through the service — only tokenized vault references

---

## Development Notes

- `cons` (consent) is a separate required dependency from `auth` — customer consent for data processing is distinct from operational authentication
- Both batch operations and individual events require Bytewax routing — two separate guardrail rules; consistent with `fintech_aml` and `fintech_fraud`
- Local AI models (Ollama llava:13b, moondream, Helsinki-NLP transliterators) are served from the same infrastructure as all other AI capabilities per the platform AI strategy — no cloud model dependencies
- `© 2025 Datacraft · www.datacraft.co.ke`
