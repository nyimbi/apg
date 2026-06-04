# Know Your Customer

## Overview
Know Your Customer provides the customer identity foundation for the entire APG fintech platform: tenant-scoped identity profiles, consent-backed onboarding, document verification with minimum confidence thresholds, sanctions/PEP/adverse-media/watchlist screening, KYC risk scoring, customer due diligence, enhanced due diligence for high-risk profiles, and AI-assisted review workflows. It is a hard dependency for every capability that onboards customers.

Profiles must carry consent evidence. Documents must have a tokenized storage reference and an extracted subject record. Screening hits require review before a verification decision can be recorded. High-risk profiles (score > 75) require enhanced due diligence review. Open review flags block verification decisions. All KYC events stream to `apg.fintech.kyc.lifecycle` via Bytewax.

## Capability ID
`fintech_kyc`  Version: 1.1.0

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

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| documents.minimum_confidence | number | 0.75 | Minimum document confidence score |
| risk.high_risk_threshold | number | 75 | Score triggering EDD requirement |
| risk.medium_risk_threshold | number | 45 | Score triggering enhanced monitoring |
| decisions.expiry_days | number | 365 | KYC decision validity period |
| customers.supported_types | list | individual, sole_proprietor, business, nonprofit, government | Customer types |
| documents.supported_types | list | passport, national_id, driver_license, resident_permit, business_registration, tax_id, utility_bill, bank_statement | Document types |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-kyc/dashboard | GET | fintech_kyc:view | Overview |
| profiles | /fintech-kyc/profiles | GET/POST | fintech_kyc:manage_profiles | Profiles |
| documents | /fintech-kyc/documents | GET/POST | fintech_kyc:manage_documents | Evidence |
| screening | /fintech-kyc/screening | GET/POST | fintech_kyc:screen | Screening |
| risk | /fintech-kyc/risk | GET/POST | fintech_kyc:review_risk | Risk |
| reviews | /fintech-kyc/reviews | GET/POST | fintech_kyc:review | Reviews |
| agents | /fintech-kyc/agents | GET/POST | fintech_kyc:admin | Automation |
| settings | /fintech-kyc/settings | GET/POST | fintech_kyc:admin | Administration |

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

## Data Models
| Model | Key Fields |
|-------|-----------|
| KycProfile | id, tenant_id, subject_reference, legal_name, customer_type, country, consent_reference, status |
| KycDocument | id, profile_id, document_type, token_reference, extracted_subject, confidence, status |
| KycScreening | id, profile_id, sanctions_hit, pep_hit, adverse_media_hit, watchlist_hit, review_reference |
| KycRiskScore | id, profile_id, score, risk_band, source_reference |
| KycDecision | id, profile_id, decision, identity_doc_reference, address_doc_reference, screening_reference, risk_reference, expiry_date |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| kyc_profile_opened | Identity profile created |
| kyc_document_registered | Document registered and confidence-checked |
| kyc_screening_recorded | Sanctions/PEP/watchlist screening recorded |
| kyc_risk_scored | Risk score recorded |
| kyc_decision_recorded | CDD/EDD verification decision recorded |
| kyc_agent_registered | AI agent registered |

## Edge Cases Handled
- Document confidence below 0.75 is rejected at registration time — a low-confidence document cannot be submitted as evidence, preventing weak identity proofs from anchoring a verification decision
- Open review flags (unresolved screening hits or EDD reviews) block verification decisions — a profile cannot be verified while any review flag is outstanding; all flags must be resolved first
- High-risk profiles (score > 75) trigger a require-review for EDD — EDD evidence must be recorded before the final verification decision can be made
- Screening records all four dimensions (sanctions, PEP, adverse media, watchlist) in a single record; a hit on any one dimension triggers the `screening_hit` flag
- KYC decisions carry an expiry date (default 365 days); expired decisions are treated as requiring re-verification by consuming capabilities

## Composability
- **Upstream**: `cons` provides consent management; `biop` and `cvsn` provide biometric liveness checks and document OCR/extraction; `keym` manages tokenized document storage references
- **Downstream**: Every other fintech capability that onboards customers reads KYC profile references — `fintech_aml`, `fintech_fraud`, `fintech_payments`, `fintech_wallets`, `fintech_cards`, `fintech_lending`, `fintech_neobanking`, `fintech_agency`, `fintech_remittance`, `fintech_mobile`, `fintech_bnpl`
- **Peer**: `fintech_aml` is the closest peer — every AML-monitored transaction requires a linked KYC profile; the two capabilities are designed to be deployed as a pair

## Development Notes
- `cons` (consent) is a separate required dependency from `auth` — customer consent for data processing is distinct from operational authentication and is managed by a dedicated consent service
- Document token references must point to a secure document vault managed by `keym`; raw document files never flow through the KYC capability
- Customer type `sole_proprietor` sits between individual and business — it has individual-style identity documents but may require business registration evidence; the capability does not differentiate rule requirements by type
- Both batch operations and individual events require Bytewax routing — two separate guardrail rules; this is consistent with `fintech_aml` and `fintech_fraud`
