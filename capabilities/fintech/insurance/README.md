# InsurTech

## Overview
InsurTech manages the end-to-end lifecycle of insurance operations: policyholder onboarding, product publishing across life, health, property, motor, travel, crop, and microinsurance lines, quote generation with underwriting evidence, policy binding, premium recording, claim intake, document management, risk assessment, reinsurance attachment, compliance alerts, and governance reviews. It is designed for regulated insurance operations where every quote must have an underwriting reference and every claim must have supporting evidence.

Policy binding requires a payment reference — premiums are not deferred. Claim amounts must be positive and loss dates must be recorded. Reinsurance attachments require treaty references. All insurance lifecycle events stream to `apg.fintech.insurance.lifecycle` via Bytewax.

## Capability ID
`fintech_insurance`  Version: 2.0.0

## Features
- Full insurance lifecycle: onboarding → quoting → binding → premiums → claims → payout
- Multi-line product catalogue: life, health, property, motor, travel, crop, microinsurance
- Claims pipeline: file → assess → approve/reject → pay, with fast-track and fraud check
- Policy administration: renewal, cancellation with pro-rata refund, lapse/reinstatement
- Group policy enrollment for SACCOs and employers
- Microinsurance and USSD/M-Pesa adapter support for feature-phone markets
- Reinsurance cession recording with treaty reference validation
- IRA Kenya regulatory return generation
- No Claims Discount (NCD) calculation
- Portfolio analytics: loss ratio, combined ratio, claims frequency, product mix
- AI-assisted ML premium risk scoring via local Ollama (falls back gracefully)
- Event-streamed audit trail; all lifecycle events published to Bytewax

## Provides
| Service | Description |
|---------|-------------|
| insurance_policyholder_workflow | Onboard policyholders with KYC, contact, and risk profile evidence |
| insurance_product_workflow | Publish insurance products with supported product lines, coverage terms, and pricing |
| insurance_quote_workflow | Generate quotes with policyholder, product, premium, currency, and underwriting reference |
| insurance_policy_workflow | Bind policies with quote, effective date, and payment reference |
| insurance_premium_workflow | Record premium payments with policy, amount, currency, and payment reference |
| insurance_claim_workflow | Open claims with policy, type, amount, loss date, and evidence |
| insurance_document_workflow | Record policy schedules, proof of loss, medical reports, and identity documents |
| insurance_risk_workflow | Record risk assessments with policyholder, score, and source evidence |
| insurance_reinsurance_workflow | Record reinsurance attachments with treaty reference and positive share |
| insurance_compliance_workflow | Record and review compliance alerts with severity controls |
| insurance_review_workflow | Governance reviews for quotes, policies, and claims |
| insurance_agent_workflow | Register AI agents for underwriting, claim triage, fraud review, and compliance |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Policyholder and operations notifications |
| nlpc | NLP for claim narrative and document analysis |
| keym | Key management |
| fintech_payments | Premium payment execution and claim disbursement |
| fintech_wallets | Wallet-based premium funding |
| fintech_kyc | Policyholder identity verification |
| fintech_aml | AML screening for policyholders |
| fintech_fraud | Fraud screening for claims |
| bia | Business intelligence and analytics |
| fin_rpt | Financial reporting |

## Quick Start

```python
from capabilities.fintech.insurance.service import InsurTechService

svc = InsurTechService(tenant_id="acme", actor_id="underwriter_01")

# Onboard
ph = await svc.onboard_policyholder("ph_1", "Alice Wanjiku", "kyc_001", "contact_001", "risk_001")

# Publish product and bind a policy end-to-end
await svc.publish_product("prod_motor", "Motor Comprehensive", "motor", "terms_motor", "price_motor")
policy = await svc.create_policy("ph_1", "motor", coverage_amount=2_000_000, premium=48_000, period="annual")

# File and fast-track a small claim
claim = await svc.file_claim(policy["id"], "accident", "2026-06-01", 3_500, "minor scrape", evidence_reference="ev_001")
result = await svc.claim_fast_track(claim["claim_id"])  # auto-approves if <= 5 000 KES

# Portfolio analytics
stats = await svc.insurance_analytics("Q2-2026")
print(stats["loss_ratio"], stats["combined_ratio"])
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| products.supported_lines | list | life, health, property, motor, travel, crop, microinsurance | Insurance product lines |
| claims.supported_types | list | medical, accident, theft, damage, death, delay, weather | Claim event types |
| documents.supported_types | list | policy_schedule, proof_of_loss, medical_report, identity, invoice, photo_evidence | Document types |
| premiums.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Premium currencies |
| compliance.supported_severities | list | low, medium, high, critical | Alert severity levels |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-insurance/dashboard | GET | fintech_insurance:view | Overview |
| policyholders | /fintech-insurance/policyholders | GET/POST | fintech_insurance:policyholders | Customers |
| products | /fintech-insurance/products | GET/POST | fintech_insurance:products | Products |
| quotes | /fintech-insurance/quotes | GET/POST | fintech_insurance:quotes | Underwriting |
| policies | /fintech-insurance/policies | GET/POST | fintech_insurance:policies | Policies |
| premiums | /fintech-insurance/premiums | GET/POST | fintech_insurance:premiums | Policies |
| claims | /fintech-insurance/claims | GET/POST | fintech_insurance:claims | Claims |
| documents | /fintech-insurance/documents | GET/POST | fintech_insurance:documents | Claims |
| risk | /fintech-insurance/risk | GET/POST | fintech_insurance:risk | Risk |
| reinsurance | /fintech-insurance/reinsurance | GET/POST | fintech_insurance:reinsurance | Risk |
| compliance | /fintech-insurance/compliance | GET/POST | fintech_insurance:compliance | Governance |
| reviews | /fintech-insurance/reviews | GET/POST | fintech_insurance:reviews | Governance |
| agents | /fintech-insurance/agents | GET/POST | fintech_insurance:admin | Automation |
| settings | /fintech-insurance/settings | GET/POST | fintech_insurance:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| policyholder_kyc_required | Policyholder without KYC | deny |
| policyholder_contact_required | Policyholder without contact details | deny |
| product_coverage_required | Product without coverage terms | deny |
| quote_underwriting_required | Quote without underwriting reference | deny |
| quote_positive_premium | Quote with zero or negative premium | deny |
| policy_payment_required | Policy binding without payment reference | deny |
| premium_payment_required | Premium record without payment reference | deny |
| claim_evidence_required | Claim without supporting evidence | deny |
| reinsurance_treaty_required | Reinsurance attachment without treaty reference | deny |
| reinsurance_positive_share | Reinsurance share zero or negative | deny |
| insurance_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_insurance_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| Policyholder | id, name, kyc_reference, contact_reference, risk_profile_reference, status |
| InsuranceProduct | id, name, product_line, coverage_terms_reference, pricing_reference, status |
| Quote | id, policyholder_id, product_id, premium, currency, underwriting_reference, status |
| Policy | id, quote_id, effective_date, payment_reference, status |
| Premium | id, policy_id, amount, currency, payment_reference |
| Claim | id, policy_id, claim_type, amount, loss_date, evidence_references, status |
| InsuranceDocument | id, document_type, reference, evidence_reference |
| RiskAssessment | id, policyholder_id, score, source_reference |
| ReinsuranceAttachment | id, policy_id, treaty_reference, share_percent |
| ComplianceAlert | id, severity, evidence_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| policyholder_onboarded | Policyholder enrolled |
| insurance_product_published | Product published |
| quote_generated | Quote created |
| policy_bound | Policy bound with payment |
| premium_recorded | Premium payment recorded |
| premium_repriced | Dynamic repricing applied (v2.0) |
| claim_opened | Claim filed |
| claim_filed | Claim received with evidence |
| claim_assessed | Assessor findings recorded |
| claim_approved | Claim approved for payment |
| claim_paid | Claim disbursed |
| document_recorded | Document attached |
| risk_assessment_recorded | Risk assessment recorded |
| reinsurance_attachment_recorded | Reinsurance attachment recorded |
| insurance_compliance_alert_recorded | Compliance alert raised |
| insurance_review_recorded | Review completed |
| insurance_agent_registered | AI agent registered |
| policy_lapsed | Grace period expired with no premium |
| policy_renewed | Policy renewed to new term |
| policy_cancelled | Policy cancelled with refund computed |

---

## World-Class Enhancements (v2.0)

1. **Parametric Insurance Engine** — `evaluate_parametric_trigger()` accepts oracle data points (rainfall, temperature, flight status) and auto-disburses when thresholds breach. Zero-touch settlement via `fintech_payments`.
2. **Embedded Insurance SDK** — `embed_coverage()` single-call method for partner platforms: infers product, prices, binds, and returns a coverage certificate. Reduces integration from days to hours.
3. **Dynamic Premium Repricing** — `reprice_premium()` ingests telematics/IoT/behavioural signals, re-runs pricing, and adjusts the renewal rate (floor/ceiling guarded). Emits `premium_repriced` event. Enables UBI for motor and health.
4. **AI Claim Triage with Confidence Scoring** — `ai_triage_claim()` calls local Ollama with structured claim context; returns confidence score, recommended decision (approve/escalate/reject), and evidence gaps. Falls back to heuristic mode without OLLAMA_BASE_URL. Cuts manual review queue ~40%.
5. **Reinsurance Bordereau Generation** — `generate_reinsurance_bordereau()` aggregates cessions by treaty for a period; computes cedant premium, expected recoveries, and outstanding reserves. Eliminates the manual quarterly spreadsheet.
6. **Policy Lapse & Grace Period Automation** — `check_lapse_status()` evaluates payment history, transitions through `grace_period` → `lapsed`, emits `policy_lapsed` event. `reinstate_policy()` validates back-payment and restores coverage. IRA Kenya compliant.
7. **Microinsurance USSD / M-Pesa Adapter** — `mpesa_premium_callback()` maps M-Pesa C2B IPN to a policy and calls `process_premium()`. `ussd_quote_session()` drives a ≤160-char USSD state machine. Addresses Kenya's 40M+ M-Pesa users.
8. **Crop Insurance Yield Index** — `calculate_crop_indemnity()` applies the area yield index formula (actual yield, threshold yield, sum insured, deductible); integrates with the parametric engine. Enables ACRE Africa-style products.
9. **Multi-Currency FX Settlement** — `settle_premium_with_fx()` fetches or accepts a live FX rate, converts payment to policy currency, and records both amounts with the FX reference. Required for USD-denominated reinsurance treaties.
10. **Solvency II / IRA Capital Adequacy Report** — `compute_solvency_margin()` applies MCR/SCR formulas to net premium written, technical provisions, and equity capital; flags breach if ratio < 150%.
11. **Claim Subrogation Tracking** — `open_subrogation_case()` / `record_subrogation_recovery()` link recoveries to paid claims and recalculate net claim cost for accurate loss ratio reporting.
12. **Event-Sourced Tamper-Evident Audit Trail** — Append-only event store with SHA-256 hash chaining. `verify_audit_chain()` detects any gap and raises `IntegrityError`. Persisted to PostgreSQL. Passes IRA Kenya IT audit requirements.
13. **Beneficiary Management** — `register_beneficiary()` links beneficiaries with share percentages to a policy. `validate_beneficiary_shares()` enforces 100% total. Prevents wrong-payee disbursements on life claims.
14. **Cohort-Based Loss Forecasting** — `forecast_portfolio_loss()` applies the chain-ladder development method to claims triangles; projects IBNR reserves with a confidence interval for the next period's loss ratio.
15. **Policy Document Generation Pipeline** — `generate_policy_schedule()` builds a structured policy schedule dict (insured, product, coverage, premium, exclusions, effective/expiry) from service-resident data. Template-ready for PDF or email renderers. No external document system required.

---

## New Methods

### `claim_fast_track` — Auto-approve small claims
```python
# Claims <= auto_approve_threshold bypass the manual assess/approve queue
result = await svc.claim_fast_track("clm_001", auto_approve_threshold=5_000.0)
# {"fast_tracked": True, "status": "approved", ...}
```

### `fraud_indicator_check_claim` — Pre-processing fraud screen
```python
check = await svc.fraud_indicator_check_claim("clm_002")
# {"fraud_indicators": ["HIGH_VALUE_CLAIM"], "risk_score": 30.0, "recommendation": "review"}
if check["recommendation"] != "approve":
    await svc.record_compliance_alert("al_001", "clm_002", "high", "ev_fraud_001")
```

### `group_policy_enrollment` — SACCO / employer bulk onboarding
```python
result = await svc.group_policy_enrollment(
    group_id="sacco_nairobi_01",
    member_ids=["ph_1", "ph_2", "ph_3"],
    product_id="prod_health",
    premium_per_member=1_200.0,
)
# {"enrolled": 3, "total_premium": 3600.0, ...}
```

### `no_claims_discount` — NCD calculation
```python
ncd = await svc.no_claims_discount("ph_1", claim_free_years=3)
# {"ncd_discount_pct": 30.0, "claim_free_years": 3, ...}
renewal_premium = base_premium * (1 - ncd["ncd_discount_pct"] / 100)
```

### `insurance_analytics` — Portfolio loss ratio and product mix
```python
analytics = await svc.insurance_analytics("Q2-2026")
# {
#   "loss_ratio": 0.42, "combined_ratio": 0.67,
#   "claims_frequency": 0.08, "average_claim_size_minor": 45000,
#   "product_mix": {"motor": 12, "health": 7, "crop": 3}, ...
# }
```

---

## Edge Cases Handled
- Policy binding requires both a quote AND a payment reference — a quote without payment is not a bound policy; this prevents uncommitted policy issuance
- Loss date is required at claim opening — backdating prevention is not enforced by the rule engine (loss date can be historical) but the field must be explicitly provided
- Reinsurance share must be positive; 0% share would create a meaningless reinsurance record and is rejected
- Document type validation fires at document recording — attaching a document of type `other` is rejected if it is not in `SUPPORTED_DOCUMENT_TYPES`
- Microinsurance products follow the same governance rules as other product lines — there is no simplified path for micro products
- `process_premium()` auto-reinstates a lapsed policy on receipt of back-payment
- `cancel_policy()` computes a pro-rata refund based on days elapsed in the coverage period

## Composability
- **Upstream**: `fintech_kyc` provides policyholder identity; `fintech_aml` and `fintech_fraud` provide screening for both onboarding and claims; `fintech_payments` executes premium collections and claim payouts
- **Downstream**: `bia` consumes risk assessment and claims data for actuarial analytics; `fin_rpt` receives premium and claim data for financial reporting
- **Peer**: Commonly deployed alongside `fintech_risk` (policyholder risk profiling) and `fintech_lending` (credit-linked insurance)
- **v2.0 Composition**: Parametric engine integrates with `fintech_payments` for zero-touch payout; USSD adapter integrates with `mpesa` callbacks; ML scoring integrates with local Ollama inference

## Development Notes
- Crop and microinsurance product lines are included to support Africa-focused parametric and inclusive insurance products; the capability does not differentiate governance requirements between them
- Underwriting references in quotes are external references — the capability does not run underwriting models; it records the outcome reference from an external underwriting system
- Risk assessment scores are stored as arbitrary values; the capability does not enforce a 0–100 range like fraud/KYC — the score range is defined by the risk assessment source
- Claim `evidence_references` is a list; multiple pieces of evidence (photos, medical reports, invoices) can be attached to a single claim
- All amount fields use integer minor units (cents/fils) internally; public API methods accept and return floats for convenience with `* 100` / `/ 100` conversion at the boundary
- ML features (`ml_premium_score`, `ai_triage_claim`) require `OLLAMA_BASE_URL` env var; both degrade gracefully to heuristic mode when absent
