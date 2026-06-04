# InsurTech

## Overview
InsurTech manages the end-to-end lifecycle of insurance operations: policyholder onboarding, product publishing across life, health, property, motor, travel, crop, and microinsurance lines, quote generation with underwriting evidence, policy binding, premium recording, claim intake, document management, risk assessment, reinsurance attachment, compliance alerts, and governance reviews. It is designed for regulated insurance operations where every quote must have an underwriting reference and every claim must have supporting evidence.

Policy binding requires a payment reference — premiums are not deferred. Claim amounts must be positive and loss dates must be recorded. Reinsurance attachments require treaty references. All insurance lifecycle events stream to `apg.fintech.insurance.lifecycle` via Bytewax.

## Capability ID
`fintech_insurance`  Version: 1.1.0

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
| fintech_payments | Premium payment execution |
| fintech_wallets | Wallet-based premium funding |
| fintech_kyc | Policyholder identity verification |
| fintech_aml | AML screening for policyholders |
| fintech_fraud | Fraud screening for claims |
| bia | Business intelligence and analytics |
| fin_rpt | Financial reporting |

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
| claim_opened | Claim filed |
| document_recorded | Document attached |
| risk_assessment_recorded | Risk assessment recorded |
| reinsurance_attachment_recorded | Reinsurance attachment recorded |
| insurance_compliance_alert_recorded | Compliance alert raised |
| insurance_review_recorded | Review completed |
| insurance_agent_registered | AI agent registered |

## Edge Cases Handled
- Policy binding requires both a quote AND a payment reference — a quote without payment is not a bound policy; this prevents uncommitted policy issuance
- Loss date is required at claim opening — backdating prevention is not enforced by the rule engine (loss date can be historical) but the field must be explicitly provided
- Reinsurance share must be positive; 0% share would create a meaningless reinsurance record and is rejected
- Document type validation fires at document recording — attaching a document of type `other` is rejected if it is not in `SUPPORTED_DOCUMENT_TYPES`
- Microinsurance products follow the same governance rules as other product lines — there is no simplified path for micro products

## Composability
- **Upstream**: `fintech_kyc` provides policyholder identity; `fintech_aml` and `fintech_fraud` provide screening for both onboarding and claims; `fintech_payments` executes premium collections and claim payouts
- **Downstream**: `bia` consumes risk assessment and claims data for actuarial analytics; `fin_rpt` receives premium and claim data for financial reporting
- **Peer**: Commonly deployed alongside `fintech_risk` (policyholder risk profiling) and `fintech_lending` (credit-linked insurance)

## Development Notes
- Crop and microinsurance product lines are included to support Africa-focused parametric and inclusive insurance products; the capability does not differentiate governance requirements between them
- Underwriting references in quotes are external references — the capability does not run underwriting models; it records the outcome reference from an external underwriting system
- Risk assessment scores are stored as arbitrary values; the capability does not enforce a 0–100 range like fraud/KYC — the score range is defined by the risk assessment source
- Claim `evidence_references` is a list; multiple pieces of evidence (photos, medical reports, invoices) can be attached to a single claim
