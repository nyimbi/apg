# Digital Lending

## Overview
Digital Lending manages the complete credit lifecycle: loan product governance, borrower onboarding, credit application submission with affordability and bank statement evidence, underwriting decisioning with adverse-action tracking, loan offer issuance and acceptance, disbursement control with mandatory human approval, repayment scheduling, and collections case management. It enforces consumer protection at every stage — declines require adverse-action reasons, accepted offers require borrower acceptance evidence, and every disbursement requires human approval regardless of amount.

Underwriting decisions are scored 0–1000. Final approve/decline decisions require human approval. Applications above the high-amount threshold require credit committee review. Events stream to `apg.fintech.lending.lifecycle` via Bytewax.

## Capability ID
`fintech_lending`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| loan_product_governance | Register loan products with type, currency, term, rate, and repayment frequency |
| borrower_lifecycle | Onboard borrowers with KYC, income evidence, and consent |
| credit_application_workflow | Submit applications with affordability, bank statement, AML, fraud, and behavior evidence |
| underwriting_decisioning | Record scored underwriting decisions with evidence, adverse reasons, and approval gates |
| loan_offer_workflow | Issue offers with APR, term, expiry, and borrower acceptance evidence |
| disbursement_control | Record disbursements to payment, wallet, card, or bank destinations with approval |
| repayment_schedule_workflow | Schedule repayments with due amounts, dates, and frequency |
| collections_workflow | Open collection cases with reason, reviewer, and contact policy |
| lending_agent_workflow | Register AI agents for underwriting review, credit risk, and collections |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Borrower and operations notifications |
| nlpc | NLP for application and collections narrative |
| keym | Key management |
| fintech_payments | Disbursement execution |
| fintech_wallets | Wallet-based disbursement |
| fintech_cards | Card-based disbursement |
| fintech_kyc | Borrower identity verification |
| fintech_aml | AML screening for applications |
| fintech_fraud | Fraud scoring for applications |
| fintech_remittance | Remittance behavior evidence |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| products.min_term_days | number | 7 | Minimum loan term |
| products.max_term_days | number | 3650 | Maximum loan term (10 years) |
| products.max_nominal_rate | number | 0.75 | Maximum annual nominal rate (75%) |
| products.max_amount | number | 1000000 | Maximum loan amount |
| applications.high_amount_threshold | number | 100000 | Threshold requiring credit committee review |
| underwriting.min_score | number | 0 | Minimum underwriting score |
| underwriting.max_score | number | 1000 | Maximum underwriting score |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-lending/dashboard | GET | fintech_lending:view | Overview |
| products | /fintech-lending/products | GET/POST | fintech_lending:manage_products | Products |
| borrowers | /fintech-lending/borrowers | GET/POST | fintech_lending:manage_borrowers | Borrowers |
| applications | /fintech-lending/applications | GET/POST | fintech_lending:submit | Applications |
| underwriting | /fintech-lending/underwriting | GET/POST | fintech_lending:underwrite | Risk |
| offers | /fintech-lending/offers | GET/POST | fintech_lending:offer | Offers |
| disbursements | /fintech-lending/disbursements | GET/POST | fintech_lending:disburse | Funding |
| repayments | /fintech-lending/repayments | GET/POST | fintech_lending:repayments | Servicing |
| collections | /fintech-lending/collections | GET/POST | fintech_lending:collections | Servicing |
| agents | /fintech-lending/agents | GET/POST | fintech_lending:admin | Automation |
| settings | /fintech-lending/settings | GET/POST | fintech_lending:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| loan_product_rate_valid | Rate exceeds max_nominal_rate | deny |
| borrower_income_evidence_required | Borrower without income evidence | deny |
| application_affordability_required | Application without affordability evidence | deny |
| application_bank_statement_required | Application without bank statement evidence | deny |
| application_remittance_or_card_evidence_required | No behavior evidence (remittance or card) | require_review |
| high_amount_application_requires_review | Amount > 100,000 without committee review | require_review |
| underwriting_adverse_reason_required | Decline without adverse-action reason | deny |
| underwriting_final_decision_requires_approval | Final approve/decline without approval | require_review |
| offer_acceptance_required | Accepted offer without borrower acceptance evidence | deny |
| disbursement_requires_human_approval | Disbursement without approval | require_review |
| collection_contact_policy_required | Collections case without contact policy | deny |
| lending_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| LoanProduct | id, name, owner_id, product_type, currency, min_amount, max_amount, min_term_days, max_term_days, nominal_rate, repayment_frequency, status |
| Borrower | id, customer_reference, kyc_profile_id, country, income_evidence_reference, consent_reference, status |
| LoanApplication | id, borrower_id, product_id, amount, currency, purpose, affordability_reference, bank_statement_reference, aml_reference, fraud_reference, status |
| UnderwritingDecision | id, application_id, score, decision, evidence_references, adverse_reason, reviewer_id, human_approval_reference |
| LoanOffer | id, application_id, underwriting_id, amount, apr, term_days, expiry_date, status, borrower_acceptance_reference |
| LoanDisbursement | id, offer_id, funding_account_reference, rail, destination_reference, amount, human_approval_reference, status |
| RepaymentSchedule | id, offer_id, due_amount, due_date, frequency, status |
| CollectionCase | id, borrower_id, reason, reviewer_id, contact_policy_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| loan_product_registered | Product created |
| borrower_onboarded | Borrower enrolled |
| loan_application_submitted | Application submitted |
| underwriting_recorded | Underwriting decision recorded |
| loan_offer_issued | Offer issued to borrower |
| loan_disbursement_recorded | Disbursement executed |
| repayment_schedule_created | Repayment schedule created |
| collection_case_opened | Collections case opened |
| lending_agent_registered | AI agent registered |

## Edge Cases Handled
- Every disbursement requires human approval regardless of amount — there is no small-loan disbursement exemption; this prevents autonomous fund release
- Borrower acceptance evidence is required only on accepted offers (when `accepted_offer: True` in context); offers in `issued`, `expired`, or `withdrawn` status do not require acceptance evidence
- The `application_remittance_or_card_evidence_required` rule is a `require_review` (not deny) — applications without behavior evidence can proceed but are routed to credit committee review
- Adverse-action reasons are required on declines but not on counteroffers or referrals; the rule fires only when `adverse_decision: True` in context
- Loan product rate validation checks against `max_nominal_rate` (0.75 = 75% annual); rates above this are denied at product registration, preventing usurious products from being published

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide borrower identity and AML evidence; `fintech_fraud` provides application-level fraud scoring; `fintech_remittance` provides cross-border transaction behavior used in credit scoring
- **Downstream**: `fintech_payments`, `fintech_wallets`, and `fintech_cards` provide the disbursement rails; `fintech_agency` uses lending as a disbursement channel at agent outlets; `fintech_bnpl` builds on lending infrastructure for BNPL credit
- **Peer**: Deployed alongside `fintech_neobanking` (account-based lending) and `fintech_mobile` (mobile loan applications)

## Development Notes
- `SUPPORTED_PRODUCT_TYPES` includes `salary_advance` and `merchant_cash_advance` for embedded payroll and merchant lending use cases
- Underwriting score range 0–1000 follows the same convention as BNPL affordability scoring; 0 is highest risk, 1000 is lowest risk (creditworthy)
- Contact policy in collections cases is a reference to a documented outreach policy — the capability does not validate the policy content, only its presence
- `fintech_remittance` is a required adapter specifically to access remittance transaction history as behavioral credit evidence; it is not for disbursement
