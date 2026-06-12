# Digital Lending

## Overview
Digital Lending manages the complete credit lifecycle: loan product governance, borrower onboarding, credit application submission with affordability and bank statement evidence, underwriting decisioning with adverse-action tracking, loan offer issuance and acceptance, disbursement control with mandatory human approval, repayment scheduling, collections case management, and portfolio analytics. It enforces consumer protection at every stage — declines require adverse-action reasons, accepted offers require borrower acceptance evidence, and every disbursement requires human approval regardless of amount.

Underwriting decisions are scored 0–1000. Final approve/decline decisions require human approval. Applications above the high-amount threshold require credit committee review. Events stream to `apg.fintech.lending.lifecycle` via Bytewax.

v2.0 adds AI-enhanced credit scoring, dynamic risk-based pricing, open banking integration, fraud ring detection, embedded insurance, direct debit mandates, regulatory reporting, USSD channel support, covenant monitoring, loan securitisation, and a conversational servicing agent.

## Capability ID
`fintech_lending`  Version: 2.0.0

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
| credit_assessment | Composite credit score (300–850), bureau check, income verification, DSR, collateral assessment |
| loan_management | Disburse loans, process repayments, restructure, fee management, early settlement, statements |
| delinquency_collections | DPD calculation, demand notices, collector assignment, legal action, write-off |
| portfolio_analytics | Portfolio summary, PAR ratios, IFRS 9 ECL provisioning, vintage analysis |
| alternative_data_scoring | M-Pesa, telco, and e-commerce signals for thin-file borrowers |
| dynamic_pricing | Risk-based pricing using grade, PD, liquidity premium, concentration charge |
| fraud_ring_detection | Graph-based entity network analysis for coordinated fraud |
| open_banking | PSD2/CGAP-aligned statement pull and parsing |
| embedded_insurance | Credit life and involuntary unemployment cover binding and claims |
| ussd_channel | Stateful USSD session handler for feature-phone loan applications |
| direct_debit | Standing order and M-Pesa mandate setup, scheduled debit sweeps |
| tranche_disbursement | Escrow-held conditional and tranche-based disbursements |
| regulatory_reporting | CBK, SARB, BoU, Basel III, and IFRS 9 report generation |
| guarantor_support | Guarantor and co-borrower model for group and secured lending |
| early_warning_system | 30–60 day leading-indicator EWS scoring (green/amber/red) |
| product_marketplace | Personalised product recommendations ranked by TCC and eligibility |
| loan_securitisation | Loan pool structuring, SPV waterfall, investor reporting |
| conversational_agent | Ollama-hosted LLM servicing agent for WhatsApp, SMS, and in-app queries |

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

## World-Class Enhancements (v2.0)

| # | Enhancement | Method | Impact |
|---|-------------|--------|--------|
| 1 | **Real-Time Alternative Data Scoring** — M-Pesa statements, telco, e-commerce signals for thin-file borrowers | `async credit_score_alternative_data(customer_id, data_sources)` | 15–25% reduction in false-decline rate |
| 2 | **Dynamic Risk-Based Pricing** — base_rate + credit_spread(PD) + liquidity_premium + concentration_charge | `async dynamic_price_loan(application_id, portfolio_context)` | 30–80bp NIM improvement |
| 3 | **Automated Covenant Monitoring** — scheduled checks for SME revenue, ratio, and balance covenants | `async check_covenant_compliance(loan_id)` | Proactive Stage 2 prevention |
| 4 | **Open Banking Statement Pull** — Stitch/Mono/OnePipe adapters; parsed cashflow into income_verification | `async pull_open_banking_statement(customer_id, bank_id, consent_token)` | Income confidence 0.70 → 0.90+ |
| 5 | **Embedded Insurance** — credit life and involuntary unemployment cover; Britam/Jubilee adapters | `async bind_credit_insurance(loan_id, cover_type, insurer_id, premium_rate)` / `async file_insurance_claim(...)` | Reduces NPL from unforeseeable events |
| 6 | **Fraud Ring Detection via Graph Analytics** — entity graph (phone, ID, device, guarantor) with Louvain community detection | `async detect_fraud_ring(application_id)` | 40–60% portfolio fraud loss reduction |
| 7 | **Multi-Leg Tranche Disbursement with Escrow** — conditional release on inspection/invoice; construction/school fee loans | `async disburse_in_tranches(loan_id, tranches)` | Unlocks SME and asset-finance categories |
| 8 | **Regulatory Reporting** — CBK, SARB BA, BoU, Basel III capital adequacy, IFRS 9; JSON and PDF output | `async generate_regulatory_report(report_type, period, regulator)` | Eliminates 2–3 FTE manual reporting process |
| 9 | **Guarantor and Co-Borrower Support** — joint income DSR, group/Grameen lending, guarantor call trigger | `async add_guarantor(loan_id, guarantor_id, guarantee_amount)` | LGD reduced from 40% to 20–25% on covered loans |
| 10 | **Predictive Early Warning System** — payment timing drift, mobile money inflow decline, 30–60 day advance warning | `async compute_early_warning_score(loan_id)` | 20–30% reduction in Stage 2 transition rates |
| 11 | **Loan Product Marketplace** — TCC-ranked personalised recommendations with APR, fee schedule, eligibility reasons | `async get_personalised_product_recommendations(customer_id, requested_amount, purpose)` | Reduces abandonment; meets CBK Consumer Protection 2023 |
| 12 | **USSD / SMS Channel** — stateful AfricasTalking/BICS session handler for feature-phone applications and servicing | `async process_ussd_session(session_id, msisdn, input_text, step)` | Reaches 80% of African mobile subscribers vs. 30–40% smartphone |
| 13 | **Automated Direct Debit / Standing Order** — ACH/RTGS and M-Pesa Paybill mandate with scheduled sweep | `async setup_direct_debit_mandate(loan_id, account_number, bank_code, mandate_type)` / `async execute_scheduled_debits(as_of_date)` | 60–80% collections cost reduction on performing book |
| 14 | **Loan Securitisation and Investor Reporting** — SPV pool structuring, senior/mezz/equity waterfall, monthly investor pack | `async create_loan_pool(pool_id, loan_ids, pool_type)` / `async generate_investor_report(pool_id, period)` | 200–400bp cost-of-capital reduction; required at >$50M AUM |
| 15 | **Conversational Loan Servicing Agent** — Ollama Llama 3.1/Mistral + RAG over loan statement; WhatsApp, SMS, in-app | `async handle_borrower_query(loan_id, query_text, channel)` | 40–60% inbound call reduction; 3x PTP conversion vs. SMS-only |

## New Methods

### ML-Enhanced Credit Assessment

```python
svc = LendingService()
result = await svc.ml_credit_score_assess("cust_001")
# Returns composite score 300-850, risk_grade A-F, probability_of_default,
# plus ml_recommendation (approve/decline/refer) and ml_confidence when
# OLLAMA_BASE_URL is set. Falls back gracefully without Ollama.
print(result["risk_grade"], result["probability_of_default"])
# "B"  0.042
```

### Debt Service Ratio Check

```python
# After income_verification has been run for the customer:
dsr = svc.debt_service_ratio(
    customer_id="cust_001",
    new_loan_amount=500_000,
    new_loan_rate=0.18,
    tenor_months=24,
)
# dsr["passes"] is True when total_emi / net_monthly_income <= 0.40
print(dsr["dsr"], dsr["passes"])
# 0.3712  True
```

### Loan Eligibility and Tiered Offer Generation

```python
svc.credit_score_calculate("cust_001")          # populates credit_scores
svc.income_verification("cust_001", "employed", 120_000, ["payslip_jan.pdf"])
eligibility = svc.calculate_loan_eligibility("cust_001", "prod_personal_loan")
# {"eligible": True, "credit_grade": "B", "max_amount": 840000.0, ...}

offers = svc.generate_loan_offers("app_001")
# Returns conservative (60%), standard (80%), and aggressive (100%) tiers
# each with amount, annual_rate, tenor_months, monthly_emi, total_cost
```

### IFRS 9 ECL Provisioning

```python
provision = svc.provision_calculation(method="ifrs9")
# Stage 1 (DPD=0): 12-month ECL using PD * LGD * EAD
# Stage 2 (DPD 1-90): lifetime ECL
# Stage 3 (DPD>90): lifetime ECL + full impairment
print(provision["total_ecl"], provision["coverage_ratio"])
```

### Loan Restructure

```python
result = svc.restructure_loan(
    loan_id="loan_001",
    new_terms={"annual_rate": 0.14, "tenor_months": 36, "capitalise_arrears": True},
    reason="borrower_hardship",
    approved_by="credit_committee_001",
)
# Rebuilds reducing-balance schedule from today; appends to restructure_history
print(result["new_monthly_emi"], result["restructure_history_count"])
```

## Development Notes
- `SUPPORTED_PRODUCT_TYPES` includes `salary_advance` and `merchant_cash_advance` for embedded payroll and merchant lending use cases
- Underwriting score range 0–1000 follows the same convention as BNPL affordability scoring; 0 is highest risk, 1000 is lowest risk (creditworthy)
- Contact policy in collections cases is a reference to a documented outreach policy — the capability does not validate the policy content, only its presence
- `fintech_remittance` is a required adapter specifically to access remittance transaction history as behavioral credit evidence; it is not for disbursement
- Credit score range for `credit_score_calculate` is 300–850 (three pillars: behavioural 45%, demographic 20%, bureau 35%); `ml_credit_score_assess` wraps this with Ollama classification
- v2.0 async methods (`credit_score_alternative_data`, `dynamic_price_loan`, `detect_fraud_ring`, etc.) require adapter implementations for production data sources; stubs are defined in the service contract
