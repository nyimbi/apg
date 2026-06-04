# Buy Now Pay Later

## Overview
Buy Now Pay Later manages the lifecycle of deferred payment products for consumers and merchants: BNPL program governance, consumer and merchant onboarding, checkout session capture, affordability decisioning, repayment plan creation, installment scheduling, merchant settlement, and dispute handling. It enforces consumer protection through mandatory KYC, AML, fraud evidence, and explicit fee disclosure at every stage where a consumer commits to debt.

Affordability decisions are scored 0–1000 and require human approval before finalization. Declines must carry an adverse action reason. Fee disclosure and customer acceptance are prerequisites for plan creation. All lifecycle events stream to `apg.fintech.bnpl.lifecycle` via Bytewax.

## Capability ID
`fintech_bnpl`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| bnpl_merchant_program_governance | Register BNPL programs with country, currency, settlement policy, and fee disclosure |
| consumer_bnpl_lifecycle | Onboard consumers with KYC, AML, fraud, and consent evidence |
| merchant_checkout_workflow | Capture checkout sessions across web, mobile, POS, marketplace, and API channels |
| affordability_decisioning | Score and record affordability decisions with evidence and adverse-reason requirements |
| bnpl_plan_workflow | Create BNPL plans (pay-in-3, pay-in-4, monthly installments, invoice split) with fee disclosure |
| installment_schedule_workflow | Schedule and track individual installments with due dates and status |
| merchant_settlement_workflow | Record merchant settlements with reconciliation and payment rail evidence |
| bnpl_dispute_workflow | Handle consumer disputes with evidence and reviewer assignment |
| bnpl_agent_workflow | Register AI agents for affordability, merchant risk, and compliance review |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Consumer and merchant notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Payment execution for checkout |
| fintech_wallets | Wallet funding for plans |
| fintech_cards | Card-based checkout |
| fintech_kyc | Consumer identity verification |
| fintech_aml | AML screening for consumers and checkouts |
| fintech_fraud | Fraud scoring for checkout sessions |
| fintech_lending | Underlying credit product infrastructure |
| fintech_neobanking | Account-based funding |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| programs.max_installment_count | number | 24 | Maximum plan installments |
| affordability.score_min | number | 0 | Minimum affordability score |
| affordability.score_max | number | 1000 | Maximum affordability score |
| checkout.high_value_threshold | number | 100000 | Checkout requiring human review |
| settlements.high_value_threshold | number | 100000 | Settlement requiring approval |
| plans.min_term_days | number | 1 | Minimum plan duration |
| plans.max_term_days | number | 730 | Maximum plan duration (2 years) |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-bnpl/dashboard | GET | fintech_bnpl:view | Overview |
| programs | /fintech-bnpl/programs | GET/POST | fintech_bnpl:manage_programs | Programs |
| consumers | /fintech-bnpl/consumers | GET/POST | fintech_bnpl:manage_consumers | Consumers |
| merchants | /fintech-bnpl/merchants | GET/POST | fintech_bnpl:manage_merchants | Merchants |
| checkouts | /fintech-bnpl/checkouts | GET/POST | fintech_bnpl:manage_checkouts | Checkout |
| affordability | /fintech-bnpl/affordability | GET/POST | fintech_bnpl:decisioning | Risk |
| plans | /fintech-bnpl/plans | GET/POST | fintech_bnpl:plans | Plans |
| installments | /fintech-bnpl/installments | GET/POST | fintech_bnpl:installments | Plans |
| settlements | /fintech-bnpl/settlements | GET/POST | fintech_bnpl:settlements | Settlement |
| disputes | /fintech-bnpl/disputes | GET/POST | fintech_bnpl:disputes | Servicing |
| agents | /fintech-bnpl/agents | GET/POST | fintech_bnpl:admin | Automation |
| settings | /fintech-bnpl/settings | GET/POST | fintech_bnpl:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| program_fee_disclosure_required | Program without fee disclosure | deny |
| consumer_kyc_required | Consumer without KYC evidence | deny |
| checkout_fraud_required | Checkout without fraud evidence | deny |
| checkout_aml_required | Checkout without AML evidence | deny |
| high_value_checkout_requires_review | Checkout > 100,000 without review | require_review |
| affordability_score_in_range | Score outside 0–1000 range | deny |
| declined_affordability_requires_adverse_reason | Decline without adverse reason | deny |
| final_affordability_requires_approval | Final decision without human approval | require_review |
| plan_fee_disclosure_required | Plan without fee disclosure | deny |
| plan_customer_acceptance_required | Plan without customer acceptance | deny |
| settlement_hold_or_high_value_release_requires_approval | Hold or high-value settlement without approval | require_review |
| bnpl_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| MerchantProgram | id, tenant_id, name, owner_id, country, currency, settlement_policy_reference, fee_disclosure_reference, max_installments, status |
| BNPLConsumer | id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference, status |
| MerchantProfile | id, program_id, legal_entity_reference, category, country, risk_tier, settlement_account, status |
| CheckoutSession | id, merchant_id, consumer_id, channel, category, amount, currency, payment_reference, fraud_reference, aml_reference, status |
| AffordabilityDecision | id, checkout_id, score, decision, evidence_references, adverse_reason, reviewer_id, human_approval_reference |
| BNPLPlan | id, checkout_id, affordability_id, plan_type, principal, currency, term_days, fee_disclosure_reference, customer_acceptance_reference |
| Installment | id, plan_id, due_amount, due_date, status |
| MerchantSettlement | id, merchant_id, plan_id, status, amounts, reconciliation_reference, payment_rail_reference |
| BNPLDispute | id, plan_id, reason, evidence_references, reviewer_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| bnpl_program_registered | New BNPL program created |
| bnpl_consumer_onboarded | Consumer enrolled |
| bnpl_merchant_registered | Merchant registered to program |
| checkout_session_created | Checkout session opened |
| affordability_decision_recorded | Affordability score and decision recorded |
| bnpl_plan_created | BNPL repayment plan created |
| installment_scheduled | Individual installment scheduled |
| merchant_settlement_recorded | Merchant settlement processed |
| bnpl_dispute_opened | Consumer dispute filed |
| bnpl_agent_registered | AI agent registered |

## Edge Cases Handled
- Affordability score range is validated as 0–1000; scores outside this range are denied regardless of the decision label
- An approved affordability decision is required before a plan can be created — the `affordability_approved` flag must be set in context; a checkout alone is insufficient
- Fee disclosure must be attached to both the program AND the individual plan; program-level disclosure does not satisfy plan-level disclosure
- Customer acceptance (explicit consent to repayment terms) is required at plan creation, not at checkout; this is separate from the checkout-level consent
- Down payment validation checks that it does not exceed the plan principal; an overpaid down payment is denied

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide consumer evidence; `fintech_fraud` provides checkout-level fraud scoring; `fintech_lending` provides the underlying credit product infrastructure
- **Downstream**: `fintech_payments` executes the actual payment captures and merchant settlements; `fintech_wallets` provides wallet-based funding for plans
- **Peer**: Commonly deployed alongside `fintech_mobile` (mobile checkout channel), `fintech_embedded` (BNPL as embedded product in partner apps), and `fintech_cards` (card-backed installment plans)

## Development Notes
- The `plan_affordability_approved` rule checks the `affordability_approved` boolean in context — callers must set this after verifying the linked affordability decision has status `approve`
- Merchant settlement holds (`held` status) require human approval before release, even for low-value settlements, when `approval_required: True` is set in context
- `SUPPORTED_PLAN_TYPES` (pay_in_3, pay_in_4, monthly_installments, invoice_split) are the only valid plan shapes; custom installment patterns require a new type
- Installment status transitions follow: scheduled → due → paid/missed/waived; the rule engine validates supported statuses but not transition order
