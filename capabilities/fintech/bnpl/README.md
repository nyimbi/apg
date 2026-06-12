# Buy Now Pay Later

## Overview
Buy Now Pay Later manages the full lifecycle of deferred payment products for consumers and merchants: BNPL program governance, consumer and merchant onboarding, checkout session capture, affordability decisioning, repayment plan creation, installment scheduling, merchant settlement, and dispute handling. It enforces consumer protection through mandatory KYC, AML, fraud evidence, and explicit fee disclosure at every stage where a consumer commits to debt.

Affordability decisions are scored 0–1000 and require human approval before finalization. Declines must carry an adverse action reason. Fee disclosure and customer acceptance are prerequisites for plan creation. All lifecycle events stream to `apg.fintech.bnpl.lifecycle` via Bytewax.

## Capability ID
`fintech_bnpl`  Version: 2.0.0

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
| dynamic_risk_pricing | Real-time personalised APR based on credit score, plan type, and merchant category |
| adaptive_credit_reassessment | Periodic credit limit adjustment based on repayment history |
| instalment_restructuring | Hardship programme with term extension and fee waiver |
| fraud_velocity_controls | Rolling-window checkout fraud detection by merchant category |
| regulatory_exposure_check | Per-consumer multi-provider BNPL cap enforcement (CBK/BoU) |
| webhook_delivery | At-least-once event delivery with SLA tracking |
| consumer_behaviour_scoring | Thin-file scoring from purchase patterns (0–100) |
| bulk_settlement_reconciliation | Three-way reconciliation across all merchant plans |
| portfolio_stress_testing | IFRS 9-aligned ECL under macro shock scenarios |
| data_subject_request | GDPR/DPA 2019 erasure, portability, and restriction automation |
| ab_plan_offers | A/B testing framework for plan offer variants |
| delinquency_triage | RFM-scored collections queue with auto-escalation |
| merchant_widget_config | Signed SDK configuration payloads for embedded BNPL storefronts |

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

## Quick Start

```python
import asyncio
from capabilities.fintech.bnpl.service import BuyNowPayLaterService

svc = BuyNowPayLaterService(tenant_id="acme", actor_id="ops")

# Register a merchant program
svc.register_merchant_program(
    program_id="prog-ke-001", tenant_id="acme", name="Kenya BNPL",
    owner_id="ops", country="KE", currency="KES",
    settlement_policy_reference="pol-001",
    fee_disclosure_reference="fee-001",
    max_installments=12,
)

# End-to-end application (async)
async def run():
    result = await svc.apply_for_bnpl(
        customer_id="cust-001",
        merchant_id="merch-001",
        purchase_amount=15_000.00,
        plan_type="pay_in_4",
    )
    print(result["status"], result.get("repayment_schedule"))

asyncio.run(run())
```

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
| bnpl_application_approved | End-to-end application approved |
| credit_limit_reassessed | Consumer credit limit adjusted |
| instalment_processed | Single instalment payment recorded |
| early_repayment_processed | Full early repayment with discount applied |
| late_payment_handled | Late fee charged and statuses updated |
| merchant_settlement_run | Merchant batch settlement completed |
| debt_recovery_initiated | Debt recovery escalation initiated |

## New Methods

### `apply_for_bnpl` — end-to-end checkout in one call
Runs credit check, creates checkout session, records affordability decision, creates plan, and generates the repayment schedule atomically.

```python
result = await svc.apply_for_bnpl(
    customer_id="cust-007",
    merchant_id="merch-electronics-001",
    purchase_amount=45_000.00,
    plan_type="pay_in_4",        # pay_in_3 | pay_in_4 | pay_in_12 | pay_in_6
)
# result["status"] == "approved"
# result["repayment_schedule"] == [{"sequence": 1, "due_date": "...", "amount": ...}, ...]
```

### `generate_repayment_plan` — explicit schedule generation
Calculates equal instalments using the annuity formula and persists `InstallmentSchedule` records.

```python
schedule = await svc.generate_repayment_plan(bnpl_id="plan-001", instalments=4)
# schedule["total_repayable"], schedule["total_interest"], schedule["schedule"]
```

### `process_instalment` — mark payment received
Updates the in-memory schedule, creates a payment record, and reduces the consumer's outstanding balance.

```python
payment = await svc.process_instalment(bnpl_id="plan-001", instalment_number=2)
# payment["remaining_instalments"], payment["plan_status"]
```

### `late_payment_handling` — overdue fee + status update
Applies a daily 0.1% penalty (capped at 30%), marks affected instalments `overdue`, and fires a notification if the notify adapter is wired.

```python
result = await svc.late_payment_handling(bnpl_id="plan-001", days_overdue=14)
# result["late_fee_charged"], result["cumulative_late_fees"]
```

### `bnpl_analytics` — portfolio performance snapshot
Aggregates GMV, approval rate, overdue count, late fees, early repayments, and plan-type breakdown for a reporting period.

```python
report = await svc.bnpl_analytics(period="2026-Q2")
# report["total_volume"], report["approval_rate_pct"], report["by_plan_type"]
```

### `bulk_settlement_reconciliation` via `merchant_settlement`
Runs MDR deduction and produces a structured settlement record covering all plans for a merchant in a period.

```python
settlement = await svc.merchant_settlement(merchant_id="merch-001", period="2026-05")
# settlement["gross_amount"], settlement["mdr_fee"], settlement["net_amount"]
```

## World-Class Enhancements (v2.0)

1. **Dynamic Risk-Based Pricing** — real-time APR personalisation by credit score, plan type, and merchant category; tiered prime/near-prime/subprime bands (+12–18% revenue per approved plan).
2. **Adaptive Credit Limit Reassessment** — `reassess_credit_limit` reruns the scoring model on repayment history on a 30-day cron cadence; limits adjust up or down automatically.
3. **Instalment Restructuring / Hardship Programme** — `restructure_plan` extends term, recalculates instalments, waives late fees up to policy cap; reviewer approval required above principal threshold.
4. **Multi-Currency FX Settlement** — `cross_currency_settlement` fetches live FX rates, converts GMV, records gross (consumer currency) and net (settlement currency); enables pan-African merchant expansion.
5. **Webhooks & Event Delivery SLA Tracking** — `register_merchant_webhook` + `deliver_webhook_event` with at-least-once queue, exponential backoff, per-webhook latency, and `webhook_health` endpoint.
6. **Consumer Behavioural Scoring** — `consumer_behaviour_score` (0–100) from purchase frequency, basket size, days-to-pay, and channel diversity; 30% blend into credit score for thin-file approvals.
7. **Merchant Category-Aware Fraud Velocity Controls** — `check_fraud_velocity` with rolling 1h/24h windows and category anomaly detection returning `low/medium/high/block` risk level.
8. **Instalment Deferral (Skip-a-Payment)** — `defer_instalment` pushes one instalment to end of schedule with a policy-capped fee; max-deferrals-per-plan enforced.
9. **Bulk Settlement Reconciliation** — `bulk_settlement_reconciliation` aggregates all plans, runs three-way match (principal vs. collected vs. rail confirmation), and flags discrepancies.
10. **Portfolio-Level Stress Testing** — `stress_test_portfolio` applies `rate_hike_200bps`, `unemployment_spike_5pct`, or `fx_depreciation_20pct` scenarios and returns IFRS 9 Stage ECL estimates.
11. **Consent & Privacy Rights Automation** — `process_data_subject_request` handles erasure, portability, and restriction with PII pseudonymisation and 30-day DPA 2019 SLA tracking.
12. **A/B Testing Framework for Plan Offers** — `get_plan_offer` routes consumers to configured offer variants; `record_offer_conversion` closes the loop; `experiment_results` aggregates outcomes.
13. **Real-Time Delinquency Triage Queue** — `build_delinquency_queue` scores overdue instalments by RFM model and returns prioritised actions (`auto_retry`, `sms_nudge`, `agent_call`, `debt_recovery`).
14. **Regulatory Limit Engine** — `check_regulatory_exposure` validates consumer multi-provider BNPL exposure against CBK/BoU caps before approval; persists the exposure delta on success.
15. **Embedded BNPL Merchant Widget Config** — `generate_merchant_widget_config` returns a signed, tenant-scoped JSON payload with eligible plan types, localised copy, APR disclosures, and a rotating signing token.

## Edge Cases Handled
- Affordability score range is validated as 0–1000; scores outside this range are denied regardless of the decision label
- An approved affordability decision is required before a plan can be created — the `affordability_approved` flag must be set in context; a checkout alone is insufficient
- Fee disclosure must be attached to both the program AND the individual plan; program-level disclosure does not satisfy plan-level disclosure
- Customer acceptance (explicit consent to repayment terms) is required at plan creation, not at checkout; this is separate from the checkout-level consent
- Down payment validation checks that it does not exceed the plan principal; an overpaid down payment is denied
- Early repayment applies a 5% discount on outstanding interest; subsequent calls return `already_repaid`
- Late fees accumulate at 0.1%/day capped at 30% of the overdue amount; cumulative total is tracked per plan

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide consumer evidence; `fintech_fraud` provides checkout-level fraud scoring; `fintech_lending` provides the underlying credit product infrastructure
- **Downstream**: `fintech_payments` executes the actual payment captures and merchant settlements; `fintech_wallets` provides wallet-based funding for plans
- **Peer**: Commonly deployed alongside `fintech_mobile` (mobile checkout channel), `fintech_embedded` (BNPL as embedded product in partner apps), and `fintech_cards` (card-backed installment plans)

## Development Notes
- The `plan_affordability_approved` rule checks the `affordability_approved` boolean in context — callers must set this after verifying the linked affordability decision has status `approve`
- Merchant settlement holds (`held` status) require human approval before release, even for low-value settlements, when `approval_required: True` is set in context
- `SUPPORTED_PLAN_TYPES` (pay_in_3, pay_in_4, monthly_installments, invoice_split) are the only valid plan shapes; custom installment patterns require a new type
- Installment status transitions follow: scheduled → due → paid/missed/waived/overdue; the rule engine validates supported statuses but not transition order
- `BNPLService` is an alias for `BuyNowPayLaterService` for backwards compatibility
- ML eligibility scoring (`ml_bnpl_eligibility`) requires `OLLAMA_BASE_URL` env var; degrades gracefully to `{"ml_enhanced": false}` when unavailable
