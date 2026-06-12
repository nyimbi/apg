# Digital Neobanking

## Overview
Digital Neobanking provides the core banking layer for digital-first banks: program governance, customer onboarding with full AML/KYC/fraud evidence chain, deposit account opening (current, savings, joint, business, youth, merchant), payment rail linking, transaction posting with risk reference, savings pot management, virtual cards, peer transfers, bill splitting, cashback, spending analytics, budget enforcement, overdraft protection, chargeback lifecycle, multi-currency FX transfers, structured consent, balance attestations, and customer service case handling. It is the account ledger that other capabilities — mobile, cards, lending, remittance — use as their underlying account infrastructure.

Transaction currency must match account currency; cross-currency moves go through `fx_convert_and_transfer`. High-impact transactions require human approval. Initial account balances cannot be negative. All neobanking events stream to `apg.fintech.neobanking.lifecycle` via Bytewax.

## Capability ID
`fintech_neobanking`  Version: 2.0.0

## Quick Start

```python
from capabilities.fintech.neobanking.service import NeobanksService

svc = NeobanksService(tenant_id="acme", actor_id="ops-agent")

# Register a bank program
await svc.register_program("prog-1", "Acme Bank", "owner-1", "KE", "KES", "settle-001")

# Onboard a customer
await svc.onboard_customer("cust-1", "CUST-REF-001", "kyc-profile-42",
                           "KE", "consent-001", "aml-001", "fraud-001")

# Open an account and apply a feature bundle
acct = await svc.open_account("cust-1", "savings", "KES", "prog-1")
await svc.account_features_bundle(acct["id"], "standard")

# Post a transaction
await svc.post_transaction("tx-1", acct["id"], "deposit", 50000.0,
                           "KES", "initial_deposit", "risk-ref-001")
```

## Provides
| Service | Description |
|---------|-------------|
| neobank_program_governance | Register bank programs with owner, settlement account, country, and currency |
| digital_customer_onboarding | Onboard customers with KYC, AML, fraud, and consent evidence |
| deposit_account_lifecycle | Open, close, freeze, and bundle-upgrade accounts |
| payment_rail_linking | Link bank transfer, card, wallet, mobile money, and internal transfer rails |
| account_transaction_posting | Post transactions with type, amount, currency match, and risk reference |
| virtual_card_management | Issue, freeze, unfreeze, and apply per-MCC controls to virtual cards |
| peer_transfers_and_fx | Peer transfers, bill splitting, and multi-currency FX conversion |
| savings_pot_workflow | Create, fund, and auto-sweep named savings goals within accounts |
| spending_analytics_and_budgets | Analytics, category budgets with 75%/100% alerts, and subscription tracking |
| cashback_calculation | Calculate and credit cashback based on feature bundle |
| overdraft_protection | Configure tiered overdraft limits with daily interest accrual |
| chargeback_lifecycle | Open disputes with provisional credit, resolve with upheld/rejected rulings |
| statement_workflow | Issue single or bulk account statements |
| consent_management | Record, revoke, and audit structured consent (Kenya DPA Article 30) |
| balance_attestation | Generate HMAC-signed proof-of-funds for third-party verification |
| customer_risk_scoring | Aggregate 0-100 risk scores from velocity, savings, overdraft, and freeze signals |
| account_webhooks | Register HMAC-signed event webhooks per account with delivery logging |
| customer_service_case_workflow | Open service cases with reason, reviewer, and evidence |
| neobanking_agent_workflow | Register AI agents for account risk, payments review, and customer service |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Customer and operations notifications |
| nlpc | NLP processing |
| keym | Key management (attestation signing) |
| fintech_payments | Payment rail execution |
| fintech_wallets | Wallet rail integration |
| fintech_cards | Card rail and debit card linkage |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud signal scoring |
| fintech_fx | Exchange rates for cross-currency transfers |
| fintech_lending | Loan account linkage |
| fintech_remittance | Cross-border remittance integration |

## World-Class Enhancements (v2.0)

1. **Real-Time Fraud Signal Integration** — `post_transaction` hooks into `fintech_fraud`; every response includes a `fraud_signal` field gated on velocity, geo-anomaly, and device-fingerprint thresholds.

2. **Multi-Currency FX Conversion** — `fx_convert_and_transfer()` applies mid-market rate + configurable spread, posts the FX fee as a separate transaction, and settles both legs atomically with `fx_rate`, `fx_fee`, and `original_currency` on the credit leg.

3. **Savings Pot Auto-Sweep Rules** — `savings_pot_autosweep_rule()` attaches end-of-day, percentage-of-balance, or after-credit triggers to pots; `execute_autosweep_rules()` fires them and emits `savings_autosweep_executed` events.

4. **Tiered Overdraft with Daily Interest Accrual** — `overdraft_interest_accrual()` computes overdrawn EOD balance, posts a `transfer_out` fee transaction daily (annual_rate/365 + flat daily fee), and tracks a running overdrawn balance ledger.

5. **Programmable Virtual Cards with Per-MCC Controls** — `virtual_card_update_controls()` sets per-MCC allow/block lists, per-country filters, and time-of-day windows enforced at transaction posting.

6. **Account-Level Event Webhooks** — `register_account_webhook()` stores URL + event filter + HMAC-SHA256 secret per account; includes delivery log and `replay_webhook()` for failed deliveries.

7. **Intelligent Spending Budgets** — `set_spending_budget()` records category limits; `spending_budget_check()` returns remaining, burn rate, projected over-budget date, and fires `budget_75pct_warning` / `budget_exceeded` notifications (once per month per threshold).

8. **Regulatory Reporting Pipeline** — `regulatory_report()` with pluggable CBK, RBA, and CMA jurisdiction templates; `submit_regulatory_report()` archives the report and records the acknowledgment reference.

9. **Idempotency Keys** — All mutating operations accept an idempotency key; duplicates return the cached response with a 24-hour TTL, preventing double-charges in mobile retry scenarios.

10. **Balance Attestation** — `generate_balance_attestation()` produces an HMAC-SHA256 signed JSON payload (tenant, account, balance, timestamp) valid until EOD; `verify_balance_attestation()` enables counterparty verification.

11. **Chargeback Workflow** — `open_chargeback()` issues provisional credit immediately and tracks dispute status through `merchant_response`, `arbitration`, and `final_ruling`; `resolve_chargeback()` reverses credit atomically on rejection.

12. **Customer Risk Score** — `compute_customer_risk_score()` aggregates velocity, overdraft utilisation, savings ratio, and freeze history into a 0-100 score (low/medium/high) that drives limit and alert thresholds.

13. **Bulk Statement Generation** — `bulk_issue_statements()` generates statements for all active accounts in a date range in one call; backgrounds jobs auto-queue when `item_count > 500`.

14. **Account Linking (Joint & Subsidiary)** — `link_accounts()` models parent/child and joint relationships; joint accounts require multi-party consent; freeze/close propagates to linked accounts.

15. **Structured Consent Lifecycle** — `record_consent()` captures type (`account_opening`, `data_sharing`, `marketing`, `overdraft`, `biometric`), channel, and evidence hash; `revoke_consent()` and `list_consent_history()` satisfy Kenya Data Protection Act Article 30.

## New Methods

### FX Transfer
```python
result = await svc.fx_convert_and_transfer(
    from_account="acct-kes", to_account="acct-usd",
    amount=100_000.0, from_currency="KES", to_currency="USD",
    fx_rate=130.5, fx_spread_pct=0.5, reference="school_fees_q1",
)
# result keys: transfer_id, original_amount, original_currency,
#              converted_amount, target_currency, effective_rate, fx_fee
```

### Savings Auto-Sweep
```python
# Attach a rule: sweep 10% of balance to pot every end-of-day
rule = await svc.savings_pot_autosweep_rule(
    account_id="acct-1", pot_id="pot-holiday",
    trigger="percentage_of_balance", value=10.0,
)
# Run all EOD rules for the tenant
summary = await svc.execute_autosweep_rules(trigger="end_of_day")
```

### Spending Budget + Check
```python
await svc.set_spending_budget("acct-1", category="card_purchase", monthly_limit=20_000.0)
status = await svc.spending_budget_check("acct-1", "card_purchase")
# status keys: monthly_limit, spent_this_month, remaining,
#              utilisation_pct, burn_rate_daily, days_until_over_budget
```

### Chargeback Lifecycle
```python
cb = await svc.open_chargeback(
    case_id="case-42", customer_id="cust-1", account_id="acct-1",
    disputed_transaction_id="tx-suspicious", reason="unauthorized",
)
# provisional credit posted immediately
ruling = await svc.resolve_chargeback("case-42", ruling="upheld")
```

### Risk Score + Balance Attestation
```python
score = await svc.compute_customer_risk_score("cust-1")
# score: {"risk_score": 22, "tier": "low", "signals": {...}}

attest = await svc.generate_balance_attestation("acct-1", purpose="proof_of_funds")
# attest: {"balance": 150000.0, "currency": "KES", "signature": "...", "expires_at": "..."}
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| accounts.supported_types | list | current, savings, joint, business, youth, merchant | Account types |
| transactions.supported_types | list | deposit, withdrawal, transfer_in, transfer_out, card_purchase, fee, refund, interest | Transaction types |
| transactions.high_value_threshold | number | 100000 | Amount requiring approval |
| rails.supported_rails | list | bank_transfer, card, wallet, mobile_money, internal_transfer | Payment rails |
| service_cases.supported_reasons | list | account_access, card_issue, payment_dispute, kyc_review, fraud_review, fee_query, statement_query | Case reasons |
| overdraft.default_rate_pa | number | 0.18 | Annual overdraft interest rate |
| consent.supported_types | list | account_opening, data_sharing, marketing, overdraft, biometric | Consent types |
| autosweep.supported_triggers | list | end_of_day, percentage_of_balance, after_credit | Autosweep triggers |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-neobanking/dashboard | GET | fintech_neobanking:view | Overview |
| programs | /fintech-neobanking/programs | GET/POST | fintech_neobanking:manage_programs | Programs |
| customers | /fintech-neobanking/customers | GET/POST | fintech_neobanking:manage_customers | Customers |
| accounts | /fintech-neobanking/accounts | GET/POST | fintech_neobanking:manage_accounts | Accounts |
| rails | /fintech-neobanking/rails | GET/POST | fintech_neobanking:manage_rails | Payments |
| transactions | /fintech-neobanking/transactions | GET/POST | fintech_neobanking:post_transactions | Payments |
| fx_transfer | /fintech-neobanking/fx-transfer | POST | fintech_neobanking:post_transactions | Payments |
| savings | /fintech-neobanking/savings | GET/POST | fintech_neobanking:savings | Accounts |
| autosweep | /fintech-neobanking/savings/autosweep | GET/POST | fintech_neobanking:savings | Accounts |
| budgets | /fintech-neobanking/budgets | GET/POST | fintech_neobanking:manage_accounts | Analytics |
| statements | /fintech-neobanking/statements | GET/POST | fintech_neobanking:statements | Servicing |
| statements_bulk | /fintech-neobanking/statements/bulk | POST | fintech_neobanking:statements | Servicing |
| cases | /fintech-neobanking/cases | GET/POST | fintech_neobanking:cases | Servicing |
| chargebacks | /fintech-neobanking/chargebacks | GET/POST | fintech_neobanking:cases | Servicing |
| consent | /fintech-neobanking/consent | GET/POST | fintech_neobanking:manage_customers | Customers |
| risk_scores | /fintech-neobanking/risk-scores | GET | fintech_neobanking:view | Analytics |
| attestations | /fintech-neobanking/attestations | POST | fintech_neobanking:manage_accounts | Accounts |
| webhooks | /fintech-neobanking/webhooks | GET/POST | fintech_neobanking:admin | Integration |
| overdraft | /fintech-neobanking/overdraft | GET/POST | fintech_neobanking:manage_accounts | Accounts |
| agents | /fintech-neobanking/agents | GET/POST | fintech_neobanking:admin | Automation |
| settings | /fintech-neobanking/settings | GET/POST | fintech_neobanking:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| program_settlement_required | Program without settlement account | deny |
| customer_consent_required | Customer without consent | deny |
| account_initial_balance_non_negative | Negative initial balance | deny |
| transaction_currency_matches_account | Transaction currency differs from account | deny |
| transaction_risk_reference_required | Transaction without risk reference | deny |
| high_impact_transaction_requires_approval | High-impact transaction without approval | require_review |
| savings_target_positive | Savings pot with zero or negative target | deny |
| statement_period_required | Statement without specified period | deny |
| case_reviewer_required | Service case without reviewer | deny |
| neobanking_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_neobanking_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |
| fx_transfer_requires_different_currencies | FX transfer with identical from/to currency | deny |
| fx_rate_must_be_positive | FX transfer with zero or negative rate | deny |
| chargeback_transaction_must_belong_to_account | Disputed transaction on wrong account | deny |
| consent_type_must_be_supported | Consent record with unknown type | deny |
| autosweep_trigger_must_be_supported | Auto-sweep rule with unknown trigger | deny |
| overdraft_interest_requires_active_overdraft | Accrual on account with no overdraft limit | skip |
| budget_category_must_be_unique_per_account | Duplicate category budget on same account | replace |

## Data Models
| Model | Key Fields |
|-------|-----------|
| BankProgram | id, name, owner_id, settlement_account, country, currency, status |
| DigitalCustomer | id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference, status |
| DepositAccount | id, program_id, customer_id, account_type, currency, balance, overdraft_limit, feature_bundle, status |
| PaymentRail | id, account_id, rail_type, provider_reference, wallet_or_card_reference, status |
| AccountTransaction | id, account_id, transaction_type, amount, currency, risk_reference, fx_rate, fx_fee, status |
| SavingsPot | id, account_id, name, target_amount, current_amount, status |
| AutosweepRule | id, account_id, pot_id, trigger, value, execution_count, last_executed_at |
| AccountStatement | id, account_id, period_start, period_end, transaction_count |
| ServiceCase | id, customer_id, account_id, reason, reviewer_id, evidence_references, status |
| Chargeback | id, customer_id, account_id, disputed_transaction_id, disputed_amount, provisional_credit_tx_id, status, ruling |
| SpendingBudget | id, account_id, category, monthly_limit, status |
| ConsentRecord | id, customer_id, consent_type, channel, evidence_hash, status, recorded_at, revoked_at |
| BalanceAttestation | id, account_id, balance, currency, purpose, signature, algorithm, issued_at, expires_at |
| AccountWebhook | id, account_id, url, event_filter, secret_hash, delivery_count, last_delivery_at |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| bank_program_registered | Program created |
| digital_customer_onboarded | Customer enrolled |
| deposit_account_opened | Account opened |
| payment_rail_linked | Rail linked to account |
| account_transaction_posted | Transaction posted |
| fx_transfer_completed | Cross-currency transfer settled |
| savings_pot_created | Savings pot created |
| savings_autosweep_executed | Auto-sweep rule fired |
| savings_goal_reached | Pot balance meets target |
| budget_75pct_warning | Spending reaches 75% of budget |
| budget_exceeded | Spending exceeds 100% of budget |
| statement_issued | Statement generated |
| service_case_opened | Service case opened |
| chargeback_opened | Dispute opened with provisional credit |
| chargeback_resolved | Dispute ruled upheld or rejected |
| consent_recorded | Consent event captured |
| consent_revoked | Consent withdrawn |
| overdraft_interest_accrued | Daily overdraft fee posted |
| neobanking_agent_registered | AI agent registered |
| virtual_card_issued | Virtual card created |
| peer_transfer_completed | Peer transfer settled |
| balance_attestation_generated | Signed attestation issued |
| customer_risk_score_computed | Risk score computed |

## Edge Cases Handled
- Transaction currency must match account currency exactly — a KES transaction against a USD account is denied; currency conversion goes through `fx_convert_and_transfer` which posts a separate FX fee transaction
- Initial account balance of zero is valid; negative initial balance is denied
- All transactions require a risk reference — this forces every transaction to be linked to a fraud or AML risk assessment
- Savings pots are sub-accounts within a deposit account; the pot target must be positive but the current amount can be zero (newly created pot)
- Service case reasons are validated against a fixed list; free-text reason entry is not supported
- Chargeback provisional credit is issued immediately on dispute opening; reversed atomically if the ruling is rejected
- Auto-sweep rules skip silently when account balance is insufficient — no error, no partial sweep
- Budget alerts fire at most once per month per threshold per category — the `_budget_alerts` guard prevents duplicate notifications
- Balance attestations are valid only until end-of-day of issuance; counterparties must re-request for the next day
- Consent records are append-only; revocation sets status to 'revoked' with timestamp rather than deleting the record
- Customer risk scores are advisory only — they do not gate transactions; callers may use scores to tighten limits independently
- FX spread is applied against the mid-market rate; the spread amount is posted as a separate fee transaction on the debit leg for audit transparency

## Composability
- **Upstream**: `fintech_kyc`, `fintech_aml`, and `fintech_fraud` provide the evidence chain required for customer onboarding; `fintech_payments` and `fintech_wallets` provide the payment rail execution; `fintech_fx` provides exchange rates for cross-currency transfers
- **Downstream**: `fintech_mobile` uses neobank accounts as the primary account backing for mobile payments; `fintech_cards` issues debit cards against neobank accounts; `fintech_lending` links loan accounts to neobank current accounts
- **Peer**: Deployed alongside `fintech_mobile` (channel layer) and `fintech_cards` (debit card issuance) in a full neobank stack

## Development Notes
- `internal_transfer` rail is for account-to-account transfers within the same neobank program; it does not require an external provider reference
- `merchant` account type supports business accounts for merchants using the neobank as their primary business account; distinct from the merchant accounts used in `fintech_gateway`
- Statement period is caller-specified — the rule engine only checks that a period is present; the service layer handles period validation (start < end, reasonable range)
- `high_impact` is a caller-computed flag; the service layer evaluates the transaction against the `high_value_threshold` and sets the flag before invoking the rule engine
- Autosweep rules, webhooks, budgets, chargebacks, and consent records are stored in lazily-initialised `dict` attributes on the service instance; they persist in-memory and must be wired to the `store` adapter for durability
- The `generate_balance_attestation` HMAC key is `tenant_id`; in production, wire `keym` to use a tenant-specific signing key
- `NeobanksService`, `DigitalNeobankingService`, and `NeobankingService` are all aliases for the same class
