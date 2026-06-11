# Digital Neobanking

## Overview
Digital Neobanking provides the core banking layer for digital-first banks: program governance, customer onboarding with full AML/KYC/fraud evidence chain, deposit account opening (current, savings, joint, business, youth, merchant), payment rail linking, transaction posting with risk reference, savings pot management, account statement generation, and customer service case handling. It is the account ledger that other capabilities — mobile, cards, lending, remittance — use as their underlying account infrastructure.

Transaction currency must match account currency. High-impact transactions require human approval. Initial account balances cannot be negative. All neobanking events stream to `apg.fintech.neobanking.lifecycle` via Bytewax.

## Capability ID
`fintech_neobanking`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| neobank_program_governance | Register bank programs with owner, settlement account, country, and currency |
| digital_customer_onboarding | Onboard customers with KYC, AML, fraud, and consent evidence |
| deposit_account_lifecycle | Open accounts with type, currency, program, and customer linkage |
| payment_rail_linking | Link bank transfer, card, wallet, mobile money, and internal transfer rails |
| account_transaction_posting | Post transactions with type, amount, currency match, and risk reference |
| savings_pot_workflow | Create and track named savings goals within accounts |
| statement_workflow | Issue account statements with period specification |
| customer_service_case_workflow | Open service cases with reason, reviewer, and evidence |
| neobanking_agent_workflow | Register AI agents for account risk, payments review, and customer service |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Customer and operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Payment rail execution |
| fintech_wallets | Wallet rail integration |
| fintech_cards | Card rail and debit card linkage |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud signal scoring |
| fintech_lending | Loan account linkage |
| fintech_remittance | Cross-border remittance integration |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| accounts.supported_types | list | current, savings, joint, business, youth, merchant | Account types |
| transactions.supported_types | list | deposit, withdrawal, transfer_in, transfer_out, card_purchase, fee, refund, interest | Transaction types |
| transactions.high_value_threshold | number | 100000 | Amount requiring approval |
| rails.supported_rails | list | bank_transfer, card, wallet, mobile_money, internal_transfer | Payment rails |
| service_cases.supported_reasons | list | account_access, card_issue, payment_dispute, kyc_review, fraud_review, fee_query, statement_query | Case reasons |

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
| DepositAccount | id, program_id, customer_id, account_type, currency, balance, status |
| PaymentRail | id, account_id, rail_type, provider_reference, wallet_or_card_reference, status |
| AccountTransaction | id, account_id, transaction_type, amount, currency, risk_reference, status |
| SavingsPot | id, account_id, name, target_amount, current_amount, status |
| AutosweepRule | id, account_id, pot_id, trigger, value, execution_count, last_executed_at |
| AccountStatement | id, account_id, period_start, period_end, transaction_count |
| ServiceCase | id, customer_id, account_id, reason, reviewer_id, evidence_references, status |
| Chargeback | id, customer_id, account_id, disputed_transaction_id, disputed_amount, status, ruling |
| SpendingBudget | id, account_id, category, monthly_limit, status |
| ConsentRecord | id, customer_id, consent_type, channel, evidence_hash, status, recorded_at |
| BalanceAttestation | id, account_id, balance, currency, purpose, signature, issued_at, expires_at |
| AccountWebhook | id, account_id, url, event_filter, delivery_count, last_delivery_at |

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
| budget_75pct_warning | Spending reaches 75 % of budget |
| budget_exceeded | Spending exceeds 100 % of budget |
| statement_issued | Statement generated |
| service_case_opened | Service case opened |
| chargeback_opened | Dispute opened with provisional credit |
| chargeback_resolved | Dispute ruled upheld or rejected |
| consent_recorded | Consent event captured |
| consent_revoked | Consent withdrawn |
| overdraft_interest_accrued | Daily overdraft fee posted |
| neobanking_agent_registered | AI agent registered |

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
- The `generate_balance_attestation` HMAC key is the `tenant_id`; in production, wire `keym` to use a tenant-specific signing key
