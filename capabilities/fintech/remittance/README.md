# Cross-Border Remittance

## Overview
Cross-Border Remittance manages the lifecycle of international money transfers: corridor and currency eligibility checks, FX quote creation with rate and fee locking, transfer creation with dual-side KYC and source-of-funds evidence, AML screening with sanctions blocking, fraud decisioning, payout release with provider receipt, and refund handling. Same-country transfers are architecturally blocked — the capability is strictly cross-border.

Transfers require a quote lock, both sender and beneficiary KYC, AML screen, source-of-funds evidence, and a supported fraud decision. Sanctions hits are a hard deny with no override path. High-value transfers and AML/fraud review outcomes require human approval. Events stream to `apg.fintech.remittance.lifecycle` via Bytewax.

## Capability ID
`fintech_remittance`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| remittance_corridor_governance | Define and govern supported send/receive country corridors |
| remittance_quote_lifecycle | Create FX quotes with amount, rate, fee, expiry, and quote lock |
| cross_border_transfer_workflow | Create transfers with sender/beneficiary KYC, AML, fraud, funding, purpose, and source-of-funds |
| remittance_payout_workflow | Release payouts with provider receipt and settlement reference |
| remittance_refund_workflow | File refunds with reason and reviewer assignment |
| remittance_agent_workflow | Register AI agents for compliance review, payout, and treasury roles |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Sender and beneficiary notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Funding and payment execution |
| fintech_wallets | Wallet-based funding and payout rails |
| fintech_kyc | Sender and beneficiary identity verification |
| fintech_aml | Sanctions, PEP, and AML screening |
| fintech_fraud | Fraud risk scoring |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| corridors.supported_countries | list | KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE, IN | Supported send/receive countries |
| corridors.same_country_blocked | bool | true | Same-country transfers blocked |
| quotes.supported_currencies | list | KES, UGX, TZS, RWF, GHS, NGN, ZAR, GBP, USD, AED, INR, EUR | Supported currencies |
| transfers.high_value_threshold | number | 100000 | Amount requiring approval |
| payouts.supported_methods | list | bank_account, mobile_money, wallet, cash_pickup, card_push | Payout delivery methods |
| transfers.supported_purpose_codes | list | family_support, education, medical, trade, salary, savings, emergency | Transfer purposes |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-remittance/dashboard | GET | fintech_remittance:view | Overview |
| corridors | /fintech-remittance/corridors | GET/POST | fintech_remittance:govern_corridors | Corridors |
| quotes | /fintech-remittance/quotes | GET/POST | fintech_remittance:quote | Quotes |
| transfers | /fintech-remittance/transfers | GET/POST | fintech_remittance:transfer | Transfers |
| payouts | /fintech-remittance/payouts | GET/POST | fintech_remittance:payout | Payouts |
| refunds | /fintech-remittance/refunds | GET/POST | fintech_remittance:refund | Exceptions |
| agents | /fintech-remittance/agents | GET/POST | fintech_remittance:admin | Automation |
| settings | /fintech-remittance/settings | GET/POST | fintech_remittance:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| same_country_blocked | Source and destination country are the same | deny |
| corridor_supported | Corridor not in supported list | deny |
| send_amount_positive | Quote send amount <= 0 | deny |
| fx_rate_positive | FX rate <= 0 | deny |
| fee_non_negative | Negative fee | deny |
| quote_expiry_required | Quote without expiry | deny |
| quote_lock_required | Transfer without quote lock | deny |
| sender_kyc_required | Transfer without sender KYC | deny |
| beneficiary_kyc_required | Transfer without beneficiary KYC | deny |
| source_of_funds_required | Transfer without source-of-funds | deny |
| aml_screen_required | Transfer without AML screen | deny |
| sanctions_hit_blocks_transfer | Sanctions hit detected | deny |
| fraud_block_denies_transfer | Fraud decision is `block` | deny |
| aml_review_requires_approval | AML review outcome without human approval | require_review |
| fraud_review_requires_approval | Fraud hold/review without human approval | require_review |
| high_value_requires_approval | Amount > 100,000 without approval | require_review |
| provider_receipt_required | Payout without provider receipt | deny |
| settlement_reference_required | Payout without settlement reference | deny |
| refund_reviewer_required | Refund without reviewer | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| RemittanceCorridor | id, send_country, receive_country, currency_pair, policy_reference, status |
| RemittanceQuote | id, send_country, receive_country, source_currency, destination_currency, send_amount, fx_rate, fee, expiry, status |
| RemittanceTransfer | id, quote_id, sender_reference, beneficiary_reference, sender_kyc_reference, beneficiary_kyc_reference, funding_reference, payout_method, purpose_code, source_of_funds_reference, aml_screen_reference, fraud_decision, status |
| RemittancePayout | id, transfer_id, payout_method, provider_receipt_reference, settlement_reference, status |
| RemittanceRefund | id, transfer_id, reason, reviewer_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| remittance_quote_created | FX quote created |
| remittance_transfer_created | Transfer initiated |
| remittance_payout_released | Payout released to beneficiary |
| remittance_refund_filed | Refund filed |
| remittance_agent_registered | AI agent registered |

## Edge Cases Handled
- Sanctions hits are a hard deny with no override path — the only resolution is to resolve the sanctions hit at source and re-submit; this is distinct from AML review (which allows human approval to proceed)
- Both sender AND beneficiary KYC are required — one-sided KYC is not accepted; even in corridors where beneficiary KYC is operationally difficult, the rule engine enforces both
- FX rate must be strictly positive; a rate of zero would imply free money transfer and is rejected
- Quote expiry is required at quote creation — expired quotes cannot be used to create transfers; the service layer must check expiry before invoking the transfer creation rule engine
- Source-of-funds evidence is required for every transfer — not just high-value ones; this reflects the regulatory requirement for cross-border fund provenance documentation

## Composability
- **Upstream**: `fintech_kyc` provides sender and beneficiary identity; `fintech_aml` provides sanctions and AML screening; `fintech_fraud` provides fraud risk decisions; `fintech_payments` and `fintech_wallets` provide funding rails
- **Downstream**: `fintech_agency` uses remittance as a cross-border payout mechanism at agent outlets; `fintech_mobile` initiates remittances via the mobile channel; `fintech_lending` uses remittance transaction history as credit behavior evidence
- **Peer**: Deployed alongside `fintech_kyc` (identity), `fintech_aml` (sanctions), and `fintech_fraud` (fraud) — all three are required for every transfer

## Development Notes
- `cash_pickup` payout method requires a physical agent network; the capability records the payout method but agent network management is in `fintech_agency`
- `card_push` payout method (Visa Direct, Mastercard Send) requires card token references from `fintech_cards`
- Purpose codes map to SWIFT/ISO 20022 purpose codes; `family_support`, `education`, `medical` are the primary Africa remittance use cases
- Both batch operations and individual events require Bytewax routing; three separate guardrail rules cover batches, events, and privileged agent actions
