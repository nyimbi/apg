# Digital Cards

## Overview
Digital Cards provides executable card issuing and operations workflows: program governance, cardholder onboarding, virtual and physical card issuance, token provisioning (wallet, device, merchant, network tokens), authorization decisions with fraud and AML controls, and dispute intake. It is the issuing layer that sits between a payment wallet and the card network, enforcing per-authorization fraud scoring and AML result checks before any card transaction is approved.

Physical cards require a shipping address; virtual cards require a wallet reference. All card operations are linked to a funding account and cardholder consent. High-impact authorizations (above threshold or anomaly-flagged) trigger a require-review gate. Events stream to `apg.fintech.cards.lifecycle` via Bytewax.

## Capability ID
`fintech_cards`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| card_program_governance | Register card programs with BIN range, supported currencies, and settlement account |
| cardholder_card_lifecycle | Onboard cardholders with KYC; issue virtual and physical cards |
| tokenized_card_credentialing | Provision wallet, device, merchant, and network tokens with key domain references |
| card_authorization_control | Evaluate authorizations with fraud decision, AML result, currency, and category checks |
| card_dispute_workflow | File and manage card disputes with evidence and reviewer assignment |
| card_agent_workflow | Register AI agents for authorization review, token governance, and dispute handling |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Cardholder and operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| encr | Encryption for card data |
| fintech_payments | Payment authorization and settlement |
| fintech_wallets | Wallet backing and hold management |
| fintech_kyc | Cardholder identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud signal scoring per authorization |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cards.supported_types | list | virtual, physical | Card form factors |
| cards.supported_products | list | debit, prepaid, expense, fleet, merchant | Card product types |
| tokens.supported_types | list | wallet, device, merchant, network | Token use cases |
| authorizations.high_value_threshold | number | 100000 | Authorization amount requiring approval |
| authorizations.supported_fraud_decisions | list | clear, review, hold, block | Valid fraud decision values |
| authorizations.supported_aml_results | list | clear, review, blocked | Valid AML result values |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-cards/dashboard | GET | fintech_cards:view | Overview |
| programs | /fintech-cards/programs | GET/POST | fintech_cards:manage_programs | Programs |
| cardholders | /fintech-cards/cardholders | GET/POST | fintech_cards:manage_cardholders | Cards |
| cards | /fintech-cards/cards | GET/POST | fintech_cards:issue | Cards |
| tokens | /fintech-cards/tokens | GET/POST | fintech_cards:tokenize | Tokens |
| authorizations | /fintech-cards/authorizations | GET/POST | fintech_cards:authorize | Controls |
| disputes | /fintech-cards/disputes | GET/POST | fintech_cards:dispute | Exceptions |
| agents | /fintech-cards/agents | GET/POST | fintech_cards:admin | Automation |
| settings | /fintech-cards/settings | GET/POST | fintech_cards:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| program_bin_range_required | Program without BIN range | deny |
| cardholder_kyc_required | Cardholder without KYC evidence | deny |
| physical_shipping_required | Physical card without shipping address | deny |
| wallet_reference_required | Card issuance without wallet reference | deny |
| card_consent_required | Card issuance without consent evidence | deny |
| token_key_domain_required | Token without key domain reference | deny |
| fraud_block_denies_authorization | Fraud decision is `block` | deny |
| aml_block_denies_authorization | AML result is `blocked` | deny |
| high_impact_authorization_requires_approval | High-impact authorization without approval | require_review |
| dispute_evidence_required | Dispute without evidence | deny |
| card_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| CardProgram | id, name, owner_id, bin_range, currency, settlement_account, status |
| Cardholder | id, customer_reference, kyc_profile_id, country, status |
| Card | id, program_id, cardholder_id, card_type, product, wallet_reference, funding_account, masked_pan, status |
| CardToken | id, card_id, token_type, token_reference, key_domain, device_or_merchant_reference |
| CardAuthorization | id, card_id, amount, currency, merchant_category, fraud_decision, aml_result, status |
| CardDispute | id, transaction_reference, reason, evidence_references, reviewer_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| card_program_registered | New card program created |
| cardholder_onboarded | Cardholder enrolled |
| card_issued | Card issued to cardholder |
| card_token_provisioned | Token provisioned for card |
| card_authorization_decided | Authorization evaluated |
| card_dispute_filed | Dispute filed |
| card_agent_registered | AI agent registered |

## Edge Cases Handled
- Fraud `block` and AML `blocked` decisions both independently deny authorization — both checks fire; a `block` from either source is sufficient to deny even if the other is `clear`
- Physical card issuance requires a shipping address; virtual card issuance does not — the rule is conditional on `physical_card: True` in context
- Token provisioning requires device OR merchant reference (not both) — the rule checks for the presence of at least one; the type determines which is semantically required
- Card consent is distinct from cardholder KYC — a verified cardholder still requires explicit card-issuance consent as a separate evidence attachment
- Authorization currency must be in `SUPPORTED_CURRENCIES`; authorizations in unsupported currencies are denied before fraud/AML checks run

## Composability
- **Upstream**: `fintech_kyc` provides cardholder identity; `fintech_fraud` and `fintech_aml` provide per-authorization signals that gate the `card_authorization_decided` decision
- **Downstream**: `fintech_payments` handles authorization, capture, and settlement; `fintech_wallets` provides wallet holds for authorized amounts; `fintech_mobile` uses card linking for mobile payment flows
- **Peer**: Deployed alongside `fintech_wallets` (wallet-backed funding) and `fintech_neobanking` (debit cards for neobank customers)

## Development Notes
- The `high_impact` flag in authorization context is caller-computed — the rule engine does not evaluate the amount against the threshold directly; callers must set `high_impact: True` when amount > `high_value_threshold`
- BIN range is stored as a string (e.g., "411111") and validated at program registration; uniqueness enforcement is the responsibility of the service layer
- `encr` (encryption capability) is a required dependency; card data (PAN, CVV) must never traverse the system unencrypted
- Token types `wallet` and `device` require a key domain reference; `merchant` and `network` tokens additionally require device or merchant reference
