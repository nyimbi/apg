# Cryptocurrency Services

## Overview
Cryptocurrency Services provides governed digital asset operations: asset registry, custody account management, balance snapshots, order management, trade execution recording, transfer requests with approval gates, compliance screening (wallet, transaction, sanctions, travel rule), market price snapshots, and governance reviews. It is the regulated operational layer over blockchain infrastructure, providing the audit trail and compliance controls that raw chain operations lack.

All transfers require explicit approval. Sanctions hits and fraud blocks result in hard denies. Compliance screening supports sanctions, travel rule, and wallet screening workflows. Events stream to `apg.fintech.crypto.lifecycle` via Bytewax.

## Capability ID
`fintech_crypto`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| crypto_asset_workflow | Register digital assets with symbol, type, network reference, precision, owner, and evidence |
| crypto_custody_workflow | Open custody accounts with model, provider reference, policy, owner, and evidence |
| crypto_balance_workflow | Record balance snapshots with fiat valuation, currency, and evidence |
| crypto_order_workflow | Create orders with side, type, quantity, limit price, policy, requester, and evidence |
| crypto_trade_workflow | Record trade executions with venue, price, quantity, fee, and settlement reference |
| crypto_transfer_workflow | Request asset transfers with destination, approval, and multi-status tracking |
| crypto_screening_workflow | Screen wallets, transactions, assets, counterparties, sanctions, and travel rule |
| crypto_price_workflow | Record price snapshots from exchange, oracle, custodian, manual, and aggregator sources |
| crypto_review_workflow | Governance reviews for assets, contracts, and operational decisions |
| crypto_agent_workflow | Register AI agents for portfolio monitoring, trade review, and compliance screening |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_blockchain | Network and wallet infrastructure |
| fintech_wallets | Wallet references for custody |
| fintech_risk | Risk assessment for crypto operations |
| fintech_compliance | Compliance obligation evidence |
| fintech_regtech | Regulatory framework context |
| fintech_aml | AML screening integration |
| fintech_kyc | Customer identity for custody accounts |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| assets.supported_types | list | native_coin, stablecoin, utility_token, security_token, governance_token, tokenized_deposit | Asset classifications |
| custody_accounts.supported_custody_models | list | self_custody, mpc, hsm, exchange_custody, smart_contract, custodial | Custody models |
| orders.supported_types | list | market, limit, stop_limit, rfq, rebalance | Order types |
| transfers.approval_required | bool | true | All transfers require approval |
| screening.supported_types | list | wallet, transaction, asset, counterparty, sanctions, travel_rule | Screening categories |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-crypto/dashboard | GET | fintech_crypto:view | Overview |
| assets | /fintech-crypto/assets | GET/POST | fintech_crypto:assets | Assets |
| custody | /fintech-crypto/custody | GET/POST | fintech_crypto:custody | Custody |
| balances | /fintech-crypto/balances | GET/POST | fintech_crypto:balances | Portfolio |
| orders | /fintech-crypto/orders | GET/POST | fintech_crypto:orders | Trading |
| trades | /fintech-crypto/trades | GET/POST | fintech_crypto:trades | Trading |
| transfers | /fintech-crypto/transfers | GET/POST | fintech_crypto:transfers | Treasury |
| screening | /fintech-crypto/screening | GET/POST | fintech_crypto:screening | Compliance |
| prices | /fintech-crypto/prices | GET/POST | fintech_crypto:prices | Market Data |
| reviews | /fintech-crypto/reviews | GET/POST | fintech_crypto:reviews | Governance |
| agents | /fintech-crypto/agents | GET/POST | fintech_crypto:admin | Automation |
| settings | /fintech-crypto/settings | GET/POST | fintech_crypto:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| asset_symbol_required | Asset without ticker symbol | deny |
| asset_precision_valid | Negative asset precision | deny |
| custody_policy_required | Custody account without policy reference | deny |
| limit_order_requires_price | Limit order without limit price | deny |
| transfer_approval_required | Transfer without approval reference | deny |
| non_clear_screening_requires_reviewer | Non-clear screening without reviewer | deny |
| price_observed_at_required | Price snapshot without observation timestamp | deny |
| crypto_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_crypto_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| CryptoAsset | id, symbol, asset_type, network_reference, token_contract, precision, owner_id, evidence_reference |
| CustodyAccount | id, custody_model, provider_reference, policy_reference, owner_id, evidence_reference |
| CryptoBalance | id, account_id, asset_id, amount, valuation, currency, evidence_reference |
| CryptoOrder | id, account_id, asset_id, side, order_type, quantity, limit_price, policy_reference, requester_id |
| CryptoTrade | id, order_id, venue, execution_price, quantity, fee, status, settlement_reference |
| CryptoTransfer | id, account_id, asset_id, transfer_type, destination, amount, approval_reference, status |
| CryptoScreening | id, reference, screening_type, status, reviewer_id, evidence_reference |
| CryptoPrice | id, asset_id, source, price, currency, observed_at, evidence_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| crypto_asset_registered | Digital asset registered |
| crypto_custody_account_opened | Custody account opened |
| crypto_balance_recorded | Balance snapshot recorded |
| crypto_order_created | Order created |
| crypto_trade_recorded | Trade execution recorded |
| crypto_transfer_requested | Transfer initiated |
| crypto_screening_recorded | Compliance screening completed |
| crypto_price_recorded | Price snapshot recorded |
| crypto_review_recorded | Governance review recorded |
| crypto_agent_registered | AI agent registered |

## Edge Cases Handled
- All transfers require an explicit approval reference regardless of amount — there is no low-value transfer exemption for crypto, given settlement finality
- Limit orders require a limit price; market, RFQ, and rebalance orders do not — the rule is conditional on order type
- Non-clear screening results (review, blocked, escalated) require a reviewer assignment before the screening record is accepted
- Asset precision must be non-negative; precision of 0 is valid (integer-only assets) but negative precision is rejected
- Price snapshots require an `observed_at` timestamp to prevent stale prices being mistaken for current market data

## Composability
- **Upstream**: `fintech_blockchain` provides the underlying network and wallet infrastructure; `fintech_kyc` provides identity for custody account holders; `fintech_aml` provides sanctions screening
- **Downstream**: `fintech_defi` uses crypto custody accounts and asset registry as position backing; `fintech_portfolio` tracks crypto holdings via balance snapshots
- **Peer**: Deployed alongside `fintech_blockchain` (chain infrastructure) and `fintech_defi` (protocol-level DeFi operations)

## Development Notes
- `custody_model_supported` validates against `SUPPORTED_CUSTODY_MODELS`; `exchange_custody` is included for exchange-held assets that are not self-custodied
- Travel rule screening is a distinct screening type that captures VASP-to-VASP transfer information; it is separate from sanctions screening
- Trade `settlement_reference` is required at trade recording time — settlement is not deferred to a separate step; the trade record must carry evidence that settlement has been arranged
- The `large_transfer_requires_approval` governance flag in DEFAULT_CONFIGURATION applies to the transfer workflow; the threshold is implementation-defined in the service layer
