# Cryptocurrency Services

## Overview
`fintech_crypto` provides governed digital asset operations: asset registry, custody account management, balance snapshots, order management, trade execution recording, transfer requests with approval gates, compliance screening (wallet, transaction, sanctions, travel rule), market price snapshots, and governance reviews. It is the regulated operational layer over blockchain infrastructure — audit trail and compliance controls that raw chain operations lack.

All transfers require explicit approval. Sanctions hits and fraud blocks result in hard denies. Events stream to `apg.fintech.crypto.lifecycle` via Bytewax.

**Capability ID**: `fintech_crypto` | **Version**: 2.0.0 | © 2025 Datacraft

---

## Quick Start

```python
from capabilities.fintech.crypto.service import CryptocurrencyService

svc = CryptocurrencyService(tenant_id="acme", actor_id="trader-1")

# Create wallet and buy crypto
wallet = await svc.create_crypto_wallet("cust-001", "ETH")
buy = await svc.buy_crypto("cust-001", "ETH", 500.0, wallet_id=wallet["wallet_id"])

# Check portfolio
portfolio = await svc.portfolio_summary("cust-001")
print(portfolio["total_usd_value"])

# Screen an address before transfer
screen = await svc.screen_address("0xabc...", "ETH")
assert screen["status"] == "clear"

# Send crypto (requires screen first)
tx = await svc.send_crypto(wallet["wallet_id"], "0xdest...", 0.1, "ETH")
```

---

## Core Services

| Service | Description |
|---------|-------------|
| `crypto_asset_workflow` | Register digital assets with symbol, type, network reference, precision, owner, and evidence |
| `crypto_custody_workflow` | Open custody accounts with model, provider reference, policy, owner, and evidence |
| `crypto_balance_workflow` | Record balance snapshots with fiat valuation, currency, and evidence |
| `crypto_order_workflow` | Create orders with side, type, quantity, limit price, policy, requester, and evidence |
| `crypto_trade_workflow` | Record trade executions with venue, price, quantity, fee, and settlement reference |
| `crypto_transfer_workflow` | Request asset transfers with destination, approval, and multi-status tracking |
| `crypto_screening_workflow` | Screen wallets, transactions, assets, counterparties, sanctions, and travel rule |
| `crypto_price_workflow` | Record price snapshots from exchange, oracle, custodian, manual, and aggregator sources |
| `crypto_review_workflow` | Governance reviews for assets, contracts, and operational decisions |
| `crypto_agent_workflow` | Register AI agents for portfolio monitoring, trade review, and compliance screening |

---

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
| prices/stream | /fintech-crypto/prices/stream | GET | fintech_crypto:view | Market Data |
| reserves/proof | /fintech-crypto/reserves/proof | GET | public | Transparency |

---

## New Methods

### `portfolio_summary` — Cross-wallet holdings view
```python
summary = await svc.portfolio_summary("cust-001", base_currency="USD")
# {
#   "total_usd_value": 12450.0,
#   "holdings": [
#     {"coin": "ETH", "amount": 2.5, "usd_price": 3480.0, "allocation_pct": 69.9},
#     {"coin": "BTC", "amount": 0.05, "usd_price": 67500.0, "allocation_pct": 27.1},
#   ]
# }
```

### `staking_enrol` / `unstake` — Native staking with reward calculation
```python
# Stake 2 ETH for 90 days (4.2% APY)
position = await svc.staking_enrol(wallet_id, "ETH", 2.0, lock_days=90)
# {"staking_id": "...", "estimated_reward": 0.00207, "apy_pct": 4.2, "unlock_at": "..."}

# Early exit incurs 10% reward penalty; normal exit does not
result = await svc.unstake(wallet_id, position["staking_id"])
```

### `crypto_to_crypto_swap` — USD-bridge pricing with configurable fees
```python
swap = await svc.crypto_to_crypto_swap("BTC", "ETH", 0.1, wallet_id=wallet_id)
# {"from_amount": 0.1, "to_amount": 1.924, "fee_pct": 0.3, "implied_rate": 19.396}
```

### `limit_order` — Conditional execution against live price table
```python
order = await svc.limit_order("cust-001", "BTC", "buy", 0.01, limit_price=65_000.0)
# status: "filled" if current price <= limit_price, else "pending"
```

### `dca_plan_setup` — Dollar-cost averaging scheduler
```python
plan = await svc.dca_plan_setup("cust-001", "BTC", 100.0, frequency="weekly")
# {"plan_id": "...", "coin": "BTC", "amount_per_period_usd": 100.0, "status": "active"}
```

### `crypto_loan` — Crypto-backed lending with LTV
```python
loan = await svc.crypto_loan("cust-001", "BTC", 0.1, "USDC", ltv=0.5)
# {"borrow_amount": 3375.0, "ltv": 0.5, "interest_rate_annual_pct": 8.5}
```

---

## World-Class Enhancements (v2.0)

1. **Real-Time WebSocket Price Streaming** — Replace static price dict with async Binance/Coinbase WebSocket feed; sub-100ms updates; SSE endpoint `/fintech-crypto/prices/stream`.

2. **FIFO/LIFO Cost-Basis Lot Tracking** — Proper lot ledger for `tax_report()`: buy creates cost-basis lot, sell matches FIFO/LIFO/HIFO. Yields accurate short/long-term classification compliant with KE ITAT and IRS 8949.

3. **On-Chain Transaction Confirmation Polling** — `poll_confirmation(tx_hash, coin, confirmations=6)` via configurable RPC (Infura/Alchemy/local node). States: `broadcast → confirming(n/6) → confirmed → settled` with streaming events.

4. **HSM Key Derivation Integration** — `create_crypto_wallet()` calls AWS CloudHSM / HashiCorp Vault Transit for BIP-44 child key derivation `m/44'/coin_type'/account'/0/index`. Private keys never leave HSM boundary.

5. **Multi-Signature Approval Workflow** — `request_multisig_approval()` and `cosign_transfer()` for transfers above configurable USD threshold (default 10,000). M-of-N cosigner model; integrates with `fintech_compliance` approval chain.

6. **DeFi Protocol Yield Aggregator** — `defi_yield_scan(wallet_id, protocols=['aave','compound','curve'])` queries on-chain APY, ranks yield opportunities against holdings, estimates impermanent loss for LP positions.

7. **Automated Portfolio Rebalancing Engine** — `portfolio_rebalance()` computes minimum swap set to reach target allocations within tolerance band; `execute_rebalance(plan_id)` fires swaps atomically.

8. **Cross-Chain Bridge Integration** — `bridge_asset(from_chain, to_chain, coin, amount, wallet_id)` via LayerZero/Wormhole. Async status tracking, refund handling on failure, bridging fees tracked separately.

9. **Institutional Order Book via FIX Protocol** — FIX 4.4 session manager for B2C2/Wintermute/Cumberland. Orders > 50,000 USD route to RFQ; `accept_quote(quote_id)` finalises best-execution block trades.

10. **FATF Travel Rule Automation** — `travel_rule_package(transfer_id)` assembles IVMS101 bundle and auto-submits to counterparty VASP via TRP/OpenVASP. Blocks transfer until counterparty ACKs.

11. **Transaction Velocity Anomaly Detection** — `check_velocity(customer_id, coin, amount)` compares against 1h/24h/7d rolling sums; raises `VelocityBreachError` and logs compliance event on threshold breach.

12. **Gas Fee Estimation and MEV Protection** — `estimate_gas(coin, tx_type, priority)` returns EIP-1559 `base_fee`/`priority_fee`/`max_fee` at slow/standard/fast targets. ETH transfers optionally route through Flashbots Protect.

13. **Proof-of-Reserves Attestation** — `generate_reserves_proof(tenant_id)` computes Merkle tree of custodied balances, signs root with HSM key. Public endpoint `/fintech-crypto/reserves/proof` for independent verification.

14. **AI-Driven Sentiment-Augmented Price Prediction** — `price_sentiment_forecast(coin, horizon_hours=24)` ingests 500 headlines, runs locally-hosted Ollama Mistral for sentiment scoring, combines with on-chain metrics (exchange inflows, funding rate). No external AI API calls.

15. **Event-Sourced Tamper-Evident Audit Ledger** — Append-only SHA-256 hash chain replacing flat `audit_events` list. `verify_audit_chain(tenant_id)` recomputes from genesis and reports broken links. PostgreSQL `GENERATED ALWAYS AS` computed hash column for DB-level integrity. SOC 2 Type II compliant.

---

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
| transfers.multisig_threshold_usd | float | 10000.0 | USD threshold triggering M-of-N approval |
| screening.supported_types | list | wallet, transaction, asset, counterparty, sanctions, travel_rule | Screening categories |
| staking.supported_coins | list | ETH, SOL, ADA, DOT, ATOM, AVAX, BNB | Stakeable assets |

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

## Streaming Events

Events emitted to `apg.fintech.crypto.lifecycle` via Bytewax.

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

## Composability

- **Upstream**: `fintech_blockchain` provides network and wallet infrastructure; `fintech_kyc` provides identity for custody account holders; `fintech_aml` provides sanctions screening
- **Downstream**: `fintech_defi` uses crypto custody accounts and asset registry as position backing; `fintech_portfolio` tracks crypto holdings via balance snapshots
- **Peer**: Deployed alongside `fintech_blockchain` (chain infrastructure) and `fintech_defi` (protocol-level DeFi operations)
- **Composition keywords**: `crypto`, `wallet`, `custody`, `staking`, `swap`, `defi`, `blockchain`, `sanctions`, `travel_rule`

## Edge Cases

- All transfers require an explicit approval reference — no low-value exemption given settlement finality
- Limit orders require a limit price; market, RFQ, and rebalance orders do not
- Non-clear screening results (review, blocked, escalated) require reviewer assignment before record is accepted
- Asset precision of 0 is valid (integer-only assets); negative precision is rejected
- Price snapshots require `observed_at` to prevent stale prices being treated as current

---

*© 2025 Datacraft — www.datacraft.co.ke | nyimbi@gmail.com*
