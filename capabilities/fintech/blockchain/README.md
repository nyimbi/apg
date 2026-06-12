# Blockchain Services

## Overview
Blockchain Services provides governed, multi-network blockchain infrastructure for fintech applications: private chain provisioning, wallet/custody management, smart contract deployment and invocation, on-chain transaction recording, tamper-evident evidence anchoring, oracle feed management, token issuance, cross-chain bridges, supply chain tracking, digital certificates, NFT minting, CBDC issuance, and node health monitoring.

The capability is deliberately provider-neutral — live chain RPC calls, signing keys, custody providers, and oracle connectivity remain adapter boundaries. A deterministic governance layer enforces evidence, ownership, and approval requirements before any state change is recorded. Events stream to `apg.fintech.blockchain.lifecycle` via Bytewax.

## Capability ID
`fintech_blockchain`  Version: 2.0.0

## Provides
| Service | Description |
|---------|-------------|
| blockchain_network_workflow | Register and govern supported blockchain networks with chain ID, RPC reference, and environment |
| blockchain_wallet_workflow | Register custody wallets with explicit custody model, key policy, and owner evidence |
| smart_contract_workflow | Deploy, invoke, and upgrade smart contracts with artifact, approval, and evidence requirements |
| chain_transaction_workflow | Record and verify on-chain transactions with hash, signer, asset, settlement status, and high-value approval |
| evidence_anchor_workflow | Anchor payload hashes to chains for tamper-evident evidence trails; Merkle batch anchoring supported |
| oracle_feed_workflow | Register oracle feeds for price, identity, compliance, FX rate, and proof-of-reserve data |
| node_health_workflow | Record node health snapshots with block height, status, and evidence |
| blockchain_review_workflow | Governance reviews for contract deployments and network changes |
| blockchain_agent_workflow | Register AI agents for network operations, contract review, and custody policy roles |
| token_workflow | Issue ERC-20/ERC-721 tokens; CBDC issuance; balance queries; transfers |
| supply_chain_workflow | Record and query product journey events anchored on-chain |
| certificate_workflow | Issue and verify tamper-proof digital certificates backed by on-chain NFTs |
| cross_chain_workflow | Bridge assets across chains; query transfer status |
| analytics_workflow | Compute per-chain analytics: TPS, gas, block stats, value throughput |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Operational notifications |
| nlpc | NLP for review narrative |
| keym | Key management and custody policy references |
| fintech_risk | Risk assessment for on-chain operations |
| fintech_compliance | Compliance evidence for network and contract governance |
| fintech_regtech | Regulatory obligation mapping for blockchain activity |
| fintech_wallets | Wallet references backing blockchain custody |

## Quick Start

```python
from capabilities.fintech.blockchain.service import BlockchainService

svc = BlockchainService(tenant_id="acme", actor_id="ops-1")

# Provision a private consortium chain
chain = await svc.create_private_blockchain(
    "acme-settlement",
    consensus="ibft",
    permissioned=True,
    initial_validators=["val-1", "val-2", "val-3"],
)

# Deploy an ERC-20 token contract
contract = await svc.deploy_smart_contract(
    chain["chain_id"], contract_code, deployer="0xdeployer"
)

# Issue a token
token = await svc.token_issuance(
    chain_id=chain["chain_id"],
    token_name="AcmeCoin",
    total_supply=1_000_000,
    owner="0xowner",
    token_symbol="ACM",
)

# Anchor a compliance record
anchor = await svc.audit_trail_on_chain(
    record_id="kyc-123",
    record_hash="sha256:abcdef...",
    chain_id=chain["chain_id"],
)
```

## New Methods

### 1. `create_private_blockchain` — Consortium chain provisioning
```python
chain = await svc.create_private_blockchain(
    name="trade-finance-net",
    consensus="pbft",           # pbft|raft|pow|pos|ibft|tendermint|clique|dpos
    permissioned=True,
    chain_type="evm",
    block_gas_limit=8_000_000,
    initial_validators=["bank-a", "bank-b", "bank-c"],
)
# Returns: chain_id, genesis_hash, network_id, validator list, status
```

### 2. `deploy_smart_contract` + `invoke_smart_contract` — Full contract lifecycle
```python
# Deploy
contract = await svc.deploy_smart_contract(
    chain_id, solidity_source, deployer="0xdev",
    contract_type="erc20",
    constructor_args={"name": "Token", "symbol": "TKN", "totalSupply": 1_000_000},
)

# Invoke — in-memory ERC-20/ERC-721 execution, gas estimation included
result = await svc.invoke_smart_contract(
    contract["contract_address"], method="transfer",
    params={"to": "0xrecipient", "amount": 500},
    caller="0xsender", chain_id=chain_id,
)
# result["events_emitted"] -> [{"event": "Transfer", "from": ..., "to": ..., "value": 500}]
```

### 3. `token_issuance` + `transfer_token` — Token lifecycle
```python
token = await svc.token_issuance(
    chain_id=chain_id, token_name="KenyaShilling",
    total_supply=10_000_000, owner="0xcb",
    token_symbol="CBDC-KES", decimals=2,
)

tx = await svc.transfer_token(
    token["token_id"], from_address="0xcb", to_address="0xbank", amount=500_000
)
# tx["from_balance_after"], tx["to_balance_after"]

bal = await svc.token_balance(token["token_id"], "0xbank")
```

### 4. `cbdc_issuance` — Central Bank Digital Currency
```python
cbdc = await svc.cbdc_issuance(
    chain_id=chain_id,
    currency_code="KES",
    amount=1_000_000_000,   # in minor units (cents)
    central_bank="CBK",
)
# Returns token record + cbdc_type="retail" + central_bank metadata
```

### 5. `cross_chain_transfer` — Bridge protocol
```python
xfer = await svc.cross_chain_transfer(
    from_chain="ethereum-mainnet",
    to_chain="polygon-mainnet",
    amount=1_000_000,
    token="USDC",
    sender="0xsender",
    recipient="0xrecipient",
)
# lock_tx_hash + release_tx_hash + bridge_fee + estimated_finality_minutes

status = await svc.cross_chain_transfer_status(xfer["transfer_id"])
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| networks.supported_types | list | ethereum, polygon, solana, bitcoin, hyperledger_fabric, private_evm | Supported chain types |
| networks.supported_environments | list | mainnet, testnet, sandbox, consortium, private | Deployment environments |
| contracts.supported_types | list | token, multisig, settlement, identity, oracle, bridge, escrow | Contract categories |
| transactions.high_value_transaction_requires_approval | bool | true | Flag requiring approval above threshold |
| custody_models | list | self_custody, mpc, hsm, smart_contract, custodial | Supported custody models |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-blockchain/dashboard | GET | fintech_blockchain:view | Overview |
| networks | /fintech-blockchain/networks | GET/POST | fintech_blockchain:networks | Networks |
| wallets | /fintech-blockchain/wallets | GET/POST | fintech_blockchain:wallets | Custody |
| contracts | /fintech-blockchain/contracts | GET/POST | fintech_blockchain:contracts | Contracts |
| transactions | /fintech-blockchain/transactions | GET/POST | fintech_blockchain:transactions | Ledger |
| anchors | /fintech-blockchain/anchors | GET/POST | fintech_blockchain:anchors | Evidence |
| oracles | /fintech-blockchain/oracles | GET/POST | fintech_blockchain:oracles | Data |
| nodes | /fintech-blockchain/nodes | GET/POST | fintech_blockchain:nodes | Operations |
| reviews | /fintech-blockchain/reviews | GET/POST | fintech_blockchain:reviews | Governance |
| agents | /fintech-blockchain/agents | GET/POST | fintech_blockchain:admin | Automation |
| settings | /fintech-blockchain/settings | GET/POST | fintech_blockchain:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| network_chain_id_required | Network without chain ID | deny |
| network_rpc_required | Network without RPC reference | deny |
| wallet_key_policy_required | Wallet without key policy | deny |
| custody_model_supported | Unsupported custody model | deny |
| contract_approval_required | Contract deployment without approval | deny |
| contract_artifact_required | Contract without artifact reference | deny |
| transaction_hash_required | Transaction without on-chain hash | deny |
| high_value_transaction_requires_approval | High-value transaction without approval | deny |
| anchor_payload_required | Anchor without payload hash | deny |
| blockchain_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_blockchain_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| BlockchainNetwork | id, tenant_id, network_type, environment, chain_id, rpc_reference, owner_id, evidence_reference |
| BlockchainWallet | id, network_id, wallet_reference, custody_model, key_policy_reference, owner_id, evidence_reference |
| SmartContractDeployment | id, network_id, contract_type, artifact_reference, owner_id, approval_reference, evidence_reference, status |
| ChainTransaction | id, network_id, transaction_hash, transaction_type, asset_reference, amount, signer, settlement_status, evidence_reference |
| EvidenceAnchor | id, network_id, payload_hash, reference, anchored_at, evidence_reference |
| OracleFeed | id, network_id, feed_type, source, owner_id, evidence_reference |
| NodeHealth | id, network_id, endpoint, status, block_height, evidence_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| blockchain_network_registered | New network registered |
| blockchain_wallet_registered | Custody wallet registered |
| smart_contract_deployed | Contract deployment recorded |
| chain_transaction_recorded | On-chain transaction recorded |
| evidence_anchor_recorded | Payload hash anchored to chain |
| oracle_feed_registered | Oracle feed registered |
| node_health_recorded | Node health snapshot recorded |
| blockchain_review_recorded | Governance review completed |
| blockchain_agent_registered | AI agent registered |
| token_issued | New fungible token issued |
| nft_minted | NFT certificate minted |
| cbdc_issued | CBDC token issued |
| cross_chain_transfer_initiated | Bridge transfer initiated |
| supply_chain_event_recorded | Product journey event anchored |

## World-Class Enhancements (v2.0)

Planned in priority order (see `WORLD_CLASS_IMPROVEMENTS.md` for full detail):

| # | Enhancement | One-line summary |
|---|-------------|-----------------|
| 1 | **Merkle Tree Batch Anchoring** | Accumulate 1000 records, anchor single Merkle root — 99.9% gas reduction |
| 2 | **Async Event-Driven Settlement** | submit → mempool → included → finalized pipeline via asyncio queues + Bytewax; eliminates false-positive settlement confirmation |
| 3 | **Programmable Compliance Rules Engine** | Tenant-configurable DSL rules (`when(amount>10k AND jurisdiction=="US") REQUIRE aml.passed`) compiled at registration, zero marginal eval cost |
| 4 | **DeFi Liquidity Pool Management** | AMM subsystem: pool create/provision/withdraw, constant-product swap, impermanent loss calc |
| 5 | **Decentralised Identity (DID/VC)** | W3C `did:datacraft:chain:address` resolution, VC issuance/revocation, VP verification — portable KYC across all APG capabilities |
| 6 | **Cross-Chain Oracle Aggregation** | Chainlink CCIP + Band + Pyth multi-source feeds, median aggregation, configurable staleness threshold — eliminates single-oracle exploit vector |
| 7 | **Tokenised Real-World Assets (RWA)** | Legal wrapper attachment (PPSA/UCC), custody attestation, fractional NFT ownership, on-chain cap table — targets $16T RWA market |
| 8 | **DAO Governance Framework** | Full lifecycle: quorum, voting period, timelock execution queue, veto guardian, on-chain treasury management |
| 9 | **Blockchain Forensics & Analytics** | Address clustering, fund flow tracing, mixing detection, OFAC/UN sanctions screening — replaces $50K+/yr chain analysis vendor |
| 10 | **Formal Smart Contract Verification** | Static symbolic execution (Mythril/Slither-inspired) catches reentrancy, overflow, unchecked-call at deploy time |
| 11 | **Layer-2 Rollup State Channels** | Open channel → signed state updates → dispute period → L2 rollup proof; sub-millisecond settlement for micropayments |
| 12 | **Confidential Transactions** | Pedersen commitments + range proofs replace plain `amount_minor` — institutional privacy for settlement amounts |
| 13 | **MPC Key Ceremony** | Distributed key generation, t-of-n threshold signing, key rotation — HSM-class security without HSM hardware |
| 14 | **ZK Proof Attestation** | ZK-SNARK/STARK circuit proofs replace plain hash anchors — GDPR-compliant privacy-preserving compliance attestation |
| 15 | **Real-Time MEV Protection** | Private mempool (Flashbots-style relay), VDF fair ordering — eliminates front-running and sandwich attacks |

## Edge Cases Handled
- High-value transactions require an explicit approval reference before being recorded — not just a review flag; the rule is a hard deny without it
- Evidence anchors require both a payload hash AND a timestamp (`anchored_at`) — a hash without a timestamp is rejected to prevent backdating
- Contracts must pass through approval before deployment — the approval_required rule fires even in sandbox environments
- Node block heights must be non-negative; a block height of zero is valid (genesis) but negative values are denied
- Oracle feeds require a source reference to prevent orphaned feed registrations with no data lineage
- `cross_chain_transfer` rejects identical source and destination chains at the assertion boundary

## Composability
- **Upstream**: `fintech_wallets` provides wallet references for on-chain custody; `fintech_compliance` and `fintech_regtech` provide evidence and obligation context for governance reviews
- **Downstream**: `fintech_crypto` depends on Blockchain Services for network and wallet references; `fintech_defi` uses it for protocol and position registry; evidence anchors serve `fintech_compliance` as tamper-evident proof
- **Peer**: Commonly deployed with `fintech_crypto` (asset trading) and `fintech_defi` (protocol positions) in a full digital asset stack

## Development Notes
- `SUPPORTED_NETWORK_TYPES` covers both public chains (ethereum, solana, bitcoin) and private/consortium networks (hyperledger_fabric, private_evm); new types require contract update
- Evidence anchoring is one-way: once anchored, a payload hash is immutable; no update or delete operations are permitted by the rule engine
- The `_ne` suffix on condition keys inverts match direction: `event_stream_ne: "bytewax"` fires when the stream processor is anything other than bytewax
- Key policies stored in `keym` are referenced by ID; the capability does not manage private keys directly
- Async methods (`create_private_blockchain`, `deploy_smart_contract`, `record_transaction`, etc.) use `await`; sync governance methods (`register_network`, `deploy_contract`, `record_transaction_sync`) are safe to call without an event loop
- `FintechBlockchainService` and `BlockchainServicesService` are aliases for `BlockchainService`

---
© 2025 Datacraft — www.datacraft.co.ke
