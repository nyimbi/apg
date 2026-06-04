# Blockchain Services

## Overview
Blockchain Services provides governed, multi-network blockchain infrastructure for fintech applications: network registration, wallet and custody management, smart contract deployment, on-chain transaction recording, evidence anchoring, oracle feed management, node health monitoring, and review workflows. It is deliberately provider-neutral — live chain RPC calls, signing keys, custody providers, and oracle connectivity remain adapter boundaries.

The capability enables use cases including tokenized assets, settlement finality anchoring, on-chain identity evidence, and cross-chain bridges, all within a deterministic governance layer that enforces evidence, ownership, and approval requirements before any state change is recorded. Events stream to `apg.fintech.blockchain.lifecycle` via Bytewax.

## Capability ID
`fintech_blockchain`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| blockchain_network_workflow | Register and govern supported blockchain networks with chain ID, RPC reference, and environment |
| blockchain_wallet_workflow | Register custody wallets with explicit custody model, key policy, and owner evidence |
| smart_contract_workflow | Deploy and track smart contracts with artifact, approval, and evidence requirements |
| chain_transaction_workflow | Record on-chain transactions with hash, signer, asset, settlement status, and high-value approval |
| evidence_anchor_workflow | Anchor payload hashes to chains for tamper-evident evidence trails |
| oracle_feed_workflow | Register oracle feeds for price, identity, compliance, FX rate, and proof-of-reserve data |
| node_health_workflow | Record node health snapshots with block height, status, and evidence |
| blockchain_review_workflow | Governance reviews for contract deployments and network changes |
| blockchain_agent_workflow | Register AI agents for network operations, contract review, and custody policy roles |

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

## Edge Cases Handled
- High-value transactions require an explicit approval reference before being recorded — not just a review flag; the rule is a hard deny without it
- Evidence anchors require both a payload hash AND a timestamp (`anchored_at`) — a hash without a timestamp is rejected to prevent backdating
- Contracts must pass through approval before deployment — the approval_required rule fires even in sandbox environments
- Node block heights must be non-negative; a block height of zero is valid (genesis) but negative values are denied
- Oracle feeds require a source reference to prevent orphaned feed registrations with no data lineage

## Composability
- **Upstream**: `fintech_wallets` provides wallet references for on-chain custody; `fintech_compliance` and `fintech_regtech` provide evidence and obligation context for governance reviews
- **Downstream**: `fintech_crypto` depends on Blockchain Services for network and wallet references; `fintech_defi` uses it for protocol and position registry; evidence anchors serve `fintech_compliance` as tamper-evident proof
- **Peer**: Commonly deployed with `fintech_crypto` (asset trading) and `fintech_defi` (protocol positions) in a full digital asset stack

## Development Notes
- `SUPPORTED_NETWORK_TYPES` covers both public chains (ethereum, solana, bitcoin) and private/consortium networks (hyperledger_fabric, private_evm); new types require contract update
- Evidence anchoring is one-way: once anchored, a payload hash is immutable; no update or delete operations are permitted by the rule engine
- The `_ne` suffix on condition keys inverts match direction: `event_stream_ne: "bytewax"` fires when the stream processor is anything other than bytewax
- Key policies stored in `keym` are referenced by ID; the capability does not manage private keys directly
