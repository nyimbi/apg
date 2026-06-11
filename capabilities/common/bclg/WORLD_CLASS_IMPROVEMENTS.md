# BCLG - World-Class Improvement Proposals

**Capability**: Blockchain Ledger (bclg) | **Domain**: common
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## Improvement 1: Merkle Tree-Based Block Verification

**Category**: Cryptographic Integrity

**Justification**: Current block hashes are computed from a flat concatenation of transaction hashes. A Merkle tree structure enables O(log n) inclusion proofs, so any verifier can confirm a single transaction is in a block without downloading the full block. This is the foundational data structure used by Bitcoin, Ethereum, and every production-grade distributed ledger — without it, BCLG cannot interoperate with external verifiers or light clients.

**Implementation**:
```python
async def build_merkle_root(self, transaction_hashes: list[str]) -> str:
    """Compute SHA-256 Merkle root from leaf hashes."""
    import hashlib
    nodes = [bytes.fromhex(h) if len(h) == 64 else hashlib.sha256(h.encode()).digest()
             for h in transaction_hashes]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [hashlib.sha256(nodes[i] + nodes[i+1]).digest()
                 for i in range(0, len(nodes), 2)]
    return nodes[0].hex() if nodes else "0" * 64

async def merkle_inclusion_proof(
    self, tenant_id: str, ledger_id: str, transaction_hash: str
) -> dict[str, Any]: ...
```

**Competitor Reference**: Ethereum uses a Patricia Merkle Trie for state, transaction, and receipt proofs. Hyperledger Fabric uses Merkle DAGs in its block store (gossip protocol, `core/ledger/`).

---

## Improvement 2: Decimal-Precision Financial Arithmetic

**Category**: Correctness / Financial Safety

**Justification**: All monetary amounts in the current implementation use `float`, which is subject to IEEE-754 rounding errors. A transaction for `0.1 + 0.2` yields `0.30000000000000004`. Financial ledgers that store, accumulate, or compare float amounts will silently accrue errors. Python's `decimal.Decimal` with `ROUND_HALF_EVEN` is the correct type for all monetary values — it is what banks, payment processors, and accounting standards mandate.

**Implementation**:
```python
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation

async def submit_transaction(self, ..., amount: Decimal | str | float, ...) -> dict[str, Any]:
    guard_tenant_id(tenant_id)
    try:
        amount_d = Decimal(str(amount)).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)
    except InvalidOperation:
        raise ValueError("invalid_transaction_amount")
    if amount_d <= 0:
        raise PermissionError("positive_transaction_amount_required")
    ...
```

**Competitor Reference**: Stellar (Horizon API) stores all asset amounts as int64 stroops (10^-7 XLM) to avoid float. Solana stores lamports as u64. Ripple uses 64-bit integer drops. All avoid floating point at the storage layer.

---

## Improvement 3: Event Sourcing with Append-Only Ledger Log

**Category**: Architecture / Auditability

**Justification**: The current service mutates in-memory dicts with `dataclasses.replace`. This destroys history — there is no way to reconstruct system state at a prior point in time. An append-only event log (command sourcing) ensures every state transition is recorded, enables temporal queries ("show me wallet balance at T-30d"), and makes the audit trail unforgeable. This is the architecture behind Diem's DiemDB, Solana's ledger log, and Hyperledger Fabric's block store.

**Implementation**:
- Introduce `LedgerEventLog(events: list[LedgerAuditEvent])` at service init.
- All mutations emit a `LedgerEvent` before updating state.
- Add `async def replay_state(self, tenant_id: str, up_to_sequence: int) -> dict[str, Any]` that rebuilds state from events up to sequence N.
- Expose `async def ledger_event_stream(self, tenant_id: str, after_sequence: int = 0)` for real-time consumers.

**Competitor Reference**: Hyperledger Fabric's orderer service produces an immutable append-only channel block log. Diem's DiemDB stores a write-ahead log and compacts it with a Merkle accumulator.

---

## Improvement 4: Zero-Knowledge Proof Integration Hooks

**Category**: Privacy / Compliance

**Justification**: Regulated finance (GDPR, PCI-DSS, FATF travel rule) increasingly requires proving transaction validity without revealing counterparty identities or amounts. ZK-SNARK/STARK proof hooks let BCLG attach a `proof_ref` to transactions that external provers (e.g., gnark, circom, risc0) can verify without seeing the underlying data. Without this, BCLG cannot serve privacy-preserving DeFi, confidential payments, or regulatory selective disclosure use cases.

**Implementation**:
```python
async def attach_zk_proof(
    self, tenant_id: str, transaction_id: str,
    proof_system: str,  # "groth16" | "plonk" | "stark"
    proof_ref: str,     # content-addressed proof URI
    verifier_key_ref: str,
    circuit_id: str,
    actor: str = "prover",
) -> dict[str, Any]: ...
```

**Competitor Reference**: Zcash uses zk-SNARKs (Groth16) for shielded transactions. Aztec Network's Noir language compiles to UltraPlonk proofs. StarkWare's StarkEx powers dYdX and Immutable X with STARK proofs.

---

## Improvement 5: Multi-Signature Threshold Signing

**Category**: Security / Governance

**Justification**: The current custody model binds one key to one custodian. Enterprise ledger operations (treasury transfers, contract deployments) require M-of-N threshold approval — for example, 3 of 5 key holders must co-sign before a transaction is valid. This is mandatory for institutional custody (SOC 2, ISO 27001 §A.9) and is the model used by BitGo, Fireblocks, and Gnosis Safe.

**Implementation**:
```python
async def create_multisig_policy(
    self, policy_id: str, tenant_id: str, ledger_id: str,
    signers: list[str], threshold: int,
    actor: str = "admin",
) -> dict[str, Any]: ...

async def submit_multisig_signature(
    self, policy_id: str, tenant_id: str,
    transaction_id: str, signer: str, partial_signature: str,
) -> dict[str, Any]: ...

async def finalize_multisig_transaction(
    self, policy_id: str, tenant_id: str, transaction_id: str,
) -> dict[str, Any]: ...
```

**Competitor Reference**: Gnosis Safe requires M-of-N owner approval for every on-chain transaction. Fireblocks MPC wallet uses threshold ECDSA (GG20 protocol). BitGo uses 3-of-3 multisig with hot, warm, cold keys.

---

## Improvement 6: Automated Compliance Screening (AML/KYC Hook)

**Category**: Regulatory Compliance

**Justification**: Financial regulators (FATF, FinCEN, EU AMLD) require transaction screening against sanctions lists, PEP lists, and velocity thresholds. The current compliance_report method is purely retrospective — it cannot block a transaction before it commits. A pre-submit screening hook that calls a configurable compliance adapter (Chainalysis, Elliptic, or local rule set) lets BCLG enforce AML/KYC controls at the transaction boundary.

**Implementation**:
```python
async def screen_transaction(
    self, tenant_id: str, from_account: str, to_account: str,
    amount: Decimal, asset: str,
    compliance_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Return screening result before submission. Blocked result raises PermissionError."""
    ...

async def register_compliance_rule(
    self, rule_id: str, tenant_id: str,
    rule_type: str,  # "velocity" | "sanctions" | "threshold"
    parameters: dict[str, Any],
    actor: str = "compliance-officer",
) -> dict[str, Any]: ...
```

**Competitor Reference**: Chainalysis KYT (Know Your Transaction) API provides real-time risk scoring. Elliptic Lens screens wallet addresses against OFAC/UN sanctions. Circle's USDC uses automated compliance checks before minting.

---

## Improvement 7: Async-Native Service with asyncpg Backend

**Category**: Performance / Scalability

**Justification**: All current service methods are synchronous. Under concurrent load (multiple tenants submitting transactions simultaneously), the GIL and blocking I/O will serialize operations that could run in parallel. Async-native methods with an asyncpg (PostgreSQL) backend replace the in-memory dict store with a persistent, concurrent-safe, indexed store. This is how production ledger services (Stellar Horizon, Hedera Mirror Node) handle thousands of TPS.

**Implementation**:
```python
class BclgService:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def submit_transaction(self, ...) -> dict[str, Any]:
        guard_tenant_id(tenant_id)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                ...
```

All 8+ new methods added in this enhancement use `async def` signatures in anticipation of this migration, making them forward-compatible.

**Competitor Reference**: Stellar's Horizon server uses PostgreSQL with asyncio for ingestion pipelines handling 1000+ TPS. Hedera Mirror Node uses PostgreSQL + async Java reactive streams.

---

## Improvement 8: Smart Contract State Machine with Lifecycle Events

**Category**: Smart Contract Runtime

**Justification**: Deployed contracts currently have no state transitions beyond `deployed`. Real smart contracts have a lifecycle: `draft` → `audited` → `deployed` → `paused` → `deprecated` → `destroyed`. Each transition should emit a lifecycle event, optionally trigger a Bytewax stream mutation, and enforce governance rules (e.g., only a reviewer who did not deploy can pause). Without lifecycle management, BCLG cannot support contract upgrades, emergency pauses, or formal deprecation.

**Implementation**:
```python
async def pause_contract(
    self, contract_id: str, tenant_id: str, actor: str, reason: str,
) -> dict[str, Any]: ...

async def unpause_contract(
    self, contract_id: str, tenant_id: str, actor: str, justification: str,
) -> dict[str, Any]: ...

async def deprecate_contract(
    self, contract_id: str, tenant_id: str, replacement_contract_id: str | None,
    actor: str, deprecation_notice: str,
) -> dict[str, Any]: ...
```

**Competitor Reference**: OpenZeppelin Contracts provides `Pausable` (pause/unpause) and `Ownable` lifecycle controls. Ethereum EIPs 897 and 1967 define proxy upgrade patterns with lifecycle semantics.

---

## Improvement 9: Token Governance — Supply Cap and Burn

**Category**: Tokenomics / DeFi

**Justification**: The current `token_mint` method mints unbounded supply with no accounting. Production token contracts enforce: maximum supply cap, current circulating supply, burn (deflationary mechanics), and supply change events. Without these, BCLG cannot represent real ERC-20/ERC-777 token semantics or enforce governance-mandated supply limits.

**Implementation**:
```python
async def token_burn(
    self, burn_id: str, tenant_id: str, ledger_id: str,
    from_account: str, amount: Decimal, asset: str,
    actor: str, key_custody_id: str,
) -> dict[str, Any]: ...

async def token_supply(
    self, tenant_id: str, ledger_id: str, asset: str,
) -> dict[str, Any]:
    """Return minted, burned, and circulating supply for an asset."""
    ...
```

**Competitor Reference**: Uniswap V3 tracks per-pool liquidity with precise int128/uint128 fixed-point arithmetic. MakerDAO's Dai has a hard supply ceiling (debt ceiling) enforced on-chain. ERC-20 `totalSupply()` is a required interface method.

---

## Improvement 10: Cross-Tenant Federation and Atomic Swaps

**Category**: Interoperability / Multi-Party Finance

**Justification**: Enterprises running separate tenant ledgers (e.g., buyer and supplier tenants) need atomic cross-tenant swaps — transaction A on tenant-1 commits if and only if transaction B on tenant-2 commits, with cryptographic linkage. HTLC (Hash Time-Locked Contracts) provide this guarantee without a trusted intermediary. Without federation, BCLG tenants are isolated silos that cannot settle bilaterally.

**Implementation**:
```python
async def create_htlc(
    self, htlc_id: str, initiator_tenant_id: str, responder_tenant_id: str,
    initiator_ledger_id: str, responder_ledger_id: str,
    initiator_amount: Decimal, responder_amount: Decimal,
    asset: str, hash_lock: str, timeout_seconds: int,
    actor: str,
) -> dict[str, Any]: ...

async def claim_htlc(
    self, htlc_id: str, tenant_id: str, preimage: str, actor: str,
) -> dict[str, Any]: ...

async def refund_htlc(
    self, htlc_id: str, tenant_id: str, actor: str,
) -> dict[str, Any]: ...
```

**Competitor Reference**: Bitcoin's lightning network uses HTLCs for payment channels. Interledger Protocol (ILP) uses conditional payments with HTLC semantics for cross-ledger settlement. Cosmos IBC (Inter-Blockchain Communication) uses packet timeouts and acknowledgements.

---

## Improvement 11: Ledger Snapshot and Point-in-Time Recovery

**Category**: Resilience / Disaster Recovery

**Justification**: The in-memory store has no persistence. A service restart loses all ledger state. Point-in-time snapshots (serialized to PostgreSQL or object storage) enable: disaster recovery, ledger state export/import, environment cloning (staging mirrors production), and forensic analysis at a specific block height. This is a tier-1 operational requirement for any production ledger.

**Implementation**:
```python
async def create_snapshot(
    self, snapshot_id: str, tenant_id: str,
    ledger_id: str | None = None,
    actor: str = "ops",
) -> dict[str, Any]:
    """Serialize current ledger state to a content-addressed snapshot."""
    ...

async def restore_from_snapshot(
    self, snapshot_id: str, tenant_id: str, actor: str,
) -> dict[str, Any]: ...

async def list_snapshots(self, tenant_id: str) -> list[dict[str, Any]]: ...
```

**Competitor Reference**: Bitcoin Core's chainstate uses LevelDB snapshots for UTXO set checkpoints. Ethereum clients (Geth, Besu) support state snapshots for fast sync. Hedera Mirror Node supports daily state snapshots for disaster recovery.

---

## Improvement 12: Governance Voting on Ledger Parameters

**Category**: DAO / Decentralized Governance

**Justification**: Network parameter changes (consensus profile, network policy, participant admission) should be subject to on-ledger governance votes rather than unilateral admin action. A proposal → vote → tally → enact lifecycle is foundational to permissioned consortium ledgers (Hyperledger Besu IBFT, Quorum QBFT) and enterprise DAOs. Without it, BCLG has no mechanism for multi-stakeholder parameter negotiation.

**Implementation**:
```python
async def create_governance_proposal(
    self, proposal_id: str, tenant_id: str, ledger_id: str,
    proposer: str, title: str, description: str,
    proposed_changes: dict[str, Any], voting_deadline_iso: str,
) -> dict[str, Any]: ...

async def cast_vote(
    self, proposal_id: str, tenant_id: str,
    voter: str, vote: str,  # "yes" | "no" | "abstain"
    rationale: str = "",
) -> dict[str, Any]: ...

async def tally_votes(
    self, proposal_id: str, tenant_id: str, actor: str = "governance",
) -> dict[str, Any]: ...
```

**Competitor Reference**: Compound Governor Bravo is the reference implementation for on-chain governance. Snapshot.org provides off-chain voting with on-chain execution. Hyperledger Besu IBFT uses validator voting for network membership changes.

---

## Improvement 13: DeFi Protocol Adapters (AMM, Lending)

**Category**: DeFi / Composability

**Justification**: BCLG's composability story currently covers key management (KEYM), encryption (ENCR), and composition (COMP). Adding DeFi protocol adapters — Automated Market Maker (AMM) swap quoting, lending pool deposit/borrow operations — makes BCLG useful for tokenized asset platforms, treasury yield optimization, and collateralized lending. These adapter hooks call external protocol contracts via the existing `invoke_contract` pathway.

**Implementation**:
```python
async def amm_quote_swap(
    self, tenant_id: str, ledger_id: str, contract_id: str,
    token_in: str, token_out: str, amount_in: Decimal,
) -> dict[str, Any]:
    """Get swap output quote from an AMM pool contract."""
    ...

async def lending_deposit(
    self, deposit_id: str, tenant_id: str, ledger_id: str,
    contract_id: str, depositor: str, asset: str, amount: Decimal,
    key_custody_id: str, actor: str = "user",
) -> dict[str, Any]: ...

async def lending_borrow(
    self, borrow_id: str, tenant_id: str, ledger_id: str,
    contract_id: str, borrower: str, asset: str, amount: Decimal,
    collateral_ref: str, key_custody_id: str, actor: str = "user",
) -> dict[str, Any]: ...
```

**Competitor Reference**: Uniswap V3 is the reference AMM (concentrated liquidity, tick-based). Aave V3 is the reference lending protocol with isolation mode, efficiency mode, and risk parameters. Compound V3 (Comet) is the reference money market.

---

## Improvement 14: Real-Time Risk Engine with Velocity Limits

**Category**: Risk Management / Fraud Prevention

**Justification**: The current service enforces a single hard threshold (`> 100_000` triggers review) baked into the evaluation rules. Production risk engines need: per-account velocity limits (max N transactions per hour), per-asset exposure limits, anomaly detection on transaction graph topology, and dynamic risk scoring that can be updated at runtime without redeploying. This is how Stripe Radar, Chainalysis, and Elliptic detect fraud.

**Implementation**:
```python
async def set_velocity_limit(
    self, rule_id: str, tenant_id: str, ledger_id: str,
    account_id: str | None,  # None = applies to all accounts
    asset: str, max_amount: Decimal, window_seconds: int,
    actor: str = "risk-manager",
) -> dict[str, Any]: ...

async def check_velocity(
    self, tenant_id: str, ledger_id: str,
    account_id: str, asset: str, proposed_amount: Decimal,
) -> dict[str, Any]:
    """Return velocity window state and whether the proposed amount would breach limits."""
    ...

async def get_risk_score(
    self, tenant_id: str, account_id: str, ledger_id: str,
) -> dict[str, Any]: ...
```

**Competitor Reference**: Stripe Radar uses ML-based real-time risk scoring with velocity rules (N transactions per minute per card). Chainalysis Reactor maps transaction graph topology for risk attribution. Elliptic's wallet screening uses graph neural networks trained on labeled blockchain data.

---

## Improvement 15: WebSocket/SSE Event Streaming for Real-Time Dashboards

**Category**: Observability / User Experience

**Justification**: All current query methods are pull-based. Operators running the BCLG dashboard have no way to receive instant notification of high-value transactions, pending review alerts, consensus forks, or compliance flags. Push-based event streaming (Server-Sent Events or WebSocket) over the existing Flask-AppBuilder stack delivers sub-second latency for ledger events without polling overhead. This is how Etherscan, Infura, and Alchemy websocket endpoints work.

**Implementation**:
```python
async def subscribe_ledger_events(
    self, tenant_id: str, ledger_id: str | None = None,
    event_types: list[str] | None = None,
    after_sequence: int = 0,
) -> AsyncIterator[LedgerAuditEvent]:
    """Async generator yielding audit events as they occur — SSE/WS adapter point."""
    ...

async def get_event_cursor(
    self, tenant_id: str, ledger_id: str,
) -> dict[str, Any]:
    """Return current event sequence number for resumable subscription."""
    ...
```

**Competitor Reference**: Infura's WebSocket endpoint (`wss://mainnet.infura.io/ws/v3/`) streams Ethereum events in real time. Alchemy's `alchemy_pendingTransactions` subscription pushes mempool events. The Graph Protocol indexes on-chain events and exposes them via GraphQL subscriptions.
