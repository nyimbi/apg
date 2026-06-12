# BCLG - Blockchain Ledger Services

BCLG provides governed distributed-ledger services for APG applications. It
covers tenant ledger registration, key-custody binding, signed transaction
submission, high-value transaction review, smart contract deployment and
lifecycle, deterministic hash evidence, token operations, NFT minting, wallet
management, DeFi protocol hooks, audit events, ZK-proof integration, multi-sig
governance, cross-tenant federation, event sourcing, real-time streaming,
Bytewax lifecycle stream metadata, AI ledger-agent registration, compliance
screening, UI route metadata, and visual theming.

The package is dependency-light. It does not require a live chain node, wallet
provider, HSM, custody provider, regulatory engine, web server, database, or
live Bytewax worker. Production deployments connect those systems through
adapters after BCLG has validated tenant context, ownership, custody,
signature, approval, stream, and audit guardrails.

## What BCLG Provides

- Tenant-qualified ledger network registry with owner, consensus profile,
  network policy, participants, fork-monitoring posture, and status.
- Key-custody bindings that gate all ledger mutations.
- Signed transaction submission with deterministic transaction and block hashes.
- High-value transaction review workflow (requester → reviewer → commit/reject).
- Smart contract compile, deploy, invoke, pause, unpause, and deprecate lifecycle.
- Token mint, burn, transfer, and supply tracking with circulating-supply accounting.
- NFT minting with content-addressed token hash and block anchor.
- Wallet creation with automatic key-custody binding.
- AI ledger-agent registration for `codex`, `claude_code`, `opencode`, and `pi`
  runtimes with explicit role, scope, disclosure, and policy evidence.
- Gas estimation for transactions based on amount and network load.
- Cross-chain bridge registration and certificate anchoring.
- Compliance reporting, bulk transaction submission, and transaction export.
- Consensus health monitoring per ledger.
- Bytewax lifecycle stream metadata for batch ledger mutation and generated
  application composition.
- API helpers and route-ready view models for generated APG Python applications.
- UI route metadata for dashboards, ledgers, transactions, review queues,
  contracts, key custody, agents, audit, analytics, compliance, and settings.

## Package Structure

- `SPECIFICATION.md` — functional scope, lifecycle rules, adapter boundaries,
  acceptance criteria.
- `PLAN.md` — implementation and review sequence.
- `cap_spec.md` — points older tooling to the active specification.
- `capability_contract.py` — configuration, guardrails, UI routes, theme,
  provides/requires metadata, and Bytewax stream metadata.
- `models.py` — tenant-scoped ledger records.
- `ledger_engine.py` — deterministic hash helpers.
- `service.py` — full executable lifecycle (v2.0).
- `api.py` — generated-application helper calls.
- `views.py` — route-ready UI state.
- `app.py`, `semantic_model.json`, `package_manifest.json`,
  `release_report.json` — package publication evidence.
- `tests/` — focused contract and package verification.

## Basic Usage

```python
from capabilities.common.bclg.service import BclgService

service = BclgService()
tenant_id = "tenant-ledger"

ledger = service.register_ledger(
    ledger_id="supply-chain-ledger",
    tenant_id=tenant_id,
    name="Supply Chain Ledger",
    owner="ledger-owner",
    consensus_profile="proof-of-authority",
    network_policy="tenant-private",
    participants=["warehouse", "procurement", "finance"],
)

custody = service.bind_key_custody(
    binding_id="custody-1",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    key_id="key-001",
    custodian="key-manager",
)

transaction = service.submit_transaction(
    transaction_id="txn-1",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    from_account="warehouse",
    to_account="supplier",
    amount=2500,
    asset="USD",
    signature="sig:warehouse:txn-1",
    key_custody_id=custody["id"],
)

assert transaction["status"] == "committed"
assert len(transaction["block_hash"]) == 64
```

## High-Value Transaction Review

High-value transactions (> 100,000) are held for review regardless of caller
input. They commit only after an independent approval from a reviewer who is
not the submitter.

```python
pending = service.submit_transaction(
    transaction_id="txn-high",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    from_account="treasury",
    to_account="supplier",
    amount=250000,
    asset="USD",
    signature="sig:treasury:txn-high",
    key_custody_id=custody["id"],
    actor="treasury-operator",
)

review = service.request_transaction_review(
    review_id="review-high",
    tenant_id=tenant_id,
    transaction_id=pending["id"],
    requested_by="treasury-operator",
    justification="High-value supplier settlement.",
)

approved = service.decide_transaction_review(
    review_id=review["id"],
    tenant_id=tenant_id,
    reviewer="risk-reviewer",
    decision="approved",
    notes="Invoice and limit checks passed.",
)

assert approved["status"] == "committed"
```

## Ledger-Agent Governance

AI agents must declare runtime, role, scope, and contribution disclosure before
appearing in generated application surfaces.

```python
agent = service.register_ledger_agent(
    agent_id="txn-review-agent",
    tenant_id=tenant_id,
    name="Transaction Review Agent",
    runtime="claude-code",
    role="transaction-reviewer",
    scope="Summarize high-value transaction review evidence.",
    contribution_disclosed=True,
    policy_ref="bclg-agent-policy",
)

assert agent["runtime"] == "claude_code"
assert agent["role"] == "transaction_reviewer"
```

## Bytewax Guardrail

Batch ledger mutation must declare the Bytewax lifecycle stream.

```python
service.validate_batch_ledger_mutation(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
)
```

## Composition Contract

`get_capability_contract()` returns the executable APG contract:

- `provides`: ledger registry, transaction governance, smart contract
  governance, key-custody governance, audit evidence, and ledger agents.
- `requires`: ENCR, KEYM, and COMP.
- `configuration`: ledger, transaction, smart contract, ledger-agent,
  governance, observability, adapter, UI, and theme settings.
- `rule_engine`: deterministic guardrails for tenant context, ledger owner,
  signatures, custody, reviews, contracts, agents, audit, and Bytewax batch
  mutation.
- `ui`: route metadata for generated APG Python applications.
- `theme`: compact ledger operations tokens and component metadata.
- `streaming`: Bytewax processor, topic, state collections, lifecycle events,
  and batch mutation guardrail.

---

## World-Class Enhancements (v2.0)

Fifteen production-grade improvements aligned with Bitcoin, Ethereum,
Hyperledger Fabric, Stellar, and institutional custody standards.

| # | Category | Improvement |
|---|----------|-------------|
| 1 | Cryptographic Integrity | **Merkle Tree Block Verification** — O(log n) inclusion proofs; `build_merkle_root` + `merkle_inclusion_proof`. Enables light-client verification without full block download. |
| 2 | Financial Correctness | **Decimal-Precision Arithmetic** — All monetary amounts use `decimal.Decimal` with `ROUND_HALF_EVEN`. Eliminates IEEE-754 float drift mandatory for regulated finance. |
| 3 | Architecture | **Event Sourcing / Append-Only Log** — Every state transition emits a `LedgerEvent` before mutating state. Enables `replay_state(up_to_sequence)` and `ledger_event_stream` for temporal queries and real-time consumers. |
| 4 | Privacy | **Zero-Knowledge Proof Hooks** — `attach_zk_proof` attaches a `proof_ref` (Groth16/PLONK/STARK) to any transaction. Enables GDPR-compliant confidential payments without revealing counterparty data. |
| 5 | Security | **Multi-Signature Threshold Signing** — `create_multisig_policy` / `submit_multisig_signature` / `finalize_multisig_transaction`. M-of-N threshold approval for institutional custody (SOC 2, ISO 27001 §A.9). |
| 6 | Compliance | **AML/KYC Compliance Screening** — `screen_transaction` blocks submissions pre-commit; `register_compliance_rule` adds velocity, sanctions, or threshold rules at runtime. |
| 7 | Performance | **Async-Native with asyncpg Backend** — All new methods use `async def`. Production path: `asyncpg.Pool` replaces in-memory dict for concurrent-safe, indexed persistence. |
| 8 | Smart Contracts | **Contract Lifecycle State Machine** — `pause_contract` / `unpause_contract` / `deprecate_contract`. Full `draft → audited → deployed → paused → deprecated → destroyed` progression with governance guards. |
| 9 | Tokenomics | **Token Supply Cap and Burn** — `token_burn` + `token_supply` return minted, burned, and circulating supply. Enforces governance-mandated supply limits (ERC-20 semantics). |
| 10 | Interoperability | **Cross-Tenant HTLC Atomic Swaps** — `create_htlc` / `claim_htlc` / `refund_htlc`. Hash Time-Locked Contracts for atomic bilateral settlement across tenant ledgers without a trusted intermediary. |
| 11 | Resilience | **Ledger Snapshot and Point-in-Time Recovery** — `create_snapshot` / `restore_from_snapshot` / `list_snapshots`. Content-addressed snapshots for disaster recovery, environment cloning, and forensic analysis. |
| 12 | Governance | **On-Ledger Governance Voting** — `create_governance_proposal` / `cast_vote` / `tally_votes`. Proposal → vote → tally → enact lifecycle for consortium parameter changes (IBFT/QBFT pattern). |
| 13 | DeFi | **AMM and Lending Protocol Adapters** — `amm_quote_swap` for Uniswap-style quoting; `lending_deposit` / `lending_borrow` for Aave-style collateralized lending via `invoke_contract`. |
| 14 | Risk Management | **Real-Time Velocity Risk Engine** — `set_velocity_limit` / `check_velocity` / `get_risk_score`. Per-account, per-asset sliding-window limits with dynamic runtime updates (Stripe Radar pattern). |
| 15 | Observability | **WebSocket/SSE Event Streaming** — `subscribe_ledger_events` is an async generator that yields `LedgerAuditEvent` objects as they occur, suitable for direct SSE or WebSocket adapters. `get_event_cursor` enables resumable subscriptions. |

---

## New Methods

### `verify_transaction` — Hash Integrity Check

```python
result = service.verify_transaction(
    tenant_id=tenant_id,
    transaction_id="txn-1",
)
# {"transaction_id": "txn-1", "valid": True, "stored_hash": "...", "computed_hash": "..."}
assert result["valid"] is True
```

### `wallet_create` — Wallet with Automatic Key Custody

```python
wallet = service.wallet_create(
    wallet_id="wallet-alice",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    owner="alice",
    wallet_type="standard",
)
# Returns address, public_key, custody_binding_id
txn = service.token_transfer(
    transaction_id="txn-transfer-1",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    from_account="alice",
    to_account="bob",
    amount=100,
    asset="TOKEN",
    signature="sig:alice:txn-transfer-1",
    key_custody_id=wallet["custody_binding_id"],
)
```

### `smart_contract_compile` + `invoke_contract` — Contract Lifecycle

```python
artifact = service.smart_contract_compile(
    artifact_id="artifact-erc20",
    tenant_id=tenant_id,
    source_code="pragma solidity ^0.8.20; contract Token { ... }",
    compiler_version="solidity-0.8.20",
    actor="developer",
)

# After deploy_contract(...) with approved review:
result = service.invoke_contract(
    invocation_id="call-1",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    contract_id="contract-erc20",
    method="transfer",
    args={"to": "0xabc", "amount": 500},
    actor="user",
)
assert result["status"] == "success"
```

### `certificate_anchor` — Immutable Document Provenance

```python
import hashlib

cert_hash = hashlib.sha256(b"<DER-encoded-cert>").hexdigest()
anchor = service.certificate_anchor(
    anchor_id="anchor-cert-001",
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    certificate_hash=cert_hash,
    issuer="PKI-Root-CA",
    actor="compliance-officer",
)
# anchor["block_hash"] is the immutable ledger proof of the certificate
```

### `compliance_report` — Automated Posture Assessment

```python
report = service.compliance_report(
    tenant_id=tenant_id,
    framework="iso27001",
)
# {"compliance_score": 85.0, "status": "compliant", "checks": [...]}
```

### `block_explorer` — Chain State Inspection

```python
explorer = service.block_explorer(
    tenant_id=tenant_id,
    ledger_id=ledger["id"],
    limit=50,
)
# {"chain_head": "<hash>", "committed_transaction_count": N, "transactions": [...]}
```

---

## API Reference

| Method | Description |
|--------|-------------|
| `register_ledger` | Create tenant ledger network |
| `list_ledgers` | List ledger networks |
| `bind_key_custody` | Bind a key to a ledger |
| `list_key_custody` | List custody bindings |
| `submit_transaction` | Submit and (if warranted) stage for review |
| `request_transaction_review` | Open review for pending transaction |
| `decide_transaction_review` | Approve or reject a pending review |
| `approve_transaction` | Compatibility helper: request + decide in one call |
| `list_transactions` | List all transactions |
| `list_transaction_reviews` | List all transaction reviews |
| `request_contract_deployment_approval` | Open deployment review |
| `decide_contract_deployment_approval` | Approve or reject deployment |
| `deploy_contract` | Deploy contract with approved review |
| `list_contracts` | List deployed contracts |
| `list_contract_deployment_approvals` | List deployment reviews |
| `smart_contract_compile` | Compile source, produce bytecode artifact |
| `invoke_contract` | Invoke a deployed contract method |
| `token_mint` | Mint tokens to an account |
| `token_transfer` | Transfer tokens (thin wrapper over submit_transaction) |
| `nft_mint` | Mint an NFT with content-addressed token hash |
| `wallet_create` | Create wallet with automatic custody binding |
| `digital_signature` | Produce HMAC-SHA256 signature record for a payload |
| `certificate_anchor` | Anchor certificate hash on-ledger |
| `verify_transaction` | Verify transaction hash integrity |
| `block_explorer` | Paginated committed transaction view |
| `cross_chain_bridge` | Register cross-chain bridge operation |
| `consensus_monitor` | Consensus health metrics for a ledger |
| `gas_estimate` | Fee estimate for a transaction |
| `audit_trail_verify` | Full audit trail for a subject |
| `list_audit_events` | List all audit events |
| `export_transactions` | Export transactions as JSON or CSV |
| `bulk_submit_transactions` | Submit a batch of transactions |
| `compliance_report` | Automated compliance posture report |
| `validate_batch_ledger_mutation` | Bytewax batch mutation guardrail |
| `register_ledger_agent` | Register an AI agent as ledger participant |
| `list_ledger_agents` | List registered agents |
| `ledger_summary` | Aggregate KPI counts |
| `health_check` | Service health status |
| `dashboard` | Combined summary + health |
| `describe` | Return capability contract |
| `evaluate` | Evaluate guardrail rules |
| `create_record` | Compatibility: auditable ledger note |
| `list_records` | Compatibility: alias for list_transactions |

---

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/bclg/__init__.py \
    capabilities/common/bclg/capability_contract.py \
    capabilities/common/bclg/models.py \
    capabilities/common/bclg/ledger_engine.py \
    capabilities/common/bclg/service.py \
    capabilities/common/bclg/api.py \
    capabilities/common/bclg/views.py \
    capabilities/common/bclg/app.py \
    capabilities/common/bclg/test_capability_contract.py \
    capabilities/common/bclg/tests/test_package_contract.py

./.venv/bin/pytest -q \
    capabilities/common/bclg/test_capability_contract.py \
    capabilities/common/bclg/tests/test_package_contract.py

./.venv/bin/python -c \
    "from capabilities.common.bclg import app; r=app.self_test(); print(r); assert r['passed']"

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
```

Full repository suites, live chain nodes, wallets, HSMs, custody providers,
regulatory engines, rendered browser UI, live Bytewax workers, and load tests
are separate integration concerns.

---

## Composability

BCLG composes with:

| Capability | Role |
|------------|------|
| **KEYM** | Key material for custody bindings, multisig, and ZK-proof keys |
| **ENCR** | Payload encryption before on-ledger storage |
| **COMP** | Composition event bus for ledger mutation triggers and downstream notification |

---

© 2025 Datacraft · www.datacraft.co.ke
