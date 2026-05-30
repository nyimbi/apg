# BCLG - Blockchain Ledger Services

BCLG provides governed distributed-ledger services for APG applications. It
covers tenant ledger registration, key-custody binding, signed transaction
submission, high-value transaction review, smart contract deployment approval,
deterministic hash evidence, audit events, Bytewax lifecycle stream metadata,
AI ledger-agent registration, UI route metadata, and visual theming.

The package is dependency-light. It does not require a live chain node, wallet
provider, HSM, custody provider, regulatory engine, web server, database, or
live Bytewax worker. Production deployments connect those systems through
adapters after BCLG has validated tenant context, ownership, custody,
signature, approval, stream, and audit guardrails.

## What BCLG Provides

- Tenant-qualified ledger network registry with owner, consensus profile,
  network policy, participants, fork-monitoring posture, and status.
- Key-custody bindings that gate ledger mutation.
- Signed transaction submission with deterministic transaction and block
  hashes.
- High-value transaction review workflow with requester, reviewer, decision,
  notes, and commit/reject behavior.
- Smart contract deployment approval workflow with artifact hash, rollback
  plan, independent review, and deterministic deployment hash.
- AI ledger-agent registration for `codex`, `claude_code`, `opencode`, and
  `pi` runtimes with explicit role, scope, disclosure, and policy evidence.
- Bytewax lifecycle stream metadata for batch ledger mutation and generated
  application composition.
- API helpers and route-ready view models for generated APG Python
  applications.
- UI route metadata for dashboards, ledgers, transactions, review queues,
  contracts, key custody, agents, audit, analytics, compliance, and settings.

## Package Structure

- `SPECIFICATION.md` defines functional scope, lifecycle rules, adapter
  boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review sequence for this packet.
- `cap_spec.md` points older tooling to the active specification.
- `capability_contract.py` declares configuration, guardrails, UI routes,
  theme, provides/requires metadata, and Bytewax stream metadata.
- `models.py` defines tenant-scoped ledger records.
- `ledger_engine.py` provides deterministic hash helpers.
- `service.py` implements the executable lifecycle.
- `api.py` exposes generated-application helper calls.
- `views.py` exposes route-ready UI state.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` provide package publication evidence.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide
  focused verification.

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

High-value transactions are held for review even if a caller supplies a review
boolean. They commit only after an explicit independent approval.

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

BCLG treats AI ledger agents as governed participants in review workflows. An
agent must declare a supported runtime, supported role, explicit scope, and
contribution disclosure before it can be shown in generated application
surfaces.

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

## Verification

Focused checks for this package:

```bash
./.venv/bin/python -m py_compile capabilities/common/bclg/__init__.py capabilities/common/bclg/capability_contract.py capabilities/common/bclg/models.py capabilities/common/bclg/ledger_engine.py capabilities/common/bclg/service.py capabilities/common/bclg/api.py capabilities/common/bclg/views.py capabilities/common/bclg/app.py capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bclg import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
```

Full repository suites, live chain nodes, wallets, HSMs, custody providers,
regulatory engines, rendered browser UI, live Bytewax workers, and load tests
are separate integration concerns.
