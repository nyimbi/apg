# BCLG Capability Specification

## Identity

- Capability ID: `bclg`
- Display name: Blockchain Ledger Services
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `bclg_ledger_ops`

## Purpose

BCLG is the tenant-scoped distributed-ledger control plane for APG
applications. It governs ledger-network registration, key-custody binding,
signed ledger transaction submission, high-value transaction review, smart
contract deployment approval, deterministic hash evidence, block-head updates,
audit evidence, UI route metadata, and visual theming.

The package must remain usable without a live blockchain node, HSM, wallet
provider, external custody service, regulatory reporting engine, or production
web server. Those systems remain adapter boundaries. Local package proof
focuses on deterministic ledger governance, tenant isolation, approval state,
ledger mutation guardrails, view models, semantic evidence, and composition
behavior.

## Users And Outcomes

- Application builders can compose a ledger runtime without importing a live
  chain node or custody provider.
- Finance and supply-chain teams can record signed asset transfers with
  deterministic transaction and block hashes.
- Security owners can require active key-custody binding before ledger
  mutation.
- Risk reviewers can approve or reject high-value transactions with reviewer
  evidence and notes.
- Smart contract owners can request and approve deployment with artifact hash,
  rollback plan, and independent review evidence.
- Ledger operators can register AI ledger agents for transaction, contract,
  custody, ledger, and fork review support while preserving runtime, role,
  scope, disclosure, and audit evidence.
- Auditors can inspect tenant-scoped ledger, custody, transaction, contract,
  review, and audit state.

## Domain Model

BCLG owns these package-level records:

- `LedgerNetwork`: tenant ledger network, owner, consensus profile, network
  policy, participants, fork-monitoring posture, and status.
- `KeyCustodyBinding`: managed key-custody binding for a tenant ledger.
- `LedgerTransaction`: signed transaction with deterministic transaction hash,
  review state, approval evidence, and block hash.
- `TransactionReviewApproval`: independent review request and decision for a
  high-value transaction.
- `SmartContractArtifact`: deployed smart contract artifact with deployment
  hash and approval evidence.
- `ContractDeploymentApproval`: independent review request and decision for a
  smart contract deployment.
- `LedgerAgent`: governed AI ledger-agent registration with tenant, runtime,
  role, scope, disclosure, policy, and status evidence.
- `LedgerAuditEvent`: tenant-scoped governance event for ledger, custody,
  transaction, review, contract, block, and compatibility actions.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

The focused lifecycle is:

1. Register tenant ledgers with owner, consensus profile, network policy,
   participants, and fork-monitoring posture.
2. Bind active key-custody evidence to a tenant ledger before ledger mutation.
3. Submit signed transactions against active custody.
4. Commit standard transactions immediately with deterministic transaction and
   block hashes.
5. Hold high-value transactions in `pending_review` until a review decision is
   recorded.
6. Approve or reject high-value transaction review with an independent reviewer
   and notes.
7. Commit approved reviewed transactions and keep rejected transactions out of
   the ledger head.
8. Request smart contract deployment approval with artifact hash, rollback
   plan, and requester evidence.
9. Approve or reject smart contract deployment with an independent reviewer and
   notes.
10. Deploy contracts only when the ledger, custody binding, artifact hash,
    rollback plan, and approval guardrails pass.
11. Register ledger agents only when they use a supported runtime, supported
    role, explicit scope, registration state, and contribution disclosure.
12. Validate batch ledger mutation intent against the Bytewax lifecycle stream.
13. List tenant ledger, custody, transaction, review, contract, agent, and audit
    state.
14. Emit tenant-scoped audit events for every lifecycle transition.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: all ledger operations require tenant context.
- `ledger_requires_owner`: ledger creation requires an accountable owner.
- `transaction_requires_signature`: ledger transactions require signatures.
- `key_custody_required`: mutating ledger operations require active custody.
- `contract_requires_review`: contract deployment requires approved review
  evidence.
- `high_value_transaction_requires_review`: high-value transactions require
  review before commit.
- `transaction_review_requires_independent_reviewer`: transaction reviewers
  cannot approve their own submitted transactions.
- `contract_deployment_review_requires_independent_reviewer`: contract
  deployment reviewers cannot approve their own requests.
- `ledger_agent_requires_registration`: AI ledger agents must be registered
  before use in review workflows.
- `ledger_agent_runtime_supported`: AI ledger agents must use a supported
  runtime: `codex`, `claude_code`, `opencode`, or `pi`.
- `ledger_agent_role_supported`: AI ledger agents must use a supported role.
- `ledger_agent_requires_scope`: AI ledger agents must declare explicit
  operating scope.
- `ledger_agent_requires_disclosure`: AI-assisted ledger contributions must be
  disclosed.
- `ledger_state_change_requires_audit`: BCLG lifecycle state changes require
  audit evidence.
- `batch_ledger_mutation_requires_bytewax`: batch ledger mutation intent must
  use Bytewax event streams.

Service methods must enforce these rules and expose the same lifecycle through
API helpers and view models. Caller-supplied booleans are not sufficient
approval evidence for high-value transactions or contract deployment.

## UI And Theme

BCLG exposes route and view-model surfaces for:

- dashboard;
- ledger console;
- key-custody matrix;
- transaction monitor;
- transaction review queue;
- smart contract registry;
- contract deployment review queue;
- ledger-agent panel;
- audit and compliance surfaces;
- analytics dashboard;
- settings and tenant configuration.

The `bclg_ledger_ops` theme must provide semantic tokens and component
metadata for ledger cards, transaction monitors, transaction review queues,
contract registries, contract approval queues, key custody, and audit trails.

## Ledger-Agent Composition

BCLG supports first-class AI ledger-agent composition without invoking agent
CLIs or provider SDKs inside the dependency-light package. The package records
the agent registration and validates runtime, role, scope, disclosure, and
tenant guardrails. Production adapters can later bind registrations to Codex,
Claude Code, OpenCode, Pi, or other approved runtimes through platform
orchestration.

Supported roles are:

- `ledger_reviewer`;
- `transaction_reviewer`;
- `contract_reviewer`;
- `custody_reviewer`;
- `fork_reviewer`.

Generated applications must display agent scope and contribution disclosure in
review surfaces so human approvers can distinguish agent-assisted summaries
from direct reviewer decisions.

## Streaming

BCLG publishes Bytewax lifecycle stream metadata through the capability
contract and generated semantic model:

- processor: `bytewax`;
- topic: `apg.bclg.lifecycle`;
- state collections: ledgers, key custody, transactions, transaction reviews,
  contract reviews, contracts, ledger agents, audit events, and ledger heads;
- events: ledger registration, key-custody binding, transaction submission,
  review decisions, contract deployment decisions, contract deployment, and
  ledger-agent registration;
- batch mutation guardrail: `batch_ledger_mutation_requires_bytewax`.

## Adapter Boundaries

These integrations remain replaceable:

- live blockchain nodes and chain anchoring providers;
- wallet settlement providers;
- hardware security modules and key-management services;
- compliance/regulatory reporting engines;
- audit ledger and notification integrations;
- production FastAPI, Flask, or Flask-AppBuilder surfaces.
- live Bytewax topology execution.

Local package tests must not require those systems.

## Acceptance Gates

- Contract validation passes.
- The dependency-light service runs the full ledger-custody-transaction-review
  contract-deployment-audit lifecycle.
- High-value transaction submission cannot be committed by caller-supplied
  `transaction_review_recorded=True`; it requires explicit review approval.
- Rejected high-value transactions remain rejected and never update the ledger
  head.
- Only one pending review can exist for a high-value transaction, and decided
  reviews cannot later mutate rejected or committed transactions.
- Contract deployment fails without approved matching deployment approval.
- Approval decisions fail when reviewer identity or notes are missing or the
  reviewer is not independent.
- Tenant-qualified state allows duplicate IDs across tenants without collision.
- API helpers and view models expose the same lifecycle state.
- Ledger agents can be registered with supported runtime, supported role,
  scope, disclosure, and policy evidence.
- Unsupported ledger-agent runtime, missing scope, or undisclosed agent
  contribution fails closed.
- Batch ledger mutation validation accepts Bytewax and denies other stream
  providers.
- Generated semantic model exposes ledger-agent route, provides/requires
  metadata, and Bytewax stream metadata.
- Publish-plan and implementation-audit checks pass.
- Legacy generated-package naming is removed from package tests.

## Focused Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/bclg/__init__.py capabilities/common/bclg/capability_contract.py capabilities/common/bclg/models.py capabilities/common/bclg/ledger_engine.py capabilities/common/bclg/service.py capabilities/common/bclg/api.py capabilities/common/bclg/views.py capabilities/common/bclg/app.py capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bclg import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
```
