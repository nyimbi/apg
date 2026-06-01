# Decentralized Finance Capability Specification

## Purpose

`fintech_defi` makes decentralized-finance workflows first-class APG
application components. It provides a dependency-light package that generated
Python applications can compose with Blockchain Services, Cryptocurrency
Services, wallets, compliance, risk, RegTech, AML, KYC, audit, notifications,
NLP, and key-management capabilities.

The package is intentionally executable without live chain credentials. It
models protocol registration, position tracking, governed actions, yield
strategies, rewards, governance votes, risk assessments, reviews, Bytewax
lifecycle metadata, UI/view composition, visual theming, and provider-neutral
AI-agent registration.

## Users

- Treasury and digital-asset operations teams that need governed DeFi actions.
- Risk and compliance teams that need deterministic policy checks before
  protocol, position, action, reward, governance, and review records mutate.
- Product teams composing crypto, wallet, blockchain, compliance, and DeFi
  screens into generated APG applications.
- AI-agent teams that need explicit runtime, role, scope, and human-approval
  boundaries for DeFi automation.

## Functional Scope

The capability owns these workflows:

- Protocol registry for lending pools, liquidity pools, staking, yield vaults,
  DEXs, bridges, derivatives, and insurance pools.
- Position ledger for supply, borrow, liquidity, stake, vault-share, long,
  short, and cover positions.
- Governed action queue for deposits, withdrawals, borrows, repayments, swaps,
  staking, unstaking, reward claims, and rebalancing.
- Yield strategy registry with target APY and max risk tier controls.
- Reward accrual ledger for interest, fee share, staking reward, liquidity
  mining, and governance rewards.
- Governance vote recording for for, against, and abstain choices.
- Risk assessment and review workflows with evidence references.
- Provider-neutral AI-agent registration for Codex, Claude Code, OpenCode, and
  Pi runtimes.
- Bytewax lifecycle stream metadata for generated applications and workers.

## Rule Engine

The deterministic rule engine denies unsafe operations before mutation. It
guards tenant context, policy attachment, supported protocol/action/position/
reward/vote/risk/review values, required references, positive or non-negative
amounts, evidence capture, Bytewax routing, supported agent runtimes and roles,
and human approval for privileged agent actions.

Rules are declared in `capability_contract.py` and enforced by `service.py`.
Generated applications may evaluate the same rules without calling service
methods.

## UI And Theming

The capability publishes compact APG UI metadata for:

- Dashboard
- Protocol registry
- Position console
- Action queue
- Yield strategy workbench
- Reward ledger
- Governance console
- Risk console
- Review console
- AI-agent workbench
- Settings

Theme tokens use restrained operational colors, 8px radius, compact density,
and component status indicators for protocol, position, action, strategy,
reward, governance, risk, review, and agent views.

## Adapter Boundaries

The executable package deliberately does not perform live chain RPC, protocol
calls, oracle reads, custody operations, private-key handling, MEV mitigation,
bridge execution, liquidation execution, governance submission, or durable
Bytewax worker execution. Those remain behind adapters so generated
applications can substitute production providers without changing the APG
capability contract.

## Acceptance Criteria

- `fintech_defi` appears as a valid capability contract.
- The package exposes README, specification, plan, capability spec, manifest,
  semantic model, release evidence, service, API helpers, views, app, and tests.
- Service methods enforce deterministic guardrails before mutation.
- UI routes include an AI-agent workbench at `/fintech-defi/agents`.
- Streaming metadata uses Bytewax and stream `apg.fintech.defi.lifecycle`.
- Focused tests and APG audits pass without touching unrelated dirty paths.
