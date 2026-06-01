# Digital Wallets Capability Development Plan

## Current Slice

Promote `capabilities/fintech/wallets` from an empty package into a first-class
APG capability with executable local wallet behavior.

## Implementation Steps

1. Replace empty package metadata with APG package exports.
2. Add a concrete contract with wallet configuration, deterministic rules,
   routes, theme, dependencies, and Bytewax streaming.
3. Add wallet, instrument, ledger, and evidence models.
4. Add service methods for wallet lifecycle, instruments, credits, debits,
   transfers, holds, agents, batches, and summaries.
5. Add process-local API helpers and framework-neutral view models.
6. Add publishable app self-test and semantic model.
7. Add focused package tests for lifecycle and guardrails.
8. Update package evidence and progress log.

## Adapter Boundary

Keep live banks, card networks, mobile-money providers, ledgers, notifications,
audit sinks, identity, key management, payment gateway, WALT, Digital Payments,
and durable Bytewax workers behind adapters.

## Review Checklist

- Amount math uses `Decimal`, not floats.
- Available balance honors held funds.
- Same-wallet, cross-currency, and over-limit transfers are guarded.
- Non-positive holds/releases and over-release attempts are denied.
- Non-Bytewax batches are denied.
- Agent runtimes include Codex, Claude Code, OpenCode, and Pi.
- UI routes and theme are contract-backed.
