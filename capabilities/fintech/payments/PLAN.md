# Digital Payments Capability Development Plan

## Current Slice

Promote `capabilities/fintech/payments` from an empty package into a
first-class APG capability with executable local lifecycle behavior.

## Implementation Steps

1. Replace empty package metadata with APG package exports.
2. Add a concrete capability contract with configuration, rules, routes, theme,
   provides, requires, and Bytewax streaming.
3. Add dependency-light payment models for accounts, instruments, orders, and
   lifecycle evidence.
4. Add a service runtime for account, instrument, order, risk, authorization,
   capture, refund, payout, settlement, dispute, batch, summary, and agent
   lifecycle operations.
5. Add process-local API helpers and framework-neutral view models.
6. Add publishable app self-test and semantic model.
7. Add focused package tests that prove lifecycle execution and guardrails.
8. Update release/package/semantic artifacts.
9. Run focused verification and record progress evidence.

## Adapter Boundary

Keep card networks, mobile-money providers, banks, wallets, vaults, ledgers,
AUDL, NTFY, KEYM, ENCR, cash management, accounts receivable, and live Bytewax
workers behind adapters. The package runtime must stay importable and useful
without those services.

## Review Checklist

- Tenant and policy guardrails deny missing context.
- Financial amounts use `Decimal`, not floats.
- High-risk, blocked-risk, overcapture, overrefund, settlement variance, and
  non-Bytewax batch cases are covered.
- AI agent runtimes include Codex, Claude Code, OpenCode, and Pi.
- UI routes and theme are contract-backed.
- No Bytewax dependency is introduced.
