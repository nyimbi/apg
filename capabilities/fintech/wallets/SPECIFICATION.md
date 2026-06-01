# Digital Wallets Capability Specification

## Purpose

Digital Wallets gives generated APG applications an executable wallet layer for
consumer, merchant, agent, escrow, and treasury wallet experiences. It must be
locally runnable, tenant-scoped, deterministic, themeable, and ready to compose
with common WALT and Digital Payments capabilities.

## Scope

The capability owns:

- wallet lifecycle and ownership;
- wallet instrument registration;
- stored-value ledger entries;
- wallet credit, debit, transfer, hold, and release workflows;
- wallet limit governance;
- provider-neutral wallet-agent registration;
- deterministic rule evaluation;
- APG Python UI route/view-model metadata;
- Bytewax lifecycle stream metadata.

Live bank, card, mobile-money, ledger, notification, audit, identity, key, and
payment-provider side effects are adapters, not hard runtime dependencies.

## Functional Requirements

1. Every write must include tenant context and policy evidence.
2. Wallets require owner references, supported wallet types, and supported
   currencies.
3. Instruments require tenant-local wallets, supported types, token references,
   and verification evidence.
4. Credits, debits, holds, and hold releases require positive amounts.
5. Debits and holds cannot exceed available balance.
6. Hold releases cannot exceed held balance.
7. Transfers require distinct wallets, matching wallet currencies, and
   transfer-limit review policy when configured thresholds are exceeded.
8. Lifecycle batches must use Bytewax.
9. AI wallet agents must use supported runtimes and roles.
10. Privileged agent actions require human approval evidence.

## UI And Theming

The APG Python UI surface exposes dashboard, wallets, instruments, ledger,
limits, holds, agents, and settings screens. Theme tokens use compact density,
8px radius, and distinct operational status colors.

## Acceptance Evidence

The package is acceptable when focused py_compile, pytest, app self-test,
inspect, publish-plan, implementation-audit, lifecycle-audit, stale-marker scan,
and diff checks pass for `capabilities/fintech/wallets`.
