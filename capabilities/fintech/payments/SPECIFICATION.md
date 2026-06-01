# Digital Payments Capability Specification

## Purpose

Digital Payments gives APG-generated applications a terse but complete payment
composition surface. The capability must be executable locally, deterministic,
tenant-scoped, policy-guarded, visually themeable, and ready to bind live
payment providers behind adapters.

## Scope

The capability owns:

- payment account lifecycle;
- instrument/token reference registration;
- payment order creation;
- payment risk screening evidence;
- authorization, capture, refund, and payout state;
- settlement reconciliation evidence;
- payment dispute evidence;
- provider-neutral AI payment-agent registration;
- deterministic rule evaluation;
- APG Python UI route/view-model metadata;
- Bytewax lifecycle stream metadata.

The capability does not own live card-network, mobile-money, bank, wallet,
ledger, notification, vault, or audit side effects. Those are adapters.

## Functional Requirements

1. Every write must include tenant context and policy evidence.
2. Accounts must include owner references and supported currencies.
3. Instruments must belong to tenant-local accounts and carry token references.
4. Payment orders must use positive amounts, supported currencies, accounts,
   and instruments.
5. High-risk payments require review evidence before proceeding.
6. Blocked payments cannot be authorized.
7. High-value payments require approval evidence before authorization.
8. Captures cannot exceed authorized amount.
9. Refunds cannot exceed captured balance.
10. Payouts require destination references.
11. Settlement variance requires review evidence.
12. Disputes require accountable owners.
13. Lifecycle batches must use Bytewax.
14. AI payment agents must use supported runtimes and roles.
15. Privileged agent actions require human approval evidence.

## Configuration

Configuration includes tenant ID, account policy, instrument policy, order
policy, risk policy, money-movement rules, agent rules, governance, observability
metadata, adapters, UI settings, and theme settings.

## Rule Engine

Rules are deterministic dictionaries in `capability_contract.py` and return
`allow`, `require_review`, or `deny` decisions with matched rules and effects.
Generated applications can evaluate the same rules before calling service
methods or rely on service guardrails.

## UI And Theming

The APG Python UI surface exposes dashboard, account, instrument, order, risk,
settlement, dispute, agent, and settings screens. Theme tokens include an 8px
border radius, distinct primary/accent/status colors, compact density, and
component descriptors for repeated controls.

## Streaming

Lifecycle streams use Bytewax metadata:

- stream: `apg.fintech.payments.lifecycle`
- processor: `bytewax`
- key: `tenant_id`

## Acceptance Evidence

The package is acceptable when focused py_compile, pytest, app self-test,
inspect, publish-plan, implementation-audit, lifecycle-audit, stale-marker scan,
and diff checks pass for `capabilities/fintech/payments`.
