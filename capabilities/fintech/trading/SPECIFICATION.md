# Algorithmic Trading Specification

## Purpose

Algorithmic Trading gives generated APG applications a first-class trading
governance component. It manages strategy approval, signal lineage, backtest
evidence, risk limits, order-intent staging, execution evidence, position
snapshots, surveillance alerts, reviews, and AI-agent guardrails without
requiring live market connectivity.

## Actors

- Quant and trading teams register strategies, signals, backtests, order
  intents, executions, and positions.
- Risk and compliance teams set limits, review surveillance alerts, and approve
  sensitive lifecycle changes.
- AI agents assist with strategy review, signal-quality checks, backtest
  analysis, risk review, order checks, surveillance triage, and compliance
  summaries.

## Functional Requirements

1. Register tenant-scoped strategies with owner, name, strategy type, asset
   class, and policy reference.
2. Attach signal sources only to existing tenant strategies, with source,
   freshness, and lineage evidence.
3. Record backtests only for existing tenant strategies, with period, positive
   trade count, data source, and metrics.
4. Set risk limits only for existing tenant strategies, with metric, positive
   limit value, and approval evidence.
5. Stage order intents only when strategy, risk limit, supported order type,
   instrument, positive quantity, and approval evidence exist.
6. Record executions only for existing order intents, supported venue, positive
   filled quantity, and source evidence.
7. Record position snapshots with strategy, as-of date, exposures, and source
   evidence.
8. Record surveillance alerts with supported severity and evidence.
9. Record review decisions with supported status, reviewer, and evidence.
10. Register provider-neutral AI agents with supported runtimes and roles.
11. Deny privileged AI-agent actions unless human approval is recorded.
12. Publish APG UI routes, theme tokens, deterministic rules, Bytewax lifecycle
    metadata, semantic model, package manifest, release report, and tests.

## Rule Engine

Rules are deterministic. Service methods build a context, evaluate the
capability rule engine, and raise `PermissionError` before mutating state when
a deny rule matches. Tenant context and write-policy evidence are universal
mutation guardrails.

## Non-Goals

This package does not directly connect to trading venues, consume live market
data, route orders, settle custody instructions, calculate official tax lots,
file regulator reports, or run durable stream workers. Those concerns remain
adapter-backed integration boundaries.
