# Wealth Management Specification

## Purpose

Wealth Management gives APG generated applications a safe advisory and
portfolio-management surface. It must make suitability, mandate, policy,
approval, and audit requirements explicit before any client portfolio action is
recorded.

## Functional Scope

- Register client profiles with KYC, tax, and risk evidence.
- Capture suitability profiles with risk tolerance, investment horizon, and
  stated goals.
- Create portfolios with advisor assignment, base currency, and investment
  policy statement references.
- Create advisory mandates that bind portfolio actions to suitability evidence
  and mandate type.
- Propose portfolio rebalances with allocation targets and analysis evidence.
- Stage trade orders with instrument, side, quantity, approval, and risk
  references.
- Record performance snapshots with valuation and benchmark evidence.
- Record fee schedules with bounded advisory, performance, and platform fees.
- Register provider-neutral AI agents for advisor, suitability, portfolio,
  order, fee, and compliance review.

## Rules

The deterministic rule engine must reject missing tenant context, missing write
policy evidence, incomplete client/suitability/portfolio/mandate evidence,
unsupported mandate/order values, allocations that do not total 100 percent,
unapproved large orders, missing benchmark evidence, out-of-bounds fees,
non-Bytewax batch streams, unsupported agent runtimes/roles, and privileged AI
agent actions without human approval.

## UI And Theming

Generated applications expose dashboard, clients, suitability, portfolios,
mandates, rebalances, orders, performance, fees, agents, and settings routes.
The UI shell is `apg_python`, theming is mandatory, and the package publishes a
compact wealth-operations theme with 8px radius.

## Non-Goals

This package does not directly execute broker orders, calculate tax lots, hold
custody assets, collect invoices, render statements, connect live market data,
or run durable Bytewax topologies. Those operations remain adapter concerns.
