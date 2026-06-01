# Wealth Management Plan

## Build Order

1. Define the capability contract, configuration, deterministic rules, UI
   routes, theme, Bytewax lifecycle stream, provides, and dependencies.
2. Add models for client profiles, suitability profiles, portfolios, advisory
   mandates, rebalances, orders, performance snapshots, fee schedules, and
   evidence.
3. Implement a service layer that evaluates rules before state mutation.
4. Add API helper functions and view models for generated applications.
5. Add a publishable app entrypoint, semantic model, package manifest, and
   release report.
6. Add focused tests for contract shape, lifecycle execution, guardrails,
   API/view behavior, and publishability.
7. Run focused py_compile, pytest, app self-test, and APG audits.
8. Record verification and code review findings in the progress log.

## Review Focus

- Suitability evidence must exist before mandates and portfolio decisions.
- Rebalance allocations must be complete and total 100 percent.
- Large orders must require human approval before staging.
- Fee percentages must remain bounded.
- Live custody, broker routing, market data, billing, statements, and durable
  Bytewax workers must remain behind adapters.
