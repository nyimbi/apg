# Robo Advisory Plan

## Build Order

1. Define the capability contract, configuration, deterministic rules, UI
   routes, theme, Bytewax lifecycle stream, provides, and dependencies.
2. Add models for investor profiles, goals, model portfolios, recommendations,
   automated plans, drift records, tax-loss candidates, review records, and
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

- Model allocations must total 100 percent.
- Recommendations must bind profile, goal, model, and analysis evidence.
- Automation must require approved recommendation evidence.
- Drift and tax-loss workflows must have analysis/tax-lot evidence.
- Live brokerage, market data, custody, statements, billing, and durable
  Bytewax workers must remain behind adapters.
