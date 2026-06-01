# Algorithmic Trading Build Plan

## Slice

Promote `capabilities/fintech/trading` from a placeholder into an executable
APG capability package that matches the current fintech package standards.

## Implementation Steps

1. Replace placeholder metadata with a package export surface.
2. Add a capability contract with configuration, rules, UI routes, theme tokens,
   dependencies, provided workflows, and Bytewax lifecycle metadata.
3. Add dependency-light dataclasses for strategies, signals, backtests, risk
   limits, order intents, executions, position snapshots, surveillance alerts,
   reviews, and AI-agent evidence.
4. Add a service that enforces deterministic rules before mutating in-memory
   state.
5. Add API helpers, route/view models, and a publishable app entrypoint.
6. Add package docs, capability spec, manifest, semantic model, release report,
   and focused tests.
7. Update the fintech registry, catalog README, and progress log.
8. Run focused compile, package tests, APG inspect/publish/audit commands,
   scans, and whitespace checks.

## Verification Strategy

Battery-conscious verification uses package tests and APG audits rather than
the full repository suite. The package must pass py_compile, focused pytest,
self-test, inspect, publish-plan, implementation audit, lifecycle audit, global
implementation audit, strict package audit, stale-marker scan, messaging scan,
and `git diff --check`.

## Review Notes

Review must confirm that state mutation follows rule evaluation, strategy and
order controls require evidence, risk limits require approval, AI runtimes
remain provider-neutral, privileged agent actions require human approval, and
live venue/market-data/custody concerns remain behind adapters.
