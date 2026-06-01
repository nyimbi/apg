# APG Digital Lending Implementation Plan

## Packet 1: Contract And Domain Shape

- Define `fintech_lending` contract metadata, dependencies, configuration,
  deterministic rules, UI routes, theme tokens, and Bytewax lifecycle stream.
- Publish provider-neutral agent runtime and role configuration.
- Keep live credit and servicing providers behind adapter names.

## Packet 2: Executable Runtime

- Add dependency-light models for products, borrowers, applications,
  underwriting decisions, offers, disbursements, repayment schedules,
  collection cases, and evidence.
- Add `lending_runtime.py` helpers for normalization, score/rate conversion,
  installment estimation, and decision categorization.
- Implement `LendingService` methods that enforce rules before state changes and
  emit tenant-scoped audit events.

## Packet 3: Composition Surfaces

- Add API helpers that generated APG applications can call without importing a
  web stack.
- Add view-model helpers for dashboards, lending consoles, rules, theme, and
  route composition.
- Add `app.py` self-test, component manifest, and semantic model.

## Packet 4: Evidence And Review

- Add package manifest, release evidence, semantic model, local README, and
  focused tests.
- Update fintech capability metadata, catalog README counts, and progress log.
- Run focused verification: compile checks, package tests, app self-test, APG
  inspect, publish-plan, implementation audit, lifecycle audit, strict package
  audit, stale-marker scan, Bytewax terminology scan, and diff whitespace check.

## Code Review Checklist

- Every service mutation has tenant and policy context.
- Product, borrower, application, underwriting, offer, disbursement, repayment,
  collection, batch, and agent guardrails are enforced through the rule engine.
- Review-required outcomes fail closed until review evidence is supplied.
- Provider-specific integrations stay behind adapters.
- UI routes and theme tokens are complete enough for generated application
  composition.
- Tests cover both executable lifecycle and guardrail failures.
