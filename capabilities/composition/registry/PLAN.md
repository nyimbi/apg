# Capability Registry Build Plan

## Phase 1 - Contract

- Replace generic spec-backed contract with explicit registry contract.
- Define provides/requires, configuration schema, deterministic rules, UI routes, theme, Bytewax streaming, and agent roles.

## Phase 2 - Service

- Provide dependency-light service methods for capability registration, dependency edges, composition blueprints, validation, version release, deprecation, marketplace publication, import validation, agent registration, and dashboard summaries.
- Keep durable catalog/search stores and external marketplace publishing behind adapters.

## Phase 3 - API and Views

- Publish small API helpers that generated applications can wrap with any Python web target.
- Publish view models for dashboard, catalog, dependencies, compositions, versions, marketplace, rules, agents, and navigation.

## Phase 4 - Package Evidence

- Regenerate semantic model, package manifest, and release report from the contract.
- Replace focused package tests with registry lifecycle tests.
- Run compile, focused tests, inspect, publish-plan, implementation audit, marker scan, and diff checks.

## Phase 5 - Review

- Review for optional dependency imports on package surfaces.
- Review guardrail coverage against catalog, dependency, composition, version, marketplace, and agent lifecycles.
- Review Bytewax usage and AI agent role/runtime support.
