# Advanced CRM Analytics Build Plan

## Phase 1 - Contract

- Replace generic spec-backed contract with explicit advanced CRM contract.
- Define provides/requires, configuration schema, deterministic rules, UI routes, theme, Bytewax streaming, and agent roles.

## Phase 2 - Service

- Provide dependency-light service methods for accounts, contacts, leads, lead assignment, opportunities, activities, campaigns, forecasts, import validation, CRM agent registration, and dashboard summaries.
- Keep durable CRM stores, analytics engines, campaign delivery, and notification delivery behind adapters.

## Phase 3 - API and Views

- Publish small API helpers that generated applications can wrap with any Python web target.
- Publish view models for dashboard, accounts, contacts, leads, pipeline, activities, campaigns, forecasts, agents, and navigation.

## Phase 4 - Package Evidence

- Regenerate semantic model, package manifest, and release report from the contract.
- Replace focused package tests with advanced CRM lifecycle tests.
- Run compile, focused tests, inspect, publish-plan, implementation audit, marker scan, and diff checks.

## Phase 5 - Review

- Review for optional dependency imports on package surfaces.
- Review guardrail coverage against account, contact, lead, opportunity, campaign, forecast, and agent lifecycles.
- Review Bytewax usage and AI agent role/runtime support.
