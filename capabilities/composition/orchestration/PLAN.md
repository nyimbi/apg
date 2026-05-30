# Workflow Orchestration Build Plan

## Phase 1 - Contract

- Replace generic spec-backed contract with explicit orchestration contract.
- Define provides/requires, configuration schema, deterministic rules, UI routes, theme, Bytewax streaming, and agent roles.

## Phase 2 - Service

- Provide dependency-light service methods for workflow definition, task validation, graph validation, releases, execution starts, task advancement, assignments, batch schedule validation, agent registration, and dashboard summaries.
- Keep live scheduler, worker, storage, and notification integrations behind adapters.

## Phase 3 - API and Views

- Publish small API helpers that generated applications can wrap with any Python web target.
- Publish view models for dashboard, definitions, designer, executions, tasks, releases, rules, agents, and navigation.

## Phase 4 - Package Evidence

- Regenerate semantic model, package manifest, and release report from the contract.
- Replace focused package tests with orchestration lifecycle tests.
- Run compile, focused tests, inspect, publish-plan, implementation audit, marker scan, and diff checks.

## Phase 5 - Review

- Review for accidental optional dependency imports on package surfaces.
- Review guardrail coverage against lifecycle requirements.
- Review Bytewax usage and AI agent role/runtime support.
