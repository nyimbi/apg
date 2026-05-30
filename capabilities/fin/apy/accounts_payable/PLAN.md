# Accounts Payable Implementation Plan

## 1. Contract And Specification

- Define the AP capability intent, dependencies, provides list, UI routes, theme, stream metadata, and configuration schema.
- Encode AP guardrails as deterministic rules that can be evaluated without web, database, or queue dependencies.
- Keep Bytewax as the declared lifecycle processor for AP events and batches.

## 2. Executable Service

- Implement an in-memory tenant-scoped service for vendors, invoices, matching, approvals, holds, payments, payment batches, expenses, period close, AP agents, dashboard summaries, aging summaries, and audit events.
- Route all writes through the rule engine before state mutation.
- Emit audit events with Bytewax stream metadata after accepted lifecycle mutations.
- Preserve `APService` as a compatibility alias.

## 3. API, Views, And App Surface

- Provide dependency-light API helper functions over a singleton service.
- Provide view-model helpers for each published route.
- Publish an `app.py` entrypoint with semantic model, component manifest, and self-test methods that can run directly.

## 4. Documentation And Metadata

- Add `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
- Replace stale package notes with current APG package documentation.
- Regenerate `semantic_model.json`, `package_manifest.json`, and `release_report.json` from the executable app surface.

## 5. Focused Verification

- Compile the package modules and focused package test.
- Run the focused package test only.
- Run the package app self-test.
- Inspect the APG capability and publish plan.
- Run the package implementation audit.
- Confirm Bytewax, AP-agent, route, rule, and dependency metadata in the generated semantic model.
- Search touched APY package files for stale markers and unsupported stream naming.
