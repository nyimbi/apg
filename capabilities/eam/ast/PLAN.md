# Enterprise Asset Management Development Plan

## Slice Goal

Deliver a coherent lifecycle and guardrail packet for `eam_ast` so the capability is executable, documented, testable, theme-aware, AI-agent aware, and Bytewax-aligned.

## Implementation Steps

1. Replace the generic contract with a domain-specific EAM contract for locations, assets, maintenance plans, work orders, inspections, condition readings, inventory reservations, agents, governance, observability, adapters, UI, theme, provides/requires, and Bytewax lifecycle streaming.
2. Replace dependency-heavy runtime surfaces with dependency-light service, API, view, and app modules that compile without optional web or database services.
3. Preserve compatibility names such as `EAMAssetService`, `create_record`, and `list_records` while routing new behavior through `EnterpriseAssetManagementService`.
4. Add deterministic guardrails for tenant context, write policy, asset integrity, maintenance planning, safety approval, inspection quality, condition alerts, inventory quantities, Bytewax routing, and AI-agent approval.
5. Refresh package metadata and semantic evidence from the active contract.
6. Add README and specification documents that explain purpose, lifecycle, guardrails, APIs, UI, theming, streaming, and verification.
7. Expand focused package tests around contract shape, rule execution, service lifecycle, guardrail failures, API/view surfaces, and semantic metadata.
8. Run battery-conscious verification: compile touched package files, run the focused EAM package tests, inspect package metadata, and scan touched files for stale marker terms.

## Review Checklist

- Tenant context is enforced on write operations.
- Assets cannot be registered without owner, category, location, criticality, and fixed-asset reference when capitalized.
- Health scores are bounded between 0 and 100.
- Predictive maintenance plans require a condition source.
- Critical-asset work orders require approval and safety plan.
- Condition alerts require review.
- Inventory reservations require positive quantities.
- Batch imports and lifecycle events require Bytewax routing.
- EAM agents support Codex, Claude Code, OpenCode, and Pi.
- UI models expose useful operational data without importing unavailable web frameworks.
- Package imports remain dependency-light.

## Known Deferred Work

- Bind fixed-asset, procurement, notification, audit, and authorization adapters to live APG services.
- Deploy durable Bytewax topologies and persistent event stores.
- Render browser UI screens and run visual checks.
- Add performance, concurrency, and failure-recovery validation after the capability family is stabilized.

